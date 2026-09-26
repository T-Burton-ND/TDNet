"""Focused guards for manifest identity, ledger resume, response cap, and timing."""

import json
import importlib.util
from pathlib import Path

import pandas as pd
import requests
import pytest

from gridiron_ml.experiments.nextgen_temporal import next_game_rows
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (
    AcquisitionLedger, build_manifest, execute_request, make_request, materialize_plan,
    quota_allows, request_identity, verify_cache,
)

_executor_spec = importlib.util.spec_from_file_location(
    "nextgen_cfbd_acquire", Path(__file__).resolve().parents[1] / "scripts/nextgen_cfbd_acquire.py")
_executor = importlib.util.module_from_spec(_executor_spec)
_executor_spec.loader.exec_module(_executor)
validate_stage_e_approval = _executor.validate_stage_e_approval


def test_stage_e_gate_requires_sample_unique_features_and_explicit_choice():
    approval = {"decision": "subset", "approved_game_ids": [101],
                "sample_coverage_and_stat_meaning_verified": True,
                "response_cap_handling_verified": True,
                "incremental_value_reviewed": True,
                "quota_and_storage_reviewed": True,
                "unique_f10_f12_features": [{"generation": "F10", "name": "player_quarter_rush_share",
                                              "uniquely_enabled_by_plays_stats": True}]}
    assert validate_stage_e_approval(approval) == {101}
    for key, bad_value in (("decision", "skip"), ("approved_game_ids", []),
                           ("sample_coverage_and_stat_meaning_verified", False),
                           ("unique_f10_f12_features", [{"generation": "F10", "name": "x"}])):
        with pytest.raises(ValueError):
            validate_stage_e_approval({**approval, key: bad_value})


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.api_calls = 0

    def get_json(self, endpoint, params, max_retries=2):
        self.api_calls += 1
        return self.payload


def item(tmp_path: Path, cap=None):
    params = {"year": 2025, "week": 1, "seasonType": "regular"}
    rid = request_identity("/plays", params, "2025-w01")
    return {"request_id": rid, "endpoint": "/plays", "parameters": params,
            "parameters_json": json.dumps(params, sort_keys=True), "partition": "2025-w01",
            "schema_version": "cfbd_openapi_5.30.0_parquet_v1", "year": 2025,
            "week": 1, "game_id": None, "team": None, "planned": True,
            "status": "planned", "response_row_cap": cap,
            "cache_path": str(tmp_path / "plays.parquet")}


def test_request_identity_is_order_invariant_and_partition_sensitive():
    a = request_identity("/plays", {"year": 2025, "week": 1}, "2025-w01")
    b = request_identity("/plays", {"week": 1, "year": 2025}, "2025-w01")
    c = request_identity("/plays", {"year": 2025, "week": 1}, "2025-w02")
    assert a == b and a != c


def test_ledger_complete_cache_resume_and_hash_check(tmp_path):
    request = item(tmp_path)
    ledger = AcquisitionLedger(tmp_path)
    client = FakeClient([{"id": 1, "season": 2025, "week": 1, "yardsGained": 7}])
    first = execute_request(request, ledger, client)
    assert first["status"] == "success_complete"
    assert first["row_count"] == 1 and first["byte_size"] > 0
    assert first["schema_hash"] and first["sha256"]
    assert verify_cache(Path(first["cache_path"]), first)
    second = execute_request(request, ledger, client)
    assert second["request_id"] == first["request_id"] and client.api_calls == 1
    Path(first["cache_path"]).write_bytes(b"corrupt")
    assert verify_cache(Path(first["cache_path"]), first) is None
    execute_request(request, ledger, client)
    assert client.api_calls == 2


def test_cap_response_never_complete(tmp_path):
    request = item(tmp_path, cap=2)
    client = FakeClient([{"id": 1}, {"id": 2}])
    record = execute_request(request, AcquisitionLedger(tmp_path), client)
    assert record["status"] == "success_suspected_partial"
    assert record["completeness_status"] == "response_cap_reached"
    assert materialize_plan([request], AcquisitionLedger(tmp_path))["preserved_terminal"] == 1
    assert AcquisitionLedger(tmp_path).read(request["request_id"])["status"] == "success_suspected_partial"
    assert not quota_allows(29995, 4, reserve=10000, hard_limit=20000)
    assert quota_allows(29996, 4, reserve=10000, hard_limit=20000)


def test_empty_and_bad_query_require_review(tmp_path):
    request = item(tmp_path)
    ledger = AcquisitionLedger(tmp_path)
    empty = execute_request(request, ledger, FakeClient([]))
    assert empty["status"] == "needs_review"
    assert materialize_plan([request], ledger)["preserved_terminal"] == 1

    class BadQuery(FakeClient):
        def get_json(self, endpoint, params, max_retries=2):
            self.api_calls += 1
            response = requests.Response()
            response.status_code = 400
            response.url = "https://api.collegefootballdata.com/plays"
            raise requests.HTTPError("bad query", response=response)

    bad = execute_request(request, ledger, BadQuery(None))
    assert bad["status"] == "needs_review" and bad["http_status"] == 400


def test_mixed_year_legacy_file_is_not_verified(tmp_path):
    cache = tmp_path / "mixed.parquet"
    pd.DataFrame({"season": [2024, 2025], "week": [1, 1],
                  "data": ["a" * 400, "b" * 400]}).to_parquet(cache)
    assert verify_cache(cache, year=2025) is None


def test_executor_does_not_reuse_unproven_legacy_ledger_record(tmp_path):
    request = item(tmp_path)
    cache = Path(request["cache_path"])
    pd.DataFrame({"season": [2025], "week": [1], "data": ["a" * 400]}).to_parquet(cache)
    ledger = AcquisitionLedger(tmp_path)
    ledger.write({**request, "status": "skipped_existing_complete",
                  "row_count": 1, "byte_size": cache.stat().st_size})
    client = FakeClient([{"id": 1, "season": 2025, "week": 1, "yardsGained": 7}])
    result = execute_request(request, ledger, client)
    assert result["status"] == "success_complete"
    assert client.api_calls == 1


def test_legacy_team_reuse_waits_for_fresh_ledger_backed_schedule(tmp_path):
    games = {"endpoint": "/games", "stage": "A", "classification": "current_feature_eligible",
             "partition_strategy": "year", "default_acquire": True, "required_query_parameters": [],
             "legal_query_parameters": ["year", "seasonType"], "fixed_parameters": {"seasonType": "regular"},
             "earliest_year": 2010, "existing_cache_alias": "games", "response_row_cap": None}
    teams = {"endpoint": "/games/teams", "stage": "C", "classification": "current_feature_eligible",
             "partition_strategy": "year_week", "default_acquire": True,
             "required_query_parameters": [], "legal_query_parameters": ["year", "week"],
             "fixed_parameters": {}, "earliest_year": 2010,
             "existing_cache_alias": "game_team_stats", "response_row_cap": None}
    inventory = {"endpoints": [games, teams]}
    legacy = tmp_path / "data/raw/cfbd/v2"
    (legacy / "games").mkdir(parents=True)
    (legacy / "game_team_stats").mkdir()
    schedule_row = {"id": 1, "season": 2025, "week": 1, "season_type": "regular",
                    "completed": True, "home_classification": "fbs",
                    "away_classification": "fbs", "home_team": "A", "away_team": "B",
                    "start_date": "2025-09-06T12:00:00Z", "extra": "x" * 500}
    pd.DataFrame([schedule_row]).to_parquet(legacy / "games/2025.parquet")
    pd.DataFrame({"id": [1], "week": [1], "season": [2025],
                  "extra": ["y" * 500]}).to_parquet(legacy / "game_team_stats/2025.parquet")
    root = tmp_path / "artifacts"
    before, summary = build_manifest(inventory, tmp_path, root, years=range(2025, 2026))
    assert not summary["schedule_authoritative"]
    assert summary["missing_fresh_schedule_years"] == [2025]
    assert all(row["status"] == "planned" for row in before)
    fresh = next(row for row in before if row["endpoint"] == "/games")
    assert fresh["request_id"] == make_request(games, {"year": 2025, "seasonType": "regular"},
                                                "2025", root, year=2025)["request_id"]
    execute_request(fresh, AcquisitionLedger(root), FakeClient([schedule_row]))
    after, summary = build_manifest(inventory, tmp_path, root, years=range(2025, 2026))
    assert summary["schedule_authoritative"]
    assert all(row["status"] == "skipped_existing_complete" for row in after)
    # The fresh schedule adds a game absent from the old team file: no reuse.
    second = {**schedule_row, "id": 2}
    fresh_path = Path(fresh["cache_path"])
    pd.DataFrame([schedule_row, second]).to_parquet(fresh_path)
    record = AcquisitionLedger(root).read(fresh["request_id"])
    from gridiron_ml.pipeline.fetch.nextgen_acquisition import schema_hash, sha256_file
    record.update(row_count=2, byte_size=fresh_path.stat().st_size,
                  sha256=sha256_file(fresh_path), schema_hash=schema_hash(pd.read_parquet(fresh_path)))
    AcquisitionLedger(root).write(record)
    after, _ = build_manifest(inventory, tmp_path, root, years=range(2025, 2026))
    assert next(row for row in after if row["endpoint"] == "/games/teams")["status"] == "planned"
    stats = {"endpoint": "/plays/stats", "stage": "E", "classification": "future_feature_candidate",
             "partition_strategy": "game", "default_acquire": False,
             "required_query_parameters": ["gameId"], "legal_query_parameters": ["gameId"],
             "fixed_parameters": {}, "earliest_year": 2012,
             "existing_cache_alias": None, "response_row_cap": 2000}
    full, _ = build_manifest({"endpoints": [games, teams, stats]}, tmp_path, root,
                             years=range(2025, 2026), include_plays_stats=True)
    stat_rows = [row for row in full if row["endpoint"] == "/plays/stats"]
    assert len(stat_rows) == 2 and all(row["status"] == "planned" for row in stat_rows)
    assert all(AcquisitionLedger(root).read(row["request_id"]) is None for row in stat_rows)
    # A file with a matching hash can still lack a usable completed schedule.
    pd.DataFrame([{**schedule_row, "completed": False}]).to_parquet(fresh_path)
    record.update(row_count=1, byte_size=fresh_path.stat().st_size,
                  sha256=sha256_file(fresh_path), schema_hash=schema_hash(pd.read_parquet(fresh_path)))
    AcquisitionLedger(root).write(record)
    invalid, invalid_summary = build_manifest(inventory, tmp_path, root, years=range(2025, 2026))
    assert not invalid_summary["schedule_authoritative"]
    pending_game = next(row for row in invalid if row["endpoint"] == "/games")
    assert pending_game["status"] == "planned"
    client = FakeClient([schedule_row])
    execute_request(pending_game, AcquisitionLedger(root), client)
    assert client.api_calls == 1


def test_regular_context_targets_following_game_and_excludes_postseason():
    games = pd.DataFrame([
        (1, 2025, 1, "regular", "2025-08-30", "A", "FCS", "fbs", "fcs", 21, 7),
        (2, 2025, 2, "regular", "2025-09-06", "A", "B", "fbs", "fbs", 10, 17),
        (3, 2025, 3, "postseason", "2025-12-20", "A", "B", "fbs", "fbs", 90, 0),
    ], columns=["id", "season", "week", "season_type", "start_date", "home_team",
                "away_team", "home_classification", "away_classification",
                "home_points", "away_points"])
    games["completed"] = True
    rows = next_game_rows(games)
    a = rows.loc[rows.team.eq("A")].iloc[0]
    assert a.source_game_id == 1 and a.game_id == 2
    assert a.prior_mean_margin == 14 and a.next_game_margin == -7
    assert 3 not in rows.game_id.to_list()
