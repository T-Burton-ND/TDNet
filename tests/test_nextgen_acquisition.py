"""Focused guards for manifest identity, ledger resume, response cap, and timing."""

import json
from pathlib import Path

import pandas as pd
import requests

from gridiron_ml.experiments.nextgen_temporal import next_game_rows
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (
    AcquisitionLedger, execute_request, materialize_plan, quota_allows, request_identity, verify_cache,
)


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
    assert not quota_allows(6200, 300, 2)
    assert quota_allows(7000, 300, 2)


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
