"""Stage-A schedule, manifest, Week-0, and keyed matchup boundaries."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from gridiron_ml.experiments.nextgen_artifacts import (
    NextgenFeatureBuilder, NextgenModelBoundary, canonical_family_path,
    write_canonical_feature_family,
)
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (
    AcquisitionLedger, make_request, sha256_file, verify_cache,
)


ROOT = Path(__file__).resolve().parents[1]
NAMES = tuple(record["name"] for record in json.loads(
    (ROOT / "configs/experiments/nextgen_seed_features_v1.json").read_text()))


def game(game_id, year, week, home, away, date, away_class="fbs"):
    return {"id": game_id, "season": year, "week": week, "season_type": "regular",
            "completed": True, "home_team": home, "away_team": away,
            "home_classification": "fbs", "away_classification": away_class,
            "start_date": date, "padding": "x" * 400}


def stage_a_fixture(tmp_path):
    inventory = json.loads((ROOT / "configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json").read_text())
    endpoint = next(item for item in inventory["endpoints"] if item["endpoint"] == "/games")
    ledger = AcquisitionLedger(tmp_path)
    for year in range(2010, 2026):
        rows = [game(year * 1000, year, 1, f"H{year}", f"V{year}", f"{year}-09-01T12:00:00Z")]
        if year == 2025:
            rows = [game(100, year, 0, "A", "FCS", "2025-08-30T12:00:00Z", "fcs"),
                    game(101, year, 0, "B", "FCS2", "2025-08-30T13:00:00Z", "fcs"),
                    game(102, year, 1, "A", "B", "2025-09-06T12:00:00Z"),
                    game(103, year, 2, "A", "B", "2025-09-13T12:00:00Z")]
        params = {"year": year, **endpoint["fixed_parameters"]}
        request = make_request(endpoint, params, str(year), tmp_path, year=year)
        path = Path(request["cache_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        metadata = verify_cache(path, year=year)
        assert metadata is not None
        ledger.write({**request, **metadata, "status": "success_complete"})
    records = json.loads((ROOT / "configs/experiments/nextgen_seed_features_v1.json").read_text())
    manifest = tmp_path / "family_manifest.json"
    manifest.write_text(json.dumps(records))
    return manifest


def dynamic_rows():
    rows = []
    for team, source_id, source_time, value in (
            ("A", 100, "2025-08-30T12:00:00Z", 2.0),
            ("B", 101, "2025-08-30T13:00:00Z", 5.0)):
        rows.append({"season": 2025, "season_type": "regular", "team": team,
                     "feature_kind": "dynamic", "target_game_id": 102,
                     "target_start_utc": "2025-09-06T12:00:00Z",
                     "feature_available_utc": "2025-08-31T00:00:00Z",
                     "latest_source_game_id": source_id,
                     "latest_source_game_utc": source_time,
                     "latest_source_season_type": "regular",
                     "static_availability_documentation": None,
                     NAMES[0]: value, NAMES[1]: value / 2})
    return pd.DataFrame(rows)


def static_rows():
    rows = []
    for target, kickoff in ((102, "2025-09-06T12:00:00Z"),
                            (103, "2025-09-13T12:00:00Z")):
        for team, value in (("A", 2.0), ("B", 5.0)):
            rows.append({"season": 2025, "season_type": "regular", "team": team,
                         "feature_kind": "static_week0", "target_game_id": target,
                         "target_start_utc": kickoff,
                         "feature_available_utc": "2025-08-25T00:00:00Z",
                         "latest_source_game_id": None, "latest_source_game_utc": None,
                         "latest_source_season_type": None,
                         "static_availability_documentation": "frozen Week-0 roster snapshot",
                         NAMES[0]: value, NAMES[1]: value / 2})
    return pd.DataFrame(rows)


def write(tmp_path, frame, manifest):
    return write_canonical_feature_family(tmp_path, "F09", "rushing_state", frame, NAMES, manifest)


def test_matchup_uses_game_team_keys_across_shuffled_reset_indexes(tmp_path):
    manifest = stage_a_fixture(tmp_path)
    frame = dynamic_rows()
    write(tmp_path, frame, manifest)
    first = NextgenModelBoundary.from_canonical(tmp_path, "F09", "rushing_state", manifest)
    expected = first.matchup((102, "A"), (102, "B"))
    shuffled = frame.iloc[::-1].copy().reset_index(drop=True)
    shuffled.index = [40, 20]
    write(tmp_path, shuffled, manifest)
    second = NextgenModelBoundary.from_canonical(tmp_path, "F09", "rushing_state", manifest)
    pd.testing.assert_frame_equal(expected, second.matchup((102, "A"), (102, "B")))
    assert second.for_fit().index.tolist() == [(102, "A"), (102, "B")]
    assert second.for_shap().index.tolist() == [(102, "A"), (102, "B")]
    for home, away in (((103, "A"), (102, "B")), ((102, "B"), (102, "A")),
                       ((102, "C"), (102, "B"))):
        with pytest.raises(ValueError):
            second.matchup(home, away)


def test_duplicate_missing_wrong_target_and_fake_source_fail(tmp_path):
    manifest = stage_a_fixture(tmp_path)
    frame = dynamic_rows()
    bad_frames = [pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
                  frame.iloc[[0]], frame.assign(target_game_id=103),
                  frame.assign(target_start_utc="2025-09-07T12:00:00Z"),
                  frame.assign(team=["A", "C"]),
                  frame.assign(latest_source_game_id=[999, 101]),
                  frame.assign(latest_source_game_id=[101, 100]),
                  frame.assign(latest_source_game_utc="2025-08-29T12:00:00Z"),
                  frame.assign(latest_source_game_id=102,
                               latest_source_game_utc="2025-09-06T12:00:00Z")]
    for bad in bad_frames:
        with pytest.raises(ValueError):
            write(tmp_path, bad, manifest)
    assert not canonical_family_path(tmp_path, "F09", "rushing_state").exists()


def test_week0_cutoff_and_frozen_value_fail_closed(tmp_path):
    manifest = stage_a_fixture(tmp_path)
    records = json.loads(manifest.read_text())
    for record in records:
        record["static_or_dynamic"] = "static_preseason"
        record["availability_rule"] = "week0_roster_snapshot"
        record["raw_endpoints"] = ["/roster"]
    manifest.write_text(json.dumps(records))
    frame = static_rows()
    for bad in (frame.assign(feature_available_utc="2025-08-31T00:00:00Z"),
                frame.assign(**{NAMES[0]: [2.0, 5.0, 3.0, 5.0]})):
        with pytest.raises(ValueError):
            write(tmp_path, bad, manifest)
    write(tmp_path, frame, manifest)
    assert (tmp_path / "results/preflight/week0_freeze_v1.json").exists()
    boundary = NextgenModelBoundary.from_canonical(tmp_path, "F09", "rushing_state", manifest)
    assert len(boundary.for_fit()) == 4
    changed = frame.copy()
    changed.loc[changed.team.eq("A"), NAMES[0]] = 9.0
    with pytest.raises(ValueError):
        write(tmp_path, changed, manifest)


def test_manifest_flags_and_hash_mismatch_fail_even_for_safe_names(tmp_path):
    manifest = stage_a_fixture(tmp_path)
    frame = dynamic_rows()
    clean = json.loads(manifest.read_text())
    for flag in ("market_derived", "target_derived", "pregame_win_probability_derived"):
        bad = copy.deepcopy(clean)
        bad[0][flag] = True
        manifest.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            write(tmp_path, frame, manifest)
    bad = copy.deepcopy(clean)
    bad[0]["temporal_cutoff"] = "same_game"
    manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        write(tmp_path, frame, manifest)
    manifest.write_text(json.dumps(clean))
    with pytest.raises(ValueError, match="generation"):
        write_canonical_feature_family(tmp_path, "F10", "rushing_state", frame, NAMES, manifest)
    write(tmp_path, frame, manifest)
    manifest.write_text(json.dumps(clean, indent=2))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        NextgenModelBoundary.from_canonical(tmp_path, "F09", "rushing_state", manifest)
    with pytest.raises(ValueError):
        NextgenModelBoundary(frame, NAMES, pd.DataFrame())


def test_model_load_rechecks_schedule_even_if_artifact_hash_is_refreshed(tmp_path):
    manifest = stage_a_fixture(tmp_path)
    path = write(tmp_path, dynamic_rows(), manifest)
    frame = pd.read_parquet(path)
    frame.loc[0, "latest_source_game_id"] = 999
    frame.to_parquet(path, index=False)
    sidecar = path.with_suffix(".provenance.json")
    data = json.loads(sidecar.read_text())
    data["data_sha256"] = sha256_file(path)
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Source game provenance"):
        NextgenModelBoundary.from_canonical(tmp_path, "F09", "rushing_state", manifest)


def test_builder_cannot_override_checked_materialization():
    with pytest.raises(TypeError):
        class Bypass(NextgenFeatureBuilder):
            def build_frame(self):
                return dynamic_rows()

            def materialize(self, root):
                return root / "unchecked.parquet"
