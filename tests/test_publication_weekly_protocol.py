from pathlib import Path

import pandas as pd

from gridiron_ml.publication.weekly_protocol import (
    build_snapshot_completeness,
    inspect_endpoint,
    validate_deadline_precedes_kickoff,
    validate_deadline_utc,
)


def test_endpoint_inspection_records_schema_hash_and_failure(tmp_path: Path):
    path = tmp_path / "games.parquet"
    pd.DataFrame({"id": [1], "season": [2026]}).to_parquet(path, index=False)
    report = inspect_endpoint(path, endpoint="games", spec={"required_columns": ["id", "season", "week"]})
    assert report["status"] == "fail_missing_columns"
    assert report["sha256"]


def test_snapshot_certification_requires_all_declared_endpoints(tmp_path: Path):
    endpoint = tmp_path / "games" / "2026.parquet"
    endpoint.parent.mkdir()
    pd.DataFrame(
        {"id": [1], "season": [2026], "week": [1], "start_date": ["2026-09-01"],
         "completed": [False], "home_id": [1], "away_id": [2]}
    ).to_parquet(endpoint, index=False)
    report = build_snapshot_completeness(
        raw_cache_dir=tmp_path,
        season=2026,
        endpoints={"games": True, "lines": True},
        required_endpoints=["games", "lines"],
    )
    assert report["status"] == "fail"
    assert report["certification"] == "weekly_snapshot_not_certified"
    assert {item["endpoint"] for item in report["endpoints"]} == {"games", "lines"}


def test_deadline_requires_thursday_new_york_cutoff():
    record = validate_deadline_utc("2026-09-11T03:59:00Z", local_date="2026-09-10")
    assert record["deadline_timezone"] == "America/New_York"


def test_deadline_allows_owner_selected_pre_kickoff_thursday_cutoff():
    record = validate_deadline_utc("2026-09-10T15:59:00Z", local_date="2026-09-10")
    assert record["deadline_local"] == "2026-09-10T11:59:00-04:00"
    assert record["deadline_utc"] == "2026-09-10T15:59:00Z"


def test_deadline_allows_early_week_cutoff_for_slates_with_early_kickoffs():
    record = validate_deadline_utc("2026-10-06T23:00:00Z", local_date="2026-10-06")
    assert record["deadline_local"] == "2026-10-06T19:00:00-04:00"
    validate_deadline_precedes_kickoff(
        record["deadline_utc"], "2026-10-07T00:00:00Z"
    )
    try:
        validate_deadline_precedes_kickoff(
            record["deadline_utc"], "2026-10-06T23:00:00Z"
        )
    except ValueError as exc:
        assert "must precede" in str(exc)
    else:
        raise AssertionError("A cutoff at kickoff should be rejected")


def test_deadline_rejects_cutoff_after_declared_thursday():
    try:
        validate_deadline_utc("2026-09-11T04:00:00Z", local_date="2026-09-10")
    except ValueError as exc:
        assert "no later than Thursday" in str(exc)
    else:
        raise AssertionError("Friday-local deadline should be rejected")
