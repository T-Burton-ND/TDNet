from datetime import date
import json

from gridiron_ml.publication.amendments import (
    DEADLINE_ZONE,
    append_amendment,
    deadline_record,
    thursday_deadline_utc,
    verify_amendment_ledger,
)


def _record(identifier):
    return {
        "amendment_id": identifier,
        "timestamp": "2026-07-27T15:00:00Z",
        "week": 1,
        "affected_games": ["g1"],
        "affected_models": ["m1"],
        "affected_files": ["public/predictions.parquet"],
        "reason": "score correction",
        "impact_on_predictions": "none",
        "impact_on_future_fingerprints": "future weeks only",
        "commit": "abc",
        "authorizer": "owner",
    }


def test_thursday_deadline_is_recorded_in_both_timezones():
    record = deadline_record(date(2026, 8, 27))
    assert record["deadline_timezone"] == "America/New_York"
    assert record["deadline_utc"].endswith("Z")
    assert thursday_deadline_utc(date(2026, 8, 27)).astimezone(DEADLINE_ZONE).weekday() == 3


def test_amendment_ledger_is_hash_linked_and_append_only(tmp_path):
    path = tmp_path / "amendments.jsonl"
    first = append_amendment(path, _record("a1"))
    original = path.read_bytes()
    second = append_amendment(path, _record("a2"))
    assert second["previous_record_sha256"] == first["record_sha256"]
    assert path.read_bytes().startswith(original)
    assert verify_amendment_ledger(path)["valid"]
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
