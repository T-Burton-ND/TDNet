"""Append-only correction ledger and prospective deadline helpers."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


DEADLINE_ZONE_NAME = "America/New_York"
DEADLINE_ZONE = ZoneInfo(DEADLINE_ZONE_NAME)


def thursday_deadline_utc(day: date) -> datetime:
    """Return the declared Thursday 23:59 local deadline in UTC."""
    if day.weekday() != 3:
        raise ValueError(f"Prospective publication deadline must be Thursday; got {day.isoformat()}")
    local = datetime.combine(day, time(23, 59, 0), tzinfo=DEADLINE_ZONE)
    return local.astimezone(timezone.utc)


def deadline_record(day: date) -> dict[str, str]:
    local = datetime.combine(day, time(23, 59, 0), tzinfo=DEADLINE_ZONE)
    utc = thursday_deadline_utc(day)
    return {
        "deadline_local": local.isoformat(),
        "deadline_timezone": DEADLINE_ZONE_NAME,
        "deadline_utc": utc.isoformat().replace("+00:00", "Z"),
    }


def _canonical_record(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def append_amendment(ledger_path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append one hash-linked amendment without changing prior bytes."""
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_hash = None
    if path.exists():
        existing = path.read_text(encoding="utf-8").splitlines()
        if existing:
            previous = json.loads(existing[-1])
            previous_hash = previous.get("record_sha256")
    payload = dict(record)
    required = {"amendment_id", "timestamp", "week", "affected_games", "affected_models", "affected_files", "reason", "impact_on_predictions", "impact_on_future_fingerprints", "commit", "authorizer"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Amendment is missing {missing}")
    payload["previous_record_sha256"] = previous_hash
    payload["record_sha256"] = sha256(_canonical_record(payload)).hexdigest()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return payload


def verify_amendment_ledger(ledger_path: str | Path) -> dict[str, Any]:
    path = Path(ledger_path)
    failures = []
    previous_hash = None
    rows = []
    if not path.exists():
        return {"valid": True, "records": 0, "failures": []}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"line {line_number}: invalid JSON ({exc})")
            continue
        claimed = record.get("record_sha256")
        expected_previous = record.get("previous_record_sha256")
        if expected_previous != previous_hash:
            failures.append(f"line {line_number}: broken previous-record hash link")
        unsigned = dict(record)
        unsigned.pop("record_sha256", None)
        actual = sha256(_canonical_record(unsigned)).hexdigest()
        if claimed != actual:
            failures.append(f"line {line_number}: record hash mismatch")
        previous_hash = claimed
        rows.append(record)
    return {"valid": not failures, "records": len(rows), "failures": failures, "last_record_sha256": previous_hash}
