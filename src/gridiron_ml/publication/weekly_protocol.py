"""Prospective weekly snapshot contracts.

This module deliberately operates on locally cached endpoint files.  The fetch
layer is responsible for recording request timestamps; this layer decides
whether the resulting files are complete enough to support a weekly snapshot.
An empty response is evidence, not success: it is allowed only for endpoints
explicitly declared as optional in the weekly configuration.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .amendments import deadline_record, thursday_deadline_utc


DEFAULT_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "games": ("id", "season", "week", "start_date", "completed", "home_id", "away_id"),
    "game_team_stats": ("id", "season", "week", "teams"),
    "stats_advanced_game": ("game_id", "season", "week", "team", "opponent"),
    "havoc_game": ("game_id", "season", "week", "team", "opponent"),
    "lines": ("id", "season", "week", "lines"),
    "pregame_wp": ("season", "week", "game_id"),
    "ppa_games": ("game_id", "season", "week", "team", "opponent"),
}


def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded chunks."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec_for(
    endpoint: str,
    completeness_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    configured = dict((completeness_config or {}).get(endpoint, {}) or {})
    configured.setdefault("required_columns", list(DEFAULT_REQUIRED_COLUMNS.get(endpoint, ())))
    configured.setdefault("allow_empty", False)
    configured.setdefault("minimum_rows", 0)
    return configured


def inspect_endpoint(
    path: str | Path,
    *,
    endpoint: str,
    spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return reproducible schema, count, hash, and missing-field evidence."""
    path = Path(path)
    result: dict[str, Any] = {
        "endpoint": endpoint,
        "path": str(path),
        "exists": path.exists(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": 0,
        "columns": [],
        "required_columns": list((spec or {}).get("required_columns", ())),
        "missing_columns": [],
        "sha256": None,
        "status": "missing",
        "error": None,
    }
    if not path.exists():
        return result
    result["sha256"] = sha256_file(path)
    try:
        frame = pd.read_parquet(path)
        result["rows"] = int(len(frame))
        result["columns"] = [str(column) for column in frame.columns]
        result["missing_columns"] = sorted(
            set(result["required_columns"]) - set(result["columns"])
        )
    except Exception as exc:  # pragma: no cover - parquet engine errors are environment-specific
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    allow_empty = bool((spec or {}).get("allow_empty", False))
    minimum_rows = int((spec or {}).get("minimum_rows", 0))
    if result["missing_columns"]:
        result["status"] = "fail_missing_columns"
    elif not allow_empty and result["rows"] == 0:
        result["status"] = "fail_empty"
    elif result["rows"] < minimum_rows:
        result["status"] = "fail_below_minimum_rows"
    else:
        result["status"] = "pass"
    return result


def build_snapshot_completeness(
    *,
    raw_cache_dir: str | Path,
    season: int,
    endpoints: Mapping[str, bool],
    completeness_config: Mapping[str, Any] | None = None,
    required_endpoints: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Inspect all enabled endpoint snapshots and certify only complete data."""
    raw_cache_dir = Path(raw_cache_dir)
    selected = [name for name, enabled in endpoints.items() if enabled]
    required = set(required_endpoints or selected)
    endpoint_reports = []
    failures = []
    for endpoint in selected:
        spec = _spec_for(endpoint, completeness_config)
        report = inspect_endpoint(
            raw_cache_dir / endpoint / f"{int(season)}.parquet",
            endpoint=endpoint,
            spec=spec,
        )
        report["required_for_certification"] = endpoint in required
        endpoint_reports.append(report)
        if report["required_for_certification"] and report["status"] != "pass":
            failures.append({"endpoint": endpoint, "status": report["status"]})

    return {
        "season": int(season),
        "raw_cache_dir": str(raw_cache_dir.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_endpoints": sorted(required),
        "endpoint_count": len(endpoint_reports),
        "endpoints": endpoint_reports,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "certification": (
            "weekly_snapshot_certified"
            if not failures
            else "weekly_snapshot_not_certified"
        ),
    }


def write_snapshot_completeness(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write an immutable-by-convention JSON completeness report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_deadline_utc(deadline_utc: str, *, local_date: str | date) -> dict[str, str]:
    """Require the supplied UTC deadline to equal Thursday 23:59 New York time."""
    day = date.fromisoformat(local_date) if isinstance(local_date, str) else local_date
    expected = thursday_deadline_utc(day)
    text = str(deadline_utc).strip()
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline-utc must include an explicit UTC offset or Z suffix")
    parsed_utc = parsed.astimezone(timezone.utc)
    if parsed_utc != expected:
        raise ValueError(
            f"deadline-utc {deadline_utc!r} does not equal the declared Thursday deadline "
            f"{expected.isoformat().replace('+00:00', 'Z')}"
        )
    return deadline_record(day)
