"""Manifest-first CFBD acquisition for the next-generation experiment.

Planning is read-only with respect to CFBD. The request ledger is separate from
the cross-process outbound-attempt budget in :mod:`cfbd_fetch_v2`.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .cfbd_fetch_v2 import CFBDClient, CallBudget, write_parquet

SCHEMA_VERSION = "cfbd_openapi_5.30.0_parquet_v1"
STATUSES = {"planned", "skipped_existing_complete", "success_complete",
            "success_suspected_partial", "failed_retryable", "failed_final",
            "structurally_unavailable", "needs_review"}
COMPLETE = {"skipped_existing_complete", "success_complete"}
TERMINAL_UNAVAILABLE = {"structurally_unavailable", "failed_final"}
PARTIAL = {"success_suspected_partial"}
REVIEW_REQUIRED = {"needs_review"}
# Legacy games only support a provisional call estimate. Fresh ledger-backed
# schedules are required before team-game legacy reuse or later stages.
ELIGIBLE_LEGACY_REUSE = {"/games/teams"}


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def request_identity(endpoint: str, params: dict, partition: str,
                     schema_version: str = SCHEMA_VERSION) -> str:
    payload = {"endpoint": endpoint, "parameters": params, "partition": partition,
               "schema_version": schema_version}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".nextgen_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def regular_fbs_games(frame: pd.DataFrame) -> pd.DataFrame:
    """Only completed regular games with at least one FBS team inform coverage."""
    required = {"id", "season", "week", "season_type", "completed",
                "home_classification", "away_classification"}
    if not required <= set(frame):
        raise ValueError(f"Schedule cache missing columns: {sorted(required-set(frame))}")
    mask = (frame.season_type.astype(str).str.lower().eq("regular") &
            frame.completed.fillna(False).astype(bool) &
            (frame.home_classification.astype(str).str.lower().eq("fbs") |
             frame.away_classification.astype(str).str.lower().eq("fbs")))
    return frame.loc[mask].copy()


def schema_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_json({c: str(t) for c, t in frame.dtypes.items()}).encode()).hexdigest()


class AcquisitionLedger:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, request_id: str) -> Path:
        return self.root / "request_ledger" / request_id[:2] / f"{request_id}.json"

    def read(self, request_id: str) -> dict | None:
        path = self.path(request_id)
        return json.loads(path.read_text()) if path.exists() else None

    def write(self, record: dict) -> None:
        if record["status"] not in STATUSES:
            raise ValueError("Unknown acquisition status")
        atomic_json(self.path(record["request_id"]), record)


def verify_cache(path: Path, record: dict | None = None,
                 year: int | None = None, week: int | None = None) -> dict | None:
    if not path.is_file() or path.stat().st_size < 600:
        return None
    try:
        frame = pd.read_parquet(path)
        if frame.empty:
            return None
        if year is not None:
            season_col = next((c for c in ("season", "year") if c in frame), None)
            if season_col and not pd.to_numeric(frame[season_col], errors="coerce").eq(year).all():
                return None
        if week is not None:
            if "week" not in frame or not pd.to_numeric(frame.week, errors="coerce").eq(week).any():
                return None
        digest = sha256_file(path)
        result = {"row_count": len(frame), "byte_size": path.stat().st_size,
                  "sha256": digest, "schema_hash": schema_hash(frame)}
        if record is not None and any(record.get(key) != result[key] for key in result):
            return None
        return result
    except (OSError, ValueError, ImportError):
        return None


def game_team_coverage(path: Path, expected_games: pd.DataFrame, week: int) -> bool:
    """Check team-game IDs against the fresh completed FBS schedule."""
    try:
        frame = pd.read_parquet(path, columns=["id", "week"])
        observed = set(pd.to_numeric(frame.loc[frame.week.eq(week), "id"], errors="coerce").dropna().astype(int))
        expected = set(expected_games.loc[expected_games.week.eq(week), "id"].astype(int))
        return bool(expected) and expected <= observed
    except (OSError, ValueError, KeyError):
        return False


def make_request(endpoint: dict, params: dict, partition: str, root: Path,
                 year: int | None = None, week: int | None = None,
                 game_id: int | None = None) -> dict:
    rid = request_identity(endpoint["endpoint"], params, partition)
    slug = endpoint["endpoint"].strip("/").replace("/", "_")
    return {"request_id": rid, "endpoint": endpoint["endpoint"],
            "parameters_json": canonical_json(params), "parameters": params,
            "year": year, "week": week, "game_id": game_id, "team": params.get("team"),
            "partition": partition, "schema_version": SCHEMA_VERSION,
            "stage": endpoint["stage"], "classification": endpoint["classification"],
            "response_row_cap": endpoint["response_row_cap"],
            "cache_path": str(root / "raw_cache" / "v1" / slug / f"{rid}.parquet"),
            "legacy_alias": endpoint.get("existing_cache_alias"),
            "planned": True, "status": "planned"}


def verified_schedule_year(games_endpoint: dict, artifact_root: Path,
                           ledger: AcquisitionLedger, year: int) -> tuple[pd.DataFrame, str] | None:
    """Read one fresh Stage-A schedule only through its exact successful ledger record."""
    params = {"year": year, **games_endpoint["fixed_parameters"]}
    expected = make_request(games_endpoint, params, str(year), artifact_root, year=year)
    prior = ledger.read(expected["request_id"])
    if not (prior and prior.get("status") == "success_complete"
            and prior.get("cache_path") == expected["cache_path"]
            and prior.get("parameters_json") == expected["parameters_json"]
            and prior.get("endpoint") == "/games"
            and verify_cache(Path(expected["cache_path"]), prior, year=year)):
        return None
    try:
        frame = regular_fbs_games(pd.read_parquet(expected["cache_path"]))
        required = {"id", "season", "start_date", "home_team", "away_team"}
        if not required <= set(frame) or frame.empty or frame.id.isna().any() or not frame.id.is_unique:
            return None
        if not pd.to_numeric(frame.season, errors="coerce").eq(year).all():
            return None
        if (frame.home_team.isna().any() or frame.away_team.isna().any()
                or frame.home_team.astype(str).str.strip().eq("").any()
                or frame.away_team.astype(str).str.strip().eq("").any()
                or frame.home_team.eq(frame.away_team).any()
                or pd.to_datetime(frame.start_date, utc=True, errors="coerce").isna().any()):
            return None
        return frame, prior["sha256"]
    except (OSError, ValueError, KeyError):
        return None


def load_authoritative_schedule(artifact_root: Path, inventory: dict,
                                years: range = range(2010, 2026)) -> tuple[pd.DataFrame, str]:
    """Require every Stage-A year and return the schedule plus its durable digest."""
    endpoint = next(item for item in inventory["endpoints"] if item["endpoint"] == "/games")
    ledger = AcquisitionLedger(artifact_root)
    frames = []
    hashes = {}
    for year in years:
        verified = verified_schedule_year(endpoint, artifact_root, ledger, year)
        if verified is None:
            raise RuntimeError(f"Fresh ledger-backed Stage-A /games schedule missing for {year}")
        frame, digest = verified
        frames.append(frame)
        hashes[str(year)] = digest
    schedule = pd.concat(frames, ignore_index=True)
    if not schedule.id.is_unique:
        raise ValueError("Authoritative schedule contains duplicate game IDs")
    digest = hashlib.sha256(canonical_json(hashes).encode()).hexdigest()
    return schedule, digest


def build_manifest(inventory: dict, project_root: Path, artifact_root: Path,
                   years: range = range(2010, 2026), include_plays_stats: bool = False) -> tuple[list[dict], dict]:
    """Enumerate calls; legacy schedules support estimates only until Stage A."""
    legacy_root = project_root / "data/raw/cfbd/v2"
    ledger = AcquisitionLedger(artifact_root)
    games_endpoint = next(item for item in inventory["endpoints"] if item["endpoint"] == "/games")
    fresh_schedule = {}
    missing_fresh_years = []
    for year in years:
        verified = verified_schedule_year(games_endpoint, artifact_root, ledger, year)
        if verified is not None:
            fresh_schedule[year] = verified[0]
            continue
        missing_fresh_years.append(year)
    authoritative = not missing_fresh_years
    schedule = fresh_schedule if authoritative else {}
    if not authoritative:
        for year in years:
            path = legacy_root / "games" / f"{year}.parquet"
            if verify_cache(path, year=year) is None:
                raise RuntimeError(f"Cannot estimate calls without a games cache: {path}")
            schedule[year] = regular_fbs_games(pd.read_parquet(path))
    requests_by_id = {}
    skipped = []
    legacy_validation = {}
    for endpoint in inventory["endpoints"]:
        if not endpoint["default_acquire"] and not (include_plays_stats and endpoint["endpoint"] == "/plays/stats"):
            skipped.append({"endpoint": endpoint["endpoint"], "reason": "not_default_acquisition"})
            continue
        path = endpoint["endpoint"]
        strategy = endpoint["partition_strategy"]
        first_year = endpoint["earliest_year"]
        parts = []
        if strategy == "static_once":
            parts.append(({}, "static", None, None, None))
        elif strategy == "year_range":
            parts.append(({"startYear": min(years), "endYear": max(years)},
                          f"{min(years)}-{max(years)}", None, None, None))
        elif strategy in {"year", "year_week", "game"}:
            for year in years:
                if year < first_year:
                    continue
                if strategy == "year":
                    parts.append(({"year": year}, str(year), year, None, None))
                elif strategy == "year_week":
                    for week in sorted(schedule[year].week.dropna().astype(int).unique()):
                        parts.append(({"year": year, "week": int(week)},
                                      f"{year}-w{week:02d}", year, int(week), None))
                else:
                    for game_id in sorted(schedule[year].id.dropna().astype(int).unique()):
                        parts.append(({"gameId": int(game_id)}, str(game_id), year, None, int(game_id)))
        else:
            skipped.append({"endpoint": path, "reason": f"unsupported_partition:{strategy}"})
        for params, partition, year, week, game_id in parts:
            params.update(endpoint["fixed_parameters"])
            if not set(endpoint["required_query_parameters"]) <= set(params):
                raise ValueError(f"Missing required API query parameter for {path}: {params}")
            if not set(params) <= set(endpoint["legal_query_parameters"]):
                raise ValueError(f"Illegal API query parameter for {path}: {params}")
            item = make_request(endpoint, params, partition, artifact_root, year, week, game_id)
            rid = item["request_id"]
            if rid in requests_by_id:
                raise ValueError(f"Duplicate request identity: {rid}")
            prior = ledger.read(rid)
            if (prior and prior.get("status") == "success_complete"
                    and prior.get("endpoint") == path
                    and prior.get("parameters_json") == item["parameters_json"]
                    and prior.get("cache_path") == item["cache_path"]
                    and (path != "/games" or year in fresh_schedule)
                    and (path != "/games/teams" or (authoritative and
                         game_team_coverage(Path(item["cache_path"]), schedule[year], week)))):
                meta = verify_cache(Path(prior["cache_path"]), prior)
                if meta:
                    item.update({"status": "skipped_existing_complete", "cache_path": prior["cache_path"],
                                 "reuse_source": "verified_request_ledger", **meta})
            if prior and prior.get("status") in TERMINAL_UNAVAILABLE | PARTIAL | REVIEW_REQUIRED:
                item.update({"status": prior["status"], "error_summary": prior.get("error_summary"),
                             "http_status": prior.get("http_status")})
            if item["status"] == "planned" and item["legacy_alias"] and year is not None:
                legacy = legacy_root / item["legacy_alias"] / f"{year}.parquet"
                if path not in ELIGIBLE_LEGACY_REUSE or not authoritative:
                    if legacy.exists():
                        item["legacy_candidate_path"] = str(legacy)
                        item["legacy_reuse_decision"] = ("fresh_schedule_required" if not authoritative
                                                          else "unverified_request_scope_or_coverage")
                    requests_by_id[rid] = item
                    continue
                key = (str(legacy), week)
                if key not in legacy_validation:
                    legacy_validation[key] = verify_cache(legacy, year=year, week=week)
                    if path == "/games/teams" and legacy_validation[key]:
                        if not game_team_coverage(legacy, schedule[year], week):
                            legacy_validation[key] = None
                if legacy_validation[key]:
                    item.update({"status": "skipped_existing_complete", "cache_path": str(legacy),
                                 "reuse_source": "structurally_valid_legacy_superset",
                                 **legacy_validation[key]})
            requests_by_id[rid] = item
    manifest = list(requests_by_id.values())
    counts = defaultdict(lambda: Counter())
    for item in manifest:
        category = ("reused" if item["status"] in COMPLETE else
                    item["status"] if item["status"] in TERMINAL_UNAVAILABLE | PARTIAL | REVIEW_REQUIRED else "new")
        counts[item["endpoint"]][category] += 1
    new = sum(c["new"] for c in counts.values())
    reused = sum(c["reused"] for c in counts.values())
    legacy_unverified = sum("legacy_candidate_path" in item for item in manifest)
    summary = {"schema_version": SCHEMA_VERSION, "years": [min(years), max(years)],
               "schedule_authoritative": authoritative,
               "fresh_schedule_years": sorted(fresh_schedule),
               "missing_fresh_schedule_years": missing_fresh_years,
               "schedule_planning_basis": ("fresh_ledger_backed" if authoritative
                                           else "provisional_legacy_estimate_only"),
               "include_plays_stats": include_plays_stats, "total_requests": len(manifest),
               "new_planned_calls": new, "reused_cached_partitions": reused,
               "legacy_candidates_not_reused": legacy_unverified,
               "unresolved_partial_requests": sum(c["success_suspected_partial"] for c in counts.values()),
               "unresolved_review_requests": sum(c["needs_review"] for c in counts.values()),
               "by_endpoint": {k: dict(v) for k, v in sorted(counts.items())},
               "skipped_endpoints": skipped, "excluded_due_budget": [],
               "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    return manifest, summary


def materialize_plan(manifest: list[dict], ledger: AcquisitionLedger) -> dict:
    """Persist one ledger record per intended request without losing work."""
    counts = Counter()
    for item in manifest:
        prior = ledger.read(item["request_id"])
        if (prior and item["status"] in COMPLETE and prior.get("status") in COMPLETE
                and verify_cache(Path(prior["cache_path"]), prior)):
            counts["preserved_complete"] += 1
            continue
        if prior and prior.get("status") in TERMINAL_UNAVAILABLE | PARTIAL | REVIEW_REQUIRED:
            counts["preserved_terminal"] += 1
            continue
        if (prior and prior.get("status") == item["status"] == "planned"
                and prior.get("parameters_json") == item["parameters_json"]
                and prior.get("stage") == item["stage"]):
            counts["preserved_planned"] += 1
            continue
        record = {**item, "attempt_count": prior.get("attempt_count", 0) if prior else 0,
                  "http_status": prior.get("http_status") if prior else None,
                  "row_count": item.get("row_count"), "byte_size": item.get("byte_size"),
                  "sha256": item.get("sha256"), "schema_hash": item.get("schema_hash"),
                  "completeness_status": ("structurally_valid_legacy_cache"
                                          if item["status"] in COMPLETE else "not_checked"),
                  "fetched_at_utc": prior.get("fetched_at_utc") if prior else None,
                  "error_summary": prior.get("error_summary") if prior else None}
        ledger.write(record)
        counts[record["status"]] += 1
    return dict(counts)


def execute_request(item: dict, ledger: AcquisitionLedger, client: CFBDClient) -> dict:
    """Execute one manifest row, recording terminal or retryable state immediately."""
    rid = item["request_id"]
    previous = ledger.read(rid)
    if (previous and previous["status"] in COMPLETE
            and (item["status"] in COMPLETE or
                 (previous["status"] == "success_complete"
                  and item["endpoint"] not in {"/games", "/games/teams"}))
            and verify_cache(Path(previous["cache_path"]), previous)):
        return previous
    if item["status"] in COMPLETE and verify_cache(Path(item["cache_path"]), item):
        record = {**item, "attempt_count": 0, "http_status": None,
                  "completeness_status": "structurally_valid_legacy_cache",
                  "fetched_at_utc": None, "error_summary": None}
        ledger.write(record)
        return record
    record = {**item, "status": "planned", "attempt_count": 0, "http_status": None,
              "row_count": None, "byte_size": None, "sha256": None, "schema_hash": None,
              "completeness_status": "not_checked", "fetched_at_utc": None, "error_summary": None}
    try:
        before = client.api_calls
        payload = client.get_json(item["endpoint"], item["parameters"], max_retries=2)
        record["attempt_count"] = client.api_calls - before
        if not isinstance(payload, list):
            raise ValueError("CFBD response is not a list")
        if not payload:
            record.update(status="needs_review", completeness_status="empty_response_unverified",
                          error_summary="Empty 200 response requires coverage review")
        else:
            frame = pd.json_normalize(payload)
            cap = item.get("response_row_cap")
            partial = (cap is not None and len(frame) >= cap) or (cap is None and len(frame) == 2000)
            path = Path(item["cache_path"])
            write_parquet(frame, path, snake=True)
            meta = verify_cache(path)
            if meta is None:
                raise RuntimeError("Atomic Parquet write failed validation")
            record.update(meta)
            record.update(status="success_suspected_partial" if partial else "success_complete",
                          completeness_status=("response_cap_reached" if cap is not None else
                                               "possible_common_2000_row_cap") if partial else "complete")
        record["http_status"] = 200
        record["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    except requests.HTTPError as exc:
        record["attempt_count"] = max(record["attempt_count"], client.api_calls - before)
        code = exc.response.status_code if exc.response is not None else None
        record["http_status"] = code
        record["status"] = "needs_review" if code in (400, 401, 403, 404) else "failed_retryable"
        record["error_summary"] = f"HTTP {code}: {str(exc)[:240]}"
    except Exception as exc:
        record["attempt_count"] = max(record["attempt_count"], client.api_calls - before)
        record["status"] = "failed_retryable"
        record["error_summary"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    ledger.write(record)
    return record


def quota_allows(remaining_calls: int, reserved_attempts: int, *,
                 reserve: int, hard_limit: int) -> bool:
    """Provider can cover every remaining local slot while retaining reserve."""
    return (isinstance(remaining_calls, int) and isinstance(reserved_attempts, int)
            and 0 <= reserved_attempts <= hard_limit <= 20000
            and remaining_calls >= reserve + hard_limit - reserved_attempts)
