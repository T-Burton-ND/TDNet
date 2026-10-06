#!/usr/bin/env python3
"""Acquire bounded 2026 inputs for the separately labeled scientific what-if.

This does not write the prospective fingerprint archive or frozen publication
bundles. Every outbound attempt is reserved in the shared CFBD call budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import (  # noqa: E402
    CFBDClient, CallBudget, load_cfbd_key_file,
)

ARTIFACTS = ROOT / "data/what_if_2026_fingerprints"
SHARED_BUDGET = Path("/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/results/cfbd_api_call_budget.json")
HARD_LIMIT = 20_000
RETRIES = 2
RATE_DELAY = 0.25


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temp.replace(path)


def _save_response(endpoint: str, params: dict, payload: object) -> dict:
    if not isinstance(payload, list):
        raise ValueError(f"{endpoint} returned a non-list payload")
    frame = pd.json_normalize(payload)
    if frame.empty:
        raise ValueError(f"{endpoint} returned an empty response for {params}")
    rel = endpoint.strip("/").replace("/", "_")
    partition = "_".join(f"{key}_{params[key]}" for key in sorted(params)) or "static"
    target = ARTIFACTS / "raw_cache" / rel / f"{partition}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(target)
    receipt = {
        "endpoint": endpoint,
        "parameters": params,
        "status": "needs_review_possible_2000_row_cap" if len(frame) == 2000 else "success_complete",
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "cache_path": str(target.relative_to(ROOT)),
        "sha256": _digest(target),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if len(frame) == 2000:
        raise RuntimeError(f"Response may be capped; review required before accepting {endpoint} {params}")
    return receipt


def _request(client: CFBDClient, endpoint: str, params: dict) -> dict:
    payload = client.get_json(endpoint, params, max_retries=RETRIES)
    return _save_response(endpoint, params, payload)


def _targets() -> tuple[pd.DataFrame, list[int], list[str]]:
    games = pd.read_parquet(ROOT / "data/raw/cfbd/v2/games/2026.parquet")
    games = games.loc[
        games.season_type.astype(str).str.lower().eq("regular")
        & games.completed.fillna(False).astype(bool)
        & (games.home_classification.astype(str).str.lower().eq("fbs")
           | games.away_classification.astype(str).str.lower().eq("fbs"))
    ].copy()
    games["kickoff_utc"] = pd.to_datetime(games.start_date, utc=True, errors="coerce")
    if games.empty or games.kickoff_utc.isna().any():
        raise ValueError("Current 2026 completed FBS game schedule is unavailable or incomplete")
    weeks = sorted(games.week.dropna().astype(int).unique().tolist())
    fbs = pd.read_parquet(ROOT / "data/raw/cfbd/v2/teams_fbs/2026.parquet")
    teams = sorted(fbs.loc[fbs.classification.astype(str).str.lower().eq("fbs"), "school"].astype(str).unique())
    return games.sort_values(["week", "kickoff_utc", "id"]), weeks, teams


def _setup() -> tuple[CFBDClient, CallBudget]:
    key = load_cfbd_key_file(ROOT / ".env")
    os.environ["CFBD_API_KEY"] = key
    budget = CallBudget(SHARED_BUDGET, HARD_LIMIT)
    client = CFBDClient(call_budget=budget, max_retries=RETRIES)
    return client, budget


def probe() -> None:
    client, budget = _setup()
    quota = client.get_json("/info", {}, max_retries=0)
    if isinstance(quota, list):
        quota = quota[0] if quota else {}
    games, weeks, teams = _targets()
    first_week = weeks[0]
    team = teams[0]
    first_game = int(games.iloc[0].id)
    probes = [
        ("/plays", {"year": 2026, "week": first_week, "seasonType": "regular", "classification": "fbs"}),
        ("/drives", {"year": 2026, "week": first_week, "seasonType": "regular", "classification": "fbs"}),
        ("/games/players", {"year": 2026, "week": first_week, "seasonType": "regular", "classification": "fbs"}),
        ("/roster", {"team": team, "year": 2026}),
    ]
    reports = []
    for endpoint, params in probes:
        try:
            reports.append(_request(client, endpoint, params))
        except Exception as exc:
            reports.append({"endpoint": endpoint, "parameters": params,
                            "status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:180]}"})
        time.sleep(RATE_DELAY)
    # /plays/stats was separately sampled before the broad request plan.
    sample = ARTIFACTS / "raw_cache/plays_stats_sample" / f"{first_game}.parquet"
    receipt = {
        "status": "probe_complete",
        "scope": "2026 current-season what-if only",
        "completed_context_games_with_fbs": len(games),
        "fbs_fbs_games": int((games.home_classification.astype(str).str.lower().eq("fbs")
                               & games.away_classification.astype(str).str.lower().eq("fbs")).sum()),
        "weeks": weeks,
        "fbs_teams": len(teams),
        "sample_plays_stats": {"game_id": first_game,
                                "path": str(sample.relative_to(ROOT)) if sample.exists() else None,
                                "sha256": _digest(sample) if sample.exists() else None,
                                "rows": len(pd.read_parquet(sample)) if sample.exists() else None},
        "endpoint_probes": reports,
        "provider_quota_snapshot": {k: quota.get(k) for k in ("remainingCalls", "usedCalls")},
        "shared_budget_reserved": budget.status()["reserved"],
        "api_attempts": client.api_calls,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(ARTIFACTS / "acquisition_probe.json", receipt)
    print(json.dumps(receipt, indent=2, default=str))


def execute() -> None:
    client, budget = _setup()
    quota = client.get_json("/info", {}, max_retries=0)
    if isinstance(quota, list):
        quota = quota[0] if quota else {}
    remaining = quota.get("remainingCalls")
    games, weeks, teams = _targets()
    game_ids = games.id.astype(int).tolist()
    # Each endpoint is partitioned to keep payloads below common provider caps.
    requests = []
    for week in weeks:
        base = {"year": 2026, "week": week, "seasonType": "regular", "classification": "fbs"}
        requests.extend((endpoint, base.copy()) for endpoint in ("/plays", "/drives", "/games/players"))
    requests.extend(("/plays/stats", {"gameId": game_id}) for game_id in game_ids)
    requests.extend(("/roster", {"team": team, "year": 2026}) for team in teams)
    minimum_attempts = len(requests) + 1  # quota call
    state = budget.status()
    if (not isinstance(remaining, int) or remaining < 10_000 + minimum_attempts
            or state["reserved"] + minimum_attempts > HARD_LIMIT):
        raise RuntimeError("Provider quota or shared CFBD budget cannot cover the bounded acquisition")
    plan = {
        "status": "running",
        "scope": "2026 completed regular-season games involving at least one FBS team; target scoring remains FBS-vs-FBS",
        "game_count": len(game_ids),
        "weeks": weeks,
        "team_count": len(teams),
        "planned_requests": len(requests),
        "reserved_before": state["reserved"],
        "quota_remaining_before": remaining,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(ARTIFACTS / "acquisition_plan.json", plan)
    receipts = []
    for index, (endpoint, params) in enumerate(requests, 1):
        # Skip only a response whose hash still matches its prior receipt.
        partition = "_".join(f"{key}_{params[key]}" for key in sorted(params))
        target = ARTIFACTS / "raw_cache" / endpoint.strip("/").replace("/", "_") / f"{partition}.parquet"
        old_path = ARTIFACTS / "receipts" / f"{hashlib.sha256((endpoint+json.dumps(params,sort_keys=True)).encode()).hexdigest()}.json"
        old = json.loads(old_path.read_text()) if old_path.exists() else None
        if old and target.exists() and old.get("sha256") == _digest(target) and old.get("status") == "success_complete":
            receipts.append(old)
            continue
        try:
            item = _request(client, endpoint, params)
            item["attempts"] = client.api_calls
        except Exception as exc:
            item = {"endpoint": endpoint, "parameters": params, "status": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:180]}"}
            receipts.append(item)
            _write_json(ARTIFACTS / "acquisition_receipts.json", {
                "status": "failed", "completed_requests": index, "receipts": receipts,
                "shared_budget_reserved": budget.status()["reserved"],
                "api_attempts_this_process": client.api_calls,
            })
            raise
        key = hashlib.sha256((endpoint + json.dumps(params, sort_keys=True)).encode()).hexdigest()
        _write_json(ARTIFACTS / "receipts" / f"{key}.json", item)
        receipts.append(item)
        if index % 25 == 0:
            _write_json(ARTIFACTS / "acquisition_receipts.json", {
                "status": "running", "completed_requests": index,
                "planned_requests": len(requests),
                "shared_budget_reserved": budget.status()["reserved"],
                "api_attempts_this_process": client.api_calls,
                "receipts": receipts,
            })
        time.sleep(RATE_DELAY)
    _write_json(ARTIFACTS / "acquisition_receipts.json", {
        "status": "complete", "planned_requests": len(requests),
        "completed_requests": len(receipts),
        "shared_budget_reserved": budget.status()["reserved"],
        "api_attempts_this_process": client.api_calls,
        "receipts": receipts,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"status": "complete", "planned_requests": len(requests),
                      "shared_budget_reserved": budget.status()["reserved"],
                      "api_attempts_this_process": client.api_calls}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "execute"))
    args = parser.parse_args()
    probe() if args.mode == "probe" else execute()
