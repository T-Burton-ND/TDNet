#!/usr/bin/env python3
"""Future staged CFBD executor; never starts acquisition without --execute-stage.

Run nextgen_preflight.py first. This module is not invoked by the preflight pass.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import (  # noqa: E402
    CFBDClient, CallBudget, load_cfbd_key_file,
)
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (  # noqa: E402
    AcquisitionLedger, atomic_json, build_manifest, execute_request, quota_allows,
)


def validate_stage_e_approval(approval: dict) -> set[int]:
    """Require sample evidence and a concrete value decision before any Stage-E pull."""
    features = approval.get("unique_f10_f12_features")
    if not isinstance(features, list) or not features or any(
        not isinstance(item, dict)
        or item.get("generation") not in {"F10", "F12"}
        or not isinstance(item.get("name"), str)
        or not item["name"].strip()
        or item.get("uniquely_enabled_by_plays_stats") is not True
        for item in features
    ):
        raise ValueError("Stage E needs concrete F10/F12 features uniquely enabled by /plays/stats")
    game_ids = approval.get("approved_game_ids")
    if (approval.get("decision") not in {"full", "subset"}
            or not isinstance(game_ids, list) or not game_ids
            or any(not isinstance(game_id, int) or game_id <= 0 for game_id in game_ids)
            or len(game_ids) != len(set(game_ids))
            or not all(approval.get(key) is True for key in (
                "sample_coverage_and_stat_meaning_verified",
                "response_cap_handling_verified",
                "incremental_value_reviewed",
                "quota_and_storage_reviewed"))):
        raise ValueError("Stage E needs sample, cap, incremental-value, quota, storage, and full/subset approval")
    return set(game_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Future staged nextgen acquisition")
    parser.add_argument("--execute-stage", choices=["A", "B", "C", "D", "E"], required=True)
    parser.add_argument("--max-requests", type=int, default=None)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/experiments/nextgen_fingerprints_v1.json").read_text())
    root = Path(config["artifact_root"])
    inventory = json.loads((ROOT / "configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json").read_text())
    approved_game_ids = None
    if args.execute_stage == "E":
        gate = root / "results/preflight/plays_stats_approval.json"
        approval = json.loads(gate.read_text()) if gate.exists() else {}
        approved_game_ids = validate_stage_e_approval(approval)
    manifest, summary = build_manifest(inventory, ROOT, root,
                                       include_plays_stats=args.execute_stage == "E")
    if args.execute_stage != "A" and not summary["schedule_authoritative"]:
        raise RuntimeError("Fetch all 16 fresh /games schedules in Stage A, then rerun preflight")
    if summary["unresolved_partial_requests"] or summary["unresolved_review_requests"]:
        raise RuntimeError("A capped or anomalous request remains unresolved; review before more acquisition")
    saved_manifest = root / "results/preflight/cfbd_request_manifest_v1.jsonl"
    if not saved_manifest.exists():
        raise RuntimeError("Missing persisted preflight manifest")
    with saved_manifest.open() as handle:
        saved = {entry["request_id"]: entry for entry in map(json.loads, handle)}
    saved_ids = set(saved)
    current_default_ids = {item["request_id"] for item in manifest if item["endpoint"] != "/plays/stats"}
    if saved_ids != current_default_ids:
        raise RuntimeError("Request identities changed since preflight; rebuild and review the manifest")
    saved_summary_path = root / "results/preflight/cfbd_plan_summary_v1.json"
    if args.execute_stage != "A":
        saved_summary = json.loads(saved_summary_path.read_text()) if saved_summary_path.exists() else {}
        if not saved_summary.get("schedule_authoritative"):
            raise RuntimeError("Rerun preflight after all fresh /games schedules are acquired")
    if any(item["status"] == "planned" and saved[item["request_id"]]["status"] == "skipped_existing_complete"
           for item in manifest if item["endpoint"] != "/plays/stats"):
        raise RuntimeError("A reused cache failed validation since preflight; refresh the manifest")
    pending = [item for item in manifest if item["status"] == "planned"]
    stage_items = [item for item in pending if item["stage"] == args.execute_stage]
    if approved_game_ids is not None:
        available_ids = {item["game_id"] for item in manifest if item["endpoint"] == "/plays/stats"}
        if not approved_game_ids <= available_ids:
            raise RuntimeError("Stage E approval contains game IDs outside the fresh schedule")
        if approval["decision"] == "full" and approved_game_ids != available_ids:
            raise RuntimeError("Stage E full approval must cover the fresh full-game schedule")
        stage_items = [item for item in stage_items if item["game_id"] in approved_game_ids]
    if args.max_requests is not None:
        if args.max_requests <= 0:
            raise ValueError("--max-requests must be positive")
        stage_items = stage_items[:args.max_requests]
    if not stage_items:
        print(f"Stage {args.execute_stage}: no pending requests")
        return
    policy = config["cfbd_api_call_budget"]
    budget = CallBudget(Path(policy["ledger"]), policy["hard_limit"])
    if budget.status()["reserved"] + len(stage_items) > policy["hard_limit"]:
        raise RuntimeError("Selected stage's minimum first attempts exceed the local hard cap")
    os.environ["CFBD_API_KEY"] = load_cfbd_key_file(ROOT / ".env")
    client = CFBDClient(call_budget=budget, max_retries=2)
    quota = client.get_json("/info", {}, max_retries=0)
    if isinstance(quota, list):
        quota = quota[0] if quota else {}
    remaining = quota.get("remainingCalls")
    reserved = budget.status()["reserved"]
    if not isinstance(remaining, int) or not quota_allows(
        remaining, reserved, reserve=policy["minimum_reserve"],
        hard_limit=policy["hard_limit"]):
        raise RuntimeError("Live quota cannot cover remaining local budget and provider reserve")
    free = shutil.disk_usage(root).free
    plan_summary = root / "results/preflight/cfbd_plan_summary_v1.json"
    if not plan_summary.exists():
        raise RuntimeError("Missing preflight plan/storage estimate")
    if args.execute_stage == "E":
        from nextgen_preflight import storage_estimate  # local script, no acquisition on import
        selected_plan = [item for item in manifest
                         if item["endpoint"] != "/plays/stats" or item["game_id"] in approved_game_ids]
        estimate = storage_estimate(selected_plan, root)
    else:
        estimate = json.loads(plan_summary.read_text())["storage_estimate"]
    if (estimate["estimated_total_bytes"] > estimate["soft_limit_bytes"] or
            estimate["estimated_total_bytes"] > free):
        raise RuntimeError("Storage estimate exceeds soft limit or current free space")
    ledger = AcquisitionLedger(root)
    counts = Counter()
    for item in stage_items:
        record = execute_request(item, ledger, client)
        counts[record["status"]] += 1
        if record["status"] in {"success_suspected_partial", "needs_review", "failed_final", "failed_retryable"}:
            # Stop on a cap, auth/permanent error, or unstable request.
            break
        if budget.status()["reserved"] >= policy["hard_limit"]:
            break
        time.sleep(0.25)
    report = {"stage": args.execute_stage, "attempted_request_count": len(stage_items),
              "statuses": dict(counts), "api_attempts_this_run": client.api_calls,
              "budget_reserved": budget.status()["reserved"], "quota_snapshot": quota,
              "at_utc": datetime.now(timezone.utc).isoformat(),
              "plan_new_calls_at_start": summary["new_planned_calls"]}
    atomic_json(root / "results/preflight" / f"stage_{args.execute_stage}_latest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
