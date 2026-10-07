#!/usr/bin/env python3
"""Matched historical search over F16 A plus the 98 distinct F12 B/C fields."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import pandas as pd

from constraint_free_broad_universe import load_broad_historical
from constraint_free_search import ROOT, digest, evaluate_task, write_json

CONFIG = ROOT / "configs/experiments/constraint_free_v1.json"
BASE_SEARCH = ROOT / "scripts/constraint_free_search.py"
UNIVERSE = ROOT / "scripts/constraint_free_broad_universe.py"
SAFETY = ROOT / "src/gridiron_ml/experiments/constraint_free.py"
DEFAULT_OUTPUT = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/broad_search")


def bindings() -> dict:
    return {str(path): digest(path) for path in (CONFIG, BASE_SEARCH, UNIVERSE, SAFETY,
                                                Path(__file__))}


def plan(output: Path, stage: int) -> dict:
    config = json.loads(CONFIG.read_text())
    if config["sge_task_concurrency_max"] > 10:
        raise ValueError("TC may never exceed 10")
    tasks = [{"tier": tier, "architecture": architecture, "representation": kind}
             for tier in config["tiers"] for architecture in config["architectures"]
             for kind in config[f"stage{stage}_representations"]]
    if len(tasks) != 2 * len(config["architectures"]) * len(config[f"stage{stage}_representations"]):
        raise ValueError("F18/F19 broad search budgets differ")
    universes = {}
    for tier in config["tiers"]:
        X, meta, records, evidence = load_broad_historical(tier)
        universes[tier] = {"games": len(meta), "features": X.shape[1],
                           "market_features": [record.name for record in records if record.market_derived],
                           "source_evidence": evidence}
    manifest = {"stage": stage, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "universe": "F16_A_plus_distinct_F12_BC", "bindings": bindings(),
                "tasks": tasks, "universes": universes,
                "development_years": config["development_years"],
                "task_concurrency_max": config["sge_task_concurrency_max"],
                "historical_market_quote_timing": "unverified; development only"}
    path = output / f"stage{stage}" / "manifest.json"
    if path.exists():
        raise FileExistsError(f"Immutable manifest exists: {path}")
    write_json(path, manifest)
    return {"manifest": str(path), "sha256": digest(path), "tasks": len(tasks),
            "per_tier": len(tasks) // 2}


def run_task(manifest_path: Path, task_id: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["bindings"] != bindings():
        raise ValueError("Bound source code or config changed after planning")
    if task_id < 1 or task_id > len(manifest["tasks"]):
        raise IndexError("Task ID outside manifest")
    task = manifest["tasks"][task_id - 1]
    out = manifest_path.parent / "results" / f"task_{task_id:03d}.json"
    if out.exists():
        prior = json.loads(out.read_text())
        if prior.get("status") == "success" and prior.get("manifest_sha256") == digest(manifest_path):
            return prior
        raise FileExistsError(f"Nonreusable task output: {out}")
    started = time.monotonic()
    result = {**task, "task_id": task_id, "status": "running",
              "manifest_sha256": digest(manifest_path),
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        result.update(evaluate_task(task["tier"], task["architecture"],
                                    task["representation"], json.loads(CONFIG.read_text()),
                                    loader=load_broad_historical))
        result["status"] = "success"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=12))
    result["runtime_seconds"] = time.monotonic() - started
    write_json(out, result)
    return result


def summarize(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for task_id in range(1, len(manifest["tasks"]) + 1):
        path = manifest_path.parent / "results" / f"task_{task_id:03d}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result.get("manifest_sha256") != digest(manifest_path):
            raise ValueError(f"Stale result {path}")
        rows.append({key: result.get(key) for key in (
            "task_id", "tier", "architecture", "representation", "status",
            "mean_mae", "mean_brier", "worst_year_mae", "model_fits", "runtime_seconds", "error")})
    table = pd.DataFrame(rows)
    dest = manifest_path.parent / "leaderboard.csv"
    table.to_csv(dest, index=False)
    return {"completed": len(rows), "expected": len(manifest["tasks"]),
            "success": int(table.status.eq("success").sum()) if len(table) else 0,
            "failed": int(table.status.eq("failed").sum()) if len(table) else 0,
            "leaderboard": str(dest)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "summarize"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    if args.action == "plan":
        result = plan(args.output, args.stage)
    elif args.action == "run":
        if not args.manifest or not args.task_id:
            parser.error("run requires --manifest and --task-id")
        result = run_task(args.manifest, args.task_id)
    else:
        if not args.manifest:
            parser.error("summarize requires --manifest")
        result = summarize(args.manifest)
    print(json.dumps(result, indent=2, default=str), flush=True)
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
