#!/usr/bin/env python3
"""Matched successive-halving screen for additional F18/F19 reducers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from constraint_free_broad_universe import load_broad_historical
from constraint_free_search import digest, fit_predict, make_architecture, write_json
from gridiron_ml.experiments.constraint_free import ROSTER
from gridiron_ml.experiments.constraint_free_extra_reducers import CANDIDATES, extra_representation

CONFIG = ROOT / "configs/experiments/constraint_free_extra_v1.json"
CORE = ROOT / "src/gridiron_ml/experiments/constraint_free_extra_reducers.py"
UNIVERSE = ROOT / "scripts/constraint_free_broad_universe.py"
BASE = ROOT / "scripts/constraint_free_search.py"
BASE_CONFIG = ROOT / "configs/experiments/constraint_free_v1.json"
SAFETY = ROOT / "src/gridiron_ml/experiments/constraint_free.py"
DEFAULT_OUTPUT = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/broad_search/extra_reducers")


def bindings() -> dict:
    return {str(path): digest(path) for path in (CONFIG, CORE, UNIVERSE, BASE,
                                                BASE_CONFIG, SAFETY, Path(__file__))}


def evaluate(tier: str, architecture: str, kind: str, years_to_score: list[int]) -> dict:
    X, meta, records, evidence = load_broad_historical(tier)
    if architecture not in ROSTER:
        raise ValueError("Architecture is not in the scientific roster")
    years = meta.season.to_numpy(int)
    y = meta.next_game_margin.to_numpy(float)
    base_config = json.loads(BASE_CONFIG.read_text())
    folds = []
    for year in years_to_score:
        train, test = years < year, years == year
        if train.sum() < 1000 or test.sum() < 100:
            raise ValueError(f"Invalid rolling-origin fold: {year}")
        transform = extra_representation(kind, X.shape[1], seed=1729)
        fit = transform.fit_transform(X[train], y[train])
        future = transform.transform(X[test])
        if not np.isfinite(fit).all() or not np.isfinite(future).all():
            raise ValueError("Reducer produced a nonfinite value")
        model, backend = make_architecture(architecture, base_config)
        prediction = fit_predict(model, backend, fit, y[train], future)
        if prediction.shape != (test.sum(),) or not np.isfinite(prediction).all():
            raise ValueError("Architecture produced an invalid prediction")
        actual = y[test]
        probability = expit(prediction / max(float(np.std(y[train])), 3.0))
        folds.append({"year": year, "games": int(test.sum()),
                      "mae": float(np.abs(actual - prediction).mean()),
                      "rmse": float(np.sqrt(np.square(actual - prediction).mean())),
                      "brier": float(np.square(probability - (actual > 0)).mean()),
                      "winner_accuracy": float(((prediction > 0) == (actual > 0)).mean()),
                      "representation_features": int(fit.shape[1])})
    return {"tier": tier, "architecture": architecture, "representation": kind,
            "folds": folds, "mean_mae": float(np.mean([f["mae"] for f in folds])),
            "mean_brier": float(np.mean([f["brier"] for f in folds])),
            "worst_year_mae": float(max(f["mae"] for f in folds)),
            "model_fits": len(folds), "preprocessing_fits": len(folds),
            "source_features": len(evidence["source_features"]),
            "historical_market_quote_timing": "unverified" if tier == "F19" else "none"}


def plan(output: Path, stage: int) -> dict:
    cfg = json.loads(CONFIG.read_text())
    if cfg["sge_task_concurrency_max"] > 10 or tuple(cfg["candidates"]) != CANDIDATES:
        raise ValueError("Reducer menu or TC cap differs from checked implementation")
    if stage == 1:
        candidates = cfg["candidates"]
        tasks = [{"tier": tier, "architecture": architecture, "representation": kind}
                 for tier in cfg["tiers"] for architecture in cfg["architectures"]
                 for kind in candidates]
        prior_sha = None
        years = cfg["screen_years"]
    else:
        prior = output / "stage1/leaderboard.csv"
        table = pd.read_csv(prior)
        expected = len(cfg["tiers"]) * len(cfg["architectures"]) * len(cfg["candidates"])
        if len(table) != expected or table.duplicated(["tier", "architecture", "representation"]).any():
            raise ValueError("Stage 1 screen is incomplete")
        tasks = []
        for tier in cfg["tiers"]:
            for architecture in cfg["architectures"]:
                ranked = table.loc[table.tier.eq(tier) & table.architecture.eq(architecture)
                                   & table.status.eq("success")].sort_values(
                    ["mean_mae", "mean_brier", "representation"])
                if len(ranked) < cfg["promotion_per_cell"]:
                    raise ValueError(f"Insufficient successful candidates: {tier} {architecture}")
                tasks.extend({"tier": tier, "architecture": architecture,
                              "representation": kind}
                             for kind in ranked.representation.head(cfg["promotion_per_cell"]))
        prior_sha = digest(prior)
        years = cfg["confirm_years"]
    for tier in cfg["tiers"]:
        load_broad_historical(tier)
    manifest = {"stage": stage, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "universe": "F16_A_plus_distinct_F12_BC", "bindings": bindings(),
                "screen_leaderboard_sha256": prior_sha,
                "tasks": tasks, "years_to_score": years,
                "task_concurrency_max": cfg["sge_task_concurrency_max"],
                "historical_market_quote_timing": "unverified; development only"}
    path = output / f"stage{stage}" / "manifest.json"
    if path.exists():
        raise FileExistsError(f"Immutable manifest already exists: {path}")
    write_json(path, manifest)
    return {"manifest": str(path), "sha256": digest(path),
            "tasks": len(tasks), "per_tier": len(tasks) // 2}


def run_task(manifest_path: Path, task_id: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["bindings"] != bindings():
        raise ValueError("Reducer source changed after planning")
    if manifest["stage"] == 2 and digest(manifest_path.parent.parent / "stage1/leaderboard.csv") != manifest["screen_leaderboard_sha256"]:
        raise ValueError("Stage 1 promotion source changed")
    if task_id < 1 or task_id > len(manifest["tasks"]):
        raise IndexError("Task ID outside manifest")
    task = manifest["tasks"][task_id - 1]
    out = manifest_path.parent / "results" / f"task_{task_id:03d}.json"
    if out.exists():
        prior = json.loads(out.read_text())
        if prior.get("status") == "success" and prior.get("manifest_sha256") == digest(manifest_path):
            return prior
        raise FileExistsError(f"Nonreusable task receipt: {out}")
    result = {**task, "task_id": task_id, "status": "running",
              "manifest_sha256": digest(manifest_path),
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        result.update(evaluate(task["tier"], task["architecture"],
                               task["representation"], manifest["years_to_score"]))
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
        row = json.loads(path.read_text())
        if row.get("manifest_sha256") != digest(manifest_path):
            raise ValueError(f"Stale receipt: {path}")
        rows.append({key: row.get(key) for key in (
            "task_id", "tier", "architecture", "representation", "status",
            "mean_mae", "mean_brier", "worst_year_mae", "model_fits",
            "runtime_seconds", "error")})
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
