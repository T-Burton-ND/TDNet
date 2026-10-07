#!/usr/bin/env python3
"""Run matched F18/F19 temporal representation baselines on the SGE cluster.

No 2026 result or target feature path is referenced by this search program.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from gridiron_ml.experiments.constraint_free import (
    ROSTER, SNAPSHOT_MARKET_COLUMNS, assert_feature_contract,
    metadata_for_matrix, representation,
)
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import build_estimator
from nextgen_rounds_scientific_train import make_model
from nextgen_rounds_train import load_stage_matrix

CONFIG = ROOT / "configs/experiments/constraint_free_v1.json"
SETPOINTS = ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json"
DATA = ROOT / "data/nextgen_rounds_2026"
DEFAULT_OUTPUT = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, sort_keys=True, indent=2, default=str) + "\n")
    temp.replace(path)


def load_historical(tier: str):
    stage = "F16" if tier == "F18" else "F17_market"
    values, meta, _, evidence = load_stage_matrix(DATA, stage)
    if not meta.season.between(2013, 2025).all():
        raise ValueError("Historical search input includes a season outside 2013–2025")
    names = [f"matchup__{name}" for name in evidence["source_features"]]
    if tier == "F19":
        all_market = list(evidence["market_features"])
        market_positions = [len(names) + all_market.index(name) for name in SNAPSHOT_MARKET_COLUMNS]
        positions = list(range(len(names))) + market_positions
        values = values[:, positions]
        names.extend(SNAPSHOT_MARKET_COLUMNS)
    records = metadata_for_matrix(names, tier)
    assert_feature_contract(names, records, tier)
    if values.shape[1] != len(names):
        raise ValueError("Historical feature shape differs from provenance")
    if meta.target_game_id.duplicated().any():
        raise ValueError("Historical game identities are not unique")
    return np.asarray(values, dtype=float), meta.reset_index(drop=True), records, evidence


def make_architecture(architecture: str, config: dict):
    if architecture in ("M2", "M4"):
        points = json.loads(SETPOINTS.read_text())
        ident = config["m2_setpoint" if architecture == "M2" else "m4_setpoint"]
        point = next(p for p in points[architecture] if p["id"] == ident)
        return build_estimator(architecture, point), "sklearn"
    model, _ = make_model(architecture, int(config["seed"]), "F16")
    if architecture == "M3":
        model.params["n_estimators"] = int(config["m3_search_estimators"])
        model.params["n_jobs"] = 1
    if architecture == "M5":
        model.max_epochs = int(config["m5_search_epochs"])
        model.patience = int(config["m5_search_patience"])
    return model, "tdnet"


def fit_predict(model, backend: str, X_train, y_train, X_test):
    if backend == "sklearn":
        model.fit(X_train, y_train)
        return np.asarray(model.predict(X_test), dtype=float).reshape(-1)
    columns = [f"repr_{i:04d}" for i in range(X_train.shape[1])]
    model.train(pd.DataFrame(X_train, columns=columns), y_train)
    return np.asarray(model.predict_margin(pd.DataFrame(X_test, columns=columns)), dtype=float).reshape(-1)


def evaluate_task(tier: str, architecture: str, kind: str, config: dict,
                  loader=load_historical) -> dict:
    X, meta, records, evidence = loader(tier)
    years = meta.season.to_numpy(int)
    y = meta.next_game_margin.to_numpy(float)
    if architecture not in ROSTER:
        raise ValueError("Scientific architecture roster changed")
    rows = []
    for year in config["development_years"]:
        train = years < int(year)
        test = years == int(year)
        if train.sum() < 1000 or test.sum() < 100:
            raise ValueError(f"Invalid rolling-origin fold for {year}")
        transform = representation(kind, records, seed=int(config["seed"]))
        fit = transform.fit_transform(X[train], y[train])
        future = transform.transform(X[test])
        if not np.isfinite(fit).all() or not np.isfinite(future).all():
            raise ValueError("Representation produced nonfinite values")
        model, backend = make_architecture(architecture, config)
        prediction = fit_predict(model, backend, fit, y[train], future)
        if prediction.shape != (test.sum(),) or not np.isfinite(prediction).all():
            raise ValueError("Architecture produced invalid predictions")
        # Fixed, training-only scale is intentionally simple during search.
        # Final calibration will use true historical out-of-fold residuals.
        probability = expit(prediction / max(float(np.std(y[train])), 3.0))
        actual = y[test]
        outcome = actual > 0
        rows.append({
            "year": int(year), "games": int(test.sum()),
            "mae": float(np.abs(actual - prediction).mean()),
            "rmse": float(np.sqrt(np.square(actual - prediction).mean())),
            "brier": float(np.square(probability - outcome).mean()),
            "winner_accuracy": float(((prediction > 0) == outcome).mean()),
            "representation_features": int(fit.shape[1]),
        })
    return {
        "tier": tier, "architecture": architecture, "representation": kind,
        "folds": rows, "mean_mae": float(np.mean([r["mae"] for r in rows])),
        "mean_brier": float(np.mean([r["brier"] for r in rows])),
        "worst_year_mae": float(max(r["mae"] for r in rows)),
        "source_features": len(evidence["source_features"]),
        "market_features": [r.name for r in records if r.market_derived],
        "historical_market_quote_timing": "unverified" if tier == "F19" else "none",
        "model_fits": len(rows), "preprocessing_fits": len(rows),
    }


def plan(output: Path, stage: int) -> dict:
    config = json.loads(CONFIG.read_text())
    if config["sge_task_concurrency_max"] > 10:
        raise ValueError("TC may never exceed 10")
    candidates = config[f"stage{stage}_representations"]
    tasks = [
        {"tier": tier, "architecture": architecture, "representation": kind}
        for tier in config["tiers"] for architecture in config["architectures"]
        for kind in candidates
    ]
    if len(tasks) != 2 * len(config["architectures"]) * len(candidates):
        raise ValueError("F18/F19 task budget is asymmetric")
    # These reads validate both feature universes before any cluster submission.
    universes = {}
    for tier in config["tiers"]:
        X, meta, records, evidence = load_historical(tier)
        universes[tier] = {
            "games": len(meta), "features": X.shape[1],
            "market_features": [r.name for r in records if r.market_derived],
            "source_evidence": evidence,
        }
    manifest = {
        "stage": stage, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "starting_git_sha": "780605fe39e9c0c2f6fb9048372df490af50f79d",
        "config_sha256": digest(CONFIG), "source_sha256": digest(Path(__file__)),
        "safety_source_sha256": digest(ROOT / "src/gridiron_ml/experiments/constraint_free.py"),
        "tasks": tasks, "universes": universes,
        "folds": config["development_years"],
        "task_concurrency_max": config["sge_task_concurrency_max"],
        "output": str(output.resolve()),
    }
    path = output / f"stage{stage}/manifest.json"
    if path.exists():
        raise FileExistsError(f"Immutable stage manifest already exists: {path}")
    write_json(path, manifest)
    return {"manifest": str(path), "sha256": digest(path), "tasks": len(tasks), "per_tier": len(tasks)//2}


def run_task(manifest_path: Path, task_id: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != digest(CONFIG) or manifest["source_sha256"] != digest(Path(__file__)):
        raise ValueError("Search code or config changed after planning")
    if manifest["safety_source_sha256"] != digest(ROOT / "src/gridiron_ml/experiments/constraint_free.py"):
        raise ValueError("Safety code changed after planning")
    task = manifest["tasks"][task_id - 1]
    out = manifest_path.parent / "results" / f"task_{task_id:03d}.json"
    if out.exists():
        saved = json.loads(out.read_text())
        if saved.get("status") == "success" and saved.get("manifest_sha256") == digest(manifest_path):
            return saved
        raise FileExistsError(f"Nonreusable task output exists: {out}")
    started = time.monotonic()
    result = {**task, "task_id": task_id, "status": "running",
              "manifest_sha256": digest(manifest_path),
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        result.update(evaluate_task(task["tier"], task["architecture"],
                                    task["representation"], json.loads(CONFIG.read_text())))
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
            raise ValueError(f"Stale task output: {path}")
        rows.append({k: row.get(k) for k in (
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "run", "summarize"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--stage", type=int, choices=(1, 2), default=1)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--task-id", type=int)
    args = p.parse_args()
    if args.action == "plan":
        result = plan(args.output, args.stage)
    elif args.action == "run":
        if not args.manifest or not args.task_id or args.task_id < 1:
            p.error("run requires --manifest and positive --task-id")
        result = run_task(args.manifest, args.task_id)
    else:
        if not args.manifest:
            p.error("summarize requires --manifest")
        result = summarize(args.manifest)
    print(json.dumps(result, indent=2, default=str), flush=True)
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
