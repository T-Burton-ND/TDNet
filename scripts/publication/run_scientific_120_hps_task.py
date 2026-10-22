#!/usr/bin/env python3
"""Run one metrics-only, five-fold CPU HPS candidate on frozen historical inputs."""
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
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from gridiron_ml.experiments.hyperparameter_search import build_tuned_model_config
from gridiron_ml.experiments.constraint_free import metadata_for_matrix
from gridiron_ml.models import TDLinear, TDSpline, TDTree, TDBoosted, TDMLP, TDKNN
from constraint_free_fit_finalists import _representation

DATA_ROOT = Path("/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1/data")
RESULT_ROOT = Path("/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1/results")
MANIFEST = ROOT / "publication/2026/research/scientific_120_hps_v1/candidate_tasks.jsonl"
PLAN = ROOT / "publication/2026/research/scientific_120_hps_v1/plan.json"
WINNERS = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/broad_search/finalists.json")
BASE = {
    "M1": "configs/models/linear/config_ridge.yaml",
    "M2": "configs/models/spline/config_spline_ridge.yaml",
    "M3": "configs/models/tree/config_random_forest.yaml",
    "M4": "configs/models/boosted/config_hist_gradient_boosted.yaml",
    "M5": "configs/models/neural/config_mlp.yaml",
    "M10": "configs/models/knn/config_knn_distance.yaml",
}
CLASSES = {"M1": TDLinear, "M2": TDSpline, "M3": TDTree,
           "M4": TDBoosted, "M5": TDMLP, "M10": TDKNN}
META = {"target_game_id", "season", "week", "target_start_utc", "home_team", "away_team", "actual_margin"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def task_at(task_id: int) -> dict:
    with MANIFEST.open() as f:
        for index, line in enumerate(f, start=1):
            if index == task_id:
                task = json.loads(line)
                if task["task_id"] != task_id:
                    raise ValueError("Task index differs from manifest identity")
                return task
    raise ValueError(f"Missing task {task_id}")


def verify_inputs(task: dict, data_root: Path) -> tuple[pd.DataFrame, dict]:
    plan = json.loads(PLAN.read_text())
    if sha256(MANIFEST) != plan["manifest_sha256"]:
        raise ValueError("Frozen candidate manifest changed")
    if int(os.environ.get("NSLOTS", "1")) > 2:
        raise ValueError("More than two CPU cores allocated to one study task")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise ValueError("This study is CPU only")
    receipt = json.loads((data_root / task["fingerprint"] / "receipt.json").read_text())
    cohort = json.loads((data_root / "cohort_receipt.json").read_text())
    path = data_root / task["fingerprint"] / "features.parquet"
    if (receipt["status"] != "materialized" or receipt["games"] != 7358 or
            receipt["cohort_sha256"] != cohort["cohort_sha256"] or
            sha256(path) != receipt["matrix_sha256"]):
        raise ValueError("Input matrix or common cohort changed")
    frame = pd.read_parquet(path)
    if (len(frame) != 7358 or not frame.target_game_id.is_unique or
            sorted(frame.season.unique().tolist()) != list(range(2013, 2026)) or
            set(frame.columns) != META | set(receipt["feature_names"])):
        raise ValueError("Historical matrix identity or schema differs from receipt")
    return frame, receipt


def winner_for(task: dict) -> dict | None:
    if task["fingerprint"] not in {"F18", "F19"}:
        return None
    finalist = json.loads(WINNERS.read_text())
    cell = [row for row in finalist["winners"]
            if row["tier"] == task["fingerprint"] and row["architecture"] == task["architecture"]]
    if len(cell) != 1:
        raise ValueError("Missing frozen F18/F19 representation winner")
    winner = cell[0]
    if sha256(Path(winner["receipt"])) != winner["receipt_sha256"]:
        raise ValueError("Frozen F18/F19 representation receipt changed")
    return winner


def baseline(winner: dict | None, X: pd.DataFrame, y: np.ndarray, fit: np.ndarray,
             val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not winner or not winner.get("genome") or winner["genome"]["target_mode"] != "residual":
        return tuple(np.zeros(int(mask.sum())) for mask in (fit, val, test))
    if winner["tier"] == "F18":
        value = float(y[fit].mean())
        return tuple(np.full(int(mask.sum()), value) for mask in (fit, val, test))
    spread = X["market_home_spread"].to_numpy(float)
    if not np.isfinite(spread[fit | val | test]).all():
        raise ValueError("F19 residual baseline requires observed market spreads")
    return -spread[fit], -spread[val], -spread[test]


def model_for(task: dict):
    level = task["architecture"]
    config = build_tuned_model_config(base_config_path=ROOT / BASE[level], params=task["params"])
    config["seed"] = int(task["fit_seed"])
    config["loss_function"] = "MAE"
    if level == "M3":
        config.setdefault("params", {})["n_jobs"] = 1
        config["params"]["random_state"] = int(task["fit_seed"])
    if level == "M5":
        config["torch_threads"] = 1
    if task["fingerprint"] in {"F7", "F8", "F17", "F19"}:
        config["allow_market_features_for_training"] = True
    return CLASSES[level](config), config


def market_spread_for(frame: pd.DataFrame, data_root: Path) -> np.ndarray:
    """Use one common archived spread ledger for every cell's secondary ATS score."""
    market = pd.read_parquet(data_root / "F17" / "features.parquet",
                             columns=["target_game_id", "market_home_spread"])
    if not market.target_game_id.equals(frame.target_game_id):
        raise ValueError("Archived spread ledger differs from the common game order")
    spread = market.market_home_spread.to_numpy(float)
    if not np.isfinite(spread).all():
        raise ValueError("Archived spread ledger has missing or nonfinite values")
    return spread


def evaluate(task: dict, frame: pd.DataFrame, receipt: dict, winner: dict | None,
             market_spread: np.ndarray) -> tuple[list[dict], list[dict]]:
    y = frame.actual_margin.to_numpy(float)
    years = frame.season.to_numpy(int)
    feature_names = receipt["feature_names"]
    raw = frame[feature_names].replace([np.inf, -np.inf], np.nan)
    rows, histories = [], []
    for fold in task["folds"]:
        year = int(fold["test_season"])
        if (fold["fit_seasons"] != [2013, year - 2] or
                fold["inner_validation_season"] != year - 1):
            raise ValueError(f"Historical fold boundary changed for {year}")
        fit = (years >= 2013) & (years <= year - 2)
        val = years == year - 1
        test = years == year
        if fit.sum() < 1000 or val.sum() < 100 or test.sum() < 100 or year > 2025:
            raise ValueError(f"Invalid historical fold {year}")
        if winner:
            records = metadata_for_matrix(feature_names, winner["tier"])
            transformer = _representation(winner, records, len(feature_names))
            train_X = transformer.fit_transform(raw.loc[fit].to_numpy(float), y[fit])
            val_X = transformer.transform(raw.loc[val].to_numpy(float))
            test_X = transformer.transform(raw.loc[test].to_numpy(float))
            columns = [f"repr_{i:04d}" for i in range(train_X.shape[1])]
            train_X = pd.DataFrame(train_X, columns=columns)
            val_X = pd.DataFrame(val_X, columns=columns)
            test_X = pd.DataFrame(test_X, columns=columns)
        else:
            train_X, val_X, test_X = (raw.loc[mask].reset_index(drop=True)
                                         for mask in (fit, val, test))
        train_base, val_base, test_base = baseline(winner, raw, y, fit, val, test)
        model, _ = model_for(task)
        model.train(train_X, y[fit] - train_base,
                    X_val=val_X, y_val=y[val] - val_base)
        if task["architecture"] == "M5" and getattr(model, "backend_", "") != "pytorch":
            raise ValueError("M5 CPU PyTorch backend unavailable; fallback changes the study")
        predicted = test_base + np.asarray(model.predict_margin(test_X), float).reshape(-1)
        inner_prediction = val_base + np.asarray(model.predict_margin(val_X), float).reshape(-1)
        if not np.isfinite(predicted).all() or not np.isfinite(inner_prediction).all():
            raise ValueError(f"Nonfinite prediction in fold {year}")
        labels = (y[val] > 0).astype(int)
        if len(np.unique(labels)) == 2:
            calibrator = LogisticRegression(C=1000.0, max_iter=1000)
            calibrator.fit(inner_prediction.reshape(-1, 1), labels)
            probability = calibrator.predict_proba(predicted.reshape(-1, 1))[:, 1]
            calibration = {"method": "prior_year_logistic", "slope": float(calibrator.coef_[0, 0]),
                           "intercept": float(calibrator.intercept_[0])}
        else:
            probability = expit(predicted / max(float(np.std(y[fit])), 3.0))
            calibration = {"method": "prior_fit_scale_fallback"}
        actual_ats = y[test] + market_spread[test]
        predicted_ats = predicted + market_spread[test]
        decisive = actual_ats != 0
        ats_correct = (predicted_ats[decisive] > 0) == (actual_ats[decisive] > 0)
        rows.append({"fingerprint": task["fingerprint"], "architecture": task["architecture"],
                     "trial_id": task["trial_id"], "test_season": year, "fit_games": int(fit.sum()),
                     "inner_validation_games": int(val.sum()), "test_games": int(test.sum()),
                     "margin_mae": float(np.abs(y[test] - predicted).mean()),
                     "brier": float(np.square(probability - (y[test] > 0)).mean()),
                     "winner_accuracy": float(((predicted > 0) == (y[test] > 0)).mean()),
                     "ats_accuracy": float(ats_correct.mean()) if decisive.any() else None,
                     "ats_decisions": int(decisive.sum()),
                     "ats_pushes": int((~decisive).sum()),
                     "calibration": calibration,
                     "feature_count_after_representation": int(train_X.shape[1]),
                     "best_epoch": getattr(model, "best_epoch_", None)})
        if task["architecture"] == "M5":
            history = pd.DataFrame(model.training_history_)
            if not {"epoch", "train_loss", "validation_loss"}.issubset(history):
                raise ValueError("M5 omitted required epoch losses")
            for item in history.to_dict("records"):
                histories.append({"fingerprint": task["fingerprint"],
                                  "architecture": task["architecture"],
                                  "trial_id": task["trial_id"], "test_season": year, **item})
    return rows, histories


def run(task_id: int, data_root: Path, output_root: Path) -> dict:
    task = task_at(task_id)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"task_{task_id:05d}.json"
    history_path = output_root / f"task_{task_id:05d}_history.csv"
    frame, receipt = verify_inputs(task, data_root)
    winner = winner_for(task)
    market_receipt = json.loads((data_root / "F17" / "receipt.json").read_text())
    binding = {"manifest_sha256": sha256(MANIFEST), "matrix_sha256": receipt["matrix_sha256"],
               "ats_spread_matrix_sha256": market_receipt["matrix_sha256"],
               "worker_sha256": sha256(Path(__file__)), "task": task,
               "winner_receipt_sha256": winner["receipt_sha256"] if winner else None}
    if output.exists():
        prior = json.loads(output.read_text())
        if prior.get("status") == "success":
            if prior.get("binding") != binding:
                raise ValueError("Existing result has different code, input, or configuration")
            if task["architecture"] == "M5" and sha256(history_path) != prior.get("history_sha256"):
                raise ValueError("Existing M5 history changed")
            return prior
    started = time.monotonic()
    result = {"status": "running", "task_id": task_id, "binding": binding,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "cpu_slots": int(os.environ.get("NSLOTS", "1")),
              "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    try:
        spread = market_spread_for(frame, data_root)
        rows, history = evaluate(task, frame, receipt, winner, spread)
        if history:
            tmp = history_path.with_suffix(".tmp.csv")
            pd.DataFrame(history).to_csv(tmp, index=False)
            tmp.replace(history_path)
            result["history_sha256"] = sha256(history_path)
        result.update(status="success", fold_metrics=rows, history_rows=len(history),
                      checkpoints_written=0)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=12))
    result["runtime_seconds"] = time.monotonic() - started
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    result = run(args.task_id, args.data_root, args.output_root)
    print(json.dumps({k: v for k, v in result.items() if k in {"status", "task_id", "runtime_seconds", "error"}}, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
