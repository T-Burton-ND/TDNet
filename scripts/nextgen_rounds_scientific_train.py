#!/usr/bin/env python3
"""Train the remaining scientific architectures on the F12-corrected A ladder."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gridiron_ml.experiments.hyperparameter_search import build_tuned_model_config
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import (
    future_metrics,
    residual_probability,
)
from gridiron_ml.models import TDKNN, TDLinear, TDMLP, TDTree
from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file
from nextgen_rounds_train import load_stage_matrix


MODEL_CLASSES = {"M1": TDLinear, "M3": TDTree, "M5": TDMLP, "M10": TDKNN}
CONFIG_PATH = ROOT / "configs/experiments/nextgen_rounds_scientific_models_v1.json"
DEFAULT_DATA_ROOT = ROOT / "data/nextgen_rounds_2026"


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text())
    if cfg.get("selection_source", {}).get("sha256") != "1a7280231b69e467674f6642c44117a6f7090c1b5d39af25a68a4e55799633d0":
        raise ValueError("Scientific model parameter provenance changed")
    if set(cfg["models"]) != set(MODEL_CLASSES):
        raise ValueError("Scientific architecture matrix changed")
    return cfg


def make_model(model_name: str, seed: int, stage: str, *, smoke: bool = False):
    cfg = load_config()
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported scientific model: {model_name}")
    model_spec = cfg["models"][model_name]
    base = ROOT / model_spec["base_config"]
    model_cfg = build_tuned_model_config(base_config_path=base,
                                         params=model_spec["params"])
    model_cfg["seed"] = int(seed)
    if model_name == "M3":
        model_cfg.setdefault("params", {})["random_state"] = int(seed)
    if stage == "F17_market":
        model_cfg["allow_market_features_for_training"] = True
    if smoke:
        if model_name == "M3":
            model_cfg["params"]["n_estimators"] = 12
        if model_name == "M5":
            model_cfg.update(hidden_layers=[8], batch_size=16, max_epochs=4,
                             patience=2, normalization="batch_norm")
    return MODEL_CLASSES[model_name](model_cfg), model_cfg


def _binding(data_root: Path, stage: str, model_name: str, seed: int,
             model_cfg: dict, evidence: dict, manifest_sha256: str | None) -> dict:
    cfg = load_config()
    source_files = [
        Path(__file__), Path(__file__).with_name("nextgen_rounds_train.py"),
        CONFIG_PATH, ROOT / cfg["models"][model_name]["base_config"],
    ]
    return {
        "stage": stage, "model": model_name, "seed": int(seed),
        "model_config": model_cfg,
        "input": evidence,
        "config_sha256": sha256_file(CONFIG_PATH),
        "code_sha256": {str(path): sha256_file(path) for path in source_files},
        "selection_source": cfg["selection_source"],
        "training_years": [2013, 2023],
        "calibration_cutoffs": [2019, 2021],
        "evaluation_years": [2024, 2025],
        "market_research_only": stage == "F17_market",
        "data_root": str(data_root.resolve()),
        "manifest_sha256": manifest_sha256,
    }


def run_task(data_root: Path, stage: str, model_name: str, seed: int,
             manifest_sha256: str | None = None) -> dict:
    cfg = load_config()
    if stage not in cfg["stages"] or model_name not in MODEL_CLASSES or seed not in cfg["seeds"]:
        raise ValueError("Task is outside the frozen scientific model matrix")
    x, meta, spreads, evidence = load_stage_matrix(data_root, stage)
    feature_names = [f"matchup__{name}" for name in evidence["source_features"]]
    feature_names.extend(evidence["market_features"])
    if len(feature_names) != x.shape[1] or len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature matrix names do not match its columns")
    X = pd.DataFrame(x, columns=feature_names)
    years = meta.season.to_numpy(dtype=int)
    y = meta.next_game_margin.to_numpy(dtype=float)
    train_mask = (years >= 2013) & (years <= 2023)
    eval_mask = (years == 2024) | (years == 2025)
    if (train_mask.sum() != 6179 or (years == 2024).sum() != 626
            or (years == 2025).sum() != 553 or eval_mask.sum() != 1179):
        raise ValueError("The corrected F12 A game splits changed")

    model, model_cfg = make_model(model_name, seed, stage)
    binding = _binding(data_root, stage, model_name, seed, model_cfg, evidence,
                       manifest_sha256)
    out = data_root / "scientific_model_runs" / "experiments" / stage / model_name / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    predictions_path = out / "predictions.parquet"
    if result_path.exists():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "success":
            if prior.get("binding") != binding or not predictions_path.exists() or prior.get(
                    "predictions_sha256") != sha256_file(predictions_path):
                raise ValueError("Existing successful result has changed")
            return prior

    started = time.monotonic()
    result = {
        "status": "incomplete", "stage": stage, "model": model_name, "seed": seed,
        "binding": binding,
        "training_games": int(train_mask.sum()),
        "development_games": {"2024": 626, "2025": 553},
        "feature_count": int(X.shape[1]),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        residuals = []
        for cutoff in cfg["split"]["calibration_cutoffs"]:
            fit_mask = years <= cutoff
            cal_mask = (years > cutoff) & (years <= cutoff + 2)
            if not fit_mask.any() or cal_mask.sum() == 0:
                raise ValueError(f"Empty rolling calibration fold at cutoff {cutoff}")
            calibrator_model, _ = make_model(model_name, seed, stage)
            calibrator_model.train(X.loc[fit_mask], y[fit_mask])
            residuals.extend(y[cal_mask] - calibrator_model.predict_margin(X.loc[cal_mask]))

        model.train(X.loc[train_mask], y[train_mask])
        predicted = np.asarray(model.predict_margin(X.loc[eval_mask]), dtype=float).reshape(-1)
        probabilities = residual_probability(predicted, residuals)
        eval_meta = meta.loc[eval_mask].reset_index(drop=True)
        prediction = pd.DataFrame({
            "target_game_id": eval_meta.target_game_id.astype(int),
            "season": eval_meta.season.astype(int),
            "actual_margin": y[eval_mask],
            "predicted_margin": predicted,
            "home_win_probability": probabilities,
        })
        for year in (2024, 2025):
            year_mask = eval_meta.season.eq(year).to_numpy()
            result[f"metrics_{year}"] = future_metrics(
                y[eval_mask][year_mask], predicted[year_mask], probabilities[year_mask],
                spreads[eval_mask][year_mask])
        temp = out / "predictions.tmp.parquet"
        prediction.to_parquet(temp, index=False, compression="zstd")
        temp.replace(predictions_path)
        result.update(status="success", predictions_sha256=sha256_file(predictions_path),
                      calibration_residuals=len(residuals),
                      model_backend=getattr(model, "backend_", "sklearn"),
                      calibration_residual_method="rolling-origin absolute training residuals")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=12))
    result["runtime_seconds"] = time.monotonic() - started
    temp = out / "result.tmp.json"
    temp.write_text(json.dumps(result, sort_keys=True, indent=2, default=str) + "\n")
    temp.replace(result_path)
    return result


def smoke_test(output: Path) -> dict:
    rng = np.random.default_rng(26084)
    X = pd.DataFrame(rng.normal(size=(64, 8)), columns=[f"feature_{i}" for i in range(8)])
    X.iloc[0, 0] = np.nan
    y = 2 * X.fillna(0).iloc[:, 0].to_numpy() - X.iloc[:, 1].to_numpy()
    checks = {}
    for model_name in MODEL_CLASSES:
        model, _ = make_model(model_name, 1701, "F12_corrected", smoke=True)
        model.train(X.iloc[:48], y[:48])
        prediction = np.asarray(model.predict_margin(X.iloc[48:]), dtype=float)
        if prediction.shape != (16,) or not np.isfinite(prediction).all():
            raise ValueError(f"CPU smoke failed for {model_name}")
        checks[model_name] = {"predictions": len(prediction),
                              "backend": getattr(model, "backend_", "sklearn")}
    receipt = {
        "status": "passed", "models": checks,
        "config_sha256": sha256_file(CONFIG_PATH),
        "code_sha256": sha256_file(Path(__file__)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_test.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stage", choices=load_config()["stages"])
    parser.add_argument("--model", choices=tuple(MODEL_CLASSES))
    parser.add_argument("--seed", type=int, choices=load_config()["seeds"])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-output", type=Path)
    args = parser.parse_args()
    if args.smoke_test:
        result = smoke_test(args.smoke_output or args.data_root / "scientific_model_runs")
    else:
        if not all((args.stage, args.model, args.seed is not None)):
            parser.error("--stage, --model, and --seed are required unless --smoke-test is used")
        result = run_task(args.data_root, args.stage, args.model, args.seed)
    print(json.dumps({k: v for k, v in result.items() if k not in {"binding", "traceback"}},
                     indent=2, default=str))
    if result["status"] != "success" and result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
