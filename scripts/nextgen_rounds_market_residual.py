#!/usr/bin/env python3
"""Exploratory market-anchored M2/M4 margin-residual variants.

Each model learns (actual margin - archived spread-implied margin). The
archived spread is then added back for prediction. Quote timing remains
unverified, so these variants are retrospective research only.
"""
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
sys.path.insert(0, str(ROOT))

from gridiron_ml.experiments.nextgen_screening_reduced_parallel import (
    build_estimator, future_metrics, residual_probability,
)
from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file
from scripts.nextgen_rounds_ablate import ablation_matrix
from scripts.nextgen_rounds_train import load_stage_matrix


VARIANTS = ("F17_full_residual", "F12_market_residual", "Market_only_residual")


def matrix(output: Path, variant: str):
    if variant == "F17_full_residual":
        return load_stage_matrix(output, "F17_market")
    if variant == "F12_market_residual":
        return ablation_matrix(output, "F17_market_corrected")
    if variant == "Market_only_residual":
        return ablation_matrix(output, "F17_market_only")
    raise ValueError("Unknown market residual variant")


def run(output: Path, variant: str, model: str, setpoint_id: str):
    if variant not in VARIANTS or model not in {"M2", "M4"}:
        raise ValueError("Unknown variant or architecture")
    config = ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json"
    points = json.loads(config.read_text())
    point = next((p for p in points[model] if p["id"] == setpoint_id), None)
    if point is None:
        raise ValueError("Unknown setpoint")
    x, meta, spreads, evidence = matrix(output, variant)
    if not np.isfinite(spreads).all():
        raise ValueError("Market-anchored residual requires a recorded spread for every common-cohort game")
    out = output / "market_residual" / variant / model / setpoint_id
    out.mkdir(parents=True, exist_ok=True)
    binding = {"variant": variant, "model": model, "setpoint": point,
               "input": evidence, "seed": 1701,
               "code_sha256": sha256_file(Path(__file__)),
               "setpoints_sha256": sha256_file(config),
               "target_equation": "residual=actual_home_margin+home_spread; predicted_home_margin=predicted_residual-home_spread"}
    target = out / "result.json"
    if target.exists():
        prior = json.loads(target.read_text())
        if prior.get("status") == "success":
            if (prior.get("binding") != binding
                    or prior.get("predictions_sha256") != sha256_file(out / "predictions.parquet")):
                raise ValueError("Existing residual result changed")
            return prior
    years = meta.season.to_numpy(int)
    y = meta.next_game_margin.to_numpy(float)
    residual_target = y + spreads
    train = years <= 2023
    result = {"status": "incomplete", "variant": variant, "model": model,
              "setpoint": setpoint_id, "binding": binding,
              "training_games": int(train.sum()), "feature_count": x.shape[1],
              "development_games": {str(y): int((years == y).sum()) for y in (2024, 2025)},
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "quote_timestamp_available": False,
              "retrospective_research_only": True}
    start = time.monotonic()
    try:
        calibration_residuals = []
        for cutoff in (2019, 2021):
            fit, calibrate = years <= cutoff, (years > cutoff) & (years <= cutoff + 2)
            estimator = build_estimator(model, point)
            estimator.fit(x[fit], residual_target[fit])
            predicted_margin = estimator.predict(x[calibrate]) - spreads[calibrate]
            calibration_residuals.extend(y[calibrate] - predicted_margin)
        estimator = build_estimator(model, point)
        estimator.fit(x[train], residual_target[train])
        evaluate = years >= 2024
        predicted_margin = estimator.predict(x[evaluate]) - spreads[evaluate]
        probabilities = residual_probability(predicted_margin, calibration_residuals)
        eval_meta = meta.loc[evaluate].reset_index(drop=True)
        table = pd.DataFrame({"target_game_id": eval_meta.target_game_id.astype(int),
                              "season": eval_meta.season.astype(int),
                              "actual_margin": y[evaluate],
                              "predicted_margin": predicted_margin,
                              "home_win_probability": probabilities})
        for year in (2024, 2025):
            mask = eval_meta.season.eq(year).to_numpy()
            result[f"metrics_{year}"] = future_metrics(
                y[evaluate][mask], predicted_margin[mask], probabilities[mask], spreads[evaluate][mask])
        temp = out / "predictions.tmp.parquet"
        table.to_parquet(temp, index=False, compression="zstd")
        temp.replace(out / "predictions.parquet")
        result.update(status="success", predictions_sha256=sha256_file(out / "predictions.parquet"),
                      calibration_residuals=len(calibration_residuals))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=8))
    result["runtime_seconds"] = time.monotonic() - start
    temp = out / "result.tmp.json"
    temp.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    temp.replace(target)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/nextgen_rounds_2026")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--model", choices=("M2", "M4"), required=True)
    parser.add_argument("--setpoint", required=True)
    args = parser.parse_args()
    result = run(args.output, args.variant, args.model, args.setpoint)
    print(json.dumps({k: v for k, v in result.items() if k not in {"binding", "traceback"}}, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
