#!/usr/bin/env python3
"""One-family and market ancestry ablations on the frozen F13–F17 A cohort."""
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
from scripts.nextgen_rounds_train import load_stage_matrix


ABLATIONS = {
    "F14_only": ("F14", 339, 347, 355),
    "F15_only": ("F15", 339, 355, 363),
    "F16_only": ("F16", 339, 363, 369),
    "F17_market_corrected": ("F17_market", 339, 369, 381),
    "F17_market_only": ("F17_market", 0, 369, 381),
}


def ablation_matrix(output: Path, ablation: str):
    if ablation not in ABLATIONS:
        raise ValueError("Unknown ablation")
    source, baseline_columns, begin, end = ABLATIONS[ablation]
    x, meta, spreads, evidence = load_stage_matrix(output, source)
    if x.shape[1] != (381 if source == "F17_market" else end):
        raise ValueError("Source matrix no longer has the frozen A feature order")
    selected = np.column_stack([x[:, :baseline_columns], x[:, begin:end]])
    if selected.shape[1] != baseline_columns + end - begin:
        raise ValueError("Ablation column selection failed")
    return selected, meta, spreads, evidence


def run(output: Path, ablation: str, model: str, setpoint_id: str):
    if model not in {"M2", "M4"}:
        raise ValueError("Unknown architecture")
    config_path = ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json"
    points = json.loads(config_path.read_text())
    point = next((p for p in points[model] if p["id"] == setpoint_id), None)
    if point is None:
        raise ValueError("Unknown frozen setpoint")
    x, meta, spreads, evidence = ablation_matrix(output, ablation)
    out = output / "ablations" / ablation / model / setpoint_id
    out.mkdir(parents=True, exist_ok=True)
    binding = {"ablation": ablation, "model": model, "setpoint": point,
               "input": evidence, "seed": 1701,
               "ablation_code_sha256": sha256_file(Path(__file__)),
               "source_train_code_sha256": sha256_file(ROOT / "scripts/nextgen_rounds_train.py"),
               "setpoints_sha256": sha256_file(config_path)}
    target = out / "result.json"
    if target.exists():
        prior = json.loads(target.read_text())
        if prior.get("status") == "success":
            if (prior.get("binding") != binding
                    or prior.get("predictions_sha256") != sha256_file(out / "predictions.parquet")):
                raise ValueError("Existing ablation result provenance changed")
            return prior
    years = meta.season.to_numpy(int)
    y = meta.next_game_margin.to_numpy(float)
    train = years <= 2023
    result = {"status": "incomplete", "ablation": ablation, "model": model,
              "setpoint": setpoint_id, "binding": binding,
              "training_games": int(train.sum()),
              "development_games": {str(y): int((years == y).sum()) for y in (2024, 2025)},
              "feature_count": x.shape[1],
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "market_research_only": ablation.startswith("F17_")}
    start = time.monotonic()
    try:
        residuals = []
        for cutoff in (2019, 2021):
            fit, calibrate = years <= cutoff, (years > cutoff) & (years <= cutoff + 2)
            estimator = build_estimator(model, point)
            estimator.fit(x[fit], y[fit])
            residuals.extend(y[calibrate] - estimator.predict(x[calibrate]))
        estimator = build_estimator(model, point)
        estimator.fit(x[train], y[train])
        evaluate = years >= 2024
        prediction = estimator.predict(x[evaluate])
        probability = residual_probability(prediction, residuals)
        eval_meta = meta.loc[evaluate].reset_index(drop=True)
        table = pd.DataFrame({"target_game_id": eval_meta.target_game_id.astype(int),
                              "season": eval_meta.season.astype(int),
                              "actual_margin": y[evaluate],
                              "predicted_margin": prediction,
                              "home_win_probability": probability})
        for year in (2024, 2025):
            mask = eval_meta.season.eq(year).to_numpy()
            result[f"metrics_{year}"] = future_metrics(
                y[evaluate][mask], prediction[mask], probability[mask], spreads[evaluate][mask])
        temp = out / "predictions.tmp.parquet"
        table.to_parquet(temp, index=False, compression="zstd")
        temp.replace(out / "predictions.parquet")
        result.update(status="success", predictions_sha256=sha256_file(out / "predictions.parquet"),
                      calibration_residuals=len(residuals))
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
    parser.add_argument("--ablation", choices=tuple(ABLATIONS), required=True)
    parser.add_argument("--model", choices=("M2", "M4"), required=True)
    parser.add_argument("--setpoint", required=True)
    args = parser.parse_args()
    result = run(args.output, args.ablation, args.model, args.setpoint)
    print(json.dumps({k: v for k, v in result.items() if k not in {"binding", "traceback"}}, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
