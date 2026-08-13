#!/usr/bin/env python3
"""Produce paired 2025 canonical-vs-Vegas inference evidence."""
from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.publication.inference import MarginCalibrator, mcnemar_test, season_clustered_bootstrap  # noqa: E402
from gridiron_ml.publication.metrics import score_predictions  # noqa: E402


def compare(frame: pd.DataFrame, label: str) -> tuple[dict, pd.DataFrame]:
    x = frame.copy()
    for col in ["pred_probability_home", "pred_margin", "actual_margin", "vegas_spread", "vegas_probability_home"]:
        if col in x:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["pred_probability_home", "pred_margin", "actual_margin", "vegas_spread"])
    model_correct = x["pred_probability_home"].ge(0.5).to_numpy() == x["actual_margin"].gt(0).to_numpy()
    vegas_correct = x["vegas_spread"].lt(0).to_numpy() == x["actual_margin"].gt(0).to_numpy()
    model_brier = (x["pred_probability_home"] - x["actual_margin"].gt(0).astype(float)) ** 2
    vegas_margin_error = (-x["vegas_spread"] - x["actual_margin"]).abs()
    model_mae = (x["pred_margin"] - x["actual_margin"]).abs()
    row = {"comparison": label, "n_games": len(x), **score_predictions(x)}
    row.update({
        "vegas_winner_accuracy": float(vegas_correct.mean()),
        "vegas_margin_mae": float(vegas_margin_error.mean()),
        "vegas_probability_rows": int(x["vegas_probability_home"].notna().sum()) if "vegas_probability_home" in x else 0,
        "mcnemar": mcnemar_test(model_correct, vegas_correct),
    })
    if "vegas_probability_home" in x:
        valid = x["vegas_probability_home"].notna()
        row["vegas_brier_score"] = float(((x.loc[valid, "vegas_probability_home"] - x.loc[valid, "actual_margin"].gt(0).astype(float)) ** 2).mean())
        row["model_brier_score_on_vegas_rows"] = float(model_brier.loc[valid].mean())
        paired = pd.DataFrame({"game_id": x.loc[valid, "keys_game_id"], "week": x.loc[valid, "keys_week"], "model_minus_vegas_brier": model_brier.loc[valid].to_numpy() - ((x.loc[valid, "vegas_probability_home"] - x.loc[valid, "actual_margin"].gt(0).astype(float)) ** 2).to_numpy(), "model_minus_vegas_mae": model_mae.loc[valid].to_numpy() - vegas_margin_error.loc[valid].to_numpy()})
        for metric in ["model_minus_vegas_brier", "model_minus_vegas_mae"]:
            row[f"{metric}_game_bootstrap"] = season_clustered_bootstrap(paired.rename(columns={"game_id": "cluster"}), season_column="cluster", value_column=metric, n_resamples=2000)
            row[f"{metric}_week_bootstrap"] = season_clustered_bootstrap(paired, season_column="week", value_column=metric, n_resamples=2000)
    return row, x.assign(comparison=label, model_correct=model_correct, vegas_correct=vegas_correct)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--consensus-root", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/publication/canonical_2025/inference")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(args.predictions)
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    selected_fp = str(decisions["selected_fingerprint_market_free"])
    individual = predictions.loc[(predictions["fingerprint_id"].astype(str) == selected_fp) & (predictions["model_role"].eq(str(decisions["selected_individual_model"])))].copy()
    if args.calibration_root is not None and selected_fp == "F5":
        calibration_path = args.calibration_root / "F5" / str(decisions["selected_individual_model"]) / "selection_through_2024.json"
        if calibration_path.exists():
            record = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibrator = MarginCalibrator(intercept=float(record["intercept"]), slope=float(record["slope"]), fit_rows=int(record["fit_rows"]), fit_hash=record.get("fit_hash"))
            individual["pred_probability_home"] = calibrator.predict(individual["pred_margin"])
    products = [
        ("selected_individual", individual),
        ("all_model_consensus", pd.read_parquet(args.consensus_root / "all_model_consensus.parquet")),
        ("top10_brier_consensus", pd.read_parquet(args.consensus_root / "top10_brier_consensus.parquet")),
    ]
    rows = []
    detail = []
    for label, frame in products:
        result, game_rows = compare(frame, label)
        rows.append(result)
        detail.append(game_rows)
    pd.DataFrame(rows).to_json(args.output_root / "paired_inference.json", orient="records", indent=2)
    pd.DataFrame(rows).drop(columns=["mcnemar", "model_minus_vegas_brier_game_bootstrap", "model_minus_vegas_brier_week_bootstrap", "model_minus_vegas_mae_game_bootstrap", "model_minus_vegas_mae_week_bootstrap"], errors="ignore").to_csv(args.output_root / "paired_inference_summary.csv", index=False)
    pd.concat(detail, ignore_index=True).to_parquet(args.output_root / "game_level_comparisons.parquet", index=False)
    (args.output_root / "inference_report.json").write_text(json.dumps({"status": "canonical_2025_paired_inference_complete", "primary_comparison": "all_model_consensus_vs_vegas", "selected_individual": str(decisions["selected_individual_model"]), "selected_fingerprint": selected_fp, "season_bootstrap_note": "Only 2025 holdout rows are available here; week-block bootstrap is reported, while multi-season sensitivity requires historical OOF/development predictions."}, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows)[["comparison", "n_games", "brier_score", "vegas_brier_score", "winner_accuracy", "vegas_winner_accuracy", "margin_mae", "vegas_margin_mae"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
