#!/usr/bin/env python3
"""Verify common-cohort M2/M4 results and summarize paired round changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file
from scripts.nextgen_rounds_train import STAGES
from scripts.nextgen_rounds_ablate import ABLATIONS
from scripts.nextgen_rounds_market_residual import VARIANTS


def report(output: Path):
    results, predictions = {}, {}
    for stage in STAGES:
        for model in ("M2", "M4"):
            for index in range(1, 11):
                point = f"{model.lower()}_{index:02d}"
                folder = output / "experiments" / stage / model / point
                path = folder / "result.json"
                key = (stage, model, point)
                if not path.exists():
                    results[key] = {"stage": stage, "model": model, "setpoint": point,
                                    "status": "missing"}
                    continue
                row = json.loads(path.read_text())
                results[key] = row
                if row.get("status") == "success":
                    p = folder / "predictions.parquet"
                    if sha256_file(p) != row.get("predictions_sha256"):
                        raise ValueError(f"Prediction artifact changed: {key}")
                    d = pd.read_parquet(p).sort_values("target_game_id").reset_index(drop=True)
                    if d.target_game_id.duplicated().any():
                        raise ValueError(f"Duplicate predicted game: {key}")
                    predictions[key] = d
    rows = []
    for key, result in results.items():
        stage, model, point = key
        item = {"stage": stage, "model": model, "setpoint": point,
                "status": result["status"], "training_games": result.get("training_games")}
        for year in (2024, 2025):
            metrics = result.get(f"metrics_{year}") or {}
            for field in ("n_games", "mae", "rmse", "brier", "winner_accuracy",
                          "ats_accuracy", "chalk_accuracy", "upset_accuracy"):
                item[f"{field}_{year}"] = metrics.get(field)
        rows.append(item)
    table = pd.DataFrame(rows)
    paired = []
    for index in range(1, len(STAGES)):
        parent, stage = STAGES[index - 1], STAGES[index]
        for model in ("M2", "M4"):
            for point_index in range(1, 11):
                point = f"{model.lower()}_{point_index:02d}"
                left_key, right_key = (parent, model, point), (stage, model, point)
                if left_key not in predictions or right_key not in predictions:
                    continue
                left, right = predictions[left_key], predictions[right_key]
                if (not left.target_game_id.equals(right.target_game_id)
                        or not left.season.equals(right.season)
                        or not left.actual_margin.equals(right.actual_margin)
                        or results[left_key]["training_games"] != results[right_key]["training_games"]
                        or results[left_key]["binding"]["input"]["games_by_season"]
                        != results[right_key]["binding"]["input"]["games_by_season"]):
                    raise ValueError(f"Training/evaluation cohort differs: {left_key} vs {right_key}")
                for year in (2024, 2025):
                    mask = left.season.eq(year)
                    actual = left.loc[mask, "actual_margin"].to_numpy(float)
                    left_mae = np.abs(actual - left.loc[mask, "predicted_margin"].to_numpy(float)).mean()
                    right_mae = np.abs(actual - right.loc[mask, "predicted_margin"].to_numpy(float)).mean()
                    paired.append({"parent": parent, "stage": stage, "model": model,
                                   "setpoint": point, "season": year, "games": int(mask.sum()),
                                   "parent_mae": float(left_mae), "stage_mae": float(right_mae),
                                   "paired_mae_change": float(right_mae - left_mae)})
    pair_table = pd.DataFrame(paired)
    summary = []
    if not pair_table.empty:
        for keys, frame in pair_table.groupby(["parent", "stage", "model", "season"]):
            summary.append(dict(zip(["parent", "stage", "model", "season"], keys)) | {
                "paired_setpoints": len(frame),
                "improved_setpoints": int(frame.paired_mae_change.lt(0).sum()),
                "median_paired_mae_change": float(frame.paired_mae_change.median()),
                "difference_of_median_mae": float(frame.stage_mae.median() - frame.parent_mae.median()),
            })
    receipt = json.loads((output / "prepare_receipt.json").read_text())
    archive = Path(receipt["source_archive"])
    parent = pd.read_parquet(archive / "fingerprints/F12_F_a/values.parquet",
                             columns=["target_game_id", "season", "is_home", "next_game_margin"])
    parent = parent.loc[parent.is_home & parent.season.isin([2024, 2025])]
    market = pd.read_parquet(output / "f17_market_game_features.parquet",
                             columns=["target_game_id", "market_home_spread"])
    baseline = parent.merge(market, on="target_game_id", how="inner", validate="one_to_one")
    if len(baseline) != len(parent):
        raise ValueError("Market baseline does not cover the common evaluation cohort")
    market_baseline = {}
    for year, frame in baseline.groupby("season"):
        actual = frame.next_game_margin.to_numpy(float)
        implied = -frame.market_home_spread.to_numpy(float)
        market_baseline[str(year)] = {
            "games": len(frame), "mae": float(np.abs(actual - implied).mean()),
            "winner_accuracy": float(((implied > 0) == (actual > 0)).mean())}
    ablation_rows, ablation_pairs = [], []
    for ablation in ABLATIONS:
        for model in ("M2", "M4"):
            for index in range(1, 11):
                point = f"{model.lower()}_{index:02d}"
                folder = output / "ablations" / ablation / model / point
                path = folder / "result.json"
                if not path.exists():
                    ablation_rows.append({"ablation": ablation, "model": model,
                                          "setpoint": point, "status": "missing"})
                    continue
                item = json.loads(path.read_text())
                row = {"ablation": ablation, "model": model, "setpoint": point,
                       "status": item["status"]}
                for year in (2024, 2025):
                    metrics = item.get(f"metrics_{year}") or {}
                    for field in ("mae", "brier", "winner_accuracy", "ats_accuracy"):
                        row[f"{field}_{year}"] = metrics.get(field)
                ablation_rows.append(row)
                if item["status"] != "success":
                    continue
                prediction_path = folder / "predictions.parquet"
                if sha256_file(prediction_path) != item.get("predictions_sha256"):
                    raise ValueError(f"Ablation prediction changed: {ablation}/{model}/{point}")
                candidate = pd.read_parquet(prediction_path).sort_values(
                    "target_game_id").reset_index(drop=True)
                reference_key = ("F12_corrected", model, point)
                if reference_key not in predictions:
                    continue
                reference = predictions[reference_key]
                if (not candidate.target_game_id.equals(reference.target_game_id)
                        or not candidate.season.equals(reference.season)
                        or not candidate.actual_margin.equals(reference.actual_margin)
                        or item["training_games"] != results[reference_key]["training_games"]):
                    raise ValueError(f"Ablation cohort differs: {ablation}/{model}/{point}")
                for year in (2024, 2025):
                    mask = candidate.season.eq(year)
                    actual = candidate.loc[mask, "actual_margin"].to_numpy(float)
                    reference_mae = float(np.abs(actual - reference.loc[mask, "predicted_margin"].to_numpy(float)).mean())
                    candidate_mae = float(np.abs(actual - candidate.loc[mask, "predicted_margin"].to_numpy(float)).mean())
                    ablation_pairs.append({"ablation": ablation, "reference": "F12_corrected",
                                           "model": model, "setpoint": point, "season": year,
                                           "reference_mae": reference_mae,
                                           "ablation_mae": candidate_mae,
                                           "paired_mae_change": candidate_mae - reference_mae})
    ablation_table = pd.DataFrame(ablation_rows)
    ablation_pair_table = pd.DataFrame(ablation_pairs)
    ablation_summary = []
    if not ablation_pair_table.empty:
        for keys, frame in ablation_pair_table.groupby(["ablation", "model", "season"]):
            ablation_summary.append(dict(zip(["ablation", "model", "season"], keys)) | {
                "paired_setpoints": len(frame),
                "improved_setpoints": int(frame.paired_mae_change.lt(0).sum()),
                "median_paired_mae_change": float(frame.paired_mae_change.median()),
                "difference_of_median_mae": float(frame.ablation_mae.median() - frame.reference_mae.median())})
    residual_rows = []
    for variant in VARIANTS:
        for model in ("M2", "M4"):
            for index in range(1, 11):
                point = f"{model.lower()}_{index:02d}"
                folder = output / "market_residual" / variant / model / point
                path = folder / "result.json"
                if not path.exists():
                    residual_rows.append({"variant": variant, "model": model,
                                          "setpoint": point, "status": "missing"})
                    continue
                item = json.loads(path.read_text())
                row = {"variant": variant, "model": model, "setpoint": point,
                       "status": item["status"]}
                for year in (2024, 2025):
                    metrics = item.get(f"metrics_{year}") or {}
                    for field in ("mae", "brier", "winner_accuracy", "ats_accuracy"):
                        row[f"{field}_{year}"] = metrics.get(field)
                residual_rows.append(row)
                if item["status"] == "success":
                    prediction_path = folder / "predictions.parquet"
                    if sha256_file(prediction_path) != item.get("predictions_sha256"):
                        raise ValueError(f"Residual prediction changed: {variant}/{model}/{point}")
                    candidate = pd.read_parquet(prediction_path).sort_values(
                        "target_game_id").reset_index(drop=True)
                    reference = predictions.get(("F17_market", model, point))
                    if (reference is None or not candidate.target_game_id.equals(reference.target_game_id)
                            or not candidate.season.equals(reference.season)
                            or not candidate.actual_margin.equals(reference.actual_margin)):
                        raise ValueError(f"Residual cohort differs: {variant}/{model}/{point}")
    residual_table = pd.DataFrame(residual_rows)
    residual_summary = []
    for variant in VARIANTS:
        for model in ("M2", "M4"):
            group = residual_table.loc[residual_table.variant.eq(variant)
                                       & residual_table.model.eq(model)
                                       & residual_table.status.eq("success")]
            if group.empty:
                continue
            for year in (2024, 2025):
                mae = group[f"mae_{year}"]
                baseline_mae = market_baseline[str(year)]["mae"]
                residual_summary.append({"variant": variant, "model": model, "season": year,
                                         "successful_setpoints": len(group),
                                         "median_mae": float(mae.median()),
                                         "spread_baseline_mae": baseline_mae,
                                         "beat_spread_setpoints": int(mae.lt(baseline_mae).sum())})
    result = {"status": "complete" if table.status.eq("success").all() else "partial",
              "expected_runs": len(STAGES) * 20, "successful_runs": int(table.status.eq("success").sum()),
              "failed_runs": int(table.status.eq("failed").sum()),
              "missing_runs": int(table.status.eq("missing").sum()),
              "market_round_is_retrospective": True,
              "quote_timestamp_available": False,
              "development_years_are_design_informed": True,
              "archived_market_spread_baseline": market_baseline,
              "paired_summary": summary,
              "ablation_successful_runs": int(ablation_table.status.eq("success").sum()),
              "ablation_expected_runs": len(ABLATIONS) * 20,
              "ablation_summary_vs_corrected_F12": ablation_summary,
              "residual_successful_runs": int(residual_table.status.eq("success").sum()),
              "residual_expected_runs": len(VARIANTS) * 20,
              "residual_summary": residual_summary}
    table.to_csv(output / "run_results.csv", index=False)
    pair_table.to_csv(output / "paired_results.csv", index=False)
    ablation_table.to_csv(output / "ablation_results.csv", index=False)
    ablation_pair_table.to_csv(output / "ablation_paired_results.csv", index=False)
    residual_table.to_csv(output / "market_residual_results.csv", index=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/nextgen_rounds_2026")
    args = parser.parse_args()
    print(json.dumps(report(args.output), indent=2))


if __name__ == "__main__":
    main()
