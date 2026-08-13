#!/usr/bin/env python3
"""Merge new-tier HPS fragments and emit decision-ready comparison tables."""

from argparse import ArgumentParser
import json
from pathlib import Path

import pandas as pd

from gridiron_ml.experiments.publication import (
    atomic_write_frame,
    atomic_write_json,
    merge_experiment_chunks,
    select_finalists,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    merged = merge_experiment_chunks(
        job_manifest=args.manifest,
        output_root=args.output_root,
    )
    master = merged["master"]
    manifest = pd.read_parquet(args.manifest)
    expected_folds = int(
        manifest
        .groupby(
            [
                "objective", "feature_config", "model_level", "model_family",
                "model_config", "parameter_index", "params_json", "seed",
            ],
            dropna=False,
        )["outer_fold"]
        .nunique()
        .max()
    )
    finalists = select_finalists(
        master,
        max_per_cell=5,
        minimum_fold_count=expected_folds,
    )
    tables = args.output_root / "summary" / "tables"
    atomic_write_frame(finalists, tables / "finalists.parquet")
    finalists.to_csv(tables / "finalists.csv", index=False)

    best = (
        finalists.sort_values(["objective", "feature_config", "model_level", "cv_metric_mean"])
        .groupby(["objective", "feature_config", "model_level"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    keep = [
        "objective", "feature_config", "model_level", "model_family",
        "params_json", "cv_metric_mean", "cv_metric_se", "cv_fold_count",
        "mae", "rmse", "winner_accuracy", "brier_score",
        "favorite_correct", "upset_correct", "ats_accuracy", "ats_n",
    ]
    best = best[[column for column in keep if column in best.columns]]
    atomic_write_frame(best, tables / "best_configuration_by_cell.parquet")
    best.to_csv(tables / "best_configuration_by_cell.csv", index=False)

    comparison_rows = []
    for (objective, level), group in best.groupby(["objective", "model_level"]):
        metric = "brier_score" if objective == "winner" else "mae"
        score = dict(zip(group["feature_config"], group["cv_metric_mean"]))
        comparison_rows.append(
            {
                "objective": objective,
                "model_level": level,
                "selection_metric": metric,
                "F5_best": score.get("F5"),
                "F6_best": score.get("F6"),
                "F7_best": score.get("F7"),
                "F8_best": score.get("F8"),
                "F6_minus_F5": score.get("F6", float("nan")) - score.get("F5", float("nan")),
                "F8_minus_F6": score.get("F8", float("nan")) - score.get("F6", float("nan")),
                "F8_minus_F7": score.get("F8", float("nan")) - score.get("F7", float("nan")),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    atomic_write_frame(comparisons, tables / "tier_increment_decision_table.parquet")
    comparisons.to_csv(tables / "tier_increment_decision_table.csv", index=False)

    coverage = {
        "manifest_rows": int(len(manifest)),
        "merged_rows": int(len(master)),
        "successful_rows": int(master.get("status", pd.Series(dtype=str)).eq("success").sum()),
        "failed_rows": int(master.get("status", pd.Series(dtype=str)).eq("failed").sum()),
        "missing_rows": int(len(merged["missing_trials"])),
        "required_folds_per_configuration": expected_folds,
        "best_cell_count": int(len(best)),
        "expected_best_cell_count": int(
            manifest[["objective", "feature_config", "model_level"]]
            .drop_duplicates()
            .shape[0]
        ),
        "decision_rule": (
            "Negative tier differences favor the richer tier. Use F6_minus_F5 to assess "
            "schedule-graph value and F8_minus_F6 to assess incremental market value."
        ),
    }
    atomic_write_json(args.output_root / "summary" / "decision_coverage.json", coverage)
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
