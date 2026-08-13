#!/usr/bin/env python3
"""Create tidy metric and feature/model matrix summaries."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from gridiron_ml.experiments.publication import atomic_write_frame, read_frame


def main():
    parser = ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    results = read_frame(args.results)
    success = results.loc[results["status"] == "success"].copy()
    metric_candidates = [
        "accuracy",
        "winner_accuracy",
        "brier_score",
        "mae",
        "rmse",
        "runtime_seconds",
    ]
    metrics = [metric for metric in metric_candidates if metric in success.columns]
    tidy = success.melt(
        id_vars=[
            column
            for column in [
                "objective",
                "feature_config",
                "model_level",
                "outer_fold",
                "seed",
            ]
            if column in success.columns
        ],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    summary = (
        tidy.groupby(["objective", "feature_config", "model_level", "metric"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), count=("value", "count"))
    )
    output = Path(args.output_root) / "summary" / "tables"
    atomic_write_frame(tidy, output / "tidy_complexity_metrics.parquet")
    atomic_write_frame(summary, output / "feature_model_matrix_summary.parquet")
    summary.to_csv(output / "feature_model_matrix_summary.csv", index=False)
    print(f"tidy_rows={len(tidy)}")
    print(f"summary_rows={len(summary)}")


if __name__ == "__main__":
    main()

