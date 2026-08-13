#!/usr/bin/env python3
"""Select configurations across CV folds and render fold validation boxplots."""

from argparse import ArgumentParser
from pathlib import Path
import os

import pandas as pd

from gridiron_ml.publication.validation_figures import normalize_cv_metrics, plot_cv_metric_boxplots


def main():
    parser = ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("TDNET_NESTED_SEARCH_ROOT", "nested_search_2025")))
    args = parser.parse_args()
    all_selected = []
    for objective in ["winner", "margin"]:
        root = args.artifact_root / objective
        manifest = pd.read_csv(root / "job_manifest.csv")
        frames = []
        for row in manifest.itertuples(index=False):
            path = Path(row.metrics_path)
            if not path.exists():
                continue
            metric = pd.read_csv(path)
            if metric.empty:
                continue
            metric["base_job_index"] = row.base_job_index
            metric["outer_fold"] = row.outer_fold
            metric["cv_test_season"] = row.cv_test_season
            metric["model"] = row.model
            metric["family"] = row.family
            metric["concrete_model_type"] = f"{row.family}/{row.model}"
            frames.append(metric)
        if not frames:
            raise RuntimeError(f"No completed CV metrics found for {objective}.")
        metrics = pd.concat(frames, ignore_index=True)
        success = metrics[metrics["status"].eq("success")].copy() if "status" in metrics else metrics.copy()
        aggregate = success.groupby(["concrete_model_type", "base_job_index"], as_index=False).agg(
            mean_tuning_score=("tuning_score", "mean"), valid_folds=("outer_fold", "nunique")
        )
        aggregate = aggregate[aggregate["valid_folds"].eq(5)]
        winners = aggregate.sort_values("mean_tuning_score", ascending=False).drop_duplicates("concrete_model_type")
        selected = success.merge(winners[["concrete_model_type", "base_job_index", "mean_tuning_score"]], on=["concrete_model_type", "base_job_index"], how="inner")
        selected = normalize_cv_metrics(selected)
        summary = root / "cv_summary"
        summary.mkdir(parents=True, exist_ok=True)
        selected.to_csv(summary / "selected_configuration_fold_metrics.csv", index=False)
        winners.to_csv(summary / "selected_configurations.csv", index=False)
        plot_cv_metric_boxplots(
            selected, summary / "selected_model_cv_boxplots.png",
            title=f"{objective.title()} models: rolling-origin validation distributions",
        )
        all_selected.append(selected.assign(objective=objective))
    combined = pd.concat(all_selected, ignore_index=True)
    combined.to_csv(args.artifact_root / "selected_model_cv_fold_metrics.csv", index=False)


if __name__ == "__main__":
    main()
