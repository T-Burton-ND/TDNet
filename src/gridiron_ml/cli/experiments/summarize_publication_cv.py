#!/usr/bin/env python3
"""Merge a publication experiment and render selected-config fold boxplots."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from gridiron_ml.experiments.publication import merge_experiment_chunks
from gridiron_ml.publication.validation_figures import normalize_cv_metrics, plot_cv_metric_boxplots


def main():
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_experiment_chunks(job_manifest=args.manifest, output_root=args.output_root)["master"]
    success = merged[merged["status"].eq("success")].copy()
    success["configuration_id"] = (
        success["experiment_id"].astype(str) + "__p" + success["parameter_index"].astype(str)
        + "__s" + success["seed"].astype(str)
    )
    selected_rows = []
    for cell, group in success.groupby(["objective", "feature_config", "model_level"]):
        metric = "brier_score" if cell[0] == "winner" else "mae"
        summary = group.groupby("configuration_id", as_index=False).agg(
            mean_metric=(metric, "mean"), valid_folds=("outer_fold", "nunique")
        )
        best = summary.sort_values("mean_metric").iloc[0]
        selected_rows.append(group[group["configuration_id"].eq(best["configuration_id"])])
    selected = normalize_cv_metrics(pd.concat(selected_rows, ignore_index=True))
    selected["concrete_model_type"] = selected["model_family"].astype(str) + "/" + selected["model_level"].astype(str)
    summary_root = args.output_root / "summary/cv_validation"
    summary_root.mkdir(parents=True, exist_ok=True)
    selected.to_csv(summary_root / "selected_configuration_fold_metrics.csv", index=False)
    for objective, frame in selected.groupby("objective"):
        plot_cv_metric_boxplots(
            frame, summary_root / f"{objective}_selected_model_cv_boxplots.png",
            title=f"{objective.title()} models: rolling-origin fold validation",
        )
    print(selected.groupby(["objective", "model_level"])[["brier_score", "winner_accuracy", "margin_mae"]].agg(["mean", "std", "count"]).to_string())


if __name__ == "__main__":
    main()
