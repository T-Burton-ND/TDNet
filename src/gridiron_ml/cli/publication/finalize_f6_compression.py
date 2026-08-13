#!/usr/bin/env python3
"""Merge F6-C candidates and select the sequential leakage-safe path."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_ml.experiments.publication import atomic_write_frame, atomic_write_json, merge_experiment_chunks


def _choose_budget(history: pd.DataFrame, metric: str) -> int:
    summary = history.groupby("target_feature_count")[metric].agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
    best = summary.sort_values("mean").iloc[0]
    eligible = summary.loc[summary["mean"].le(float(best["mean"] + best["se"]))]
    return int(eligible["target_feature_count"].min())


def select_sequential_path(master: pd.DataFrame, initial_target: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    deployment = []
    for (objective, level), group in master.loc[master["status"].eq("success")].groupby(["objective", "model_level"]):
        metric = "mae" if str(objective) == "margin" else "brier_score"
        folds = sorted(pd.to_numeric(group["outer_fold"], errors="raise").astype(int).unique())
        for index, fold in enumerate(folds):
            target = int(initial_target) if index == 0 else _choose_budget(group.loc[group["outer_fold"].astype(int).lt(fold)], metric)
            match = group.loc[
                group["outer_fold"].astype(int).eq(fold)
                & group["target_feature_count"].astype(int).eq(target)
            ]
            if len(match) != 1:
                raise ValueError(f"Expected one F6-C candidate for {(objective, level, fold, target)}.")
            row = match.iloc[0].copy()
            row["selection_target_feature_count"] = target
            row["selection_rule"] = "prespecified_25" if index == 0 else "prior_folds_one_se_smallest"
            rows.append(row)
        deployment.append(
            {
                "objective": objective,
                "model_level": level,
                "deployment_target_feature_count": _choose_budget(group, metric),
                "selection_metric": metric,
                "eligible_outer_folds": len(folds),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(deployment)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_experiment_chunks(job_manifest=args.manifest, output_root=args.output_root)
    master = merged["master"]
    if len(master) != len(pd.read_parquet(args.manifest)) or not master["status"].eq("success").all():
        raise ValueError("F6-C finalization requires every candidate task to succeed.")
    selected, deployment = select_sequential_path(master)
    summary = args.output_root / "summary"
    atomic_write_frame(selected, summary / "sequential_selected_results.parquet")
    selected.to_csv(summary / "sequential_selected_results.csv", index=False)
    atomic_write_frame(deployment, summary / "deployment_feature_counts.parquet")
    deployment.to_csv(summary / "deployment_feature_counts.csv", index=False)
    report = {
        "candidate_rows": len(master),
        "successful_rows": int(master["status"].eq("success").sum()),
        "selected_rows": len(selected),
        "expected_selected_rows": 108,
        "deployment_cells": len(deployment),
    }
    atomic_write_json(summary / "compression_coverage.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
