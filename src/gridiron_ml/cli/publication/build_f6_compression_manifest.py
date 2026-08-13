#!/usr/bin/env python3
"""Build leakage-safe candidate fits for the F6-C compression study."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import pandas as pd
import yaml

from gridiron_ml.cli._paths import project_root
from gridiron_ml.experiments.publication import atomic_write_frame, atomic_write_json, materialize_split_rows
from gridiron_ml.td_run.matchups.unit_matchups import default_counterpart


MODEL_CONFIGS = {
    "M1": "configs/models/linear/config_ridge.yaml",
    "M2": "configs/models/spline/config_spline_ridge.yaml",
    "M3": "configs/models/tree/config_random_forest.yaml",
    "M4": "configs/models/boosted/config_hist_gradient_boosted.yaml",
    "M5": "configs/models/neural/config_mlp.yaml",
    "M10": "configs/models/knn/config_knn_distance.yaml",
}


def _closed_top_features(ranking: list[str], target: int, available: set[str]) -> list[str]:
    selected: list[str] = []
    present: set[str] = set()
    for feature in ranking:
        if feature not in available or feature in present:
            continue
        group = [feature]
        counterpart = default_counterpart(feature, available=available)
        if counterpart and counterpart not in present:
            group.append(counterpart)
        for item in group:
            if item not in present:
                selected.append(item)
                present.add(item)
        if len(selected) >= int(target):
            break
    if len(selected) < int(target):
        raise ValueError(f"Only {len(selected)} F6 features were available for target {target}.")
    return selected


def _selected_f6_row(table: pd.DataFrame, objective: str, level: str) -> pd.Series:
    match = table.loc[
        table["objective"].astype(str).eq(objective)
        & table["feature_config"].astype(str).eq("F6")
        & table["model_level"].astype(str).eq(level)
    ]
    if len(match) != 1 or int(match.iloc[0]["cv_fold_count"]) != 10:
        raise ValueError(f"Expected one complete corrected-F6 selection for {(objective, level)}.")
    return match.iloc[0]


def build_manifest(args) -> pd.DataFrame:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    importance = pd.read_parquet(args.shap_importance)
    corrected = pd.read_parquet(args.corrected_selection)
    source = pd.read_parquet(args.data, columns=None)
    source_features: set[str] = set()
    # Every completed SHAP cell was resolved against the authoritative F6
    # source list, so its feature axis is the compression universe.
    source_features.update(importance["source_feature"].dropna().astype(str).unique())
    source_features &= set(source.columns)
    if len(source_features) != int(cfg["source_feature_count"]):
        raise ValueError(
            f"Compression requires {cfg['source_feature_count']} F6 source features; found {len(source_features)}."
        )

    folds = materialize_split_rows(args.project_root / "configs/splits/rolling_origin.yaml")
    eligible = [fold for fold in folds if int(fold["outer_fold"]) >= int(cfg["selection"]["first_eligible_outer_fold"])]
    rows: list[dict] = []
    for fold in eligible:
        outer_fold = int(fold["outer_fold"])
        for objective in cfg["objectives"]:
            for level in cfg["model_levels"]:
                selected = _selected_f6_row(corrected, str(objective), str(level))
                prior = importance.loc[
                    importance["objective"].astype(str).eq(str(objective))
                    & importance["model_level"].astype(str).eq(str(level))
                    & pd.to_numeric(importance["outer_fold"], errors="coerce").lt(outer_fold)
                ]
                if prior.empty:
                    raise ValueError(f"No prior-fold SHAP ranking for {(objective, level, outer_fold)}.")
                ranking = (
                    prior.groupby("source_feature", as_index=False)["normalized_importance"]
                    .median()
                    .sort_values(["normalized_importance", "source_feature"], ascending=[False, True])
                    ["source_feature"].astype(str).tolist()
                )
                for target in cfg["candidate_target_counts"]:
                    subset = _closed_top_features(ranking, int(target), source_features)
                    task_id = len(rows)
                    rows.append(
                        {
                            "task_id": task_id,
                            "chunk_id": task_id,
                            "experiment_id": f"f6_compression__{objective}__{level}__k{outer_fold}__n{target}",
                            "objective": str(objective),
                            "feature_config": "F6-C",
                            "source_feature_config": "F6",
                            "model_level": str(level),
                            "model_family": str(selected["model_family"]),
                            "model_config": str(
                                selected.get("model_config")
                                if pd.notna(selected.get("model_config"))
                                else (args.project_root / MODEL_CONFIGS[str(level)]).resolve()
                            ),
                            "split_config": str((args.project_root / "configs/splits/rolling_origin.yaml").resolve()),
                            "outer_fold": outer_fold,
                            "train_seasons_json": json.dumps(fold["train_seasons"]),
                            "val_seasons_json": json.dumps(fold["val_seasons"]),
                            "test_seasons_json": json.dumps(fold["test_seasons"]),
                            "seed": int(selected.get("seed", 1701)),
                            "parameter_index": int(selected.get("parameter_index", -1)),
                            "params_json": str(selected["params_json"]),
                            "data_path": str(args.data.resolve()),
                            "feature_registry": str((args.project_root / "configs/features/feature_registry.yaml").resolve()),
                            "feature_ladders": str((args.project_root / "configs/features/feature_ladders.yaml").resolve()),
                            "output_path": str((args.output_root / "runs" / f"task_{task_id:05d}").resolve()),
                            "estimated_memory_gb": 6 if str(level) == "M5" else 3,
                            "estimated_runtime": "04:00:00" if str(level) == "M5" else "02:00:00",
                            "retain_predictions": False,
                            "retain_checkpoint": False,
                            "target_feature_count": int(target),
                            "actual_source_feature_count": len(subset),
                            "feature_subset_json": json.dumps(subset),
                            "ranking_outer_folds_json": json.dumps(sorted(prior["outer_fold"].astype(int).unique().tolist())),
                        }
                    )
    manifest = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_frame(manifest, args.output_root / "job_manifest.parquet")
    atomic_write_frame(
        manifest[["chunk_id", "task_id"]].rename(columns={"task_id": "task_start"}).assign(
            task_end=lambda x: x["task_start"], trial_count=1, sge_task_id=lambda x: x["chunk_id"] + 1
        ),
        args.output_root / "chunk_manifest.parquet",
    )
    manifest.to_csv(args.output_root / "job_manifest.csv", index=False)
    atomic_write_json(
        args.output_root / "manifest_report.json",
        {
            "tasks": len(manifest),
            "expected_tasks": 1080,
            "eligible_outer_folds": [int(x["outer_fold"]) for x in eligible],
            "candidate_target_counts": list(cfg["candidate_target_counts"]),
            "source_feature_count": len(source_features),
        },
    )
    return manifest


def main() -> None:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shap-importance", type=Path, required=True)
    parser.add_argument("--corrected-selection", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args)
    print(f"tasks={len(manifest)} output={args.output_root}")


if __name__ == "__main__":
    main()
