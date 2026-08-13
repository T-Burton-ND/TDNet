#!/usr/bin/env python3
"""Build the 1,080-cell corrected OOF evaluation manifest for heatmaps."""

from argparse import ArgumentParser
import json
from pathlib import Path

import pandas as pd

from gridiron_ml.cli._paths import project_root
from gridiron_ml.experiments.publication import atomic_write_frame, atomic_write_json, materialize_split_rows


LEVELS = ("M1", "M2", "M3", "M4", "M5", "M10")
TIERS = tuple(f"F{i}" for i in range(9))
OBJECTIVES = ("margin", "winner")
MODEL_CONFIGS = {
    "M1": "configs/models/linear/config_ridge.yaml",
    "M2": "configs/models/spline/config_spline_ridge.yaml",
    "M3": "configs/models/tree/config_random_forest.yaml",
    "M4": "configs/models/boosted/config_hist_gradient_boosted.yaml",
    "M5": "configs/models/neural/config_mlp.yaml",
    "M10": "configs/models/knn/config_knn_distance.yaml",
}


def choose_source(tier: str, *, legacy_roster: pd.DataFrame, legacy_f7: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    if tier in {"F5", "F6", "F8"}:
        return corrected
    if tier == "F7":
        return legacy_f7
    return legacy_roster


def selected_cell(table: pd.DataFrame, objective: str, tier: str, level: str) -> pd.Series:
    match = table.loc[
        table["objective"].astype(str).eq(objective)
        & table["feature_config"].astype(str).eq(tier)
        & table["model_level"].astype(str).eq(level)
    ]
    if len(match) != 1:
        raise ValueError(f"Expected one selected row for {(objective, tier, level)}; found {len(match)}.")
    row = match.iloc[0]
    # The legacy F0--F4 scientific roster was selected with a complete
    # 25-fold leave-one-season-out design; the corrected F5/F6/F8 and F7
    # sources use the newer ten-fold rolling-origin design.  Both are complete
    # selection sources and are re-evaluated below on the same ten folds.
    if int(row.get("cv_fold_count", 10)) < 10:
        raise ValueError(f"Selected row for {(objective, tier, level)} is incomplete.")
    return row


def build_manifest(args) -> pd.DataFrame:
    legacy_roster = pd.read_csv(args.legacy_roster)
    legacy_f7 = pd.read_parquet(args.legacy_f7)
    corrected = pd.read_parquet(args.corrected)
    folds = materialize_split_rows(args.project_root / "configs/splits/rolling_origin.yaml")
    rows = []
    for fold in folds:
        for objective in OBJECTIVES:
            for tier in TIERS:
                source = choose_source(tier, legacy_roster=legacy_roster, legacy_f7=legacy_f7, corrected=corrected)
                for level in LEVELS:
                    chosen = selected_cell(source, objective, tier, level)
                    task_id = len(rows)
                    rows.append(
                        {
                            "task_id": task_id,
                            "chunk_id": task_id,
                            "experiment_id": f"heatmap_eval__{objective}__{tier}__{level}",
                            "objective": objective,
                            "feature_config": tier,
                            "model_level": level,
                            "model_family": str(chosen["model_family"]),
                            "model_config": str(
                                chosen.get("model_config")
                                if pd.notna(chosen.get("model_config"))
                                else (args.project_root / MODEL_CONFIGS[level]).resolve()
                            ),
                            "split_config": str((args.project_root / "configs/splits/rolling_origin.yaml").resolve()),
                            "outer_fold": int(fold["outer_fold"]),
                            "train_seasons_json": json.dumps(fold["train_seasons"]),
                            "val_seasons_json": json.dumps(fold["val_seasons"]),
                            "test_seasons_json": json.dumps(fold["test_seasons"]),
                            "seed": int(chosen.get("seed", 1701)),
                            "parameter_index": int(chosen.get("parameter_index", -1)),
                            "params_json": str(chosen["params_json"]),
                            "data_path": str(args.data.resolve()),
                            "feature_registry": str((args.project_root / "configs/features/feature_registry.yaml").resolve()),
                            "feature_ladders": str((args.project_root / "configs/features/feature_ladders.yaml").resolve()),
                            "output_path": str((args.output_root / "runs" / f"task_{task_id:05d}").resolve()),
                            "estimated_memory_gb": 6 if level == "M5" else 3,
                            "estimated_runtime": "04:00:00" if level == "M5" else "02:00:00",
                            "retain_predictions": False,
                            "retain_checkpoint": False,
                            "selection_source": "corrected_hps" if tier in {"F5", "F6", "F8"} else "legacy_valid_f7" if tier == "F7" else "legacy_parameter_transfer",
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
    report = {
        "tasks": len(manifest), "expected_tasks": 1080,
        "tiers": list(TIERS), "objectives": list(OBJECTIVES), "levels": list(LEVELS),
        "legacy_parameter_transfer_tiers": ["F0", "F1", "F2", "F3", "F4"],
    }
    atomic_write_json(args.output_root / "manifest_report.json", report)
    return manifest


def main() -> None:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--legacy-roster", type=Path, required=True)
    parser.add_argument("--legacy-f7", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args)
    print(f"tasks={len(manifest)} output={args.output_root}")


if __name__ == "__main__":
    main()
