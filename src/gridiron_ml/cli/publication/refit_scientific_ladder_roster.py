#!/usr/bin/env python3
"""Refit the complete 54-cell scientific margin roster on the current ladder."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path

import pandas as pd

from gridiron_ml.cli._paths import project_root
from gridiron_ml.experiments.hyperparameter_search import build_tuned_model_config
from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints
from gridiron_ml.experiments.publication import atomic_write_frame, atomic_write_json, filter_frame_for_feature_config
from gridiron_ml.models import build_model_from_config
from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder


LEVELS = ("M1", "M2", "M3", "M4", "M5", "M10")
TIERS = tuple(f"F{i}" for i in range(9))
FAMILIES = {"M1": "linear", "M2": "spline", "M3": "tree", "M4": "boosted", "M5": "neural", "M10": "knn"}


def roster_cell(task_id: int) -> tuple[str, str]:
    cells = [(tier, level) for tier in TIERS for level in LEVELS]
    if task_id < 0 or task_id >= len(cells):
        raise ValueError(f"task_id must be in [0, {len(cells) - 1}].")
    return cells[task_id]


def _selected_row(selection: Path, tier: str, level: str) -> pd.Series:
    table = pd.read_parquet(selection) if selection.suffix == ".parquet" else pd.read_csv(selection)
    match = table.loc[
        table["objective"].astype(str).eq("margin")
        & table["feature_config"].astype(str).eq(tier)
        & table["model_level"].astype(str).eq(level)
    ]
    if len(match) != 1:
        raise ValueError(f"Expected one selected margin row for {(tier, level)}; found {len(match)}.")
    return match.iloc[0]


def run_task(args, task_id: int) -> dict:
    tier, level = roster_cell(task_id)
    out = args.output_root / "runs" / f"task_{task_id:03d}"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "inventory_fragment.parquet"
    if result_path.exists() and not args.force:
        return pd.read_parquet(result_path).iloc[0].to_dict()
    selected = _selected_row(args.selection, tier, level)
    raw = pd.read_parquet(args.data)
    raw = raw.loc[pd.to_numeric(raw["keys_season"], errors="coerce").le(args.training_end_season)].copy()
    frame, metadata = filter_frame_for_feature_config(
        raw, feature_config=tier,
        registry_path=args.project_root / "configs/features/feature_registry.yaml",
        ladders_path=args.project_root / "configs/features/feature_ladders.yaml",
    )
    config = build_tuned_model_config(
        base_config_path=Path(str(selected["model_config"])),
        params=json.loads(str(selected.get("params_json", "{}")) or "{}"),
    )
    config["seed"] = int(selected.get("seed", 1701))
    config["loss_function"] = "MAE"
    allow_market = tier in {"F7", "F8"}
    if allow_market:
        config["allow_market_features_for_training"] = True
    if isinstance(config.get("params"), dict) and "n_jobs" in config["params"]:
        config["params"]["n_jobs"] = 1
    model = build_model_from_config({"family": FAMILIES[level], **config})
    builder = MatchupBuilder(representation="unit_matchup")
    evaluator = TDEval(
        config={
            "model": {"family": FAMILIES[level], "allow_market_features_for_training": allow_market},
            "feature_spec": {"include_market": allow_market, "allow_market_features_for_training": allow_market},
        },
        fingerprints=StaticFrameFingerprints(frame), matchup_builder=builder, model=model,
    )
    training_seasons = list(range(2010, args.training_end_season + 1))
    evaluator.train(train_years=training_seasons, val_years=[])
    checkpoint = out / "checkpoint.pkl"
    model.save(checkpoint)
    predictions_path = None
    prediction_rows = 0
    if args.holdout_season is not None:
        # The holdout season was removed before fitting and is loaded only for evaluation.
        holdout_raw = pd.read_parquet(args.data)
        holdout_frame, _ = filter_frame_for_feature_config(
            holdout_raw, feature_config=tier,
            registry_path=args.project_root / "configs/features/feature_registry.yaml",
            ladders_path=args.project_root / "configs/features/feature_ladders.yaml",
        )
        evaluator.fingerprints = StaticFrameFingerprints(holdout_frame)
        predictions, _ = evaluator.evaluate(years=[args.holdout_season], label="scientific_2025_holdout")
        predictions_path = out / "holdout_predictions.parquet"
        atomic_write_frame(predictions, predictions_path)
        prediction_rows = int(len(predictions))
    record = {
        "task_id": task_id,
        "model_id": f"scientific_{tier}_{level}",
        "objective": "margin",
        "feature_config": tier,
        "model_level": level,
        "model_family": FAMILIES[level],
        "market_bearing": allow_market,
        "comparative_only": allow_market,
        "use_in_tdnet_poll": not allow_market,
        "use_in_weekly_consensus": not allow_market,
        "params_json": str(selected.get("params_json", "{}")),
        "training_seasons": json.dumps(training_seasons),
        "training_end_season": args.training_end_season,
        "holdout_season": args.holdout_season,
        "holdout_excluded_from_fit": args.holdout_season is None or args.holdout_season > args.training_end_season,
        "selected_feature_count": int(metadata["selected_feature_count"]),
        "selected_features_json": json.dumps(metadata["selected_features"]),
        "matchup_feature_count": int(len(builder.feature_names())),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "fingerprint_path": str(args.data.resolve()),
        "source_fingerprint_sha256": sha256_file(args.data),
        "predictions_path": str(predictions_path.resolve()) if predictions_path else "",
        "prediction_rows": prediction_rows,
        "calibration_status": "pending_cross_fitted_oof_phase",
        "status": "success",
    }
    atomic_write_frame(pd.DataFrame([record]), result_path)
    return record


def finalize(args) -> dict:
    fragments = sorted((args.output_root / "runs").glob("task_*/inventory_fragment.parquet"))
    table = pd.concat([pd.read_parquet(path) for path in fragments], ignore_index=True) if fragments else pd.DataFrame()
    expected = {(tier, level) for tier in TIERS for level in LEVELS}
    observed = set(zip(table.get("feature_config", []), table.get("model_level", [])))
    if len(table) != 54 or observed != expected:
        raise ValueError(f"Scientific ladder roster requires 54 unique F0-F8 x M cells; found {len(table)}.")
    table = table.sort_values(["feature_config", "model_level"]).reset_index(drop=True)
    # Publication evaluators use this stable label for ballots and long-form
    # predictions; never let independently refit checkpoints collapse to the
    # model loader's generic default name.
    table["final_model_name"] = table["model_id"].astype(str)
    table.to_csv(args.output_root / "final_model_inventory.csv", index=False)
    atomic_write_frame(table, args.output_root / "final_model_inventory.parquet")
    report = {
        "status": "complete", "model_count": 54, "objective": "margin",
        "tiers": list(TIERS), "levels": list(LEVELS),
        "training_end_season": args.training_end_season,
        "holdout_season": args.holdout_season,
        "market_free_prediction_cells": 42,
        "market_comparison_cells": 12,
    }
    atomic_write_json(args.output_root / "roster_manifest.json", report)
    return report


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("command", choices=["run-task", "finalize"])
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-end-season", type=int, default=2025)
    parser.add_argument("--holdout-season", type=int)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--sge-task-id", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "finalize":
        print(json.dumps(finalize(args), indent=2))
        return
    task_id = args.task_id
    if task_id is None:
        task_id = int(args.sge_task_id or os.environ["SGE_TASK_ID"]) - 1
    print(json.dumps(run_task(args, task_id), indent=2))


if __name__ == "__main__":
    main()
