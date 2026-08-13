#!/usr/bin/env python3
"""Select, refit, and consolidate the corrected-F6 wide margin roster."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gridiron_ml.cli._paths import project_root
from gridiron_ml.experiments.hyperparameter_search import build_tuned_model_config
from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints
from gridiron_ml.experiments.publication import filter_frame_for_feature_config
from gridiron_ml.models import build_model_from_config, load_model_checkpoint, validate_model_contract
from gridiron_ml.models.td_ensemble import TDEnsemble
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder


ROOT = project_root()
REUSED = {
    "M1": ("margin_linear_ridge", "linear", "ridge", "configs/models/linear/config_ridge.yaml"),
    "M2": ("margin_spline_spline_ridge", "spline", "spline_ridge", "configs/models/spline/config_spline_ridge.yaml"),
    "M3": ("margin_tree_random_forest", "tree", "random_forest", "configs/models/tree/config_random_forest.yaml"),
    "M4": ("margin_boosted_hist_gradient_boosted", "boosted", "hist_gradient_boosted", "configs/models/boosted/config_hist_gradient_boosted.yaml"),
    "M5": ("margin_neural_mlp", "neural", "mlp", "configs/models/neural/config_mlp.yaml"),
}
GAP_ROLE_TYPES = {
    "KNN_UNIFORM_EUCLIDEAN": "uniform_euclidean",
    "KNN_DISTANCE_EUCLIDEAN": "distance_euclidean",
    "KNN_UNIFORM_MANHATTAN": "uniform_manhattan",
    "KNN_DISTANCE_MANHATTAN": "distance_manhattan",
}
POLL_EXCLUDED_MODEL_IDS = {
    "margin_stat_robust",
    "margin_stat_weighted",
    "margin_stat_z_index",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_model_type(path: Path) -> str:
    return str((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("model_type") or path.stem.removeprefix("config_"))


def build_selection(args) -> dict:
    gap = pd.read_parquet(args.gap_selection)
    scientific = pd.read_parquet(args.scientific_selection)
    cfg = yaml.safe_load(args.gap_config.read_text(encoding="utf-8")) or {}
    rows = []
    for level in cfg["model_levels"]:
        match = gap.loc[
            gap["objective"].astype(str).eq("margin")
            & gap["feature_config"].astype(str).eq("F6")
            & gap["model_level"].astype(str).eq(level)
        ]
        if len(match) != 1 or int(match.iloc[0]["cv_fold_count"]) != 10:
            raise ValueError(f"Gap selection is incomplete for {level}: {len(match)} rows.")
        chosen = match.iloc[0]
        config = (ROOT / cfg["model_configs"][level]).resolve()
        family = str(cfg["model_families"][level])
        concrete = GAP_ROLE_TYPES.get(level, config_model_type(config))
        rows.append({
            "model_id": f"margin_{family}_{concrete}", "model_family": family,
            "model_type": concrete, "model_level": level, "model_config": str(config),
            "params_json": str(chosen["params_json"]), "cv_metric_mean": float(chosen["cv_metric_mean"]),
            "cv_metric_se": float(chosen["cv_metric_se"]), "cv_fold_count": 10,
            "selection_source": "corrected_f6_wide_gap_hps", "seed": int(chosen.get("seed", 1701)),
        })
    for level, (model_id, family, concrete, config_ref) in REUSED.items():
        match = scientific.loc[
            scientific["objective"].astype(str).eq("margin")
            & scientific["feature_config"].astype(str).eq("F6")
            & scientific["model_level"].astype(str).eq(level)
        ]
        if len(match) != 1 or int(match.iloc[0]["cv_fold_count"]) != 10:
            raise ValueError(f"Scientific corrected-F6 selection is incomplete for {level}.")
        chosen = match.iloc[0]
        rows.append({
            "model_id": model_id, "model_family": family, "model_type": concrete,
            "model_level": level, "model_config": str((ROOT / config_ref).resolve()),
            "params_json": str(chosen["params_json"]), "cv_metric_mean": float(chosen["cv_metric_mean"]),
            "cv_metric_se": float(chosen["cv_metric_se"]), "cv_fold_count": 10,
            "selection_source": "reused_corrected_scientific_f6_hps", "seed": int(chosen.get("seed", 1701)),
        })
    selection = pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)
    if len(selection) != 34 or selection["model_id"].duplicated().any():
        raise ValueError(f"Corrected-F6 wide roster requires 34 unique learned roles; found {len(selection)}.")
    selection.insert(0, "task_id", np.arange(1, len(selection) + 1))
    selection["feature_config"] = "F6"
    selection["data_path"] = str(args.data.resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output_root / "refit_manifest.csv", index=False)
    report = {
        "status": "ready", "learned_role_count": 34, "ensemble_role_count": 2,
        "final_roster_count": 36, "feature_config": "F6", "market_features": False,
        "gap_selection": str(args.gap_selection.resolve()),
        "gap_selection_sha256": file_sha256(args.gap_selection.resolve()),
        "scientific_selection": str(args.scientific_selection.resolve()),
        "scientific_selection_sha256": file_sha256(args.scientific_selection.resolve()),
        "data": str(args.data.resolve()), "data_sha256": file_sha256(args.data.resolve()),
    }
    (args.output_root / "selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return report


def run_task(args) -> dict:
    manifest = pd.read_csv(args.manifest)
    task_id = int(args.task_id or os.environ["SGE_TASK_ID"])
    selected = manifest.loc[manifest["task_id"].astype(int).eq(task_id)]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for task {task_id}; found {len(selected)}.")
    row = selected.iloc[0]
    model_id = str(row["model_id"])
    out = args.output_root / "runs" / model_id
    checkpoint = args.output_root / "checkpoints" / f"{model_id}.pkl"
    status_path = out / "status.json"
    train_years = list(range(2010, int(args.train_end_season) + 1))
    status = {
        "status": "running", "task_id": task_id, "model_id": model_id,
        "feature_config": "F6", "fit_train_seasons": train_years,
        "holdout_2025_excluded": int(args.train_end_season) <= 2024,
        "prospective_2026_excluded": True,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    try:
        raw = pd.read_parquet(Path(row["data_path"]))
        frame, metadata = filter_frame_for_feature_config(
            raw, feature_config="F6", registry_path=ROOT / "configs/features/feature_registry.yaml",
            ladders_path=ROOT / "configs/features/feature_ladders.yaml", strict_registry=True,
        )
        config = build_tuned_model_config(
            base_config_path=Path(row["model_config"]), params=json.loads(str(row["params_json"])),
        )
        config["seed"] = int(row["seed"])
        config["loss_function"] = "MAE"
        if isinstance(config.get("params"), dict) and "n_jobs" in config["params"]:
            config["params"]["n_jobs"] = 1
        model = build_model_from_config({"family": str(row["model_family"]), **config})
        validate_model_contract(model)
        builder = MatchupBuilder(representation="unit_matchup", safe_math=True)
        evaluator = TDEval(
            config={"model": {"family": str(row["model_family"]), "allow_market_features_for_training": False},
                    "feature_spec": {"include_market": False, "allow_market_features_for_training": False}},
            fingerprints=StaticFrameFingerprints(frame), matchup_builder=builder, model=model,
        )
        evaluator.train(train_years=train_years, val_years=[])
        if len(builder.feature_names()) != 681:
            raise ValueError(f"{model_id} resolved {len(builder.feature_names())} coordinates; expected corrected F6=681.")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        evaluator.model.save(checkpoint)
        status.update({
            "status": "success", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "selected_feature_count": int(metadata["selected_feature_count"]),
            "selected_features_json": json.dumps(metadata["selected_features"]),
            "matchup_feature_count": 681,
        })
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "success", "model_id": model_id, "checkpoint": str(checkpoint)}))
        return status
    except Exception as exc:
        status.update({"status": "failed", "error": repr(exc), "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        raise


def ensemble_record(args, *, variant: str, members: list[dict], train_years: list[int]) -> dict:
    model_id = f"margin_ensemble_{variant}"
    checkpoint = args.output_root / "checkpoints" / f"{model_id}.pkl"
    models = [load_model_checkpoint(item["checkpoint_path"]) for item in members]
    model = TDEnsemble({"model_type": variant, "model_name": model_id}, members=models)
    model.train(pd.DataFrame({"ensemble_probe": [0.0]}), np.array([0.0]))
    model.save(checkpoint)
    return {
        "model_id": model_id, "final_model_name": model_id, "model_family": "ensemble", "family": "ensemble",
        "model_type": variant, "concrete_model_type": variant, "objective": "margin",
        "feature_config": "member_roster", "selection_metric": np.nan, "outer_folds": 0,
        "selection_status": "post_cv_equal_weight", "training_note": "Ensemble of 34 corrected-F6 learned members.",
        "training_seasons": json.dumps(train_years), "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint), "fingerprint": "F6",
        "fingerprint_path": str(args.data.resolve()), "selected_features_json": "",
        "hyperparameters_json": "{}", "use_in_weekly_consensus": True, "use_in_tdnet_poll": True,
        "use_in_comparisons": True, "ensemble_members_json": json.dumps([item["model_id"] for item in members]),
        "comparative_only": False, "poll_exclusion_reason": "", "training_end_season": int(args.train_end_season),
    }


def finalize(args) -> dict:
    manifest = pd.read_csv(args.manifest)
    train_years = list(range(2010, int(args.train_end_season) + 1))
    records = []
    failures = []
    for row in manifest.to_dict("records"):
        status_path = args.output_root / "runs" / str(row["model_id"]) / "status.json"
        if not status_path.exists():
            failures.append(f"{row['model_id']}: missing status")
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "success" or status.get("fit_train_seasons") != train_years:
            failures.append(f"{row['model_id']}: invalid status or training boundary")
            continue
        checkpoint = Path(status["checkpoint_path"])
        if not checkpoint.exists() or file_sha256(checkpoint) != status["checkpoint_sha256"]:
            failures.append(f"{row['model_id']}: checkpoint missing or hash mismatch")
            continue
        records.append({
            "model_id": row["model_id"], "final_model_name": row["model_id"],
            "model_family": row["model_family"], "family": row["model_family"],
            "model_type": row["model_type"], "concrete_model_type": row["model_type"],
            "objective": "margin", "feature_config": "F6",
            "selection_metric": row["cv_metric_mean"], "outer_folds": row["cv_fold_count"],
            "selection_status": row["selection_source"], "training_note": "Corrected fixed 681-coordinate F6 contract.",
            "training_seasons": json.dumps(train_years), "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": status["checkpoint_sha256"], "fingerprint": "F6",
            "fingerprint_path": str(args.data.resolve()), "selected_features_json": status["selected_features_json"],
            "hyperparameters_json": row["params_json"], "use_in_weekly_consensus": True,
            "use_in_tdnet_poll": row["model_id"] not in POLL_EXCLUDED_MODEL_IDS,
            "use_in_comparisons": True, "ensemble_members_json": "",
            "comparative_only": False,
            "poll_exclusion_reason": (
                "known_invalid_poll_ordering_surface" if row["model_id"] in POLL_EXCLUDED_MODEL_IDS else ""
            ),
            "training_end_season": int(args.train_end_season),
        })
    if failures or len(records) != 34:
        raise RuntimeError(f"Wide F6 refit is incomplete: records={len(records)} failures={failures}")
    records.extend([
        ensemble_record(args, variant="mean_probability", members=records, train_years=train_years),
        ensemble_record(args, variant="median_margin", members=records, train_years=train_years),
    ])
    inventory = pd.DataFrame(records).sort_values(["model_family", "model_type", "model_id"]).reset_index(drop=True)
    inventory.insert(0, "roster_rank", np.arange(1, len(inventory) + 1))
    inventory_path = args.output_root / "final_model_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    ranking = inventory[["model_id", "selection_metric"]].copy()
    ranking = ranking.sort_values(["selection_metric", "model_id"], na_position="last").reset_index(drop=True)
    ranking["preseason_performance_rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(args.output_root / "preseason_model_rankings.csv", index=False)
    report = {
        "status": "complete", "model_count": 36, "learned_model_count": 34, "ensemble_model_count": 2,
        "feature_config": "F6", "matchup_feature_count": 681, "market_features": False,
        "training_seasons": train_years, "training_end_season": int(args.train_end_season),
        "holdout_2025": int(args.train_end_season) == 2024,
        "inventory_sha256": file_sha256(inventory_path), "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_root / "roster_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("command", choices=["build-selection", "run-task", "finalize"])
    parser.add_argument("--gap-selection", type=Path)
    parser.add_argument("--scientific-selection", type=Path)
    parser.add_argument("--gap-config", type=Path, default=ROOT / "configs/publication/hps_corrected_f6_wide_margin_gap.yaml")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--train-end-season", type=int, default=2025)
    args = parser.parse_args()
    if args.command == "build-selection":
        if not args.gap_selection or not args.scientific_selection:
            parser.error("build-selection requires --gap-selection and --scientific-selection")
        build_selection(args)
    elif args.command == "run-task":
        if not args.manifest:
            parser.error("run-task requires --manifest")
        run_task(args)
    else:
        if not args.manifest:
            parser.error("finalize requires --manifest")
        finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
