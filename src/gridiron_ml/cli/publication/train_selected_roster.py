#!/usr/bin/env python3
"""Refit the best completed publication trials into a weekly model roster.

Selection is performed from the consolidated rolling-origin table.  One
candidate is retained per objective and model family, then each candidate is
refit on all historical seasons through 2025.  The resulting inventory is the
small, local handoff consumed by weekly prediction, poll, and comparison
workflows; raw array fragments remain on the configured artifact filesystem.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gridiron_ml.experiments.hyperparameter_search import build_tuned_model_config
from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints
from gridiron_ml.experiments.publication import filter_frame_for_feature_config
from gridiron_ml.models import build_model_from_config
from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


def select_candidates(results: pd.DataFrame, *, families: set[str], top_per_family: int) -> pd.DataFrame:
    frame = results.loc[
        results["status"].eq("success")
        & results["model_family"].astype(str).isin(families)
        & results["objective"].astype(str).isin({"winner", "margin"})
    ].copy()
    if frame.empty:
        raise ValueError("No successful consolidated trials match the requested model families.")
    metric = frame["objective"].map({"winner": "brier_score", "margin": "mae"})
    frame["selection_metric"] = [pd.to_numeric(row[col], errors="coerce") for col, (_, row) in zip(metric, frame.iterrows())]
    frame = frame.loc[frame["selection_metric"].notna()].copy()
    keys = ["objective", "model_family", "model_level", "feature_config", "model_config", "params_json", "seed"]
    summary = (
        frame.groupby(keys, dropna=False, as_index=False)
        .agg(selection_metric=("selection_metric", "mean"), outer_folds=("outer_fold", "nunique"), trials=("task_id", "count"))
    )
    summary = summary.loc[summary["outer_folds"].ge(3)].copy()
    if summary.empty:
        raise ValueError("No candidate configuration has the required three outer folds.")
    selected = []
    for (objective, family), group in summary.groupby(["objective", "model_family"], sort=True):
        group = group.sort_values(
            ["selection_metric", "outer_folds", "feature_config", "model_level", "seed"],
            ascending=[True, False, True, True, True], kind="mergesort",
        ).head(int(top_per_family))
        selected.append(group)
    selected = pd.concat(selected, ignore_index=True)
    selected["candidate_id"] = [
        _safe_id(f"{row.objective}_{row.model_family}_{row.model_level}_{row.feature_config}")
        for row in selected.itertuples()
    ]
    # Carry one representative manifest row for the exact data/config paths.
    selected = selected.merge(
        frame.sort_values("selection_metric", kind="mergesort")
        .drop_duplicates(keys),
        on=keys,
        how="left",
        suffixes=("", "_representative"),
    )
    return selected.sort_values(["objective", "selection_metric", "candidate_id"], kind="mergesort").reset_index(drop=True)


def refit_candidate(row: pd.Series, *, output_root: Path, project_root: Path, train_years: list[int]) -> dict:
    config = build_tuned_model_config(
        base_config_path=Path(row["model_config"]),
        params=json.loads(row.get("params_json") or "{}"),
    )
    frame = pd.read_parquet(row["data_path"])
    if str(row["model_family"]) == "temporal":
        from gridiron_ml.fingerprints.temporal import build_temporal_fingerprints

        frame = build_temporal_fingerprints(frame, **config.get("temporal_fingerprint", {}))
    allow_market = str(row["feature_config"]) in {"F7", "F8"}
    frame, feature_metadata = filter_frame_for_feature_config(
        frame,
        feature_config=str(row["feature_config"]),
        registry_path=row["feature_registry"],
        ladders_path=row["feature_ladders"],
        strict_registry=True,
    )
    config["seed"] = int(row["seed"])
    config["loss_function"] = "WinnerAccuracy" if str(row["objective"]) == "winner" else "MAE"
    if allow_market:
        config["allow_market_features_for_training"] = True
    model = build_model_from_config({"family": row["model_family"], **config})
    evaluator = TDEval(
        config={
            "model": {"family": row["model_family"], "allow_market_features_for_training": allow_market},
            "feature_spec": {"include_market": allow_market, "allow_market_features_for_training": allow_market},
        },
        fingerprints=StaticFrameFingerprints(frame),
        matchup_builder=MatchupBuilder(representation="unit_matchup", safe_math=True),
        model=model,
    )
    evaluator.train(train_years=train_years, val_years=[])
    candidate_id = str(row["candidate_id"])
    checkpoint = output_root / "checkpoints" / f"{candidate_id}.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    evaluator.model.save(checkpoint)
    return {
        "model_id": candidate_id,
        "final_model_name": candidate_id,
        "model_family": str(row["model_family"]),
        "family": str(row["model_family"]),
        "model_type": str(config.get("model_type", row["model_level"])),
        "objective": str(row["objective"]),
        "feature_config": str(row["feature_config"]),
        "selection_metric": float(row["selection_metric"]),
        "outer_folds": int(row["outer_folds"]),
        "training_seasons": json.dumps(train_years),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "fingerprint": "v1_7",
        "fingerprint_path": str(Path(row["data_path"]).resolve()),
        "selected_features_json": json.dumps(feature_metadata["selected_features"]),
        "use_in_weekly_consensus": True,
        "use_in_tdnet_poll": True,
    }


def refit_naive(*, output_root: Path, project_root: Path, train_years: list[int]) -> dict:
    config = {"family": "naive", "model_type": "home_team", "model_name": "naive_home", "loss_function": "WinnerAccuracy"}
    frame = pd.read_parquet(project_root / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet")
    model = build_model_from_config(config)
    evaluator = TDEval(
        config={"model": {"family": "naive"}},
        fingerprints=StaticFrameFingerprints(frame),
        matchup_builder=MatchupBuilder(representation="unit_matchup", safe_math=True),
        model=model,
    )
    evaluator.train(train_years=train_years, val_years=[])
    checkpoint = output_root / "checkpoints" / "naive_home.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    evaluator.model.save(checkpoint)
    return {
        "model_id": "naive_home",
        "final_model_name": "naive_home",
        "model_family": "naive",
        "family": "naive",
        "model_type": "home_team",
        "objective": "winner",
        "feature_config": "F0",
        "selection_metric": np.nan,
        "outer_folds": 0,
        "training_seasons": json.dumps(train_years),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "fingerprint": "v1_7",
        "fingerprint_path": str((project_root / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet").resolve()),
        "use_in_weekly_consensus": False,
        "use_in_tdnet_poll": False,
    }


def refit_knn(*, output_root: Path, project_root: Path, train_years: list[int]) -> dict:
    """Fit the explicit historical-matchup KNN baseline on the full ladder."""
    source = project_root / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet"
    frame = pd.read_parquet(source)
    frame, feature_metadata = filter_frame_for_feature_config(
        frame,
        feature_config="F6",
        registry_path=project_root / "configs/features/feature_registry.yaml",
        ladders_path=project_root / "configs/features/feature_ladders.yaml",
        strict_registry=True,
    )
    model = build_model_from_config({
        "family": "knn", "model_type": "distance", "model_name": "knn_distance_f6",
        "params": {"n_neighbors": 10, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
        "loss_function": "MAE", "seed": 1701,
    })
    evaluator = TDEval(
        config={"model": {"family": "knn"}},
        fingerprints=StaticFrameFingerprints(frame),
        matchup_builder=MatchupBuilder(representation="unit_matchup", safe_math=True),
        model=model,
    )
    evaluator.train(train_years=train_years, val_years=[])
    checkpoint = output_root / "checkpoints" / "knn_distance_f6.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    evaluator.model.save(checkpoint)
    return {
        "model_id": "knn_distance_f6", "final_model_name": "knn_distance_f6",
        "model_family": "knn", "family": "knn", "model_type": "distance",
        "objective": "margin", "feature_config": "F6", "selection_metric": np.nan,
        "outer_folds": 0, "training_seasons": json.dumps(train_years),
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "fingerprint": "v1_7", "fingerprint_path": str(source.resolve()),
        "selected_features_json": json.dumps(feature_metadata["selected_features"]),
        "use_in_weekly_consensus": True, "use_in_tdnet_poll": True,
    }


def refit_legacy_stat(*, output_root: Path, project_root: Path, train_years: list[int]) -> dict:
    source = project_root / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet"
    frame = pd.read_parquet(source)
    config = yaml.safe_load((project_root / "configs/models/stat/config_z_index.yaml").read_text()) or {}
    model = build_model_from_config({"family": "stat", **config})
    evaluator = TDEval(
        config={"model": {"family": "stat"}},
        fingerprints=StaticFrameFingerprints(frame),
        matchup_builder=MatchupBuilder(representation="unit_matchup", safe_math=True),
        model=model,
    )
    evaluator.train(train_years=train_years, val_years=[])
    checkpoint = output_root / "checkpoints" / "stat_z_index_f6.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    evaluator.model.save(checkpoint)
    return {
        "model_id": "stat_z_index_f6", "final_model_name": "stat_z_index_f6",
        "model_family": "stat", "family": "stat", "model_type": "stat_z_index",
        "objective": "winner", "feature_config": "F6", "selection_metric": np.nan,
        "outer_folds": 0, "training_seasons": json.dumps(train_years),
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "fingerprint": "v1_7", "fingerprint_path": str(source.resolve()),
        "use_in_weekly_consensus": True, "use_in_tdnet_poll": True,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("data/experiments/publication_model_selection/tables/all_trial_results.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("models/season_2026_roster"))
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--families", nargs="+", default=["linear", "stat", "tree", "spline", "boosted", "neural", "structured_neural", "kernel", "temporal", "knn"])
    parser.add_argument("--top-per-family", type=int, default=1)
    parser.add_argument("--no-naive", action="store_true")
    args = parser.parse_args()
    results = pd.read_parquet(args.results)
    candidates = select_candidates(results, families=set(args.families), top_per_family=args.top_per_family)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    train_years = list(range(2010, 2026))
    inventory = [refit_candidate(row, output_root=output_root, project_root=args.project_root.resolve(), train_years=train_years) for _, row in candidates.iterrows()]
    if "knn" in args.families:
        inventory.append(refit_knn(output_root=output_root, project_root=args.project_root.resolve(), train_years=train_years))
    if "stat" in args.families and not (inventory and any(row["model_family"] == "stat" for row in inventory)):
        inventory.append(refit_legacy_stat(output_root=output_root, project_root=args.project_root.resolve(), train_years=train_years))
    if not args.no_naive:
        inventory.append(refit_naive(output_root=output_root, project_root=args.project_root.resolve(), train_years=train_years))
    inventory = pd.DataFrame(inventory)
    inventory = inventory.sort_values(["use_in_weekly_consensus", "objective", "selection_metric", "model_id"], ascending=[False, True, True, True], na_position="last", kind="mergesort").reset_index(drop=True)
    inventory.insert(0, "roster_rank", np.arange(1, len(inventory) + 1))
    inventory.to_csv(output_root / "final_model_inventory.csv", index=False)
    candidates.to_csv(output_root / "selected_candidates.csv", index=False)
    manifest = {"training_seasons": train_years, "model_count": len(inventory), "weekly_model_count": int(inventory["use_in_weekly_consensus"].sum()), "families": sorted(set(inventory["model_family"])), "inventory_sha256": sha256_file(output_root / "final_model_inventory.csv")}
    (output_root / "roster_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(inventory[["roster_rank", "model_id", "objective", "model_family", "selection_metric", "checkpoint_path"]].to_string(index=False))


if __name__ == "__main__":
    main()
