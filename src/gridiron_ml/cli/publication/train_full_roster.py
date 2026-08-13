#!/usr/bin/env python3
"""Select and materialize the complete TDNet 39-role x 2-objective roster.

The roster is intentionally concrete-type based.  Legacy nested-CV artifacts
provide the tuned configurations for stat/linear/tree roles, while the
publication search table provides the tuned configurations for spline,
boosted, neural, structured-neural, kernel, and temporal roles.  KNN and
naive roles are transparent baselines; the two ensemble roles are fixed
equal-weight combinations of the trained objective-specific roster.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
import csv
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gridiron_ml.experiments.hyperparameter_search import build_tuned_model_config
from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints
from gridiron_ml.experiments.publication import filter_frame_for_feature_config
from gridiron_ml.models import build_model_from_config, load_model_checkpoint
from gridiron_ml.models.td_ensemble import TDEnsemble
from gridiron_ml.pipeline.contracts.features import is_key_column, is_label_column
from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder


LEGACY_MODELS = {
    "stat": ["stat_z_index", "stat_percentile", "stat_robust", "stat_weighted"],
    "linear": [
        "ols", "ridge", "lasso", "elastic_net", "huber", "bayesian", "ard",
        "ransac", "orthogonal_matching_pursuit", "sgd", "passive_aggressive",
    ],
    "tree": ["random_forest", "extra_trees", "gradient_boosted"],
}
PUBLICATION_TYPES = {
    "spline_ridge": "spline",
    "hist_gradient_boosted": "boosted",
    "mlp": "neural",
    "structured_mlp": "structured_neural",
    "rbf_kernel_ridge": "kernel",
    "rbf_svr": "kernel",
    "gaussian_process": "kernel",
    "nystroem_ridge": "kernel",
    "decay_ridge": "temporal",
    "trend_elastic_net": "temporal",
    "temporal_random_forest": "temporal",
    "temporal_hist_gradient_boosted": "temporal",
}
KNN_TYPES = ["uniform", "distance", "compact", "full_fingerprint"]
NAIVE_TYPES = ["majority", "constant_margin", "home_team"]
ENSEMBLE_TYPES = ["mean_probability", "median_margin"]
MARKET_FEATURE_CONFIGS = frozenset({"F7", "F8"})
COMPARATIVE_BASELINE_FAMILIES = frozenset({"knn", "naive"})
POLL_EXCLUDED_MODEL_IDS = frozenset(
    {
        "winner_linear_ols",
        "winner_linear_huber",
        "winner_linear_ridge",
        "winner_linear_sgd",
    }
)


def _safe(value: object) -> str:
    return "".join(c if str(c).isalnum() or c in "-_" else "_" for c in str(value))


def _local_path(value: object, project_root: Path) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    text = str(path)
    marker = "/TDNet/"
    if marker in text:
        candidate = project_root / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return project_root / path if not path.is_absolute() else path


def _json(value: object, default):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _mapping(value: object) -> dict:
    if isinstance(value, dict):
        return value
    parsed = _json(value, None)
    if isinstance(parsed, dict):
        return parsed
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _nested_rows(objective: str, artifact_root: Path, project_root: Path) -> pd.DataFrame:
    """Read the compact metric fields from the completed nested-CV jobs."""
    root = artifact_root / objective
    manifest = pd.read_parquet(root / "job_manifest.parquet")
    rows = []
    wanted = {"status", "tuning_score", "mae", "winner_accuracy", "brier_score", "selected_features_json"}
    for row in manifest.itertuples(index=False):
        path = _local_path(row.metrics_path, project_root)
        if not path.exists():
            continue
        try:
            with path.open(newline="") as handle:
                record = next(csv.DictReader(handle))
        except (OSError, StopIteration, csv.Error):
            continue
        if record.get("status", "success") != "success":
            continue
        item = {key: getattr(row, key) for key in manifest.columns}
        item.update({key: record.get(key) for key in wanted})
        item["selection_metric"] = pd.to_numeric(record.get("tuning_score"), errors="coerce")
        item["selected_features"] = _json(record.get("selected_features_json"), [])
        rows.append(item)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No completed nested-CV metrics found under {root}.")
    frame = frame.loc[frame["selection_metric"].notna()].copy()
    aggregate = frame.groupby(["base_job_index", "family", "model"], as_index=False).agg(
        selection_metric=("selection_metric", "mean"), outer_folds=("outer_fold", "nunique")
    )
    aggregate = aggregate.loc[aggregate["outer_folds"].ge(5)].copy()
    if aggregate.empty:
        raise RuntimeError(f"No complete five-fold nested candidates found under {root}.")
    winners = aggregate.sort_values(
        ["family", "model", "selection_metric", "base_job_index"],
        ascending=[True, True, False, True], kind="mergesort",
    ).drop_duplicates(["family", "model"])
    representative = frame.sort_values(["base_job_index", "outer_fold"], kind="mergesort").drop_duplicates("base_job_index")
    selected = winners.merge(representative, on=["base_job_index", "family", "model"], how="left", suffixes=("", "_representative"))
    selected["objective"] = objective
    selected["source_kind"] = "searched_nested_cv"
    selected["model_family"] = selected["family"].astype(str)
    selected["concrete_type"] = selected["model"].astype(str)
    selected["feature_config"] = "top_k_" + selected["top_k_features"].astype(str)
    selected["model_config"] = selected["model_config_path"].map(lambda p: str(_local_path(p, project_root)))
    selected["data_path"] = selected["fingerprint_path"].map(lambda p: str(_local_path(p, project_root)))
    selected["params"] = selected["params_json"].map(lambda x: _json(x, {}))
    return selected


def _publication_rows(objective: str, results_path: Path, project_root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(results_path)
    frame = frame.loc[frame["objective"].eq(objective) & frame["status"].eq("success")].copy()
    # Vegas is an evaluation/comparison source, never a model feature.  The
    # historical HPS table contains market-bearing F7/F8 candidates, but they
    # are not eligible for the publication roster.
    frame = frame.loc[~frame["feature_config"].astype(str).isin(MARKET_FEATURE_CONFIGS)].copy()
    metric = "brier_score" if objective == "winner" else "mae"
    frame["selection_metric"] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.loc[frame["selection_metric"].notna()].copy()
    keys = ["model_family", "model_level", "feature_config", "model_config", "params_json", "seed"]
    summary = frame.groupby(keys, dropna=False, as_index=False).agg(
        selection_metric=("selection_metric", "mean"), outer_folds=("outer_fold", "nunique")
    )
    summary = summary.loc[summary["outer_folds"].ge(3)].copy()
    summary = summary.sort_values(
        ["model_family", "model_level", "selection_metric", "feature_config", "seed"],
        ascending=[True, True, True, True, True], kind="mergesort",
    )
    selected = summary.drop_duplicates(["model_family", "model_level"])
    representative = frame.sort_values(["selection_metric", "task_id"], kind="mergesort").drop_duplicates(keys)
    selected = selected.merge(representative, on=keys, how="left", suffixes=("", "_representative"))
    selected["objective"] = objective
    selected["source_kind"] = "searched_publication_cv"
    type_by_level = {"M1": "ridge", "M2": "spline_ridge", "M3": "random_forest", "M4": "hist_gradient_boosted", "M5": "mlp", "M6": "structured_mlp",
                     "K1": "rbf_kernel_ridge", "K2": "rbf_svr", "K3": "gaussian_process", "K4": "nystroem_ridge",
                     "T1": "decay_ridge", "T2": "trend_elastic_net", "T3": "temporal_random_forest", "T4": "temporal_hist_gradient_boosted"}
    selected["concrete_type"] = selected["model_level"].map(type_by_level)
    selected["params"] = selected["params_json"].map(lambda x: _json(x, {}))
    selected["model_config"] = selected["model_config"].map(lambda p: str(_local_path(p, project_root)))
    selected["data_path"] = selected["data_path"].map(lambda p: str(_local_path(p, project_root)))
    return selected.loc[selected["concrete_type"].notna()].copy()


def select_tuned(objective: str, *, results_path: Path, nested_root: Path, project_root: Path) -> pd.DataFrame:
    pub = _publication_rows(objective, results_path, project_root)
    nested = _nested_rows(objective, nested_root, project_root)
    # Publication searches supersede the overlapping ridge/random-forest roles;
    # nested CV supplies all other legacy concrete types.
    pub_types = set(pub["concrete_type"].astype(str))
    nested = nested.loc[~nested["concrete_type"].astype(str).isin(pub_types)].copy()
    selected = pd.concat([pub, nested], ignore_index=True, sort=False)
    selected_features = selected.get("selected_features", pd.Series([], dtype=object))
    market_mask = selected_features.map(
        lambda value: any(str(feature).startswith("market_") for feature in (value if isinstance(value, list) else []))
    )
    if market_mask.any():
        selected = selected.loc[~market_mask].copy()
    expected = set(sum(LEGACY_MODELS.values(), [])) | set(PUBLICATION_TYPES)
    missing = expected - set(selected["concrete_type"].astype(str))
    if missing:
        raise RuntimeError(f"Tuned search selection is missing concrete types: {sorted(missing)}")
    return selected


def _base_config(row: pd.Series, objective: str, project_root: Path) -> dict:
    config = build_tuned_model_config(
        base_config_path=_local_path(row["model_config"], project_root),
        params=_mapping(row.get("params")),
    )
    # Keep the selected estimator hyperparameters unchanged.  A serial refit
    # is deliberate here: the roster is a large collection of retained
    # checkpoints, and unrestricted forest/OpenMP parallelism can exhaust the
    # node before the inventory is written.
    if str(row.get("concrete_type", "")) in {
        "random_forest",
        "extra_trees",
        "temporal_random_forest",
    } and isinstance(config.get("params"), dict):
        config["params"]["n_jobs"] = 1
    config["model_name"] = str(row["model_id"])
    config["seed"] = int(row.get("seed", 42) if pd.notna(row.get("seed", 42)) else 42)
    config["loss_function"] = "WinnerAccuracy" if objective == "winner" else "MAE"
    return config


def _selected_frame(frame: pd.DataFrame, selected: list[str]) -> tuple[pd.DataFrame, list[str]]:
    selected = [str(c) for c in selected if str(c) in frame.columns]
    keep = [c for c in frame.columns if is_key_column(c) or is_label_column(c) or str(c).startswith("next_")]
    keep = list(dict.fromkeys(keep + selected))
    if not selected:
        raise ValueError("Selected nested-CV feature list did not match the fingerprint frame.")
    return frame.loc[:, keep].copy(), selected


def _fit_model(row: pd.Series, *, objective: str, output_root: Path, project_root: Path, train_years: list[int]) -> dict:
    model_id = str(row["model_id"])
    if str(row.get("feature_config", "")) in MARKET_FEATURE_CONFIGS:
        raise ValueError(f"Refusing to train market-bearing roster model {model_id}.")
    checkpoint = output_root / "checkpoints" / f"{model_id}.pkl"
    if checkpoint.exists():
        model = load_model_checkpoint(checkpoint)
        return _inventory_record(row, checkpoint, model, objective, train_years)
    frame = pd.read_parquet(_local_path(row["data_path"], project_root))
    config = _base_config(row, objective, project_root)
    concrete = str(row["concrete_type"])
    if str(row["source_kind"]) == "searched_nested_cv":
        frame, selected_features = _selected_frame(frame, _json(row.get("selected_features_json"), row.get("selected_features", [])))
        feature_metadata = {"selected_features": selected_features, "feature_config": str(row["feature_config"])}
    else:
        if str(row["concrete_type"]) in {"decay_ridge", "trend_elastic_net", "temporal_random_forest", "temporal_hist_gradient_boosted"}:
            from gridiron_ml.fingerprints.temporal import build_temporal_fingerprints
            frame = build_temporal_fingerprints(frame, **config.get("temporal_fingerprint", {}))
        frame, feature_metadata = filter_frame_for_feature_config(
            frame, feature_config=str(row["feature_config"]),
            registry_path=project_root / "configs/features/feature_registry.yaml",
            ladders_path=project_root / "configs/features/feature_ladders.yaml", strict_registry=True,
        )
    family = str(row["model_family"])
    allow_market = False
    evaluator = TDEval(
        config={"model": {"family": family, "allow_market_features_for_training": allow_market},
                "feature_spec": {"include_market": allow_market, "allow_market_features_for_training": allow_market}},
        fingerprints=StaticFrameFingerprints(frame),
        matchup_builder=MatchupBuilder(representation="unit_matchup", safe_math=True),
        model=build_model_from_config({"family": family, **config}),
    )
    evaluator.train(train_years=train_years, val_years=[])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    evaluator.model.save(checkpoint)
    row["selected_features_json"] = json.dumps(feature_metadata.get("selected_features", []))
    return _inventory_record(row, checkpoint, evaluator.model, objective, train_years)


def _inventory_record(row: pd.Series, checkpoint: Path, model, objective: str, train_years: list[int]) -> dict:
    fingerprint = row.get("fingerprint", "v1_7")
    if pd.isna(fingerprint) or str(fingerprint).strip().lower() in {"", "nan", "none"}:
        data_path = str(row.get("data_path", ""))
        fingerprint = next((part for part in Path(data_path).parts if part.startswith("v1_")), "v1_7")
    model_id = str(row["model_id"])
    family = str(row["model_family"])
    feature_config = str(row.get("feature_config", ""))
    comparative_only = family.lower() in COMPARATIVE_BASELINE_FAMILIES
    if feature_config in MARKET_FEATURE_CONFIGS:
        raise ValueError(f"Market-bearing feature configuration reached roster inventory: {feature_config}")
    return {
        "model_id": model_id, "final_model_name": model_id,
        "model_family": family, "family": family,
        "model_type": str(row["concrete_type"]), "concrete_model_type": str(row["concrete_type"]),
        "objective": objective, "feature_config": feature_config,
        "selection_metric": row.get("selection_metric", np.nan), "outer_folds": row.get("outer_folds", 0),
        "selection_status": str(row.get("source_kind", "unknown")),
        "training_note": str(row.get("training_note", "")),
        "training_seasons": json.dumps(train_years), "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint), "fingerprint": str(fingerprint),
        "fingerprint_path": str(row.get("data_path", "")),
        "selected_features_json": str(row.get("selected_features_json", "")),
        "hyperparameters_json": json.dumps(row.get("params", {}), sort_keys=True, default=str),
        "use_in_weekly_consensus": not comparative_only,
        "use_in_tdnet_poll": not comparative_only and model_id not in POLL_EXCLUDED_MODEL_IDS,
        "use_in_comparisons": True,
        "comparative_only": comparative_only,
        "poll_exclusion_reason": (
            "comparative_baseline" if comparative_only else
            "known_poll_ordering_failure" if model_id in POLL_EXCLUDED_MODEL_IDS else ""
        ),
    }


def _roster_model_id(objective: str, model_family: str, concrete_type: str, *, normalized_names: bool = False) -> str:
    prefix = "" if normalized_names else f"{objective}_"
    return f"{prefix}{model_family}_{concrete_type}"


def _manual_row(*, objective: str, model_family: str, concrete_type: str, feature_config: str, params: dict, source_kind: str, normalized_names: bool = False) -> pd.Series:
    return pd.Series({"model_id": f"{_roster_model_id(objective, model_family, concrete_type, normalized_names=normalized_names)}_{feature_config}", "model_family": model_family,
                      "concrete_type": concrete_type, "feature_config": feature_config, "params": params,
                      "source_kind": source_kind, "fingerprint": "v1_7",
                      "data_path": "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet"})


def _fit_manual(row: pd.Series, *, objective: str, output_root: Path, project_root: Path, train_years: list[int]) -> dict:
    model_id = str(row["model_id"]); checkpoint = output_root / "checkpoints" / f"{model_id}.pkl"
    if not checkpoint.exists():
        source = project_root / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet"
        frame = pd.read_parquet(source)
        if row["model_family"] == "knn":
            from gridiron_ml.experiments.publication import filter_frame_for_feature_config
            if row["feature_config"] == "full_fingerprint":
                selected = [c for c in frame.columns if not is_key_column(c) and not is_label_column(c) and not str(c).startswith("game_") and not str(c).startswith("market_")]
                frame, selected_features = _selected_frame(frame, selected)
            else:
                frame, meta = filter_frame_for_feature_config(frame, feature_config="F6", registry_path=project_root / "configs/features/feature_registry.yaml", ladders_path=project_root / "configs/features/feature_ladders.yaml", strict_registry=True)
                selected_features = meta["selected_features"]
        else:
            selected_features = []
        config = {"family": row["model_family"], "model_type": row["concrete_type"], "model_name": model_id,
                  "loss_function": "WinnerAccuracy" if objective == "winner" else "MAE", "seed": 1701, "params": dict(row["params"])}
        evaluator = TDEval(config={"model": {"family": row["model_family"]}}, fingerprints=StaticFrameFingerprints(frame), matchup_builder=MatchupBuilder(representation="unit_matchup", safe_math=True), model=build_model_from_config(config))
        evaluator.train(train_years=train_years, val_years=[])
        checkpoint.parent.mkdir(parents=True, exist_ok=True); evaluator.model.save(checkpoint)
        row["selected_features_json"] = json.dumps(selected_features)
    model = load_model_checkpoint(checkpoint)
    return _inventory_record(row, checkpoint, model, objective, train_years)


def _fit_ensemble(*, objective: str, variant: str, members: list[dict], output_root: Path, train_years: list[int], normalized_names: bool = False) -> dict:
    model_id = _roster_model_id(objective, "ensemble", variant, normalized_names=normalized_names)
    members = [
        item for item in members
        if str(item.get("model_family", "")).lower() not in COMPARATIVE_BASELINE_FAMILIES
    ]
    if not members:
        raise RuntimeError(f"Cannot build {model_id}: no learned members remain after baseline exclusion.")
    checkpoint = output_root / "checkpoints" / f"{model_id}.pkl"
    if not checkpoint.exists():
        model = TDEnsemble({"model_type": variant, "model_name": model_id}, members=[item["model"] for item in members])
        model.train(pd.DataFrame({"ensemble_probe": [0.0]}), np.array([0.0]))
        checkpoint.parent.mkdir(parents=True, exist_ok=True); model.save(checkpoint)
    return _inventory_record(pd.Series({"model_id": model_id, "model_family": "ensemble", "concrete_type": variant,
        "feature_config": "member_roster", "selection_metric": np.nan, "outer_folds": 0,
        "source_kind": "post_oof_equal_weight", "fingerprint": "v1_7", "data_path": "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet"}), checkpoint,
        load_model_checkpoint(checkpoint), objective, train_years) | {"ensemble_members_json": json.dumps([item["model_id"] for item in members])}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--results", type=Path, default=Path("data/experiments/publication_model_selection/tables/all_trial_results.parquet"))
    parser.add_argument("--nested-root", type=Path, default=Path(os.environ.get("TDNET_NESTED_SEARCH_ROOT", "nested_search_2025")))
    parser.add_argument("--output-root", type=Path, default=Path("models/season_2026_full_roster"))
    parser.add_argument(
        "--train-through-season",
        type=int,
        default=2025,
        help="Inclusive final season used to fit the roster checkpoints.",
    )
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--refresh-selection", action="store_true")
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=("winner", "margin"),
        default=("winner", "margin"),
        help="Objectives to refit; running one at a time keeps peak memory bounded.",
    )
    parser.add_argument(
        "--skip-comparative-baselines",
        action="store_true",
        help="Do not refit KNN/naive baselines; they are never eligible for the poll.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Record an individual model failure and finish the remaining roster.",
    )
    parser.add_argument("--exclude-model-ids", nargs="*", default=[], help="Known refit failures to leave out of this local run.")
    parser.add_argument(
        "--normalize-margin-names",
        action="store_true",
        help="When training only margin, omit the margin_ objective prefix from model IDs and generated artifacts.",
    )
    args = parser.parse_args()
    if args.train_through_season < 2010:
        raise ValueError("--train-through-season must be at least 2010.")
    project_root = args.project_root.resolve(); results = _local_path(args.results, project_root); nested_root = args.nested_root
    output_root = args.output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    normalized_names = bool(args.normalize_margin_names and list(args.objectives) == ["margin"])
    selection_cache = output_root / "full_roster_selection.csv"
    if selection_cache.exists() and not args.refresh_selection:
        selected = pd.read_csv(selection_cache)
        selected["params"] = selected.get("params", pd.Series(["{}"] * len(selected))).map(_mapping)
    else:
        selected_frames = [select_tuned(o, results_path=results, nested_root=nested_root, project_root=project_root) for o in args.objectives]
        selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected[selected["objective"].isin(args.objectives)].copy()
    selected["model_id"] = [
        _roster_model_id(str(row["objective"]), str(row["model_family"]), str(row["concrete_type"]), normalized_names=normalized_names)
        for _, row in selected.iterrows()
    ]
    selected = selected.loc[~selected["model_id"].isin(set(args.exclude_model_ids))].copy()
    rows = []
    for _, row in selected.iterrows():
        row = row.copy(); row["model_id"] = _roster_model_id(str(row["objective"]), str(row["model_family"]), str(row["concrete_type"]), normalized_names=normalized_names)
        rows.append(row)
    train_years = list(range(2010, args.train_through_season + 1))
    if args.selection_only:
        selected.to_csv(output_root / "full_roster_selection.csv", index=False)
        print(selected.groupby(["objective", "source_kind"]).size().to_string()); print(selected[["objective", "model_family", "concrete_type", "feature_config", "selection_metric"]].sort_values(["objective", "model_family", "concrete_type"]).to_string(index=False)); return
    inventory = []
    failures = []
    trained_by_objective = {"winner": [], "margin": []}
    for row in rows:
        try:
            record = _fit_model(row, objective=str(row["objective"]), output_root=output_root, project_root=project_root, train_years=train_years)
        except Exception as exc:
            if not args.allow_failures:
                raise RuntimeError(f"Failed training {row['model_id']}: {exc}") from exc
            failures.append({"model_id": str(row["model_id"]), "error": repr(exc)})
            continue
        inventory.append(record)
        if not bool(record.get("comparative_only", False)):
            trained_by_objective[str(row["objective"])].append({"model_id": record["model_id"], "model_family": record["model_family"], "model": load_model_checkpoint(record["checkpoint_path"])})
    for objective in args.objectives:
        if not args.skip_comparative_baselines:
            for concrete_type in KNN_TYPES:
                cfg = {"n_neighbors": 5 if concrete_type == "uniform" else 10, "weights": "uniform" if concrete_type == "uniform" else "distance", "metric": "euclidean", "n_jobs": -1}
                feature = "full_fingerprint" if concrete_type == "full_fingerprint" else ("compact" if concrete_type == "compact" else "F6")
                row = _manual_row(objective=objective, model_family="knn", concrete_type=concrete_type, feature_config=feature, params=cfg, source_kind="default_no_completed_search", normalized_names=normalized_names)
                record = _fit_manual(row, objective=objective, output_root=output_root, project_root=project_root, train_years=train_years); inventory.append(record)
            for concrete_type in NAIVE_TYPES:
                params = {}; row = _manual_row(objective=objective, model_family="naive", concrete_type=concrete_type, feature_config="F0", params=params, source_kind="non_tuned", normalized_names=normalized_names)
                if not (output_root / "checkpoints" / f"{row['model_id']}.pkl").exists():
                    source = project_root / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet"; frame = pd.read_parquet(source)
                    model = build_model_from_config({"family": "naive", "model_type": concrete_type, "model_name": row["model_id"], "loss_function": "WinnerAccuracy" if objective == "winner" else "MAE"})
                    evaluator = TDEval(config={"model": {"family": "naive"}}, fingerprints=StaticFrameFingerprints(frame), matchup_builder=MatchupBuilder(representation="unit_matchup", safe_math=True), model=model); evaluator.train(train_years=train_years, val_years=[]); evaluator.model.save(output_root / "checkpoints" / f"{row['model_id']}.pkl")
                record = _inventory_record(row, output_root / "checkpoints" / f"{row['model_id']}.pkl", load_model_checkpoint(output_root / "checkpoints" / f"{row['model_id']}.pkl"), objective, train_years); inventory.append(record); trained_by_objective[objective].append({"model_id": record["model_id"], "model_family": record["model_family"], "model": load_model_checkpoint(record["checkpoint_path"])})
        for variant in ENSEMBLE_TYPES:
            record = _fit_ensemble(objective=objective, variant=variant, members=trained_by_objective[objective], output_root=output_root, train_years=train_years, normalized_names=normalized_names); inventory.append(record)
    inventory = pd.DataFrame(inventory).sort_values(["objective", "model_family", "model_type"], kind="mergesort").reset_index(drop=True)
    inventory.insert(0, "roster_rank", np.arange(1, len(inventory) + 1))
    inventory.to_csv(output_root / "final_model_inventory.csv", index=False)
    selected.to_csv(output_root / "full_roster_selection.csv", index=False)
    manifest = {"model_count": int(len(inventory)), "concrete_role_count": 39, "objective_count": int(inventory["objective"].nunique()), "weekly_model_count": int(inventory["use_in_weekly_consensus"].sum()), "poll_model_count": int(inventory["use_in_tdnet_poll"].sum()), "training_seasons": train_years, "objectives": list(args.objectives), "comparative_baselines_included": not args.skip_comparative_baselines, "normalized_margin_names": normalized_names, "failed_models": failures, "inventory_sha256": sha256_file(output_root / "final_model_inventory.csv")}
    (output_root / "roster_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(inventory[["roster_rank", "model_id", "objective", "model_family", "model_type", "selection_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
