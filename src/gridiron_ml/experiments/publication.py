"""Generic manifest-driven feature-by-model publication experiments.

This module consolidates the reusable parts of the older fingerprint search:
manifest generation, temporal folds, chunk execution, compact fragments,
resumption, merging, status tables, and finalist reduction. Existing search
entry points remain available and their manifests/results remain compatible.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
import json
import math
import os
import shutil
import socket
import tempfile
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import yaml

from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints
from gridiron_ml.experiments.hyperparameter_search import (
    build_tuned_model_config,
    sample_setpoints,
    stable_seed,
)
from gridiron_ml.models import build_model_from_config, validate_model_contract
from gridiron_ml.pipeline.contracts.features import (
    is_feature_column,
    is_key_column,
    is_label_column,
    is_market_column,
)
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder


REQUIRED_MANIFEST_COLUMNS = (
    "task_id",
    "chunk_id",
    "experiment_id",
    "objective",
    "feature_config",
    "model_config",
    "split_config",
    "seed",
    "output_path",
    "estimated_memory_gb",
    "estimated_runtime",
)


@dataclass(frozen=True)
class FeatureDefinition:
    """Expanded exact feature metadata from the registry."""

    name: str
    family: str
    tier: str
    metadata: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(project_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root).resolve() / path


def expand_feature_registry(
    columns,
    *,
    registry_path: str | Path,
    strict: bool = True,
) -> dict[str, FeatureDefinition]:
    """Expand exact and patterned registry entries against concrete columns."""
    registry = load_yaml(registry_path)
    defaults = dict(registry.get("defaults", {}))
    exact = dict(registry.get("features", {}))
    patterns = list(dict(registry.get("patterns", {})).items())
    blocked = list(registry.get("blocked_patterns", []))
    expanded = {}
    missing = []
    for column in map(str, columns):
        # Fingerprint frames carry identifiers, labels, and build metadata next
        # to predictors.  Those columns are governed by the frame contract, not
        # the scientific feature registry.
        if (
            is_key_column(column)
            or is_label_column(column)
            or column.startswith("next_")
            or column.startswith("fp_")
        ):
            continue
        if any(fnmatch(column, pattern) for pattern in blocked):
            continue
        metadata = exact.get(column)
        if metadata is None:
            for pattern, pattern_metadata in patterns:
                if fnmatch(column, pattern):
                    metadata = pattern_metadata
                    break
        if metadata is None:
            if is_feature_column(column):
                missing.append(column)
            continue
        resolved = {**defaults, **dict(metadata)}
        required = {
            "family",
            "tier",
            "description",
            "source",
            "units",
            "direction",
            "availability_rule",
            "temporal_lag",
            "aggregation_window",
            "opponent_adjusted",
            "market_derived",
            "target_derived",
            "allowed_preseason",
            "allowed_weekly",
            "missing_policy",
            "code_path",
            "version",
        }
        absent = sorted(required - set(resolved))
        if absent:
            raise ValueError(f"Feature '{column}' registry entry is missing {absent}.")
        expanded[column] = FeatureDefinition(
            name=column,
            family=str(resolved["family"]),
            tier=str(resolved["tier"]),
            metadata=resolved,
        )
    if strict and missing:
        raise ValueError(
            f"Feature registry has no entry/pattern for {len(missing)} columns: {missing[:20]}"
        )
    return expanded


def selected_feature_families(feature_config: str, ladders: dict[str, Any]) -> set[str]:
    """Resolve a tier, union, or `F1+F2` expression to feature families."""
    tiers = dict(ladders.get("tiers", {}))
    visiting = set()

    def resolve(name):
        if name in visiting:
            raise ValueError(f"Cyclic feature ladder at '{name}'.")
        if name not in tiers:
            raise ValueError(f"Unknown feature tier '{name}'.")
        visiting.add(name)
        spec = dict(tiers[name])
        families = set(map(str, spec.get("include_families", [])))
        for child in spec.get("union", []):
            families.update(resolve(str(child)))
        visiting.remove(name)
        return families

    parts = [part.strip() for part in str(feature_config).split("+") if part.strip()]
    families = set()
    for part in parts:
        families.update(resolve(part))
    return families


def filter_frame_for_feature_config(
    frame: pd.DataFrame,
    *,
    feature_config: str,
    registry_path: str | Path,
    ladders_path: str | Path,
    strict_registry: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return frame with only the declared feature-tier columns plus contracts."""
    registry = expand_feature_registry(
        frame.columns, registry_path=registry_path, strict=strict_registry
    )
    ladders = load_yaml(ladders_path)
    families = selected_feature_families(feature_config, ladders)
    selected = [name for name, spec in registry.items() if spec.family in families]
    contract = [
        column
        for column in frame.columns
        if is_key_column(column)
        or is_label_column(column)
        or str(column) in {"game_is_home", "game_home_away"}
        or str(column).startswith("next_")
        or str(column).startswith("fp_")
    ]
    # Keep market columns as an evaluation sidecar for every tier. FeatureSpec
    # controls whether they enter X; only explicit F7/F8 runs opt in. Without
    # the sidecar, market-free models cannot be graded for chalk, upset, or ATS.
    market_sidecars = [column for column in frame.columns if is_market_column(column)]
    keep = list(dict.fromkeys(contract + selected + market_sidecars))
    if not selected:
        raise ValueError(f"Feature config '{feature_config}' selected no columns.")
    metadata = {
        "feature_config": feature_config,
        "families": sorted(families),
        "selected_features": selected,
        "selected_feature_count": len(selected),
    }
    return frame.loc[:, keep].copy(), metadata


def materialize_split_rows(split_path: str | Path) -> list[dict[str, Any]]:
    """Expand a split config into honest season-level outer folds."""
    cfg = load_yaml(split_path)
    strategy = str(cfg.get("strategy"))
    rows = []
    if strategy == "fixed_season_holdout":
        rows.append(
            {
                "outer_fold": 0,
                "train_seasons": list(map(int, cfg["train_seasons"])),
                "val_seasons": [],
                "test_seasons": list(map(int, cfg["holdout_seasons"])),
            }
        )
        return rows
    seasons = sorted(map(int, cfg.get("allowed_seasons", [])))
    if strategy == "season_rolling_origin":
        minimum = int(cfg.get("minimum_train_seasons", 4))
        val_count = int(cfg.get("validation_seasons", 1))
        for test_index in range(minimum + val_count, len(seasons)):
            validation = seasons[test_index - val_count : test_index]
            training = seasons[: test_index - val_count]
            rows.append(
                {
                    "outer_fold": len(rows),
                    "train_seasons": training,
                    "val_seasons": validation,
                    "test_seasons": [seasons[test_index]],
                }
            )
        return rows
    if strategy == "grouped_leave_one_season_out":
        for test in seasons:
            training = [season for season in seasons if season != test]
            rows.append(
                {
                    "outer_fold": len(rows),
                    "train_seasons": training,
                    "val_seasons": [],
                    "test_seasons": [test],
                }
            )
        return rows
    raise ValueError(f"Unsupported split strategy '{strategy}'.")


def build_experiment_manifest(
    *,
    project_root: str | Path,
    config_path: str | Path,
    data_path: str | Path,
    output_root: str | Path | None = None,
    max_trials: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build trial and chunk manifests without submitting any work."""
    root = Path(project_root).resolve()
    cfg = load_yaml(resolve_path(root, config_path))
    experiment_name = str(cfg["experiment_name"])
    artifact_root = Path(output_root or cfg["artifact_root"]).resolve()
    run_root = artifact_root / "experiments" / experiment_name
    run_root.mkdir(parents=True, exist_ok=True)
    rows = []
    search_cfg = dict(cfg.get("parameter_search", {}))
    trials_per_level = dict(search_cfg.get("trials_per_level", {}))
    spaces = dict(search_cfg.get("spaces", {}))
    search_seed = int(search_cfg.get("seed", 20260716))
    setpoints_by_level = {}
    for level in cfg["model_levels"]:
        trial_count = int(trials_per_level.get(level, search_cfg.get("trials", 1)))
        setpoints_by_level[level] = sample_setpoints(
            dict(spaces.get(level, {})),
            n=trial_count,
            seed=stable_seed(search_seed, level),
        )
    for split_ref in cfg.get("split_configs", []):
        split_path = resolve_path(root, split_ref)
        for fold in materialize_split_rows(split_path):
            for objective in cfg["objectives"]:
                for feature_config in cfg["feature_tiers"]:
                    for level in cfg["model_levels"]:
                        for parameter_index, parameters in enumerate(setpoints_by_level[level]):
                            for seed in cfg["seeds"]:
                                task_id = len(rows)
                                experiment_id = (
                                    f"{objective}__{feature_config.replace('+', '_')}__"
                                    f"{level}__{split_path.stem}__v1"
                                )
                                output_path = run_root / "runs" / f"task_{task_id:07d}"
                                rows.append(
                                {
                                    "task_id": task_id,
                                    "chunk_id": -1,
                                    "experiment_id": experiment_id,
                                    "objective": objective,
                                    "feature_config": feature_config,
                                    "model_level": level,
                                    "model_family": cfg["model_families"][level],
                                    "model_config": str(resolve_path(root, cfg["model_configs"][level])),
                                    "split_config": str(split_path),
                                    "outer_fold": int(fold["outer_fold"]),
                                    "train_seasons_json": json.dumps(fold["train_seasons"]),
                                    "val_seasons_json": json.dumps(fold["val_seasons"]),
                                    "test_seasons_json": json.dumps(fold["test_seasons"]),
                                    "seed": int(seed),
                                    "parameter_index": int(parameter_index),
                                    "params_json": json.dumps(parameters, sort_keys=True),
                                    "data_path": str(resolve_path(root, data_path)),
                                    "feature_registry": str(root / "configs/features/feature_registry.yaml"),
                                    "feature_ladders": str(root / "configs/features/feature_ladders.yaml"),
                                    "output_path": str(output_path),
                                    "estimated_memory_gb": 6 if level in {"M5", "M6"} else 3,
                                    "estimated_runtime": "04:00:00" if level in {"M5", "M6"} else "02:00:00",
                                    "retain_predictions": False,
                                    "retain_checkpoint": False,
                                    "temporal_feature_expansion": bool(
                                        cfg.get("temporal_feature_expansion", True)
                                    ),
                                }
                                )
                                if max_trials is not None and len(rows) >= int(max_trials):
                                    break
                            if max_trials is not None and len(rows) >= int(max_trials):
                                break
                        if max_trials is not None and len(rows) >= int(max_trials):
                            break
                    if max_trials is not None and len(rows) >= int(max_trials):
                        break
                if max_trials is not None and len(rows) >= int(max_trials):
                    break
            if max_trials is not None and len(rows) >= int(max_trials):
                break
        if max_trials is not None and len(rows) >= int(max_trials):
            break
    manifest = pd.DataFrame(rows)
    chunk_size = int(cfg.get("worker", {}).get("chunk_size", 4))
    manifest["chunk_id"] = manifest["task_id"] // chunk_size
    chunks = (
        manifest.groupby("chunk_id", as_index=False)
        .agg(task_start=("task_id", "min"), task_end=("task_id", "max"), trial_count=("task_id", "size"))
    )
    chunks["sge_task_id"] = chunks["chunk_id"] + 1
    atomic_write_frame(manifest, run_root / "job_manifest.parquet")
    atomic_write_frame(chunks, run_root / "chunk_manifest.parquet")
    manifest.to_csv(run_root / "job_manifest.csv", index=False)
    chunks.to_csv(run_root / "chunk_manifest.csv", index=False)
    resolved = dict(cfg)
    resolved.update(
        {
            "project_root": str(root),
            "data_path": str(resolve_path(root, data_path)),
            "output_root": str(run_root),
            "trial_count": len(manifest),
            "chunk_count": len(chunks),
            "created_at_utc": utc_now(),
        }
    )
    atomic_write_json(run_root / "resolved_config.json", resolved)
    return manifest, chunks


def run_experiment_trial(row: dict[str, Any], *, force=False, retry_incomplete=False) -> dict[str, Any]:
    """Run one temporal-fold trial and write one compact atomic fragment."""
    output = Path(row["output_path"])
    result_path = output / "result.parquet"
    status_path = output / "status.json"
    if not force and result_path.exists() and valid_result_fragment(result_path):
        if not retry_incomplete:
            return {"task_id": int(row["task_id"]), "status": "skipped_existing"}
        existing = pd.read_parquet(result_path)
        if str(existing.loc[0, "status"]) == "success":
            # A previously interrupted array can leave status.json at
            # ``running`` even though its atomic result fragment completed.
            # Resumable retries must reconcile that metadata before returning;
            # otherwise the reducer incorrectly treats valid results as live
            # work forever.
            atomic_write_json(status_path, existing.iloc[0].to_dict())
            return {"task_id": int(row["task_id"]), "status": "skipped_existing"}
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {
        key: row.get(key)
        for key in [
            "task_id",
            "chunk_id",
            "experiment_id",
            "objective",
            "feature_config",
            "model_level",
            "model_family",
            "model_config",
            "parameter_index",
            "params_json",
            "outer_fold",
            "seed",
            "source_feature_config",
            "target_feature_count",
            "actual_source_feature_count",
            "feature_subset_json",
        ]
        if key in row
    }
    result.update(
        {
            "status": "running",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_at_utc": utc_now(),
            "sge_job_id": os.environ.get("JOB_ID", ""),
            "sge_task_id": os.environ.get("SGE_TASK_ID", ""),
            "nslots": os.environ.get("NSLOTS", ""),
        }
    )
    atomic_write_json(status_path, result)
    try:
        model_cfg = build_tuned_model_config(
            base_config_path=Path(row["model_config"]),
            params=json.loads(row.get("params_json") or "{}"),
        )
        frame = pd.read_parquet(row["data_path"])
        if (
            str(row["model_family"]) == "temporal"
            and bool(row.get("temporal_feature_expansion", True))
        ):
            from gridiron_ml.fingerprints.temporal import build_temporal_fingerprints
            frame = build_temporal_fingerprints(frame, **model_cfg.get("temporal_fingerprint", {}))
        source_feature_config = str(row.get("source_feature_config") or row["feature_config"])
        frame, feature_metadata = filter_frame_for_feature_config(
            frame,
            feature_config=source_feature_config,
            registry_path=row["feature_registry"],
            ladders_path=row["feature_ladders"],
            strict_registry=True,
        )
        feature_subset_raw = row.get("feature_subset_json")
        if feature_subset_raw is not None and not pd.isna(feature_subset_raw):
            feature_subset = list(dict.fromkeys(json.loads(str(feature_subset_raw))))
            unavailable = sorted(set(feature_subset) - set(feature_metadata["selected_features"]))
            if unavailable:
                raise ValueError(
                    f"Compressed feature subset contains {len(unavailable)} features outside "
                    f"{source_feature_config}: {unavailable[:10]}"
                )
            source_columns = set(feature_metadata["selected_features"])
            passthrough = [column for column in frame.columns if column not in source_columns]
            frame = frame.loc[:, passthrough + feature_subset].copy()
            feature_metadata = {
                **feature_metadata,
                "selected_features": feature_subset,
                "selected_feature_count": len(feature_subset),
            }
        allow_market_features = "F7" in str(row["feature_config"]) or "F8" in str(row["feature_config"])
        model_cfg["seed"] = int(row["seed"])
        model_cfg["loss_function"] = (
            "WinnerAccuracy" if str(row["objective"]) == "winner" else "MAE"
        )
        if allow_market_features:
            model_cfg["allow_market_features_for_training"] = True
        model = build_model_from_config(
            {"family": row["model_family"], **model_cfg}
        )
        validate_model_contract(model)
        fingerprints = StaticFrameFingerprints(frame)
        evaluator = TDEval(
            config={
                "model": {
                    "family": row["model_family"],
                    "allow_market_features_for_training": allow_market_features,
                },
                "feature_spec": {
                    "include_market": allow_market_features,
                    "allow_market_features_for_training": allow_market_features,
                },
            },
            fingerprints=fingerprints,
            matchup_builder=MatchupBuilder(representation="unit_matchup"),
            model=model,
        )
        train_seasons = json.loads(row["train_seasons_json"])
        val_seasons = json.loads(row["val_seasons_json"])
        test_seasons = json.loads(row["test_seasons_json"])
        if 2026 in train_seasons or 2026 in val_seasons:
            raise ValueError("The prospective 2026 season cannot be used for fitting or selection.")
        evaluator.train(train_years=train_seasons, val_years=val_seasons)
        matchup_features = evaluator.matchup_builder.feature_names()
        graph_matchup_features = [
            name for name in matchup_features if "graph_" in str(name).lower()
        ]
        result["matchup_feature_count"] = int(len(matchup_features))
        result["graph_matchup_feature_count"] = int(len(graph_matchup_features))
        result["graph_matchup_features_json"] = json.dumps(graph_matchup_features)
        expected_matchup_counts = {"F5": 657, "F6": 681, "F7": 12, "F8": 693}
        expected_matchup_count = expected_matchup_counts.get(str(row["feature_config"]))
        if expected_matchup_count is not None and len(matchup_features) != expected_matchup_count:
            raise ValueError(
                f"{row['feature_config']} requires {expected_matchup_count} matchup coordinates; "
                f"resolved {len(matchup_features)}."
            )
        if str(row["feature_config"]) in {"F6", "F8"} and len(graph_matchup_features) != 24:
            raise ValueError(
                f"{row['feature_config']} requires 24 schedule-graph matchup coordinates; "
                f"resolved {len(graph_matchup_features)}."
            )
        predictions, metrics = evaluator.evaluate(years=test_seasons, label="outer_test")
        result.update(metrics.iloc[0].to_dict())
        prediction_column = next((c for c in ["pred_margin", "prediction", "y_pred"] if c in predictions), None)
        actual_column = next((c for c in ["y", "actual_margin", "y_true"] if c in predictions), None)
        if prediction_column and actual_column:
            predicted = pd.to_numeric(predictions[prediction_column], errors="coerce")
            actual = pd.to_numeric(predictions[actual_column], errors="coerce")
            valid = predicted.notna() & actual.notna()
            variance = float(predicted[valid].var(ddof=0)) if valid.any() else 0.0
            result["calibration_slope"] = (
                float(np.cov(predicted[valid], actual[valid], ddof=0)[0, 1] / variance)
                if valid.sum() >= 2 and variance > 0 else np.nan
            )
        result.update(
            {
                "status": "success",
                "selected_feature_count": feature_metadata["selected_feature_count"],
                "selected_features_json": json.dumps(feature_metadata["selected_features"]),
                "train_seasons_json": json.dumps(train_seasons),
                "val_seasons_json": json.dumps(val_seasons),
                "test_seasons_json": json.dumps(test_seasons),
                "model_metadata_json": json.dumps(model.get_metadata(), default=str, sort_keys=True),
                "prediction_rows": len(predictions),
            }
        )
        if bool(row.get("retain_predictions", False)):
            atomic_write_frame(predictions, output / "predictions.parquet")
        if bool(row.get("retain_checkpoint", False)):
            model.save(output / "checkpoint.pkl")
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    result["runtime_seconds"] = time.monotonic() - started
    result["completed_at_utc"] = utc_now()
    atomic_write_frame(pd.DataFrame([result]), result_path)
    atomic_write_json(status_path, result)
    return result


def run_experiment_chunk(
    *,
    job_manifest: str | Path,
    chunk_id: int | None = None,
    sge_task_id: int | None = None,
    workers: int = 4,
    force: bool = False,
    retry_incomplete: bool = False,
) -> list[dict[str, Any]]:
    """Run up to one manifest chunk with independent single-threaded trials."""
    manifest = read_frame(job_manifest)
    if sge_task_id is not None:
        chunk_id = int(sge_task_id) - 1
    if chunk_id is None:
        raise ValueError("Provide chunk_id or sge_task_id.")
    selected = manifest.loc[manifest["chunk_id"].astype(int) == int(chunk_id)]
    if selected.empty:
        raise IndexError(f"Chunk {chunk_id} does not exist.")
    rows = selected.to_dict(orient="records")
    if int(workers) <= 1 or len(rows) == 1:
        return [run_experiment_trial(row, force=force, retry_incomplete=retry_incomplete) for row in rows]
    results = []
    with ProcessPoolExecutor(max_workers=min(int(workers), len(rows))) as pool:
        future_map = {
            pool.submit(run_experiment_trial, row, force=force, retry_incomplete=retry_incomplete): row for row in rows
        }
        for future in as_completed(future_map):
            results.append(future.result())
    return sorted(results, key=lambda item: int(item["task_id"]))


def merge_experiment_chunks(
    *, job_manifest: str | Path, output_root: str | Path
) -> dict[str, pd.DataFrame]:
    """Merge compact fragments and write completed/failed/missing/duplicate status."""
    manifest = read_frame(job_manifest)
    fragments = []
    duplicates = []
    missing = []
    for row in manifest.to_dict(orient="records"):
        result_path = Path(row["output_path"]) / "result.parquet"
        if not result_path.exists():
            missing.append(row)
            continue
        fragment = pd.read_parquet(result_path)
        if len(fragment) != 1:
            duplicates.append({**row, "fragment_rows": len(fragment)})
            continue
        fragments.append(fragment)
    master = pd.concat(fragments, ignore_index=True) if fragments else pd.DataFrame()
    status_root = Path(output_root) / "summary" / "status"
    tables_root = Path(output_root) / "summary" / "tables"
    status_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    completed = master.loc[master.get("status", pd.Series(dtype=str)) == "success"].copy()
    failed = master.loc[master.get("status", pd.Series(dtype=str)) == "failed"].copy()
    status_tables = {
        "completed_trials": completed,
        "failed_trials": failed,
        "missing_trials": pd.DataFrame(missing),
        "duplicate_trials": pd.DataFrame(duplicates),
    }
    for name, table in status_tables.items():
        atomic_write_frame(table, status_root / f"{name}.parquet")
    atomic_write_frame(master, tables_root / "master_results.parquet")
    master.to_csv(tables_root / "master_results.csv.gz", index=False, compression="gzip")
    report = {
        "manifest_rows": len(manifest),
        "merged_rows": len(master),
        "completed_rows": len(completed),
        "failed_rows": len(failed),
        "missing_rows": len(missing),
        "duplicate_rows": len(duplicates),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(status_root / "merge_report.json", report)
    return {"master": master, **status_tables}


def select_finalists(
    results: pd.DataFrame,
    *,
    max_per_cell: int = 10,
    metric_by_objective: dict[str, str] | None = None,
    minimum_fold_count: int | None = None,
) -> pd.DataFrame:
    """Reduce successful trials by cell using mean performance and one-SE band."""
    metric_by_objective = metric_by_objective or {"winner": "brier_score", "margin": "mae"}
    success = results.loc[results["status"] == "success"].copy()
    selected = []
    cells = ["objective", "feature_config", "model_level"]
    for cell, group in success.groupby(cells, dropna=False):
        objective = str(cell[0])
        metric = metric_by_objective[objective]
        if metric not in group.columns:
            continue
        configs = ["model_family", "model_level", "model_config", "params_json", "seed"]
        summary = (
            group.groupby(configs, as_index=False)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        if minimum_fold_count is not None:
            summary = summary.loc[
                summary["count"].ge(int(minimum_fold_count))
            ].copy()
        if summary.empty:
            continue
        summary["se"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
        best = summary.sort_values("mean").iloc[0]
        threshold = float(best["mean"] + best["se"])
        eligible = summary.loc[summary["mean"] <= threshold].sort_values("mean").head(int(max_per_cell))
        representatives = []
        for _, config_row in eligible.iterrows():
            mask = pd.Series(True, index=group.index)
            for column in configs:
                mask &= group[column].astype(str).eq(str(config_row[column]))
            representative = group.loc[mask].sort_values(metric).iloc[0].copy()
            representative["cv_metric_mean"] = float(config_row["mean"])
            representative["cv_metric_se"] = float(config_row["se"])
            representative["cv_fold_count"] = int(config_row["count"])
            representatives.append(representative)
        selected.append(pd.DataFrame(representatives))
    if not selected:
        return pd.DataFrame(columns=list(results.columns) + ["selection_reason"])
    finalists = pd.concat(selected, ignore_index=True)
    finalists["selection_reason"] = "within_one_standard_error_then_best_trials"
    return finalists


def validate_experiment_output(*, job_manifest: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Validate manifest fields, merge completeness, and 2026 exclusion."""
    manifest = read_frame(job_manifest)
    missing_columns = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {missing_columns}")
    leaked = manifest["train_seasons_json"].astype(str).str.contains("2026") | manifest[
        "val_seasons_json"
    ].astype(str).str.contains("2026")
    if leaked.any():
        raise ValueError("Manifest includes prospective 2026 data in fit/selection seasons.")
    merged = merge_experiment_chunks(job_manifest=job_manifest, output_root=output_root)
    return {
        "manifest_rows": len(manifest),
        "completed_rows": len(merged["completed_trials"]),
        "failed_rows": len(merged["failed_trials"]),
        "missing_rows": len(merged["missing_trials"]),
        "duplicate_rows": len(merged["duplicate_trials"]),
        "valid": len(merged["missing_trials"]) == 0 and len(merged["duplicate_trials"]) == 0,
    }


def estimate_disk_bytes(manifest: pd.DataFrame, bytes_per_fragment=32_000) -> int:
    return int(len(manifest) * int(bytes_per_fragment))


def assert_disk_guardrail(path: str | Path, required_free_gb: float) -> None:
    candidate = Path(path).expanduser()
    # Manifest preparation commonly checks a fresh run directory.  Find the
    # nearest existing parent so the guardrail works before mkdir as well as
    # after it, without creating a partial run tree on failure.
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    free = shutil.disk_usage(candidate).free
    required = float(required_free_gb) * 1024**3
    if free < required:
        raise OSError(
            f"Disk guardrail failed at {path}: {free / 1024**3:.1f} GiB free; "
            f"{required_free_gb:.1f} GiB required."
        )


def valid_result_fragment(path: str | Path) -> bool:
    try:
        frame = pd.read_parquet(path)
        return len(frame) == 1 and frame.loc[0, "status"] in {"success", "failed"}
    except Exception:
        return False


def read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def atomic_write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
