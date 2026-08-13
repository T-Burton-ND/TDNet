"""Ablation and SHAP workflow for opponent-adjusted TDNet fingerprints.

The workflow is split into manifest, task, and merge steps so it can run as a
local smoke test or as an SGE array job on a cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import re
import sys
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridiron_ml.experiments.opponent_adjusted import (
    DEFAULT_ALL_YEARS,
    DEFAULT_TEST_YEARS,
    DEFAULT_TRAIN_YEARS,
    DEFAULT_VAL_YEARS,
    DEFAULT_VERSION_SPECS,
    OpponentAdjustedVersionSpec,
    StaticFrameFingerprints,
    available_weeks_for_poll,
    build_opponent_adjusted_experiment_frames,
    extract_vegas_metrics,
    first_row,
    safe_label,
)
from gridiron_ml.fingerprints.features import is_adjusted_feature_column, split_frame
from gridiron_ml.models import build_model_from_config
from gridiron_ml.pipeline.contracts.features import (
    DEFAULT_TRAINING_TARGET,
    FINGERPRINT_KEY_COLUMNS,
    HAS_NEXT_GAME_COLUMN,
    LABEL_COLUMNS,
    MARKET_CONTEXT_KEY_COLUMNS,
    NEXT_GAME_COLUMNS,
    is_feature_column,
)
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.td_run.season_vs_vegas import evaluate_models_vs_vegas_season
from gridiron_ml.td_run.shap_analysis import safe_name
from gridiron_ml.td_run.training import (
    DEFAULT_MODEL_SPECS,
    ModelRunSpec,
    checkpoint_path,
)


DEFAULT_ABLATION_EXPERIMENT_NAME = "opponent_adjusted_ablation_shap"
DEFAULT_SOURCE_EXPERIMENT_NAME = "opponent_adjusted_fingerprints"
DEFAULT_SHAP_MAX_BACKGROUND = 96
DEFAULT_SHAP_MAX_EXPLAIN = 256
DEFAULT_SCORING_WEIGHTS = {
    "winner_accuracy": 0.30,
    "chalk_accuracy": 0.05,
    "upset_recall": 0.30,
    "disagreement_accuracy": 0.15,
    "edge_3_plus_accuracy": 0.10,
    "record_upset_recall": 0.05,
    "margin_score": 0.05,
}
ADJUSTED_CONTEXT_MARKERS = (
    "games_played",
    "unique_opponents",
    "team_rating",
    "opponent_rating",
    "rating_edge",
)
ROLLING_SUFFIX_PATTERN = re.compile(r"_(mean_to_date|last3|ewm)$")


@dataclass(frozen=True)
class AblationSpec:
    """One feature-view ablation for a fingerprint frame."""

    name: str
    description: str
    include_raw: bool
    include_adjusted_residuals: bool
    include_adjusted_context: bool

    @property
    def safe_name(self) -> str:
        return safe_label(self.name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "include_raw": bool(self.include_raw),
            "include_adjusted_residuals": bool(self.include_adjusted_residuals),
            "include_adjusted_context": bool(self.include_adjusted_context),
        }


DEFAULT_ABLATION_SPECS = (
    AblationSpec(
        name="raw_baseline",
        description="Baseline v0 feature families only; all opponent-adjusted columns removed.",
        include_raw=True,
        include_adjusted_residuals=False,
        include_adjusted_context=False,
    ),
    AblationSpec(
        name="adjusted_residuals_only",
        description="Only opponent-adjusted residual/stat contribution features; raw v0 features and context ratings removed.",
        include_raw=False,
        include_adjusted_residuals=True,
        include_adjusted_context=False,
    ),
    AblationSpec(
        name="adjusted_all_only",
        description="All opponent-adjusted features, including residuals plus games-played/opponent/rating context; raw v0 features removed.",
        include_raw=False,
        include_adjusted_residuals=True,
        include_adjusted_context=True,
    ),
    AblationSpec(
        name="context_only",
        description="Only schedule-strength/sample-size context from the opponent-adjusted family; raw and residual features removed.",
        include_raw=False,
        include_adjusted_residuals=False,
        include_adjusted_context=True,
    ),
    AblationSpec(
        name="raw_plus_context",
        description="Raw v0 features plus adjusted sample-size/rating context; residual/stat adjusted features removed.",
        include_raw=True,
        include_adjusted_residuals=False,
        include_adjusted_context=True,
    ),
    AblationSpec(
        name="raw_plus_adjusted_all",
        description="Raw v0 features plus all opponent-adjusted residual and context features.",
        include_raw=True,
        include_adjusted_residuals=True,
        include_adjusted_context=True,
    ),
)


def default_output_root(project_root: str | Path) -> Path:
    """Return the default ablation output root."""

    return (
        Path(project_root).resolve()
        / "data"
        / "experiments"
        / DEFAULT_ABLATION_EXPERIMENT_NAME
    )


def default_source_fingerprint_root(project_root: str | Path) -> Path:
    """Return the default source opponent-adjusted fingerprint output root."""

    return (
        Path(project_root).resolve()
        / "data"
        / "experiments"
        / DEFAULT_SOURCE_EXPERIMENT_NAME
    )


def build_ablation_eval_config(
    *,
    max_background: int = DEFAULT_SHAP_MAX_BACKGROUND,
    max_explain: int = DEFAULT_SHAP_MAX_EXPLAIN,
    shap_plots: bool = False,
) -> dict[str, Any]:
    """Evaluation config for ablation jobs, weighted toward winners/upsets."""

    return {
        "artifacts": {
            "core_tables": True,
            "game_predictions": True,
            "prediction_sanity": True,
            "weekly_tables": True,
            "bucket_tables": True,
            "calibration_tables": True,
            "ats_tables": True,
            "shap": True,
            "shap_summary_plots": bool(shap_plots),
            "shap_bar_plots": bool(shap_plots),
            "png_plots": False,
        },
        "probability": {"margin_temperature": 7.0},
        "scoring_weights": dict(DEFAULT_SCORING_WEIGHTS),
        "shap": {
            "enabled": True,
            "max_background": int(max_background),
            "max_explain": int(max_explain),
            "max_display": 35,
            "random_seed": 42,
            "summary_plots": bool(shap_plots),
            "bar_plots": bool(shap_plots),
        },
        "plotting": {"dpi": 150},
    }


def apply_ablation_view(frame: pd.DataFrame, spec: AblationSpec) -> pd.DataFrame:
    """Return ``frame`` with training feature families filtered by ``spec``."""

    keep: list[str] = []
    for col in frame.columns:
        name = str(col)
        if name.startswith("fp_"):
            continue
        if should_always_keep_column(name):
            keep.append(name)
            continue
        if not is_trainable_candidate(frame, name):
            if not is_feature_column(name):
                keep.append(name)
            continue
        role = feature_role(name)
        if role == "raw" and spec.include_raw:
            keep.append(name)
        elif role == "adjusted_residual" and spec.include_adjusted_residuals:
            keep.append(name)
        elif role == "adjusted_context" and spec.include_adjusted_context:
            keep.append(name)

    keep = list(dict.fromkeys([col for col in keep if col in frame.columns]))
    return frame.loc[:, keep].copy()


def should_always_keep_column(col: str) -> bool:
    """Return whether a column is needed for keys, labels, markets, or scheduling."""

    exact = set(FINGERPRINT_KEY_COLUMNS) | set(LABEL_COLUMNS) | set(NEXT_GAME_COLUMNS)
    exact |= set(MARKET_CONTEXT_KEY_COLUMNS)
    return (
        col in exact
        or col.startswith("keys_")
        or col.startswith("market_")
        or col.startswith("game_")
    )


def is_trainable_candidate(frame: pd.DataFrame, col: str) -> bool:
    """Return whether a column can become a numeric training feature."""

    if col not in frame.columns or not is_feature_column(col):
        return False
    series = frame[col]
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def feature_role(feature: object) -> str:
    """Classify a model feature for ablation and SHAP aggregation."""

    stripped = strip_matchup_side(str(feature))
    if is_adjusted_feature_column(stripped):
        if is_adjusted_context_column(stripped):
            return "adjusted_context"
        return "adjusted_residual"
    if is_adjusted_context_column(stripped):
        return "adjusted_context"
    return "raw"


def is_adjusted_context_column(feature: object) -> bool:
    """Return whether an adjusted feature is schedule/sample/rating context."""

    name = str(feature).lower()
    return any(marker in name for marker in ADJUSTED_CONTEXT_MARKERS)


def strip_matchup_side(feature: str) -> str:
    """Remove a matchup-side prefix from a feature name."""

    return re.sub(r"^(home|away|net)_", "", str(feature), count=1)


def matchup_side(feature: object) -> str:
    """Return home/away/net/raw side tag for a feature."""

    text = str(feature)
    for side in ("home", "away", "net"):
        if text.startswith(f"{side}_"):
            return side
    return "team"


def normalized_feature(feature: object) -> str:
    """Normalize version-specific matchup features for cross-run aggregation."""

    stripped = strip_matchup_side(str(feature))
    stripped = re.sub(r"^opp_adj_v\d+_\d+_", "opp_adj_", stripped)
    return stripped


def normalized_feature_family(feature: object) -> str:
    """Return a stable family label for a model feature."""

    name = normalized_feature(feature)
    if name.startswith("opp_adj_"):
        rest = name[len("opp_adj_") :]
        rest = ROLLING_SUFFIX_PATTERN.sub("", rest)
        for marker in ADJUSTED_CONTEXT_MARKERS:
            if marker in rest:
                return f"opp_adj_{marker}"
        return rest
    for prefix in ("statOff_", "statDef_", "statGen_", "statSpe_"):
        if name.startswith(prefix):
            parts = name.split("_")
            return "_".join(parts[:2]) if len(parts) >= 2 else name
    for prefix in ("offense_", "defense_", "target_"):
        if name.startswith(prefix):
            parts = name.split("_")
            return "_".join(parts[:2]) if len(parts) >= 2 else name
    return name.split("_", 1)[0] if "_" in name else name


def summarize_ablation_features(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize feature counts for an ablated fingerprint frame."""

    x_df, _, _, _ = split_frame(frame)
    roles = pd.Series([feature_role(col) for col in x_df.columns], dtype="object")
    counts = roles.value_counts().to_dict()
    return {
        "feature_count": int(x_df.shape[1]),
        "raw_feature_count": int(counts.get("raw", 0)),
        "adjusted_residual_feature_count": int(counts.get("adjusted_residual", 0)),
        "adjusted_context_feature_count": int(counts.get("adjusted_context", 0)),
    }


def build_ablation_job_manifest(
    *,
    project_root: str | Path,
    output_root: str | Path | None = None,
    source_fingerprint_root: str | Path | None = None,
    version_specs: tuple[OpponentAdjustedVersionSpec, ...] = DEFAULT_VERSION_SPECS,
    ablation_specs: tuple[AblationSpec, ...] = DEFAULT_ABLATION_SPECS,
    model_specs: tuple[ModelRunSpec, ...] | list[ModelRunSpec] = DEFAULT_MODEL_SPECS,
    train_years: tuple[int, ...] | list[int] = DEFAULT_TRAIN_YEARS,
    val_years: tuple[int, ...] | list[int] = DEFAULT_VAL_YEARS,
    test_years: tuple[int, ...] | list[int] = DEFAULT_TEST_YEARS,
    ensure_fingerprints: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build and save the cluster/job manifest for the ablation sweep."""

    root = Path(project_root).resolve()
    output_root = Path(output_root or default_output_root(root)).resolve()
    source_fingerprint_root = Path(
        source_fingerprint_root or default_source_fingerprint_root(root)
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if ensure_fingerprints:
        ensure_source_fingerprints(
            project_root=root,
            source_fingerprint_root=source_fingerprint_root,
            version_specs=tuple(version_specs),
            train_years=train_years,
            val_years=val_years,
            test_years=test_years,
            overwrite=overwrite,
        )

    rows = []
    job_index = 0
    model_specs = tuple(ModelRunSpec.from_mapping(spec) for spec in model_specs)
    for version in version_specs:
        fingerprint_path = source_fingerprint_path(source_fingerprint_root, version)
        for ablation in ablation_specs:
            for model in model_specs:
                combo_root = (
                    output_root
                    / "runs"
                    / version.safe_label
                    / ablation.safe_name
                    / model.family
                    / model.name
                )
                rows.append(
                    {
                        "job_index": job_index,
                        "sge_task_id": job_index + 1,
                        "fingerprint": version.label,
                        "fingerprint_safe": version.safe_label,
                        "fingerprint_method": version.method,
                        "ablation": ablation.name,
                        "ablation_safe": ablation.safe_name,
                        "model": model.name,
                        "family": model.family,
                        "model_config_path": model.config_path,
                        "fingerprint_path": str(fingerprint_path),
                        "output_dir": str(combo_root),
                        "metrics_path": str(combo_root / "metrics.csv"),
                        "shap_fragment_path": str(
                            combo_root / "shap_importance_master_fragment.csv"
                        ),
                        "status_path": str(combo_root / "status.json"),
                    }
                )
                job_index += 1

    manifest = pd.DataFrame(rows)
    manifest_path = output_root / "job_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    write_ablation_manifest_metadata(
        output_root=output_root,
        source_fingerprint_root=source_fingerprint_root,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        version_specs=tuple(version_specs),
        ablation_specs=tuple(ablation_specs),
        model_specs=model_specs,
        job_count=len(manifest),
    )
    write_methods_notes(
        output_root=output_root,
        source_fingerprint_root=source_fingerprint_root,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        version_specs=tuple(version_specs),
        ablation_specs=tuple(ablation_specs),
        model_specs=model_specs,
    )
    return manifest


def ensure_source_fingerprints(
    *,
    project_root: str | Path,
    source_fingerprint_root: str | Path,
    version_specs: tuple[OpponentAdjustedVersionSpec, ...],
    train_years,
    val_years,
    test_years,
    overwrite: bool = False,
) -> None:
    """Ensure source adjusted fingerprint parquet files exist."""

    root = Path(project_root).resolve()
    source_root = Path(source_fingerprint_root).resolve()
    missing = [
        spec
        for spec in version_specs
        if overwrite or not source_fingerprint_path(source_root, spec).exists()
    ]
    if not missing:
        return
    years = tuple(sorted({int(y) for y in [*train_years, *val_years, *test_years]}))
    if not years:
        years = DEFAULT_ALL_YEARS
    build_opponent_adjusted_experiment_frames(
        project_root=root,
        output_root=source_root,
        seasons=years,
        version_specs=version_specs,
        overwrite=overwrite,
    )


def source_fingerprint_path(
    source_fingerprint_root: str | Path,
    spec: OpponentAdjustedVersionSpec,
) -> Path:
    """Return the canonical parquet path for one adjusted fingerprint."""

    return (
        Path(source_fingerprint_root).resolve()
        / "fingerprints"
        / spec.safe_label
        / "canonical_fingerprint.parquet"
    )


def load_manifest(output_root: str | Path, job_manifest: str | Path | None = None) -> pd.DataFrame:
    """Load an ablation job manifest."""

    path = Path(job_manifest) if job_manifest is not None else Path(output_root) / "job_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Ablation job manifest not found: {path}")
    return pd.read_csv(path)


def run_manifest_job(
    *,
    project_root: str | Path,
    output_root: str | Path,
    job_index: int | None = None,
    sge_task_id: int | None = None,
    job_manifest: str | Path | None = None,
    train_years: tuple[int, ...] | list[int] = DEFAULT_TRAIN_YEARS,
    val_years: tuple[int, ...] | list[int] = DEFAULT_VAL_YEARS,
    test_years: tuple[int, ...] | list[int] = DEFAULT_TEST_YEARS,
    max_background: int = DEFAULT_SHAP_MAX_BACKGROUND,
    max_explain: int = DEFAULT_SHAP_MAX_EXPLAIN,
    shap_plots: bool = False,
    keep_checkpoints: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run one manifest row by zero-based index or one-based SGE task id."""

    manifest = load_manifest(output_root=output_root, job_manifest=job_manifest)
    if sge_task_id is not None:
        selector = manifest.loc[
            pd.to_numeric(manifest["sge_task_id"], errors="coerce") == int(sge_task_id)
        ]
    elif job_index is not None:
        selector = manifest.loc[
            pd.to_numeric(manifest["job_index"], errors="coerce") == int(job_index)
        ]
    else:
        raise ValueError("Provide either job_index or sge_task_id.")
    if selector.empty:
        raise IndexError("Requested ablation manifest row does not exist.")
    row = selector.iloc[0].to_dict()
    version = version_spec_from_label(str(row["fingerprint"]))
    ablation = ablation_spec_from_name(str(row["ablation"]))
    model = ModelRunSpec(
        name=str(row["model"]),
        family=str(row["family"]),
        config_path=str(row["model_config_path"]),
    )
    return run_ablation_combo(
        project_root=project_root,
        output_dir=Path(row["output_dir"]),
        fingerprint_path=Path(row["fingerprint_path"]),
        version_spec=version,
        ablation_spec=ablation,
        model_spec=model,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        max_background=max_background,
        max_explain=max_explain,
        shap_plots=shap_plots,
        keep_checkpoints=keep_checkpoints,
        force=force,
        job_metadata=row,
    )


def run_ablation_combo(
    *,
    project_root: str | Path,
    output_dir: str | Path,
    fingerprint_path: str | Path,
    version_spec: OpponentAdjustedVersionSpec,
    ablation_spec: AblationSpec,
    model_spec: ModelRunSpec,
    train_years: tuple[int, ...] | list[int] = DEFAULT_TRAIN_YEARS,
    val_years: tuple[int, ...] | list[int] = DEFAULT_VAL_YEARS,
    test_years: tuple[int, ...] | list[int] = DEFAULT_TEST_YEARS,
    max_background: int = DEFAULT_SHAP_MAX_BACKGROUND,
    max_explain: int = DEFAULT_SHAP_MAX_EXPLAIN,
    shap_plots: bool = False,
    keep_checkpoints: bool = False,
    force: bool = False,
    job_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and explain one fingerprint/ablation/model combination."""

    root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    metrics_path = output_dir / "metrics.csv"
    status_path = output_dir / "status.json"
    if metrics_path.exists() and not force:
        return {
            "status": "skipped_existing",
            "output_dir": str(output_dir),
            "metrics_path": str(metrics_path),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_status(
        status_path,
        {
            "status": "running",
            "started_at": utc_now(),
            "fingerprint": version_spec.label,
            "ablation": ablation_spec.name,
            "model": model_spec.name,
            "family": model_spec.family,
        },
    )

    row: dict[str, Any] = base_result_row(
        version_spec=version_spec,
        ablation_spec=ablation_spec,
        model_spec=model_spec,
        output_dir=output_dir,
        job_metadata=job_metadata,
    )
    try:
        frame = pd.read_parquet(fingerprint_path)
        ablated_frame = apply_ablation_view(frame, ablation_spec)
        row.update(summarize_ablation_features(ablated_frame))
        fingerprints = StaticFrameFingerprints(ablated_frame)
        matchup_builder = MatchupBuilder(representation="unit_matchup")
        model = build_model_from_config(
            {
                "family": model_spec.family,
                "config_path": str(root / model_spec.config_path),
            }
        )
        evaluator = TDEval(
            config={
                "model": {
                    "family": model_spec.family,
                    "config_path": str(root / model_spec.config_path),
                },
                "eval": {
                    "train_years": list(train_years),
                    "test_years": list(test_years),
                    "artifact_root": str(output_dir / "train_artifacts"),
                },
            },
            fingerprints=fingerprints,
            matchup_builder=matchup_builder,
            model=model,
        )
        model = evaluator.train(train_years=train_years, val_years=val_years)
        if keep_checkpoints:
            checkpoint = model.save(
                checkpoint_path(model_spec, models_root=output_dir / "checkpoints")
            )
            row["checkpoint_path"] = str(checkpoint)
        _, metrics_df = evaluator.evaluate(years=test_years, label="test")
        train_artifact_root = evaluator.save_outputs(output_dir / "train_artifacts")
        row.update(first_row(metrics_df))
        row["train_artifact_root"] = str(train_artifact_root)

        eval_config = build_ablation_eval_config(
            max_background=max_background,
            max_explain=max_explain,
            shap_plots=shap_plots,
        )
        vegas_tables = evaluate_models_vs_vegas_season(
            fingerprints=fingerprints,
            matchup_builder=matchup_builder,
            season=int(test_years[-1]),
            model_specs=[{"name": model_spec.name, "model": model}],
            output_dir=output_dir / "season_eval",
            make_plots=False,
            eval_config=eval_config,
        )
        row.update(extract_vegas_metrics(vegas_tables, model_spec.name))
        row["poll_available_weeks"] = len(
            available_weeks_for_poll(ablated_frame, season=int(test_years[-1]))
        )
        row.update(extract_shap_status(vegas_tables, model_spec.name))
        shap_fragment = build_shap_master_fragment(
            output_dir=output_dir,
            version_spec=version_spec,
            ablation_spec=ablation_spec,
            model_spec=model_spec,
        )
        row["shap_importance_rows"] = int(len(shap_fragment))
        row["status"] = "success"
        row["error"] = ""
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        row["traceback"] = traceback.format_exc()
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    write_status(
        status_path,
        {
            **row,
            "completed_at": utc_now(),
        },
    )
    return row


def base_result_row(
    *,
    version_spec: OpponentAdjustedVersionSpec,
    ablation_spec: AblationSpec,
    model_spec: ModelRunSpec,
    output_dir: Path,
    job_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Common metadata columns for one ablation result row."""

    row = {
        "experiment": DEFAULT_ABLATION_EXPERIMENT_NAME,
        "fingerprint": version_spec.label,
        "fingerprint_safe": version_spec.safe_label,
        "fingerprint_method": version_spec.method,
        "fingerprint_description": version_spec.description,
        "ablation": ablation_spec.name,
        "ablation_safe": ablation_spec.safe_name,
        "ablation_description": ablation_spec.description,
        "model": model_spec.name,
        "family": model_spec.family,
        "model_config_path": model_spec.config_path,
        "output_dir": str(output_dir),
        "checkpoint_path": "",
    }
    if job_metadata:
        for key in ("job_index", "sge_task_id"):
            if key in job_metadata:
                row[key] = job_metadata[key]
    return row


def extract_shap_status(
    tables: dict[str, pd.DataFrame],
    model_name: str,
) -> dict[str, Any]:
    """Extract status fields from the SHAP artifact table."""

    table = tables.get("shap_artifacts", pd.DataFrame())
    if table.empty:
        return {
            "shap_status": "missing",
            "shap_method": "",
            "shap_importance_table": "",
        }
    if "model" in table.columns:
        rows = table.loc[table["model"].astype(str) == str(model_name)]
    else:
        rows = pd.DataFrame()
    if rows.empty:
        rows = table
    data = rows.iloc[0].to_dict()
    return {
        "shap_status": data.get("status", ""),
        "shap_method": data.get("method", ""),
        "shap_importance_table": data.get("importance_table", ""),
        "shap_n_explained": data.get("n_explained", np.nan),
        "shap_n_features": data.get("n_features", np.nan),
    }


def build_shap_master_fragment(
    *,
    output_dir: str | Path,
    version_spec: OpponentAdjustedVersionSpec,
    ablation_spec: AblationSpec,
    model_spec: ModelRunSpec,
) -> pd.DataFrame:
    """Load one per-model SHAP table and add cross-run analysis columns."""

    output_dir = Path(output_dir)
    importance_path = (
        output_dir
        / "season_eval"
        / "tables"
        / "shap"
        / f"{safe_name(model_spec.name)}_shap_importance.csv"
    )
    fragment_path = output_dir / "shap_importance_master_fragment.csv"
    if not importance_path.exists():
        pd.DataFrame().to_csv(fragment_path, index=False)
        return pd.DataFrame()
    frame = pd.read_csv(importance_path)
    if frame.empty or "feature" not in frame.columns:
        frame.to_csv(fragment_path, index=False)
        return frame
    frame = frame.copy()
    if "shap_rank" not in frame.columns:
        frame = frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        frame.insert(0, "shap_rank", np.arange(1, len(frame) + 1))
    if "shap_importance_share" not in frame.columns and "mean_abs_shap" in frame.columns:
        total = pd.to_numeric(frame["mean_abs_shap"], errors="coerce").sum()
        frame["shap_importance_share"] = (
            pd.to_numeric(frame["mean_abs_shap"], errors="coerce") / total
            if total > 0
            else 0.0
        )
    frame.insert(0, "fingerprint", version_spec.label)
    frame.insert(1, "fingerprint_method", version_spec.method)
    frame.insert(2, "ablation", ablation_spec.name)
    frame.insert(3, "family", model_spec.family)
    frame.insert(4, "model", model_spec.name)
    frame["feature_role"] = frame["feature"].map(feature_role)
    frame["feature_side"] = frame["feature"].map(matchup_side)
    frame["feature_normalized"] = frame["feature"].map(normalized_feature)
    frame["feature_family"] = frame["feature"].map(normalized_feature_family)
    frame["importance_table"] = str(importance_path)
    frame.to_csv(fragment_path, index=False)
    return frame


def merge_ablation_outputs(
    *,
    output_root: str | Path,
) -> dict[str, pd.DataFrame]:
    """Merge completed ablation job outputs into master tables and figures."""

    output_root = Path(output_root).resolve()
    summary_tables = output_root / "summary" / "tables"
    summary_figures = output_root / "summary" / "figures"
    summary_tables.mkdir(parents=True, exist_ok=True)
    summary_figures.mkdir(parents=True, exist_ok=True)

    metrics = read_many_csv(output_root.glob("runs/*/*/*/*/metrics.csv"))
    shap = read_many_csv(output_root.glob("runs/*/*/*/*/shap_importance_master_fragment.csv"))
    failures = (
        metrics.loc[metrics.get("status", pd.Series(dtype=str)).astype(str).eq("failed")].copy()
        if not metrics.empty and "status" in metrics.columns
        else pd.DataFrame()
    )
    metrics.to_csv(summary_tables / "master_model_ablation_metrics.csv", index=False)
    shap.to_csv(summary_tables / "master_shap_importance.csv", index=False)
    failures.to_csv(summary_tables / "failures.csv", index=False)

    save_metric_summary_tables(metrics, summary_tables)
    save_shap_summary_tables(shap, summary_tables)
    save_ablation_summary_figures(metrics, shap, summary_figures)
    write_merge_status(output_root, metrics, shap)
    return {"metrics": metrics, "shap": shap, "failures": failures}


def read_many_csv(paths) -> pd.DataFrame:
    """Read a collection of CSV files, ignoring empty placeholders."""

    frames = []
    for path in sorted(Path(p) for p in paths):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def metric_columns(frame: pd.DataFrame) -> list[str]:
    """Return key numeric columns to summarize."""

    candidates = [
        "mae",
        "rmse",
        "winner_accuracy",
        "favorite_correct",
        "upset_correct",
        "season_margin_mae",
        "season_margin_rmse",
        "season_winner_winner_accuracy",
        "season_winner_chalk_accuracy",
        "season_winner_upset_recall",
        "season_winner_upset_precision",
        "season_winner_disagreement_accuracy",
        "season_winner_edge_3_plus_accuracy",
        "season_winner_record_upset_recall",
        "score_total_score",
        "score_margin_score",
        "feature_count",
        "raw_feature_count",
        "adjusted_residual_feature_count",
        "adjusted_context_feature_count",
        "shap_importance_rows",
    ]
    return [col for col in candidates if col in frame.columns]


def successful_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return successful run rows only."""

    if metrics.empty:
        return pd.DataFrame()
    if "status" not in metrics.columns:
        return metrics.copy()
    return metrics.loc[metrics["status"].astype(str).eq("success")].copy()


def save_metric_summary_tables(metrics: pd.DataFrame, tables_dir: Path) -> None:
    """Save paper-oriented metric aggregations."""

    success = successful_metrics(metrics)
    if success.empty:
        return
    cols = metric_columns(success)
    if not cols:
        return
    groups = {
        "summary_by_ablation.csv": ["ablation"],
        "summary_by_fingerprint.csv": ["fingerprint", "fingerprint_method"],
        "summary_by_fingerprint_ablation.csv": [
            "fingerprint",
            "fingerprint_method",
            "ablation",
        ],
        "summary_by_model.csv": ["family", "model"],
        "summary_by_model_ablation.csv": ["family", "model", "ablation"],
        "summary_by_family_ablation.csv": ["family", "ablation"],
    }
    for filename, group_cols in groups.items():
        present = [col for col in group_cols if col in success.columns]
        if len(present) != len(group_cols):
            continue
        (
            success.groupby(group_cols, as_index=False)[cols]
            .mean(numeric_only=True)
            .to_csv(tables_dir / filename, index=False)
        )

    for metric in [
        "score_total_score",
        "season_winner_winner_accuracy",
        "season_winner_upset_recall",
        "season_winner_disagreement_accuracy",
        "season_margin_mae",
        "season_margin_rmse",
    ]:
        pivot_fingerprint_ablation(success, metric).to_csv(
            tables_dir / f"{metric}_fingerprint_by_ablation.csv"
        )
        pivot_model_ablation(success, metric).to_csv(
            tables_dir / f"{metric}_model_by_ablation.csv"
        )


def save_shap_summary_tables(shap: pd.DataFrame, tables_dir: Path) -> None:
    """Save master SHAP aggregation tables."""

    if shap.empty or "mean_abs_shap" not in shap.columns:
        return
    shap = shap.copy()
    shap["mean_abs_shap"] = pd.to_numeric(shap["mean_abs_shap"], errors="coerce")
    if "shap_importance_share" in shap.columns:
        shap["shap_importance_share"] = pd.to_numeric(
            shap["shap_importance_share"], errors="coerce"
        )
    else:
        shap["shap_importance_share"] = np.nan
    group_specs = {
        "shap_by_feature.csv": ["feature_normalized"],
        "shap_by_feature_family.csv": ["feature_family"],
        "shap_by_feature_role.csv": ["feature_role"],
        "shap_by_ablation_feature_role.csv": ["ablation", "feature_role"],
        "shap_by_model_feature_family.csv": ["family", "model", "feature_family"],
        "shap_by_fingerprint_feature_family.csv": [
            "fingerprint",
            "fingerprint_method",
            "feature_family",
        ],
        "shap_by_fingerprint_ablation_feature_role.csv": [
            "fingerprint",
            "ablation",
            "feature_role",
        ],
    }
    for filename, group_cols in group_specs.items():
        present = [col for col in group_cols if col in shap.columns]
        if len(present) != len(group_cols):
            continue
        out = (
            shap.groupby(group_cols, as_index=False)
            .agg(
                mean_abs_shap=("mean_abs_shap", "mean"),
                median_abs_shap=("mean_abs_shap", "median"),
                mean_importance_share=("shap_importance_share", "mean"),
                combo_count=("mean_abs_shap", "count"),
            )
            .sort_values("mean_abs_shap", ascending=False)
        )
        out.to_csv(tables_dir / filename, index=False)


def pivot_fingerprint_ablation(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot a metric as fingerprint rows by ablation columns."""

    if metric not in frame.columns:
        return pd.DataFrame()
    return frame.pivot_table(
        index=["fingerprint", "fingerprint_method"],
        columns="ablation",
        values=metric,
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)


def pivot_model_ablation(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot a metric as model rows by ablation columns."""

    if metric not in frame.columns:
        return pd.DataFrame()
    return frame.pivot_table(
        index=["family", "model"],
        columns="ablation",
        values=metric,
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)


def save_ablation_summary_figures(
    metrics: pd.DataFrame,
    shap: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Save compact summary figures for the ablation sweep."""

    success = successful_metrics(metrics)
    if not success.empty:
        specs = [
            ("score_total_score", "Composite Score by Fingerprint and Ablation", "score_fingerprint_by_ablation.png"),
            (
                "season_winner_winner_accuracy",
                "Winner Accuracy by Fingerprint and Ablation",
                "winner_accuracy_fingerprint_by_ablation.png",
            ),
            (
                "season_winner_upset_recall",
                "Upset Recall by Fingerprint and Ablation",
                "upset_recall_fingerprint_by_ablation.png",
            ),
            ("season_margin_mae", "MAE by Fingerprint and Ablation", "mae_fingerprint_by_ablation.png"),
        ]
        for metric, title, filename in specs:
            table = pivot_fingerprint_ablation(success, metric)
            if not table.empty:
                plot_heatmap(table, figures_dir / filename, title=title)

    if shap.empty or "feature_family" not in shap.columns:
        return
    shap = shap.copy()
    shap["mean_abs_shap"] = pd.to_numeric(shap["mean_abs_shap"], errors="coerce")
    grouped = (
        shap.groupby("feature_family", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"))
        .dropna(subset=["mean_abs_shap"])
        .sort_values("mean_abs_shap", ascending=False)
        .head(25)
    )
    if grouped.empty:
        return
    plot_df = grouped.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.32 * len(plot_df) + 1.5)))
    ax.barh(plot_df["feature_family"].astype(str), plot_df["mean_abs_shap"], color="#2962A3")
    ax.set_title("Top Feature Families by Mean Absolute SHAP")
    ax.set_xlabel("Mean Absolute SHAP")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "top_shap_feature_families.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(table: pd.DataFrame, path: Path, *, title: str) -> None:
    """Save a metric heatmap."""

    if table.empty:
        return
    values = table.to_numpy(dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return
    labels = [
        " / ".join(str(part) for part in idx)
        if isinstance(idx, tuple)
        else str(idx)
        for idx in table.index
    ]
    fig_width = max(8.0, 1.15 * table.shape[1] + 4.0)
    fig_height = max(4.8, 0.45 * table.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_ablation_manifest_metadata(
    *,
    output_root: Path,
    source_fingerprint_root: Path,
    train_years,
    val_years,
    test_years,
    version_specs: tuple[OpponentAdjustedVersionSpec, ...],
    ablation_specs: tuple[AblationSpec, ...],
    model_specs: tuple[ModelRunSpec, ...],
    job_count: int,
) -> None:
    """Write JSON metadata for repeatable ablation runs."""

    payload = {
        "experiment": DEFAULT_ABLATION_EXPERIMENT_NAME,
        "source_experiment": DEFAULT_SOURCE_EXPERIMENT_NAME,
        "source_fingerprint_root": str(source_fingerprint_root),
        "built_at": utc_now(),
        "train_years": [int(year) for year in train_years],
        "val_years": [int(year) for year in val_years],
        "test_years": [int(year) for year in test_years],
        "job_count": int(job_count),
        "scoring_weights": dict(DEFAULT_SCORING_WEIGHTS),
        "shap_defaults": {
            "max_background": DEFAULT_SHAP_MAX_BACKGROUND,
            "max_explain": DEFAULT_SHAP_MAX_EXPLAIN,
            "summary_plots": False,
            "bar_plots": False,
        },
        "versions": [
            {"label": spec.label, "method": spec.method, "description": spec.description}
            for spec in version_specs
        ],
        "ablations": [spec.as_dict() for spec in ablation_specs],
        "models": [
            {"name": spec.name, "family": spec.family, "config_path": spec.config_path}
            for spec in model_specs
        ],
        "environment": environment_summary(),
    }
    write_json(output_root / "ablation_manifest.json", payload)


def write_methods_notes(
    *,
    output_root: Path,
    source_fingerprint_root: Path,
    train_years,
    val_years,
    test_years,
    version_specs: tuple[OpponentAdjustedVersionSpec, ...],
    ablation_specs: tuple[AblationSpec, ...],
    model_specs: tuple[ModelRunSpec, ...],
) -> None:
    """Write human-readable methods/source notes for the experiment."""

    lines = [
        "# Opponent-Adjusted Fingerprint Ablation + SHAP",
        "",
        "This directory contains the repeatable ablation/SHAP sweep for the opponent-adjusted fingerprint experiment.",
        "",
        "## Data Split",
        "",
        f"- Train: {min(train_years)}-{max(train_years)}",
        f"- Validation: {', '.join(str(y) for y in val_years)}",
        f"- Test/reporting season: {', '.join(str(y) for y in test_years)}",
        f"- Source adjusted fingerprints: `{source_fingerprint_root}`",
        f"- Default target: `{DEFAULT_TRAINING_TARGET}`",
        "",
        "## Priority Metrics",
        "",
        "The composite score intentionally weights winner and upset behavior above raw margin error.",
        "",
        "| Metric | Weight |",
        "| --- | ---: |",
    ]
    for metric, weight in DEFAULT_SCORING_WEIGHTS.items():
        lines.append(f"| `{metric}` | {weight:.2f} |")
    lines.extend(
        [
            "",
            "## Fingerprints",
            "",
            "| Version | Method | Description |",
            "| --- | --- | --- |",
        ]
    )
    for spec in version_specs:
        lines.append(f"| {spec.label} | `{spec.method}` | {spec.description} |")
    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "| Ablation | Raw | Adjusted residuals | Adjusted context | Description |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for spec in ablation_specs:
        lines.append(
            f"| `{spec.name}` | {yes_no(spec.include_raw)} | "
            f"{yes_no(spec.include_adjusted_residuals)} | "
            f"{yes_no(spec.include_adjusted_context)} | {spec.description} |"
        )
    lines.extend(
        [
            "",
            "## Model Catalog",
            "",
            f"The sweep uses {len(model_specs)} entries from `DEFAULT_MODEL_SPECS`.",
            "",
            "| Family | Model | Config |",
            "| --- | --- | --- |",
        ]
    )
    for spec in model_specs:
        lines.append(f"| `{spec.family}` | `{spec.name}` | `{spec.config_path}` |")
    lines.extend(
        [
            "",
            "## Source Trail",
            "",
            "- Local research note: `opponent_adjusted_stats_deep_dive_tdnet.txt`.",
            "- The note frames opponent adjustment as leakage-safe adjusted game contributions rolled into team-week fingerprints.",
            "- The source fingerprint artifacts keep each method's parquet frame and, where applicable, adjusted game contribution parquet tables.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "PYTHONPATH=src python src/gridiron_ml/cli/run_opponent_ablation_shap.py build-manifest",
            "PYTHONPATH=src python src/gridiron_ml/cli/run_opponent_ablation_shap.py run-job --job-index 0 --force",
            "PYTHONPATH=src python src/gridiron_ml/cli/run_opponent_ablation_shap.py merge",
            "```",
            "",
            "For CRC/SGE execution, build the manifest once and submit the array script:",
            "",
            "```bash",
            "bash scripts/submit_opponent_ablation_shap_sge.sh",
            "```",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output_root / "method_sources.json",
        {
            "research_note": "opponent_adjusted_stats_deep_dive_tdnet.txt",
            "source_fingerprint_root": str(source_fingerprint_root),
            "source_experiment": DEFAULT_SOURCE_EXPERIMENT_NAME,
            "generated_at": utc_now(),
            "methods": [
                {"version": spec.label, "method": spec.method, "description": spec.description}
                for spec in version_specs
            ],
            "ablations": [spec.as_dict() for spec in ablation_specs],
        },
    )


def write_merge_status(output_root: Path, metrics: pd.DataFrame, shap: pd.DataFrame) -> None:
    """Write a small status payload after merging outputs."""

    success = successful_metrics(metrics)
    write_json(
        output_root / "summary" / "merge_status.json",
        {
            "merged_at": utc_now(),
            "metric_rows": int(len(metrics)),
            "successful_runs": int(len(success)),
            "failed_runs": int(len(metrics) - len(success)) if not metrics.empty else 0,
            "shap_rows": int(len(shap)),
        },
    )


def version_spec_from_label(label: str) -> OpponentAdjustedVersionSpec:
    """Resolve a version label from defaults."""

    for spec in DEFAULT_VERSION_SPECS:
        if spec.label == label or spec.safe_label == label:
            return spec
    raise KeyError(f"Unknown opponent-adjusted fingerprint version: {label!r}")


def ablation_spec_from_name(name: str) -> AblationSpec:
    """Resolve an ablation name from defaults."""

    for spec in DEFAULT_ABLATION_SPECS:
        if spec.name == name or spec.safe_name == name:
            return spec
    raise KeyError(f"Unknown ablation spec: {name!r}")


def write_status(path: Path, payload: dict[str, Any]) -> None:
    """Write a job status JSON file."""

    write_json(path, payload)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON using a compact, deterministic encoder."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def json_ready(value: Any) -> Any:
    """Convert pandas/numpy/path values into JSON-safe primitives."""

    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    if not isinstance(value, (str, bytes, bool)):
        try:
            missing = pd.isna(value)
        except Exception:
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    return value


def environment_summary() -> dict[str, Any]:
    """Return lightweight package/version context for run reproducibility."""

    summary = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "sklearn", "shap"):
        try:
            module = __import__(name)
            summary[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            summary[name] = f"unavailable: {exc}"
    return summary


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def yes_no(value: bool) -> str:
    """Return a markdown-friendly yes/no value."""

    return "yes" if bool(value) else "no"
