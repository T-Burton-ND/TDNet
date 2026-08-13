"""Time-adjusted fingerprint experiment pipeline.

This experiment builds on opponent-adjusted fingerprint frames. It adds
features that say whether a team's opponent-adjusted profile is high or low
for the season week/phase being evaluated, using only previous seasons as the
reference population for each row's season.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from gridiron_ml.experiments.opponent_ablation import (
    ablation_spec_from_name,
    apply_ablation_view,
)
from gridiron_ml.experiments.opponent_adjusted import (
    DEFAULT_TRAIN_YEARS,
    DEFAULT_VAL_YEARS,
    DEFAULT_TEST_YEARS,
    StaticFrameFingerprints,
)
from gridiron_ml.models import build_model_from_config
from gridiron_ml.pipeline.contracts.features import FINGERPRINT_KEY_COLUMNS
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.td_run.training import DEFAULT_MODEL_SPECS, ModelRunSpec


DEFAULT_EXPERIMENT_NAME = "time_adjusted_fingerprints"
DEFAULT_SOURCE_EXPERIMENT_NAME = "opponent_adjusted_fingerprints"
DEFAULT_SOURCE_LABELS = ("v1.4", "v1.7")
DEFAULT_FEATURE_PREFIXES = ("opp_adj_",)


@dataclass(frozen=True)
class TimeAdjustedVersionSpec:
    """One time-adjusted fingerprint variant."""

    label: str
    method: str
    source_label: str
    description: str

    @property
    def safe_label(self) -> str:
        return safe_label(self.label)

    @property
    def subversion(self) -> int:
        digits = re.sub(r"\D", "", self.label)
        return int(digits) if digits else 0


DEFAULT_VERSION_SPECS = (
    TimeAdjustedVersionSpec(
        "t2.1",
        "same_week_z",
        "v1.7",
        "Opponent-adjusted features z-scored against prior-season same-week baselines.",
    ),
    TimeAdjustedVersionSpec(
        "t2.2",
        "phase_z",
        "v1.7",
        "Opponent-adjusted features z-scored against prior-season season-phase baselines.",
    ),
    TimeAdjustedVersionSpec(
        "t2.3",
        "recency_week_z",
        "v1.4",
        "Same-week baselines with previous seasons exponentially weighted by recency.",
    ),
)


@dataclass(frozen=True)
class TimeAdjustedFrameArtifact:
    """Persisted fingerprint frame for one time-adjusted version."""

    spec: TimeAdjustedVersionSpec
    frame: pd.DataFrame
    path: Path


def default_output_root(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "data" / "experiments" / DEFAULT_EXPERIMENT_NAME


def default_source_fingerprint_root(project_root: str | Path) -> Path:
    return (
        Path(project_root).resolve()
        / "data"
        / "experiments"
        / DEFAULT_SOURCE_EXPERIMENT_NAME
    )


def build_time_adjusted_experiment_frames(
    *,
    project_root: str | Path,
    output_root: str | Path | None = None,
    source_fingerprint_root: str | Path | None = None,
    version_specs: tuple[TimeAdjustedVersionSpec, ...] = DEFAULT_VERSION_SPECS,
    feature_prefixes: tuple[str, ...] = DEFAULT_FEATURE_PREFIXES,
    max_features: int | None = 120,
    overwrite: bool = False,
) -> dict[str, TimeAdjustedFrameArtifact]:
    """Build and persist time-adjusted fingerprint frames."""

    root = Path(project_root).resolve()
    output_root = Path(output_root or default_output_root(root)).resolve()
    source_root = Path(source_fingerprint_root or default_source_fingerprint_root(root)).resolve()
    frames_root = output_root / "fingerprints"
    frames_root.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, TimeAdjustedFrameArtifact] = {}
    source_cache: dict[str, pd.DataFrame] = {}
    for spec in version_specs:
        path = frames_root / spec.safe_label / "canonical_fingerprint.parquet"
        if path.exists() and not overwrite:
            frame = pd.read_parquet(path)
            artifacts[spec.label] = TimeAdjustedFrameArtifact(spec=spec, frame=frame, path=path)
            continue

        source_frame = source_cache.get(spec.source_label)
        if source_frame is None:
            source_frame = load_source_frame(source_root, spec.source_label)
            source_cache[spec.source_label] = source_frame
        frame = build_time_adjusted_frame(
            source_frame,
            spec=spec,
            feature_prefixes=feature_prefixes,
            max_features=max_features,
        )
        write_frame_artifact(frame, path, spec=spec, source_root=source_root)
        artifacts[spec.label] = TimeAdjustedFrameArtifact(spec=spec, frame=frame, path=path)
    return artifacts


def load_source_frame(source_root: Path, label: str) -> pd.DataFrame:
    path = source_root / "fingerprints" / safe_label(label) / "canonical_fingerprint.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing source opponent-adjusted fingerprint: {path}")
    frame = pd.read_parquet(path)
    return apply_ablation_view(frame, ablation_spec_from_name("raw_plus_adjusted_all"))


def build_time_adjusted_frame(
    frame: pd.DataFrame,
    *,
    spec: TimeAdjustedVersionSpec,
    feature_prefixes: tuple[str, ...] = DEFAULT_FEATURE_PREFIXES,
    max_features: int | None = 120,
) -> pd.DataFrame:
    """Add week-aware features to one opponent-adjusted frame."""

    out = pd.DataFrame(frame).copy()
    features = select_time_adjustment_features(out, prefixes=feature_prefixes, max_features=max_features)
    if not features:
        raise ValueError("No numeric source features available for time adjustment.")

    if spec.method == "same_week_z":
        adjusted = time_adjusted_columns(out, features=features, label=spec.safe_label, mode="week", weighted=False)
    elif spec.method == "phase_z":
        adjusted = time_adjusted_columns(out, features=features, label=spec.safe_label, mode="phase", weighted=False)
    elif spec.method == "recency_week_z":
        adjusted = time_adjusted_columns(out, features=features, label=spec.safe_label, mode="week", weighted=True)
    else:
        raise ValueError(f"Unsupported time-adjustment method: {spec.method}")

    out = pd.concat([out.reset_index(drop=True), adjusted.reset_index(drop=True)], axis=1)
    out["fp_version_label"] = spec.label
    out["fp_method"] = spec.method
    out["fp_subversion"] = spec.subversion
    out["fp_experiment"] = DEFAULT_EXPERIMENT_NAME
    out["fp_source_label"] = spec.source_label
    return out


def select_time_adjustment_features(
    frame: pd.DataFrame,
    *,
    prefixes: tuple[str, ...],
    max_features: int | None,
) -> list[str]:
    """Choose numeric source features to time-adjust, sorted by variance."""

    key_cols = set(FINGERPRINT_KEY_COLUMNS)
    candidates = [
        col
        for col in frame.columns
        if col not in key_cols
        and any(str(col).startswith(prefix) for prefix in prefixes)
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if not candidates:
        return []
    variances = frame.loc[:, candidates].var(numeric_only=True).sort_values(ascending=False)
    ranked = [str(col) for col in variances.index if pd.notna(variances.loc[col])]
    return ranked[: int(max_features)] if max_features is not None else ranked


def time_adjusted_columns(
    frame: pd.DataFrame,
    *,
    features: list[str],
    label: str,
    mode: str,
    weighted: bool,
) -> pd.DataFrame:
    """Compute same-week or same-phase historical z-score features."""

    work = frame.reset_index(drop=True).copy()
    work["_row_id"] = np.arange(len(work))
    work["_season"] = pd.to_numeric(work["keys_season"], errors="coerce").astype("Int64")
    work["_week"] = pd.to_numeric(work["keys_week"], errors="coerce").astype("Int64")
    work["_phase"] = work["_week"].map(week_phase)
    out = pd.DataFrame(index=work.index)
    prefix = f"time_adj_{safe_label(label)}"
    group_key = "_week" if mode == "week" else "_phase"

    for (season, bucket), group in work.groupby(["_season", group_key], observed=True, sort=True):
        if pd.isna(season) or pd.isna(bucket):
            continue
        history = work.loc[(work["_season"] < int(season)) & (work[group_key] == bucket)].copy()
        if len(history) < 25:
            history = work.loc[work["_season"] < int(season)].copy()
        rows = group.index
        if history.empty:
            means = pd.Series(0.0, index=features)
            stds = pd.Series(1.0, index=features)
        else:
            values = history.loc[:, features].apply(pd.to_numeric, errors="coerce")
            if weighted:
                season_age = int(season) - pd.to_numeric(history["_season"], errors="coerce")
                weights = np.exp(-np.maximum(season_age.to_numpy(dtype=float), 0.0) / 3.0)
                means = weighted_mean(values, weights)
                stds = weighted_std(values, weights, means).replace(0.0, 1.0).fillna(1.0)
            else:
                means = values.mean(numeric_only=True)
                stds = values.std(numeric_only=True).replace(0.0, 1.0).fillna(1.0)
        current = work.loc[rows, features].apply(pd.to_numeric, errors="coerce")
        z = (current - means.reindex(features).fillna(0.0)) / stds.reindex(features).fillna(1.0)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for feature in features:
            out.loc[rows, f"{prefix}_{safe_label(feature)}_{mode}_z"] = z[feature].to_numpy(dtype=float)

    out[f"{prefix}_{mode}_index"] = (
        work["_week"].astype(float) if mode == "week" else work["_phase"].astype(float)
    )
    return out.fillna(0.0)


def weighted_mean(values: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    clean = values.fillna(values.mean(numeric_only=True)).fillna(0.0)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    denom = max(float(weights.sum()), 1e-8)
    return pd.Series((clean.to_numpy(dtype=float) * weights[:, None]).sum(axis=0) / denom, index=clean.columns)


def weighted_std(values: pd.DataFrame, weights: np.ndarray, means: pd.Series) -> pd.Series:
    clean = values.fillna(means).fillna(0.0)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    denom = max(float(weights.sum()), 1e-8)
    diff = clean.to_numpy(dtype=float) - means.reindex(clean.columns).fillna(0.0).to_numpy(dtype=float)
    var = (weights[:, None] * diff * diff).sum(axis=0) / denom
    return pd.Series(np.sqrt(np.maximum(var, 0.0)), index=clean.columns)


def week_phase(week: Any) -> str:
    if pd.isna(week):
        return "unknown"
    week = int(week)
    if week <= 0:
        return "preseason"
    if week <= 4:
        return "early"
    if week <= 9:
        return "middle"
    if week <= 13:
        return "late"
    return "postseason"


def smoke_train_time_adjusted_models(
    *,
    project_root: str | Path,
    frame_path: str | Path,
    output_root: str | Path,
    model_names: tuple[str, ...] = ("stat_weighted", "ridge", "random_forest"),
    train_years=DEFAULT_TRAIN_YEARS,
    val_years=DEFAULT_VAL_YEARS,
    test_years=DEFAULT_TEST_YEARS,
) -> pd.DataFrame:
    """Train one stat, one linear, and one tree model against a time-adjusted frame."""

    root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    frame = pd.read_parquet(frame_path)
    fingerprints = StaticFrameFingerprints(frame)
    matchup_builder = MatchupBuilder(representation="unit_matchup")
    specs = [ModelRunSpec.from_mapping(spec) for spec in DEFAULT_MODEL_SPECS if spec.name in set(model_names)]
    rows = []
    for spec in specs:
        evaluator = TDEval(
            config={
                "model": {"family": spec.family, "config_path": str(root / spec.config_path)},
                "eval": {
                    "train_years": list(train_years),
                    "test_years": list(test_years),
                    "artifact_root": str(output_root / spec.family / spec.name / "artifacts"),
                },
            },
            fingerprints=fingerprints,
            matchup_builder=matchup_builder,
            model=build_model_from_config({"family": spec.family, "config_path": str(root / spec.config_path)}),
        )
        model = evaluator.train(train_years=train_years, val_years=val_years)
        _, metrics = evaluator.evaluate(years=test_years, label="smoke")
        artifact_root = evaluator.save_outputs(output_root / spec.family / spec.name / "artifacts")
        row = {"family": spec.family, "model": spec.name, "artifact_root": str(artifact_root)}
        if metrics is not None and not metrics.empty:
            row.update(metrics.iloc[0].to_dict())
        rows.append(row)
    summary = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "time_adjusted_smoke_metrics.csv", index=False)
    return summary


def write_frame_artifact(frame: pd.DataFrame, path: Path, *, spec: TimeAdjustedVersionSpec, source_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    metadata = {
        "artifact_kind": "time_adjusted_experiment_fingerprint",
        "label": spec.label,
        "method": spec.method,
        "source_label": spec.source_label,
        "description": spec.description,
        "source_root": str(source_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))


def safe_label(value: str) -> str:
    return str(value).replace(".", "_").replace("/", "_").replace(" ", "_")


__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_SOURCE_LABELS",
    "DEFAULT_VERSION_SPECS",
    "TimeAdjustedFrameArtifact",
    "TimeAdjustedVersionSpec",
    "build_time_adjusted_experiment_frames",
    "build_time_adjusted_frame",
    "default_output_root",
    "default_source_fingerprint_root",
    "smoke_train_time_adjusted_models",
]
