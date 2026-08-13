"""Shared model-training orchestration for TDNet workflows.

The model classes own estimator-specific fitting and prediction behavior.
This module owns run-level concerns: model catalogs, artifact paths, evaluator
configuration, checkpoint saving, and summary tables. Notebooks should call
these helpers instead of reimplementing training loops.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gridiron_ml.pipeline.contracts.artifacts import (
    cleanup_fingerprint_artifacts,
    fingerprint_version_dir,
)
from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.td_run.matchups import MatchupBuilder

from .evaluator import TDEval


@dataclass(frozen=True)
class ModelRunSpec:
    """A model config entry that can be trained through ``TDEval``."""

    name: str
    family: str
    config_path: str

    @classmethod
    def from_mapping(cls, value: dict | "ModelRunSpec") -> "ModelRunSpec":
        if isinstance(value, cls):
            return value
        return cls(
            name=str(value["name"]),
            family=str(value["family"]),
            config_path=str(value["config_path"]),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "family": self.family,
            "config_path": self.config_path,
        }


@dataclass
class TrainingRun:
    """Artifacts produced by one trained TDNet model."""

    spec: ModelRunSpec
    model: object
    metrics_df: pd.DataFrame
    checkpoint_path: Path
    artifact_root: Path


@dataclass
class TrainingResult:
    """Summary of a multi-model training run."""

    runs: list[TrainingRun]
    metrics: pd.DataFrame
    checkpoints: pd.DataFrame
    fingerprints: Fingerprints
    matchup_builder: MatchupBuilder


DEFAULT_MODEL_SPECS: tuple[ModelRunSpec, ...] = (
    ModelRunSpec("stat_z_index", "stat", "configs/models/stat/config_z_index.yaml"),
    ModelRunSpec(
        "stat_percentile", "stat", "configs/models/stat/config_percentile.yaml"
    ),
    ModelRunSpec("stat_robust", "stat", "configs/models/stat/config_robust.yaml"),
    ModelRunSpec("stat_weighted", "stat", "configs/models/stat/config_weighted.yaml"),
    ModelRunSpec("ols", "linear", "configs/models/linear/config_ols.yaml"),
    ModelRunSpec("ridge", "linear", "configs/models/linear/config_ridge.yaml"),
    ModelRunSpec("lasso", "linear", "configs/models/linear/config_lasso.yaml"),
    ModelRunSpec(
        "elastic_net", "linear", "configs/models/linear/config_elastic_net.yaml"
    ),
    ModelRunSpec("huber", "linear", "configs/models/linear/config_huber.yaml"),
    ModelRunSpec("bayesian", "linear", "configs/models/linear/config_bayesian.yaml"),
    ModelRunSpec("ard", "linear", "configs/models/linear/config_ard.yaml"),
    ModelRunSpec("ransac", "linear", "configs/models/linear/config_ransac.yaml"),
    ModelRunSpec(
        "orthogonal_matching_pursuit",
        "linear",
        "configs/models/linear/config_orthogonal_matching_pursuit.yaml",
    ),
    ModelRunSpec("sgd", "linear", "configs/models/linear/config_sgd.yaml"),
    ModelRunSpec(
        "passive_aggressive",
        "linear",
        "configs/models/linear/config_passive_aggressive.yaml",
    ),
    ModelRunSpec(
        "random_forest", "tree", "configs/models/tree/config_random_forest.yaml"
    ),
    ModelRunSpec("extra_trees", "tree", "configs/models/tree/config_extra_trees.yaml"),
    ModelRunSpec(
        "gradient_boosted", "tree", "configs/models/tree/config_gradient_boosted.yaml"
    ),
    ModelRunSpec("knn_uniform", "knn", "configs/models/knn/config_knn_uniform.yaml"),
    ModelRunSpec("knn_distance", "knn", "configs/models/knn/config_knn_distance.yaml"),
)

# Kept separate so legacy training configs preserve their exact model set.
# Publication configs opt into these entries explicitly.
PUBLICATION_MODEL_SPECS: tuple[ModelRunSpec, ...] = (
    ModelRunSpec("naive_home", "naive", "configs/models/naive/config_home_team.yaml"),
    ModelRunSpec("spline_ridge", "spline", "configs/models/spline/config_spline_ridge.yaml"),
    ModelRunSpec("hist_gradient_boosted", "boosted", "configs/models/boosted/config_hist_gradient_boosted.yaml"),
    ModelRunSpec("mlp", "neural", "configs/models/neural/config_mlp.yaml"),
    ModelRunSpec(
        "structured_mlp",
        "structured_neural",
        "configs/models/neural/config_structured_mlp.yaml",
    ),
    ModelRunSpec("knn_distance", "knn", "configs/models/knn/config_knn_distance.yaml"),
)


def filter_model_specs(
    specs: Iterable[dict | ModelRunSpec] = DEFAULT_MODEL_SPECS,
    *,
    train_stat: bool = True,
    train_linear: bool = True,
    train_tree: bool = True,
    train_naive: bool = False,
    train_spline: bool = False,
    train_boosted: bool = False,
    train_neural: bool = False,
    train_structured_neural: bool = False,
    train_knn: bool = True,
    only_names: Iterable[str] | None = None,
) -> list[ModelRunSpec]:
    """Filter the default model catalog by family toggles and optional names."""

    family_toggles = {
        "stat": bool(train_stat),
        "linear": bool(train_linear),
        "tree": bool(train_tree),
        "naive": bool(train_naive),
        "spline": bool(train_spline),
        "boosted": bool(train_boosted),
        "neural": bool(train_neural),
        "structured_neural": bool(train_structured_neural),
        "knn": bool(train_knn),
    }
    requested_names = {str(name) for name in (only_names or [])}
    selected = []
    for raw_spec in specs:
        spec = ModelRunSpec.from_mapping(raw_spec)
        if not family_toggles.get(spec.family, False):
            continue
        if requested_names and spec.name not in requested_names:
            continue
        selected.append(spec)
    return selected


def train_model_specs(
    specs: Iterable[dict | ModelRunSpec],
    *,
    project_root: str | Path,
    fingerprint_version: int = 0,
    postseason: bool = False,
    train_years: Iterable[int],
    val_years: Iterable[int] | None = None,
    test_years: Iterable[int] | None = None,
    matchup_config: dict | None = None,
    models_root: str | Path | None = None,
    build_fingerprints: bool = True,
    overwrite_fingerprints: bool = False,
    clear_existing_model_artifacts: bool = True,
    fingerprints: Fingerprints | None = None,
    matchup_builder: MatchupBuilder | None = None,
) -> TrainingResult:
    """Train model specs through ``TDEval`` and return run summaries."""

    root = Path(project_root).resolve()
    specs = [ModelRunSpec.from_mapping(spec) for spec in specs]
    if not specs:
        raise ValueError("No model specs selected for training.")

    models_root = Path(models_root) if models_root is not None else root / "models"
    matchup_config = dict(matchup_config or {"representation": "unit_matchup"})
    train_years = list(train_years)
    val_years = list(val_years or [])
    test_years = list(test_years or [])

    if fingerprints is None:
        if build_fingerprints:
            if overwrite_fingerprints:
                cleanup_fingerprint_artifacts(
                    fingerprint_version_dir(root, fingerprint_version),
                    fingerprint_version,
                )
            Fingerprints(
                version=fingerprint_version,
                postseason=postseason,
                root=root,
            ).build(overwrite=overwrite_fingerprints)
        fingerprints = Fingerprints(
            version=fingerprint_version,
            postseason=postseason,
            root=root,
        )

    if matchup_builder is None:
        matchup_builder = MatchupBuilder(**matchup_config)

    runs: list[TrainingRun] = []
    for spec in specs:
        run_dir = model_run_dir(spec, models_root=models_root)
        if clear_existing_model_artifacts:
            clear_model_run_dir(spec, models_root=models_root)
        evaluator = TDEval(
            config=build_eval_config(
                spec,
                project_root=root,
                fingerprint_version=fingerprint_version,
                postseason=postseason,
                train_years=train_years,
                test_years=test_years,
                matchup_config=matchup_config,
                models_root=models_root,
            ),
            fingerprints=fingerprints,
            matchup_builder=matchup_builder,
        )
        model = evaluator.train(train_years=train_years, val_years=val_years)
        saved_checkpoint = model.save(checkpoint_path(spec, models_root=models_root))
        metrics_df = pd.DataFrame()
        if test_years:
            _, metrics_df = evaluator.evaluate(years=test_years, label="test")
        artifact_root = evaluator.save_outputs(run_dir / "artifacts")
        runs.append(
            TrainingRun(
                spec=spec,
                model=model,
                metrics_df=metrics_df,
                checkpoint_path=Path(saved_checkpoint),
                artifact_root=Path(artifact_root),
            )
        )

    return TrainingResult(
        runs=runs,
        metrics=metrics_summary_frame(runs),
        checkpoints=checkpoint_summary_frame(runs),
        fingerprints=fingerprints,
        matchup_builder=matchup_builder,
    )


def build_eval_config(
    spec: dict | ModelRunSpec,
    *,
    project_root: str | Path,
    fingerprint_version: int,
    postseason: bool,
    train_years: Iterable[int],
    test_years: Iterable[int],
    matchup_config: dict | None,
    models_root: str | Path,
) -> dict:
    """Build a ``TDEval`` config for one model spec."""

    spec = ModelRunSpec.from_mapping(spec)
    root = Path(project_root)
    return {
        "fingerprints": {
            "version": int(fingerprint_version),
            "postseason": bool(postseason),
            "root": str(root),
        },
        "matchup": dict(matchup_config or {}),
        "model": {
            "family": spec.family,
            "config_path": str(root / spec.config_path),
        },
        "eval": {
            "train_years": list(train_years),
            "test_years": list(test_years),
            "artifact_root": str(
                model_run_dir(spec, models_root=models_root) / "artifacts"
            ),
        },
    }


def model_run_dir(spec: dict | ModelRunSpec, *, models_root: str | Path) -> Path:
    spec = ModelRunSpec.from_mapping(spec)
    return Path(models_root) / _family_dir_name(spec.family) / "models" / spec.name


def checkpoint_path(spec: dict | ModelRunSpec, *, models_root: str | Path) -> Path:
    spec = ModelRunSpec.from_mapping(spec)
    prefix = {"linear": "tdlinear", "stat": "tdstat", "tree": "tdtree", "knn": "tdknn"}.get(
        spec.family,
        spec.family,
    )
    return (
        model_run_dir(spec, models_root=models_root)
        / "models"
        / f"{prefix}_{spec.name}.pkl"
    )


def metrics_summary_frame(runs: Iterable[TrainingRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        row = {"model": run.spec.name, "family": run.spec.family}
        if run.metrics_df is not None and not run.metrics_df.empty:
            row.update(run.metrics_df.iloc[0].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_summary_frame(runs: Iterable[TrainingRun]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": run.spec.name,
                "family": run.spec.family,
                "checkpoint_path": str(run.checkpoint_path),
                "artifact_root": str(run.artifact_root),
            }
            for run in runs
        ]
    )


def clear_model_run_dir(spec: dict | ModelRunSpec, *, models_root: str | Path) -> Path:
    """Remove stale checkpoint/artifact outputs for one selected model run."""

    run_dir = model_run_dir(spec, models_root=models_root)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return run_dir


def _family_dir_name(family: str) -> str:
    return str(family).strip().lower().replace(" ", "_")
