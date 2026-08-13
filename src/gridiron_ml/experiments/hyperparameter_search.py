"""Fingerprint-specific hyperparameter search workflow.

This module builds a manifest-driven study that combines opponent-adjusted
fingerprint variants, truncated feature views, and per-model hyperparameter
setpoints. It intentionally uses deterministic random sampling from YAML
spaces so the study can run on SGE arrays without a database service.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import shutil
import time
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gridiron_ml.experiments.opponent_ablation import (
    ablation_spec_from_name,
    apply_ablation_view,
)
from gridiron_ml.experiments.opponent_adjusted import (
    DEFAULT_TEST_YEARS,
    DEFAULT_TRAIN_YEARS,
    DEFAULT_VAL_YEARS,
    DEFAULT_VERSION_SPECS,
    StaticFrameFingerprints,
    extract_vegas_metrics,
)
from gridiron_ml.models import get_model_class
from gridiron_ml.pipeline.contracts.features import (
    FINGERPRINT_KEY_COLUMNS,
    LABEL_COLUMNS,
    MARKET_CONTEXT_KEY_COLUMNS,
    NEXT_GAME_COLUMNS,
    is_feature_column,
)
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.td_run.season_vs_vegas import evaluate_models_vs_vegas_season
from gridiron_ml.td_run.training import DEFAULT_MODEL_SPECS, ModelRunSpec


DEFAULT_EXPERIMENT_NAME = "fingerprint_hyperparameter_search"
DEFAULT_SOURCE_EXPERIMENT_NAME = "opponent_adjusted_fingerprints"
DEFAULT_ABLATION_EXPERIMENT_NAME = "opponent_adjusted_ablation_shap"
DEFAULT_SCORE_WEIGHTS = {
    "season_winner_chalk_accuracy": 0.35,
    "season_winner_upset_recall": 0.35,
    "season_winner_winner_accuracy": 0.20,
    "score_margin_score": 0.10,
}


@dataclass(frozen=True)
class SearchJob:
    """One manifest row for a model/fingerprint/truncation/setpoint combo."""

    job_index: int
    fingerprint: str
    model: str
    family: str
    config_path: str
    top_k_features: str
    trial_index: int
    params: dict[str, Any]
    fingerprint_path: str
    output_dir: str

    @property
    def sge_task_id(self) -> int:
        return self.job_index + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_index": self.job_index,
            "sge_task_id": self.sge_task_id,
            "fingerprint": self.fingerprint,
            "model": self.model,
            "family": self.family,
            "model_config_path": self.config_path,
            "top_k_features": self.top_k_features,
            "trial_index": self.trial_index,
            "params_json": json.dumps(self.params, sort_keys=True),
            "fingerprint_path": self.fingerprint_path,
            "output_dir": self.output_dir,
            "metrics_path": str(Path(self.output_dir) / "metrics.csv"),
            "status_path": str(Path(self.output_dir) / "status.json"),
        }


def default_output_root(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "data" / "experiments" / DEFAULT_EXPERIMENT_NAME


def default_source_fingerprint_root(project_root: str | Path) -> Path:
    return (
        Path(project_root).resolve()
        / "data"
        / "experiments"
        / DEFAULT_SOURCE_EXPERIMENT_NAME
    )


def default_ablation_summary_path(project_root: str | Path) -> Path:
    return (
        Path(project_root).resolve()
        / "data"
        / "experiments"
        / DEFAULT_ABLATION_EXPERIMENT_NAME
        / "summary"
        / "tables"
        / "master_shap_importance.csv"
    )


def build_search_manifest(
    *,
    project_root: str | Path,
    config_path: str | Path,
    output_root: str | Path | None = None,
    source_fingerprint_root: str | Path | None = None,
    model_specs: tuple[ModelRunSpec, ...] | list[ModelRunSpec] = DEFAULT_MODEL_SPECS,
) -> pd.DataFrame:
    """Build and save the hyperparameter search manifest."""

    root = Path(project_root).resolve()
    output_root = Path(output_root or default_output_root(root)).resolve()
    source_root = Path(source_fingerprint_root or default_source_fingerprint_root(root)).resolve()
    cfg = load_yaml(config_path)
    output_root.mkdir(parents=True, exist_ok=True)

    fingerprints = [str(x) for x in cfg.get("fingerprints", {}).get("labels", [])]
    if not fingerprints:
        fingerprints = [spec.label for spec in DEFAULT_VERSION_SPECS]
    top_k_values = [str(x) for x in cfg.get("feature_truncation", {}).get("top_k", ["all"])]
    seed = int(cfg.get("global", {}).get("seed", 1337))
    default_trials = int(cfg.get("global", {}).get("trials_per_model", 24))

    rows: list[dict[str, Any]] = []
    job_index = 0
    for spec in [ModelRunSpec.from_mapping(x) for x in model_specs]:
        model_cfg = cfg.get("models", {}).get(spec.name, {})
        family_cfg = cfg.get("families", {}).get(spec.family, {})
        trial_count = int(model_cfg.get("trials", family_cfg.get("trials", default_trials)))
        space = dict(family_cfg.get("params", {}))
        space.update(dict(model_cfg.get("params", {})))
        setpoints = sample_setpoints(space, n=trial_count, seed=stable_seed(seed, spec.name))
        for fingerprint in fingerprints:
            fp_path = source_root / "fingerprints" / safe_label(fingerprint) / "canonical_fingerprint.parquet"
            for top_k in top_k_values:
                for trial_index, params in enumerate(setpoints):
                    out_dir = (
                        output_root
                        / "runs"
                        / safe_label(fingerprint)
                        / spec.family
                        / spec.name
                        / f"top_{top_k}"
                        / f"trial_{trial_index:04d}"
                    )
                    rows.append(
                        SearchJob(
                            job_index=job_index,
                            fingerprint=fingerprint,
                            model=spec.name,
                            family=spec.family,
                            config_path=spec.config_path,
                            top_k_features=top_k,
                            trial_index=trial_index,
                            params=params,
                            fingerprint_path=str(fp_path),
                            output_dir=str(out_dir),
                        ).as_dict()
                    )
                    job_index += 1

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_root / "job_manifest.csv", index=False)
    metadata = {
        "experiment": DEFAULT_EXPERIMENT_NAME,
        "created_at": utc_now(),
        "project_root": str(root),
        "config_path": str(Path(config_path).resolve()),
        "source_fingerprint_root": str(source_root),
        "job_count": int(len(manifest)),
        "score_weights": dict(cfg.get("score_weights", DEFAULT_SCORE_WEIGHTS)),
        "artifacts": dict(cfg.get("artifacts", {})),
    }
    (output_root / "search_manifest.json").write_text(json.dumps(metadata, indent=2))
    write_readme(output_root, cfg, len(manifest))
    return manifest


def run_manifest_job(
    *,
    project_root: str | Path,
    output_root: str | Path,
    config_path: str | Path,
    job_index: int | None = None,
    sge_task_id: int | None = None,
    job_manifest: str | Path | None = None,
    train_years=DEFAULT_TRAIN_YEARS,
    val_years=DEFAULT_VAL_YEARS,
    test_years=DEFAULT_TEST_YEARS,
    force: bool = False,
) -> dict[str, Any]:
    """Run one hyperparameter-search manifest row."""

    output_root = Path(output_root).resolve()
    manifest_path = Path(job_manifest) if job_manifest else output_root / "job_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if sge_task_id is not None:
        selected = manifest.loc[manifest["sge_task_id"].astype(int) == int(sge_task_id)]
    elif job_index is not None:
        selected = manifest.loc[manifest["job_index"].astype(int) == int(job_index)]
    else:
        raise ValueError("Provide job_index or sge_task_id.")
    if selected.empty:
        raise IndexError("Requested hyperparameter manifest row does not exist.")
    row = selected.iloc[0].to_dict()
    if row.get("train_years_json"):
        train_years = tuple(json.loads(row["train_years_json"]))
    if row.get("val_years_json"):
        val_years = tuple(json.loads(row["val_years_json"]))
    if row.get("test_years_json"):
        test_years = tuple(json.loads(row["test_years_json"]))
    return run_search_combo(
        project_root=project_root,
        config_path=config_path,
        row=row,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        force=force,
    )


def run_search_combo(
    *,
    project_root: str | Path,
    config_path: str | Path,
    row: dict[str, Any],
    train_years=DEFAULT_TRAIN_YEARS,
    val_years=DEFAULT_VAL_YEARS,
    test_years=DEFAULT_TEST_YEARS,
    force: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one tuned model/fingerprint/truncation combo."""

    root = Path(project_root).resolve()
    cfg = load_yaml(config_path)
    output_dir = Path(row["output_dir"]).resolve()
    metrics_path = output_dir / "metrics.csv"
    status_path = output_dir / "status.json"
    if metrics_path.exists() and not force:
        return {"status": "skipped_existing", "output_dir": str(output_dir)}

    output_dir.mkdir(parents=True, exist_ok=True)
    write_status(status_path, {"status": "running", "started_at": utc_now(), **job_keys(row)})
    artifacts_cfg = dict(cfg.get("artifacts", {}))
    save_train_artifacts = bool(artifacts_cfg.get("train_artifacts", False))
    save_season_artifacts = bool(artifacts_cfg.get("season_eval", False))
    save_model_configs = bool(artifacts_cfg.get("model_configs", False))
    keep_tracebacks = bool(artifacts_cfg.get("tracebacks", False))
    params = json.loads(row.get("params_json") or "{}")
    result = {
        **job_keys(row), "params_json": json.dumps(params, sort_keys=True),
        "train_years_json": json.dumps(list(map(int, train_years))),
        "val_years_json": json.dumps(list(map(int, val_years))),
        "test_years_json": json.dumps(list(map(int, test_years))),
        "outer_fold": row.get("outer_fold"),
        "cv_test_season": row.get("cv_test_season"),
        "base_job_index": row.get("base_job_index", row.get("job_index")),
    }
    try:
        frame = pd.read_parquet(row["fingerprint_path"])
        canonical_tier = str(row.get("canonical_feature_config", "")).strip()
        if canonical_tier:
            season_key = "season" if "season" in frame.columns else "keys_season" if "keys_season" in frame.columns else None
            if season_key is None:
                raise ValueError("canonical HPS source frame has no season key")
            frame = frame.loc[pd.to_numeric(frame[season_key], errors="coerce").le(2025)].copy()
            # Canonical gap-fill rows are filtered by the version-controlled
            # F0-F8 ladder. They never use the legacy v1.x truncation path.
            from gridiron_ml.experiments.publication import filter_frame_for_feature_config
            frame, feature_metadata = filter_frame_for_feature_config(
                frame,
                feature_config=canonical_tier,
                registry_path=row["feature_registry"],
                ladders_path=row["feature_ladders"],
                strict_registry=True,
            )
            frame = apply_ablation_view(frame, ablation_spec_from_name("raw_plus_adjusted_all"))
            result["canonical_feature_config"] = canonical_tier
            result["canonical_feature_count"] = int(feature_metadata["selected_feature_count"])
        else:
            frame = apply_ablation_view(frame, ablation_spec_from_name("raw_plus_adjusted_all"))
            frame = truncate_frame_features(
                frame,
                top_k=row["top_k_features"],
                shap_path=Path(cfg.get("feature_truncation", {}).get("shap_importance_path") or default_ablation_summary_path(root)),
                model_name=str(row["model"]),
                fingerprint=str(row["fingerprint"]),
            )
        result["feature_count"] = count_trainable_features(frame)
        selected_features = [col for col in frame.columns if is_trainable_feature(frame, col)]
        result["selected_features_json"] = json.dumps(selected_features, sort_keys=True)

        tuned_config = build_tuned_model_config(
            base_config_path=root / str(row["model_config_path"]),
            params=params,
        )
        result["model_config_json"] = json.dumps(tuned_config, sort_keys=True, default=str)
        if save_model_configs:
            tuned_config_path = write_tuned_model_config(
                base_config_path=root / str(row["model_config_path"]),
                output_dir=output_dir,
                params=params,
            )
            model_ref = {"family": row["family"], "config_path": str(tuned_config_path)}
        else:
            model_ref = {"family": row["family"], **tuned_config}
        fingerprints = StaticFrameFingerprints(frame)
        matchup_builder = MatchupBuilder(representation="unit_matchup")
        model = get_model_class(row["family"])(tuned_config)
        evaluator = TDEval(
            config={
                "model": model_ref,
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
        _, metrics_df = evaluator.evaluate(years=test_years, label="test")
        result.update(first_row(metrics_df))
        if save_train_artifacts:
            result["train_artifact_root"] = str(
                evaluator.save_outputs(output_dir / "train_artifacts")
            )
        else:
            result["train_artifact_root"] = ""

        vegas_tables = evaluate_models_vs_vegas_season(
            fingerprints=fingerprints,
            matchup_builder=matchup_builder,
            season=int(list(test_years)[-1]),
            model_specs=[{"name": row["model"], "model": model}],
            output_dir=output_dir / "season_eval" if save_season_artifacts else None,
            make_plots=False,
            eval_config={"artifacts": {"shap": False, "png_plots": False}},
        )
        result.update(extract_vegas_metrics(vegas_tables, str(row["model"])))
        result["tuning_score"] = score_row(result, cfg.get("score_weights", DEFAULT_SCORE_WEIGHTS))
        result["status"] = "success"
        result["error"] = ""
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc() if keep_tracebacks else ""

    pd.DataFrame([result]).to_csv(metrics_path, index=False)
    write_status(status_path, {**result, "completed_at": utc_now()})
    return result


def merge_search_outputs(*, output_root: str | Path) -> dict[str, pd.DataFrame]:
    """Merge search jobs, choose best rows, and save surface plots."""

    output_root = Path(output_root).resolve()
    tables_dir = output_root / "summary" / "tables"
    figures_dir = output_root / "summary" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_many_csv(output_root.glob("runs/*/*/*/top_*/*/metrics.csv"))
    metrics.to_csv(tables_dir / "master_hyperparameter_metrics.csv", index=False)
    reproducible = build_reproducible_results_table(metrics)
    write_compressed_results(reproducible, tables_dir / "master_hyperparameter_results")
    if metrics.empty:
        return {"metrics": metrics, "best": pd.DataFrame()}

    success = metrics.loc[metrics.get("status", "").astype(str).eq("success")].copy()
    if success.empty:
        success.to_csv(tables_dir / "best_by_model.csv", index=False)
        return {"metrics": metrics, "best": success}
    if "tuning_score" not in success.columns:
        success["tuning_score"] = np.nan
    success["tuning_score"] = pd.to_numeric(success["tuning_score"], errors="coerce")
    best = (
        success.sort_values("tuning_score", ascending=False)
        .groupby(["family", "model"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(tables_dir / "best_by_model.csv", index=False)
    pool = best.groupby(["fingerprint", "top_k_features"], as_index=False)["tuning_score"].mean()
    pool.to_csv(tables_dir / "best_model_pool_score_by_fingerprint.csv", index=False)
    save_global_pool_plot(pool, figures_dir / "global_best_model_pool_score_surface.png")
    save_surface_plots(success, figures_dir)
    write_status(
        output_root / "summary" / "merge_status.json",
        {
            "merged_at": utc_now(),
            "metric_rows": int(len(metrics)),
            "successful_runs": int(len(success)),
            "failed_runs": int(metrics.get("status", pd.Series(dtype=str)).astype(str).eq("failed").sum()),
            "reproducible_results_rows": int(len(reproducible)),
            "reproducible_results_columns": int(len(reproducible.columns)),
        },
    )
    cleanup_runs = bool(read_search_artifact_config(output_root).get("cleanup_runs_after_merge", False))
    if cleanup_runs:
        runs_dir = output_root / "runs"
        if runs_dir.exists():
            remove_tree_with_retries(runs_dir)
    return {"metrics": metrics, "best": best}


def remove_tree_with_retries(path: Path, *, attempts: int = 3, delay_seconds: float = 1.0) -> None:
    """Remove a large task tree, retrying transient non-empty directory errors."""

    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(delay_seconds)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    if path.exists() and last_error is not None:
        raise last_error


def truncate_frame_features(
    frame: pd.DataFrame,
    *,
    top_k: str,
    shap_path: Path,
    model_name: str,
    fingerprint: str,
) -> pd.DataFrame:
    """Keep only metadata plus the top-k source features for this model."""

    if str(top_k).lower() in {"all", "none", "0"}:
        return frame.copy()
    k = int(top_k)
    always = always_keep_columns(frame)
    trainable = [col for col in frame.columns if is_trainable_feature(frame, col)]
    ranked = ranked_source_features(
        frame,
        shap_path=shap_path,
        model_name=model_name,
        fingerprint=fingerprint,
    )
    selected = [col for col in ranked if col in trainable][:k]
    if len(selected) < k:
        selected.extend([col for col in trainable if col not in selected][: k - len(selected)])
    keep = list(dict.fromkeys(always + selected))
    return frame.loc[:, keep].copy()


def ranked_source_features(
    frame: pd.DataFrame,
    *,
    shap_path: Path,
    model_name: str,
    fingerprint: str,
) -> list[str]:
    """Rank source frame features from ablation SHAP, falling back to variance."""

    if shap_path.exists():
        shap = pd.read_csv(shap_path)
        if not shap.empty and "feature" in shap.columns:
            if "model" in shap.columns:
                shap = shap.loc[shap["model"].astype(str).eq(str(model_name))]
            if "fingerprint" in shap.columns and not shap.empty:
                filtered = shap.loc[shap["fingerprint"].astype(str).eq(str(fingerprint))]
                if not filtered.empty:
                    shap = filtered
            score_col = "mean_abs_shap" if "mean_abs_shap" in shap.columns else None
            if score_col:
                shap[score_col] = pd.to_numeric(shap[score_col], errors="coerce")
                names = (
                    shap.sort_values(score_col, ascending=False)["feature"]
                    .astype(str)
                    .map(strip_matchup_prefix)
                    .drop_duplicates()
                    .tolist()
                )
                matched = match_ranked_features(names, frame.columns)
                if matched:
                    return matched
    numeric = frame.loc[:, [col for col in frame.columns if is_trainable_feature(frame, col)]]
    variances = numeric.var(numeric_only=True).sort_values(ascending=False)
    return [str(col) for col in variances.index]


def sample_setpoints(space: dict[str, Any], *, n: int, seed: int) -> list[dict[str, Any]]:
    """Sample deterministic setpoints from a YAML hyperparameter space."""

    if not space:
        return [{}]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for _ in range(max(int(n), 1)):
        row = {}
        for name, spec in space.items():
            row[name] = sample_value(dict(spec or {}), rng)
        rows.append(row)
    return rows


def sample_value(spec: dict[str, Any], rng: np.random.Generator) -> Any:
    kind = str(spec.get("type", "categorical")).lower()
    if kind == "categorical":
        choices = list(spec.get("choices", []))
        return choices[int(rng.integers(0, len(choices)))] if choices else None
    low = float(spec.get("low", 0.0))
    high = float(spec.get("high", 1.0))
    if bool(spec.get("log", False)):
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        value = rng.uniform(low, high)
    if kind == "int":
        return int(round(value))
    return float(value)


def build_tuned_model_config(
    *,
    base_config_path: Path,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build a per-trial model config with dotted override support."""

    cfg = load_yaml(base_config_path)
    for key, value in params.items():
        parts = str(key).split(".")
        if parts[0] == "config":
            set_nested(cfg, parts[1:], value)
            continue
        if len(parts) == 1:
            parts = ["params", parts[0]]
        set_nested(cfg, parts, value)
    return cfg


def write_tuned_model_config(
    *,
    base_config_path: Path,
    output_dir: Path,
    params: dict[str, Any],
) -> Path:
    """Write a per-trial model config with dotted override support."""

    cfg = build_tuned_model_config(base_config_path=base_config_path, params=params)
    out = output_dir / "model_config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def build_reproducible_results_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return one compressed-table-ready row per trial with params and feature flags."""

    if metrics.empty:
        return metrics.copy()
    out = expand_params(metrics)
    feature_sets = parse_feature_sets(out.get("selected_features_json", pd.Series(dtype=str)), len(out))
    feature_names = sorted({feature for row in feature_sets for feature in row})
    if feature_names:
        feature_flags = pd.DataFrame(
            {
                f"feature__{feature}": [feature in row for row in feature_sets]
                for feature in feature_names
            },
            index=out.index,
        )
        out = pd.concat([out.copy(), feature_flags], axis=1)
    return sanitize_results_for_storage(out)


def parse_feature_sets(values: pd.Series, length: int) -> list[set[str]]:
    """Parse selected feature JSON strings into sets for wide feature-use columns."""

    if values is None or len(values) == 0:
        return [set() for _ in range(length)]
    out: list[set[str]] = []
    for text in values.fillna("[]").astype(str):
        try:
            parsed = json.loads(text) if text.strip() else []
        except json.JSONDecodeError:
            parsed = []
        out.append({str(value) for value in parsed})
    return out


def write_compressed_results(frame: pd.DataFrame, base_path: Path) -> None:
    """Write the reproducible results table in compressed formats."""

    if frame.empty:
        frame.to_csv(base_path.with_suffix(".csv.gz"), index=False, compression="gzip")
        return
    frame = sanitize_results_for_storage(frame)
    try:
        frame.to_parquet(base_path.with_suffix(".parquet"), index=False, compression="zstd")
    except Exception:
        frame.to_parquet(base_path.with_suffix(".parquet"), index=False, compression="gzip")
    frame.to_csv(base_path.with_suffix(".csv.gz"), index=False, compression="gzip")


def sanitize_results_for_storage(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed object columns so parquet can store the results table."""

    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].where(out[col].notna(), "").astype(str)
    return out


def read_search_artifact_config(output_root: Path) -> dict[str, Any]:
    """Read artifact cleanup settings captured in search manifest metadata."""

    manifest_path = output_root / "search_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        metadata = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}
    return dict(metadata.get("artifacts", {}) or {})


def save_surface_plots(frame: pd.DataFrame, figures_dir: Path) -> None:
    """Save per-model metric surfaces for the first two varied parameters."""

    metrics = ["season_winner_chalk_accuracy", "season_winner_upset_recall", "tuning_score"]
    expanded = expand_params(frame)
    param_cols = [col for col in expanded.columns if col.startswith("param__")]
    for (family, model), group in expanded.groupby(["family", "model"]):
        varied = [col for col in param_cols if group[col].nunique(dropna=True) > 1]
        if len(varied) < 2:
            continue
        x_col, y_col = varied[:2]
        for metric in metrics:
            if metric not in group.columns:
                continue
            plot_surface_like(
                group,
                x_col=x_col,
                y_col=y_col,
                z_col=metric,
                path=figures_dir / f"{family}_{model}_{metric}_surface.png",
            )


def plot_surface_like(frame: pd.DataFrame, *, x_col: str, y_col: str, z_col: str, path: Path) -> None:
    data = frame[[x_col, y_col, z_col]].copy()
    for col in [x_col, y_col, z_col]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()
    if len(data) < 3:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.tricontourf(data[x_col], data[y_col], data[z_col], levels=12, cmap="viridis")
    ax.scatter(data[x_col], data[y_col], c="white", s=12, edgecolors="black", linewidths=0.25)
    ax.set_xlabel(x_col.replace("param__", ""))
    ax.set_ylabel(y_col.replace("param__", ""))
    ax.set_title(z_col)
    fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_global_pool_plot(pool: pd.DataFrame, path: Path) -> None:
    """Plot pooled best-model score by fingerprint and truncation policy."""

    if pool.empty:
        return
    data = pool.copy()
    data["top_k_order"] = data["top_k_features"].map(top_k_order)
    pivot = data.pivot_table(
        index="fingerprint",
        columns="top_k_order",
        values="tuning_score",
        aggfunc="mean",
    )
    if pivot.empty:
        return
    labels = (
        data.sort_values("top_k_order")
        .drop_duplicates("top_k_order")
        .set_index("top_k_order")["top_k_features"]
        .reindex(pivot.columns)
        .astype(str)
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=labels)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index.astype(str).tolist())
    ax.set_xlabel("Top-k fingerprint features")
    ax.set_ylabel("Fingerprint")
    ax.set_title("Best tuned model pool score")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_k_order(value: object) -> float:
    text = str(value).strip().lower()
    if text == "all":
        return float("inf")
    return float(pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0])


def expand_params(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    params = out.get("params_json", pd.Series(["{}"] * len(out))).fillna("{}")
    parsed = [json.loads(text) if str(text).strip() else {} for text in params.astype(str)]
    keys = sorted({key for row in parsed for key in row})
    for key in keys:
        out[f"param__{key}"] = [row.get(key) for row in parsed]
    return out


def score_row(row: dict[str, Any], weights: dict[str, float]) -> float:
    score = 0.0
    total = 0.0
    for key, weight in dict(weights).items():
        value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        if pd.notna(value):
            score += float(weight) * float(value)
            total += abs(float(weight))
    return score / total if total else float("nan")


def always_keep_columns(frame: pd.DataFrame) -> list[str]:
    exact = set(FINGERPRINT_KEY_COLUMNS) | set(LABEL_COLUMNS) | set(NEXT_GAME_COLUMNS)
    exact |= set(MARKET_CONTEXT_KEY_COLUMNS)
    return [
        col
        for col in frame.columns
        if col in exact
        or str(col).startswith("keys_")
        or str(col).startswith("market_")
        or str(col).startswith("game_")
    ]


def count_trainable_features(frame: pd.DataFrame) -> int:
    return sum(is_trainable_feature(frame, col) for col in frame.columns)


def is_trainable_feature(frame: pd.DataFrame, col: str) -> bool:
    return (
        col in frame.columns
        and is_feature_column(str(col))
        and (pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]))
    )


def match_ranked_features(names: list[str], columns) -> list[str]:
    column_set = {str(col) for col in columns}
    out = []
    for name in names:
        if name in column_set:
            out.append(name)
            continue
        suffix_matches = [col for col in column_set if col.endswith(name)]
        out.extend(sorted(suffix_matches))
    return list(dict.fromkeys(out))


def strip_matchup_prefix(name: str) -> str:
    for prefix in ("home_", "away_", "net_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def first_row(frame: pd.DataFrame) -> dict[str, Any]:
    return frame.iloc[0].to_dict() if frame is not None and not frame.empty else {}


def read_many_csv(paths) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(p) for p in paths):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        return yaml.safe_load(handle) or {}


def set_nested(target: dict[str, Any], parts: list[str], value: Any) -> None:
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def safe_label(value: str) -> str:
    return str(value).replace(".", "_").replace("/", "_").replace(" ", "_")


def stable_seed(seed: int, text: str) -> int:
    return int(seed) + sum(ord(ch) for ch in str(text))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def job_keys(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "job_index",
        "sge_task_id",
        "fingerprint",
        "model",
        "family",
        "model_config_path",
        "top_k_features",
        "trial_index",
        "fingerprint_path",
        "output_dir",
    ]
    return {key: row.get(key) for key in keys}


def write_readme(output_root: Path, cfg: dict[str, Any], job_count: int) -> None:
    output_root.joinpath("README.md").write_text(
        "\n".join(
            [
                "# Fingerprint Hyperparameter Search",
                "",
                f"Jobs: {job_count}",
                "",
                "This study searches per-model hyperparameters across opponent-adjusted",
                "fingerprint variants and top-k feature truncation policies. The merge",
                "step writes `best_by_model.csv`, the global best-model pool table, and",
                "surface-style contour plots for the first two varied parameters per model.",
                "",
                f"Score weights: `{json.dumps(cfg.get('score_weights', DEFAULT_SCORE_WEIGHTS), sort_keys=True)}`",
                "",
            ]
        )
    )
