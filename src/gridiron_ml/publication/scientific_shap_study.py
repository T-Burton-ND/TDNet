"""Cross-fitted, source-coordinate SHAP for the scientific F6 roster."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from gridiron_ml.experiments.hyperparameter_search import build_tuned_model_config
from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints
from gridiron_ml.experiments.publication import (
    atomic_write_frame,
    atomic_write_json,
    filter_frame_for_feature_config,
    materialize_split_rows,
)
from gridiron_ml.models import build_model_from_config
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder


def build_shap_manifest(*, project_root: Path, config_path: Path, output_root: Path) -> pd.DataFrame:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    scope = cfg["scope"]
    selected = pd.read_parquet(scope["hps_finalists"])
    seed_tier = str(scope.get("provisional_parameter_source_tier", scope["fingerprint"]))
    selected = selected.loc[selected["feature_config"].astype(str).eq(seed_tier)].copy()
    keys = ["objective", "model_level"]
    if selected.duplicated(keys).any():
        raise ValueError(f"HPS selection has duplicate {keys} rows for {seed_tier}.")
    selected = selected.set_index(keys)

    folds = materialize_split_rows(project_root / cfg["splits"]["config"])
    rows = []
    for fold in folds:
        for objective in scope["objectives"]:
            for level in scope["model_levels"]:
                key = (str(objective), str(level))
                if key not in selected.index:
                    raise ValueError(f"Missing provisional HPS parameters for {key} at {seed_tier}.")
                chosen = selected.loc[key]
                task_id = len(rows)
                rows.append(
                    {
                        "task_id": task_id,
                        "objective": str(objective),
                        "model_level": str(level),
                        "model_family": str(scope["model_families"][level]),
                        "model_config": str((project_root / scope["model_configs"][level]).resolve()),
                        "feature_config": str(scope["fingerprint"]),
                        "parameter_source_tier": seed_tier,
                        "params_json": str(chosen["params_json"]),
                        "outer_fold": int(fold["outer_fold"]),
                        "train_seasons_json": json.dumps(fold["train_seasons"]),
                        "val_seasons_json": json.dumps(fold["val_seasons"]),
                        "test_seasons_json": json.dumps(fold["test_seasons"]),
                        "seed": int(cfg["sampling"]["deterministic_seed"] + task_id),
                        "data_path": str(Path(scope["data"])),
                        "output_path": str((output_root / "runs" / f"task_{task_id:04d}").resolve()),
                    }
                )
    manifest = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_frame(manifest, output_root / "job_manifest.parquet")
    manifest.to_csv(output_root / "job_manifest.csv", index=False)
    atomic_write_json(
        output_root / "manifest_report.json",
        {
            "study_id": cfg["study_id"],
            "status": "provisional_corrected_f6_parameters_seeded_from_f5",
            "task_count": int(len(manifest)),
            "expected_task_count": int(len(folds) * len(scope["objectives"]) * len(scope["model_levels"])),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return manifest


def run_shap_task(*, manifest_path: Path, config_path: Path, task_id: int, force: bool = False) -> dict:
    manifest = pd.read_parquet(manifest_path)
    match = manifest.loc[manifest["task_id"].eq(int(task_id))]
    if len(match) != 1:
        raise ValueError(f"Expected one manifest row for task_id={task_id}; found {len(match)}.")
    row = match.iloc[0]
    out = Path(row["output_path"])
    result_path = out / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    try:
        frame = pd.read_parquet(row["data_path"])
        frame = frame.loc[pd.to_numeric(frame["keys_season"], errors="coerce").le(2025)].copy()
        frame, feature_meta = filter_frame_for_feature_config(
            frame,
            feature_config=str(row["feature_config"]),
            registry_path=config_path.parents[2] / "configs/features/feature_registry.yaml",
            ladders_path=config_path.parents[2] / "configs/features/feature_ladders.yaml",
            strict_registry=True,
        )
        source_features = list(feature_meta["selected_features"])
        if len(source_features) != int(cfg["scope"]["expected_source_feature_count"]):
            raise ValueError(f"Expected 227 F6 sources; resolved {len(source_features)}.")

        model_cfg = build_tuned_model_config(
            base_config_path=Path(row["model_config"]),
            params=json.loads(row["params_json"]),
        )
        model_cfg["seed"] = int(row["seed"])
        model_cfg["loss_function"] = "WinnerAccuracy" if row["objective"] == "winner" else "MAE"
        if isinstance(model_cfg.get("params"), dict) and "n_jobs" in model_cfg["params"]:
            model_cfg["params"]["n_jobs"] = 1
        model = build_model_from_config({"family": row["model_family"], **model_cfg})
        builder = MatchupBuilder(representation="unit_matchup")
        fingerprints = StaticFrameFingerprints(frame)
        evaluator = TDEval(
            config={"model": {"family": row["model_family"]}},
            fingerprints=fingerprints,
            matchup_builder=builder,
            model=model,
        )
        train_years = json.loads(row["train_seasons_json"])
        val_years = json.loads(row["val_seasons_json"])
        test_years = json.loads(row["test_seasons_json"])
        evaluator.train(train_years=train_years, val_years=val_years)
        if len(builder.feature_names()) != 681:
            raise ValueError(f"Corrected F6 must fit 681 matchup coordinates; got {len(builder.feature_names())}.")

        train_pairs = paired_source_frame(fingerprints, builder, train_years, source_features)
        test_pairs = paired_source_frame(fingerprints, builder, test_years, source_features)
        sampling = cfg["sampling"]
        background = sample_rows(train_pairs, int(sampling["background_rows_per_fold"]), int(row["seed"]))
        explain = sample_rows(test_pairs, int(sampling["explained_rows_per_fold"]), int(row["seed"]) + 1)
        values, base_values, predictions = permutation_shap(
            model=model,
            builder=builder,
            background=background,
            explain=explain,
            source_features=source_features,
            objective=str(row["objective"]),
        )
        importance = aggregate_source_importance(values, source_features)
        importance.insert(0, "outer_fold", int(row["outer_fold"]))
        importance.insert(0, "model_level", str(row["model_level"]))
        importance.insert(0, "objective", str(row["objective"]))
        atomic_write_frame(importance, out / "source_importance.parquet")
        importance.to_csv(out / "source_importance.csv", index=False)
        residual = predictions - (np.asarray(base_values).reshape(-1) + values.sum(axis=1))
        result = {
            "status": "success",
            "task_id": int(row["task_id"]),
            "objective": str(row["objective"]),
            "model_level": str(row["model_level"]),
            "outer_fold": int(row["outer_fold"]),
            "parameter_source_tier": str(row["parameter_source_tier"]),
            "source_feature_count": len(source_features),
            "matchup_feature_count": len(builder.feature_names()),
            "n_background": len(background),
            "n_explained": len(explain),
            "explainer_method": "end_to_end_permutation",
            "output_scale": "home_win_probability" if row["objective"] == "winner" else "margin_points",
            "max_abs_additivity_error": float(np.nanmax(np.abs(residual))),
            "runtime_seconds": float(time.monotonic() - started),
        }
    except Exception as exc:
        import traceback
        result = {
            "status": "failed",
            "task_id": int(row["task_id"]),
            "objective": str(row["objective"]),
            "model_level": str(row["model_level"]),
            "outer_fold": int(row["outer_fold"]),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime_seconds": float(time.monotonic() - started),
        }
    atomic_write_json(result_path, result)
    return result


def paired_source_frame(fingerprints, builder, years, source_features) -> pd.DataFrame:
    X, y, meta, market = fingerprints.training_block(years)
    missing = sorted(set(source_features) - set(X.columns))
    if missing:
        raise ValueError(f"F6 source predictors missing from model input: {missing}.")
    home, away, _, _, _ = builder._pair_game_rows(X[source_features], meta, market_df=market, y=y)
    home = home.reset_index(drop=True).add_prefix("home__")
    away = away.reset_index(drop=True).add_prefix("away__")
    return pd.concat([home, away], axis=1)


def sample_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.reset_index(drop=True)
    return frame.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)


def permutation_shap(*, model, builder, background, explain, source_features, objective):
    import shap

    count = len(source_features)
    columns = list(background.columns)

    def predict(values):
        values = np.asarray(values, dtype=float)
        home = pd.DataFrame(values[:, :count], columns=source_features)
        away = pd.DataFrame(values[:, count:], columns=source_features)
        matchup = builder.build_many(home, away)
        predicted = model.predict(matchup)
        column = "pred_proba_home_win" if objective == "winner" else "pred_margin"
        return pd.to_numeric(predicted[column], errors="coerce").to_numpy(dtype=float)

    masker = shap.maskers.Independent(background.to_numpy(dtype=float), max_samples=len(background))
    explainer = shap.Explainer(predict, masker, algorithm="permutation", feature_names=columns)
    explanation = explainer(
        explain.to_numpy(dtype=float),
        max_evals=2 * explain.shape[1] + 1,
        silent=True,
    )
    return (
        np.asarray(explanation.values, dtype=float),
        np.asarray(explanation.base_values, dtype=float),
        predict(explain.to_numpy(dtype=float)),
    )


def aggregate_source_importance(values: np.ndarray, source_features: list[str]) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    count = len(source_features)
    if values.ndim != 2 or values.shape[1] != 2 * count:
        raise ValueError(f"Expected SHAP matrix with {2 * count} columns; got {values.shape}.")
    mean_abs = np.nanmean(np.abs(values[:, :count]), axis=0) + np.nanmean(
        np.abs(values[:, count:]), axis=0
    )
    total = float(np.nansum(mean_abs))
    return pd.DataFrame(
        {
            "source_feature": source_features,
            "mean_abs_shap": mean_abs,
            "normalized_importance": mean_abs / total if total > 0 else np.zeros(count),
        }
    )


def finalize_shap(*, manifest_path: Path, output_root: Path) -> dict:
    manifest = pd.read_parquet(manifest_path)
    results = []
    importance = []
    for row in manifest.itertuples(index=False):
        out = Path(row.output_path)
        result_path = out / "result.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        table_path = out / "source_importance.parquet"
        if table_path.exists():
            importance.append(pd.read_parquet(table_path))
    result_frame = pd.DataFrame(results)
    importance_frame = pd.concat(importance, ignore_index=True) if importance else pd.DataFrame()
    summary = output_root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    atomic_write_frame(result_frame, summary / "task_status.parquet")
    atomic_write_frame(importance_frame, summary / "source_importance.parquet")
    if not importance_frame.empty:
        consensus = (
            importance_frame.groupby(["objective", "source_feature"], as_index=False)
            .agg(
                median_normalized_importance=("normalized_importance", "median"),
                mean_normalized_importance=("normalized_importance", "mean"),
                valid_cells=("normalized_importance", "count"),
            )
        )
        consensus["consensus_rank"] = consensus.groupby("objective")["median_normalized_importance"].rank(
            method="first", ascending=False
        ).astype(int)
        atomic_write_frame(consensus, summary / "source_consensus.parquet")
        consensus.to_csv(summary / "source_consensus.csv", index=False)
    report = {
        "manifest_tasks": int(len(manifest)),
        "completed_tasks": int(len(result_frame)),
        "successful_tasks": int(result_frame.get("status", pd.Series(dtype=str)).eq("success").sum()),
        "failed_tasks": int(result_frame.get("status", pd.Series(dtype=str)).eq("failed").sum()),
        "importance_rows": int(len(importance_frame)),
    }
    atomic_write_json(summary / "coverage.json", report)
    return report
