"""Build the canonical manuscript table set from merged experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PUBLICATION_TABLE_NAMES = [
    "table_01_data_summary",
    "table_02_feature_tiers",
    "table_03_model_families",
    "table_04_historical_performance",
    "table_05_feature_model_matrix",
    "table_06_calibration",
    "table_07_ablation",
    "table_08_negative_controls",
    "table_09_market_incremental_value",
    "table_10_prospective_2026_performance",
    "table_11_subgroup_sensitivity",
    "table_12_compute_cost",
]


def build_publication_tables(
    *,
    output_root: str | Path,
    experiment_results: pd.DataFrame,
    feature_tiers: pd.DataFrame,
    model_families: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    ablations: pd.DataFrame | None = None,
    negative_controls: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Create all twelve canonical tables, leaving unavailable analyses empty."""
    results = pd.DataFrame(experiment_results).copy()
    _materialize_test_season(results)
    predictions = pd.DataFrame(predictions).copy() if predictions is not None else pd.DataFrame()
    tables = {
        "table_01_data_summary": _data_summary(results, predictions),
        "table_02_feature_tiers": pd.DataFrame(feature_tiers).copy(),
        "table_03_model_families": pd.DataFrame(model_families).copy(),
        "table_04_historical_performance": _historical(results),
        "table_05_feature_model_matrix": _matrix(results),
        "table_06_calibration": _calibration(predictions),
        "table_07_ablation": pd.DataFrame(ablations).copy() if ablations is not None else pd.DataFrame(),
        "table_08_negative_controls": pd.DataFrame(negative_controls).copy() if negative_controls is not None else pd.DataFrame(),
        "table_09_market_incremental_value": _market(results),
        "table_10_prospective_2026_performance": _prospective(predictions),
        "table_11_subgroup_sensitivity": _subgroups(predictions),
        "table_12_compute_cost": _compute(results),
    }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
        table.to_parquet(output / f"{name}.parquet", index=False)
    return tables


def _data_summary(results, predictions):
    seasons = set()
    if "test_season" in results:
        seasons.update(pd.to_numeric(results["test_season"], errors="coerce").dropna().astype(int))
    if not seasons and "test_seasons_json" in results:
        for raw in results["test_seasons_json"].dropna():
            try:
                seasons.update(int(value) for value in json.loads(str(raw)))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    return pd.DataFrame(
        [
            {
                "historical_trial_rows": len(results),
                "prediction_rows": len(predictions),
                "games": predictions["game_id"].nunique() if "game_id" in predictions else 0,
                "seasons": len(seasons),
            }
        ]
    )


def _historical(results):
    keys = [c for c in ["objective", "feature_config", "model_level", "test_season"] if c in results]
    metrics = [c for c in ["winner_accuracy", "brier_score", "mae", "rmse"] if c in results]
    return results.groupby(keys, as_index=False)[metrics].mean() if keys and metrics else pd.DataFrame()


def _matrix(results):
    keys = [c for c in ["objective", "feature_config", "model_level"] if c in results]
    metrics = [c for c in ["winner_accuracy", "brier_score", "mae", "rmse", "runtime_seconds"] if c in results]
    return results.groupby(keys, as_index=False)[metrics].mean() if keys and metrics else pd.DataFrame()


def _calibration(predictions):
    required = {"model_name", "pred_home_win_probability", "actual_home_win"}
    if not required.issubset(predictions):
        return pd.DataFrame()
    rows = []
    for model, group in predictions.groupby("model_name"):
        error = group["pred_home_win_probability"].astype(float) - group["actual_home_win"].astype(float)
        rows.append({"model_name": model, "rows": len(group), "brier_score": float((error**2).mean()), "mean_probability_error": float(error.mean())})
    return pd.DataFrame(rows)


def _market(results):
    if "feature_config" not in results:
        return pd.DataFrame()
    return results.loc[results["feature_config"].astype(str).isin(["F6", "F7", "F8"])].copy()


def _prospective(predictions):
    if "season" not in predictions:
        return pd.DataFrame()
    return predictions.loc[pd.to_numeric(predictions["season"], errors="coerce") == 2026].copy()


def _subgroups(predictions):
    if predictions.empty or "winner_correct" not in predictions:
        return pd.DataFrame()
    group_columns = [c for c in ["conference", "neutral_site", "ranked_game", "spread_bucket"] if c in predictions]
    frames = []
    for column in group_columns:
        summary = predictions.groupby(["model_name", column], as_index=False).agg(games=("game_id", "nunique"), winner_accuracy=("winner_correct", "mean"))
        summary.insert(1, "subgroup", column)
        summary = summary.rename(columns={column: "level"})
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _compute(results):
    columns = [c for c in ["objective", "feature_config", "model_level", "runtime_seconds", "selected_feature_count"] if c in results]
    return results.loc[:, columns].copy() if columns else pd.DataFrame()


def _materialize_test_season(results: pd.DataFrame) -> None:
    """Expose the single held-out season stored in publication result JSON."""
    if "test_season" in results or "test_seasons_json" not in results:
        return
    seasons = []
    for raw in results["test_seasons_json"]:
        try:
            values = json.loads(str(raw))
            seasons.append(values[0] if len(values) == 1 else None)
        except (TypeError, ValueError, json.JSONDecodeError, IndexError):
            seasons.append(None)
    results["test_season"] = pd.to_numeric(seasons, errors="coerce")
