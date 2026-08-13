"""Strict out-of-fold historical-matchup KNN runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from gridiron_ml.models.td_knn import TDKNN


def _first(frame: pd.DataFrame, names: Iterable[str], default=None):
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(default, index=frame.index)


def _fold_time(frame: pd.DataFrame) -> pd.Series:
    season = pd.to_numeric(_first(frame, ["keys_season", "season"]), errors="coerce")
    week = pd.to_numeric(_first(frame, ["keys_week", "week"]), errors="coerce")
    if season.isna().any() or week.isna().any():
        raise ValueError("Strict OOF KNN requires season and week metadata for every row.")
    return season.astype(int) * 100 + week.astype(int)


def run_strict_oof_knn(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    folds: Iterable[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit KNN only on each fold's training rows and return predictions/audit.

    A fold must provide ``train_indices`` and ``test_indices`` as integer row
    positions. Strict temporal folds additionally require every training row to
    precede every scored row by the season/week key. The target game IDs are
    checked for disjointness before fitting.
    """
    missing = (set(feature_columns) | {target_column}) - set(frame.columns)
    if missing:
        raise ValueError(f"OOF KNN input missing columns: {sorted(missing)}")
    frame = frame.reset_index(drop=True)
    times = _fold_time(frame)
    game_ids = _first(frame, ["keys_game_id", "game_id"], default=np.arange(len(frame))).astype(str)
    meta_columns = [c for c in ["keys_game_id", "keys_season", "keys_week", "keys_team_home", "keys_team_away", "game_id", "season", "week", "home_team", "away_team"] if c in frame]
    outputs = []
    audits = []
    for fold_number, fold in enumerate(folds):
        train_idx = np.asarray(fold["train_indices"], dtype=int)
        test_idx = np.asarray(fold["test_indices"], dtype=int)
        if len(np.intersect1d(train_idx, test_idx)):
            raise ValueError(f"Fold {fold_number} has overlapping training and test rows.")
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"Fold {fold_number} has an empty training or test split.")
        if int(times.iloc[train_idx].max()) >= int(times.iloc[test_idx].min()):
            raise ValueError(f"Fold {fold_number} is not strictly temporal.")
        if set(game_ids.iloc[train_idx]) & set(game_ids.iloc[test_idx]):
            raise ValueError(f"Fold {fold_number} repeats a target game in training neighbors.")
        model_config = dict(config or {})
        model_config.setdefault("model_type", "distance")
        model = TDKNN(model_config)
        model.train(
            frame.iloc[train_idx][feature_columns],
            frame.iloc[train_idx][target_column],
            X_val=frame.iloc[test_idx][feature_columns],
            y_val=frame.iloc[test_idx][target_column],
            meta_train=frame.iloc[train_idx][meta_columns],
            meta_val=frame.iloc[test_idx][meta_columns],
        )
        predictions = model.predict(
            frame.iloc[test_idx][feature_columns],
            meta_df=frame.iloc[test_idx][meta_columns],
        )
        predictions.insert(0, "fold", int(fold_number))
        predictions["row_index"] = test_idx
        predictions["actual_margin"] = frame.iloc[test_idx][target_column].to_numpy()
        outputs.append(predictions)
        audit = model.neighbor_audit_.copy()
        audit.insert(0, "fold", int(fold_number))
        audit = audit.rename(columns={
            "game_id": "neighbor_game_id",
            "season": "neighbor_season",
            "week": "neighbor_week",
            "home_team": "neighbor_home_team",
            "away_team": "neighbor_away_team",
            "actual_margin": "neighbor_actual_margin",
        })
        prediction_rows = audit["prediction_row"].astype(int).to_numpy()
        target_rows = test_idx[prediction_rows]
        audit["target_row_index"] = target_rows
        audit["target_game_id"] = game_ids.iloc[target_rows].to_numpy()
        audit["target_season"] = times.iloc[target_rows].to_numpy() // 100
        audit["target_week"] = times.iloc[target_rows].to_numpy() % 100
        audits.append(audit)
    if not outputs:
        raise ValueError("No OOF folds supplied.")
    predictions = pd.concat(outputs, ignore_index=True).sort_values("row_index").reset_index(drop=True)
    audit = pd.concat(audits, ignore_index=True)
    if predictions["row_index"].duplicated().any():
        raise ValueError("OOF KNN produced duplicate predictions for a row.")
    return predictions, audit
