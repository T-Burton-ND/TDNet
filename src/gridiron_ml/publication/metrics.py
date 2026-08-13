"""Canonical scoring definitions for the TDNet retrospective and production tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_predictions(frame: pd.DataFrame, *, probability="pred_probability_home", margin="pred_margin", actual_margin="actual_margin", spread="vegas_spread") -> dict[str, float | int]:
    """Score one model on game rows, with deterministic pick'em handling.

    ``vegas_spread`` is from the home-team perspective: negative means the home
    team is favored. Missing or zero spreads are pick'em and excluded only from
    chalk/upset recall denominators.
    """
    required = {probability, margin, actual_margin, spread}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction frame missing columns: {missing}")
    x = frame.copy()
    for col in required:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=[probability, margin, actual_margin])
    actual_home = x[actual_margin].gt(0)
    predicted_home = x[probability].ge(0.5)
    valid_market = x[spread].notna() & x[spread].ne(0)
    # A negative home spread denotes a home favorite; positive denotes away favorite.
    favorite_home = x[spread].lt(0)
    actual_favorite_win = (favorite_home & actual_home) | (~favorite_home & ~actual_home)
    predicted_favorite = (favorite_home & predicted_home) | (~favorite_home & ~predicted_home)
    actual_upset_win = ~actual_favorite_win
    return {
        "number_of_games": int(len(x)),
        "brier_score": float(np.mean((x[probability].clip(0, 1) - actual_home.astype(float)) ** 2)),
        "winner_accuracy": float((predicted_home == actual_home).mean()),
        "margin_mae": float(np.abs(x[margin] - x[actual_margin]).mean()),
        "chalk_recall": float((predicted_favorite[valid_market & actual_favorite_win]).mean()) if (valid_market & actual_favorite_win).any() else np.nan,
        "upset_recall": float((~predicted_favorite[valid_market & actual_upset_win]).mean()) if (valid_market & actual_upset_win).any() else np.nan,
        "number_of_actual_chalk_wins": int((valid_market & actual_favorite_win).sum()),
        "number_of_actual_upsets": int((valid_market & actual_upset_win).sum()),
        "number_of_predicted_chalk_wins": int((valid_market & predicted_favorite).sum()),
        "number_of_predicted_upsets": int((valid_market & ~predicted_favorite).sum()),
        "number_of_pickem_games_excluded": int((~valid_market).sum()),
    }


def brier_by_group(frame: pd.DataFrame, group: str, **kwargs) -> pd.DataFrame:
    """Return canonical metrics by a declared subgroup."""
    if group not in frame:
        raise ValueError(f"Unknown subgroup column: {group}")
    rows = []
    for value, part in frame.groupby(group, dropna=False, sort=True):
        rows.append({group: value, **score_predictions(part, **kwargs)})
    return pd.DataFrame(rows)


def chalk_upset_table(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return the requested discrete favorite/underdog 2×2 counts."""
    required = {kwargs.get("probability", "pred_probability_home"), kwargs.get("actual_margin", "actual_margin"), kwargs.get("spread", "vegas_spread")}
    if not required.issubset(frame.columns):
        raise ValueError(f"Prediction frame missing columns: {sorted(required - set(frame.columns))}")
    x = frame.copy()
    probability = kwargs.get("probability", "pred_probability_home")
    actual_margin = kwargs.get("actual_margin", "actual_margin")
    spread = kwargs.get("spread", "vegas_spread")
    x = x.loc[x[spread].notna() & x[spread].ne(0)].copy()
    favorite_home = pd.to_numeric(x[spread]).lt(0)
    favorite_predicted = (favorite_home & (pd.to_numeric(x[probability]) >= 0.5)) | (~favorite_home & (pd.to_numeric(x[probability]) < 0.5))
    favorite_won = (favorite_home & pd.to_numeric(x[actual_margin]).gt(0)) | (~favorite_home & pd.to_numeric(x[actual_margin]).lt(0))
    return pd.DataFrame([
        {"prediction": "favorite_predicted", "favorite_actually_wins": int((favorite_predicted & favorite_won).sum()), "underdog_actually_wins": int((favorite_predicted & ~favorite_won).sum())},
        {"prediction": "upset_predicted", "favorite_actually_wins": int((~favorite_predicted & favorite_won).sum()), "underdog_actually_wins": int((~favorite_predicted & ~favorite_won).sum())},
    ])
