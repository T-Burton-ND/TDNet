"""Leakage-safe transparent baseline predictors for the confirmatory replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineFit:
    """Parameters learned from a historical training partition."""

    name: str
    margin_mean: float = 0.0
    home_advantage: float = 0.0
    random_seed: int = 20260727


BASELINE_NAMES = (
    "random_50",
    "home_prior",
    "season_to_date_win_rate",
    "season_to_date_point_differential",
    "season_to_date_opponent_adjusted_point_differential",
    "vegas_declared_line",
)


def fit_baseline(
    train: pd.DataFrame,
    *,
    name: str,
    actual_margin_column: str = "actual_margin",
    home_win_column: str = "actual_home_win",
) -> BaselineFit:
    """Fit only the scalar quantities needed by a transparent baseline.

    Team-season summaries are expected to be precomputed using data available
    before each row.  This function never derives them from the evaluation
    partition and therefore cannot silently turn a retrospective summary into
    a feature.
    """
    if name not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline {name!r}; expected one of {BASELINE_NAMES}")
    margins = pd.to_numeric(train.get(actual_margin_column), errors="coerce")
    margin_mean = float(margins.dropna().mean()) if margins.notna().any() else 0.0
    if name == "home_prior":
        if home_win_column not in train:
            raise ValueError(f"{home_win_column!r} is required for home_prior")
        wins = pd.to_numeric(train[home_win_column], errors="coerce")
        home_advantage = margin_mean
        if wins.notna().any():
            home_advantage = margin_mean if wins.mean() >= 0.5 else -abs(margin_mean)
    else:
        home_advantage = 0.0
    return BaselineFit(name=name, margin_mean=margin_mean, home_advantage=home_advantage)


def predict_baseline(
    frame: pd.DataFrame,
    fit: BaselineFit,
    *,
    home_win_rate_column: str = "home_win_rate_to_date",
    away_win_rate_column: str = "away_win_rate_to_date",
    home_point_diff_column: str = "home_point_diff_to_date",
    away_point_diff_column: str = "away_point_diff_to_date",
    home_opponent_adjusted_column: str = "home_opp_adj_point_diff_to_date",
    away_opponent_adjusted_column: str = "away_opp_adj_point_diff_to_date",
    vegas_margin_column: str = "market_spread_close",
) -> pd.DataFrame:
    """Return margin, uncalibrated probability, and winner for one baseline."""
    out = pd.DataFrame(index=frame.index)
    if fit.name == "random_50":
        rng = np.random.default_rng(fit.random_seed)
        probability = rng.integers(0, 2, len(frame)).astype(float)
        margin = np.zeros(len(frame), dtype=float)
    elif fit.name == "home_prior":
        margin = np.full(len(frame), fit.home_advantage, dtype=float)
        probability = np.full(len(frame), 0.5 if fit.home_advantage == 0 else (1.0 if fit.home_advantage > 0 else 0.0))
    elif fit.name == "season_to_date_win_rate":
        _require(frame, home_win_rate_column, away_win_rate_column)
        probability = pd.to_numeric(frame[home_win_rate_column], errors="coerce").sub(
            pd.to_numeric(frame[away_win_rate_column], errors="coerce")
        ).add(0.5).clip(0.0, 1.0).to_numpy()
        margin = (probability - 0.5) * 28.0
    elif fit.name == "season_to_date_point_differential":
        _require(frame, home_point_diff_column, away_point_diff_column)
        margin = _difference(frame, home_point_diff_column, away_point_diff_column)
        probability = _margin_to_probability(margin)
    elif fit.name == "season_to_date_opponent_adjusted_point_differential":
        _require(frame, home_opponent_adjusted_column, away_opponent_adjusted_column)
        margin = _difference(frame, home_opponent_adjusted_column, away_opponent_adjusted_column)
        probability = _margin_to_probability(margin)
    elif fit.name == "vegas_declared_line":
        _require(frame, vegas_margin_column)
        margin = pd.to_numeric(frame[vegas_margin_column], errors="coerce").to_numpy(dtype=float)
        probability = _margin_to_probability(margin)
    else:  # pragma: no cover - fit_baseline validates names
        raise ValueError(fit.name)
    out["baseline"] = fit.name
    out["pred_margin"] = margin
    out["pred_probability_home"] = probability
    out["pred_home_win"] = probability >= 0.5
    return out


def fit_predict_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    name: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience wrapper whose fit/test split is explicit in the call."""
    fit_kwargs = {
        key: kwargs.pop(key)
        for key in ("actual_margin_column", "home_win_column")
        if key in kwargs
    }
    return predict_baseline(test, fit_baseline(train, name=name, **fit_kwargs), **kwargs)


def _require(frame: pd.DataFrame, *columns: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Baseline requires columns {missing}")


def _difference(frame: pd.DataFrame, home: str, away: str) -> np.ndarray:
    return pd.to_numeric(frame[home], errors="coerce").sub(
        pd.to_numeric(frame[away], errors="coerce")
    ).to_numpy(dtype=float)


def _margin_to_probability(margin: np.ndarray) -> np.ndarray:
    # A fixed declared logistic link keeps the baseline transparent; learned
    # calibration belongs to the fold-safe calibration layer.
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(margin, dtype=float) / 7.0, -40, 40)))
