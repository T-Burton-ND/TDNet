"""Leakage-safe calibration and paired inference for publication tables.

This module contains small deterministic building blocks rather than a hidden
analysis pipeline. Callers must supply predictions from training-only/OOF
fits, and every returned result records the sample and cluster counts needed
to audit the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


@dataclass(frozen=True)
class MarginCalibrator:
    """A logistic map from predicted margin to home-win probability."""

    intercept: float
    slope: float
    fit_rows: int
    fit_hash: str | None = None

    def predict(self, margins: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(margins, dtype=float)
        logits = np.clip(self.intercept + self.slope * values, -35.0, 35.0)
        return np.clip(1.0 / (1.0 + np.exp(-logits)), 0.0, 1.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "logistic_regression_on_predicted_margin",
            "intercept": float(self.intercept),
            "slope": float(self.slope),
            "fit_rows": int(self.fit_rows),
            "fit_hash": self.fit_hash,
        }


def margin_to_probability(margins: Sequence[float] | np.ndarray, *, scale: float = 10.0) -> np.ndarray:
    """Convert margins to an auditable uncalibrated logistic probability."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Probability-link scale must be positive and finite.")
    values = np.asarray(margins, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Margins contain non-finite values.")
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(values / scale, -35.0, 35.0))), 0.0, 1.0)


def fit_margin_calibrator(
    margins: Sequence[float] | np.ndarray,
    actual_home_win: Sequence[int] | np.ndarray,
    *,
    fit_hash: str | None = None,
) -> MarginCalibrator:
    """Fit a calibrator on OOF/development rows only.

    The function deliberately has no evaluation-data argument. A caller that
    passes held-out rows here is violating the protocol and should be caught
    by its surrounding fold construction.
    """
    x = np.asarray(margins, dtype=float).reshape(-1, 1)
    y = np.asarray(actual_home_win, dtype=int).reshape(-1)
    valid = np.isfinite(x[:, 0]) & np.isin(y, [0, 1])
    x, y = x[valid], y[valid]
    if len(y) < 2 or len(np.unique(y)) < 2:
        raise ValueError("Calibration requires at least two rows and both outcome classes.")
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    model.fit(x, y)
    return MarginCalibrator(
        intercept=float(model.intercept_[0]),
        slope=float(model.coef_[0, 0]),
        fit_rows=int(len(y)),
        fit_hash=fit_hash,
    )


def temporal_cross_fitted_margin_calibration(
    frame: pd.DataFrame,
    *,
    season_column: str = "season",
    margin_column: str = "pred_margin",
    actual_margin_column: str = "actual_margin",
) -> pd.DataFrame:
    """Calibrate each season using only OOF rows from earlier seasons.

    These predictions estimate calibrator performance without evaluating a
    calibration map on the same games used to fit it. The first season is
    necessarily omitted because no earlier OOF calibration sample exists.
    """
    required = {season_column, margin_column, actual_margin_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Temporal calibration frame missing {missing}")
    work = frame.copy()
    for column in (season_column, margin_column, actual_margin_column):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna().sort_values([season_column], kind="stable")
    seasons = sorted(work[season_column].astype(int).unique().tolist())
    predictions: list[pd.DataFrame] = []
    for season in seasons[1:]:
        training = work.loc[work[season_column].lt(season)]
        evaluation = work.loc[work[season_column].eq(season)].copy()
        outcomes = training[actual_margin_column].gt(0).astype(int)
        if evaluation.empty or len(training) < 2 or outcomes.nunique() < 2:
            raise ValueError(f"insufficient prior OOF data to validate calibration for {season}")
        calibrator = fit_margin_calibrator(training[margin_column], outcomes)
        evaluation["actual_home_win"] = evaluation[actual_margin_column].gt(0).astype(int)
        evaluation["calibrated_probability_home"] = calibrator.predict(evaluation[margin_column])
        evaluation["calibration_train_through"] = season - 1
        evaluation["calibration_train_rows"] = len(training)
        predictions.append(evaluation)
    if not predictions:
        raise ValueError("Temporal calibration requires at least two seasons")
    return pd.concat(predictions, ignore_index=True)


def _safe_logit(probability: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    return np.log(values / (1.0 - values))


def calibration_summary(
    actual_home_win: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, object]:
    """Return calibration, scoring, sharpness, and reliability-bin metrics."""
    y = np.asarray(actual_home_win, dtype=int)
    p = np.asarray(probability, dtype=float)
    valid = np.isin(y, [0, 1]) & np.isfinite(p)
    y, p = y[valid], np.clip(p[valid], 0.0, 1.0)
    if len(y) == 0:
        raise ValueError("Calibration summary has no valid rows.")
    brier = float(np.mean((p - y) ** 2))
    loss = float(log_loss(y, np.clip(p, 1e-7, 1.0 - 1e-7), labels=[0, 1]))
    predicted = p >= 0.5
    accuracy = float(np.mean(predicted == y))
    logit_p = _safe_logit(p).reshape(-1, 1)
    if len(np.unique(y)) > 1 and len(np.unique(logit_p)) > 1:
        slope_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(logit_p, y)
        intercept = float(slope_model.intercept_[0])
        slope = float(slope_model.coef_[0, 0])
    else:
        intercept = float("nan")
        slope = float("nan")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bin_index = np.minimum(np.digitize(p, edges[1:-1], right=False), int(bins) - 1)
    reliability = []
    ece = 0.0
    for index in range(int(bins)):
        mask = bin_index == index
        count = int(mask.sum())
        if not count:
            continue
        mean_predicted = float(p[mask].mean())
        observed = float(y[mask].mean())
        ece += count / len(y) * abs(mean_predicted - observed)
        reliability.append({
            "bin": index,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "mean_predicted": mean_predicted,
            "observed_rate": observed,
        })
    return {
        "n": int(len(y)),
        "brier_score": brier,
        "log_loss": loss,
        "winner_accuracy": accuracy,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": float(ece),
        "sharpness_mean": float(p.mean()),
        "sharpness_std": float(p.std(ddof=0)),
        "reliability_bins": reliability,
        "probability_min": float(p.min()),
        "probability_max": float(p.max()),
    }


def season_clustered_bootstrap(
    differences: pd.DataFrame | Sequence[float],
    *,
    season_column: str = "season",
    value_column: str = "difference",
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 5000,
    seed: int = 20260727,
) -> dict[str, object]:
    """Bootstrap paired differences by season, preserving within-season rows."""
    if isinstance(differences, pd.DataFrame):
        required = {season_column, value_column}
        missing = required - set(differences.columns)
        if missing:
            raise ValueError(f"Bootstrap frame missing {sorted(missing)}")
        frame = differences[[season_column, value_column]].dropna().copy()
    else:
        values = np.asarray(differences, dtype=float)
        frame = pd.DataFrame({season_column: 0, value_column: values}).dropna()
    if frame.empty:
        raise ValueError("Bootstrap has no valid differences.")
    clusters = [group[value_column].to_numpy(dtype=float) for _, group in frame.groupby(season_column, sort=True)]
    if not clusters:
        raise ValueError("Bootstrap has no clusters.")
    observed = float(statistic(frame[value_column].to_numpy(dtype=float)))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_resamples), dtype=float)
    for index in range(int(n_resamples)):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        values = np.concatenate([clusters[cluster] for cluster in sampled])
        draws[index] = float(statistic(values))
    return {
        "estimate": observed,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_games": int(len(frame)),
        "n_seasons": int(len(clusters)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "cluster": season_column,
    }


def season_clustered_mean_bootstrap(
    frame: pd.DataFrame,
    *,
    season_column: str = "season",
    value_column: str = "difference",
    n_resamples: int = 5000,
    seed: int = 20260727,
) -> dict[str, object]:
    """Vectorized season-clustered bootstrap confidence interval for a mean."""
    required = {season_column, value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Bootstrap frame missing {sorted(missing)}")
    clean = frame[[season_column, value_column]].dropna().copy()
    if clean.empty:
        raise ValueError("Bootstrap has no valid differences.")
    grouped = clean.groupby(season_column, sort=True)[value_column]
    sums = grouped.sum().to_numpy(dtype=float)
    counts = grouped.size().to_numpy(dtype=float)
    if not len(sums):
        raise ValueError("Bootstrap has no clusters.")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(sums), size=(int(n_resamples), len(sums)))
    draws = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "estimate": float(clean[value_column].mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_games": int(len(clean)),
        "n_seasons": int(len(sums)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "cluster": season_column,
    }


def paired_metric_difference(
    frame: pd.DataFrame,
    *,
    actual_column: str,
    first_column: str,
    second_column: str,
    metric: str,
    season_column: str = "season",
) -> pd.DataFrame:
    """Create per-game paired differences for a declared metric."""
    required = {actual_column, first_column, second_column, season_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Paired comparison missing {sorted(missing)}")
    data = frame.loc[:, [season_column, actual_column, first_column, second_column]].dropna().copy()
    actual = data[actual_column].astype(float)
    first = data[first_column].astype(float)
    second = data[second_column].astype(float)
    if metric == "mae":
        data["difference"] = (first - actual).abs() - (second - actual).abs()
    elif metric == "rmse_component":
        data["difference"] = (first - actual) ** 2 - (second - actual) ** 2
    elif metric == "brier":
        data["difference"] = (first - actual) ** 2 - (second - actual) ** 2
    elif metric == "winner_correct":
        data["difference"] = (first == actual).astype(float) - (second == actual).astype(float)
    else:
        raise ValueError(f"Unsupported paired metric: {metric}")
    return data[[season_column, "difference"]]


def mcnemar_test(first_correct: Sequence[bool], second_correct: Sequence[bool]) -> dict[str, object]:
    """Exact paired winner comparison using McNemar's discordant cells."""
    first = np.asarray(first_correct, dtype=bool)
    second = np.asarray(second_correct, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("McNemar inputs must have equal shape.")
    first_only = int((first & ~second).sum())
    second_only = int((~first & second).sum())
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(first_only, second_only)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "n": int(len(first)),
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant": discordant,
        "exact_two_sided_p": float(p_value),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down familywise-error adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("P-values must lie in [0, 1].")
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def equivalence_result(estimate: float, ci_low: float, ci_high: float, bound: float) -> dict[str, object]:
    """Classify a paired interval as equivalence, superiority, or inconclusive."""
    if bound <= 0 or ci_low > ci_high:
        raise ValueError("Equivalence bound must be positive and interval ordered.")
    inside = ci_low >= -bound and ci_high <= bound
    if inside:
        decision = "practical_equivalence"
    elif ci_low > bound:
        decision = "inferior_first"
    elif ci_high < -bound:
        decision = "superior_first"
    else:
        decision = "inconclusive"
    return {
        "estimate": float(estimate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "equivalence_bound": float(bound),
        "decision": decision,
    }


def empirical_power_precision(
    differences: pd.DataFrame,
    *,
    sample_sizes: Iterable[int] = (500, 750, 1000),
    season_column: str = "season",
    value_column: str = "difference",
    effect_sizes: Iterable[float] = (0.25, 0.5, 1.0),
    n_resamples: int = 2000,
    seed: int = 20260727,
) -> pd.DataFrame:
    """Estimate clustered CI precision and directional power from paired history."""
    if differences.empty:
        raise ValueError("Power analysis requires historical paired differences.")
    rng = np.random.default_rng(seed)
    clusters = [group[value_column].to_numpy(dtype=float) for _, group in differences.groupby(season_column, sort=True)]
    rows = []
    for sample_size in map(int, sample_sizes):
        for effect in map(float, effect_sizes):
            estimates = []
            rejects = 0
            for _ in range(int(n_resamples)):
                sampled = []
                while sum(len(x) for x in sampled) < sample_size:
                    sampled.append(clusters[int(rng.integers(0, len(clusters)))])
                values = np.concatenate(sampled)[:sample_size] + effect
                estimate = float(values.mean())
                se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
                estimates.append(estimate)
                if np.isfinite(se) and estimate - 1.96 * se > 0:
                    rejects += 1
            rows.append({
                "sample_size": sample_size,
                "effect_size": effect,
                "mean_estimate": float(np.mean(estimates)),
                "mean_ci_width_approx": float(2 * 1.96 * np.std(estimates, ddof=1)),
                "superiority_power": float(rejects / int(n_resamples)),
                "n_resamples": int(n_resamples),
                "seed": int(seed),
            })
    return pd.DataFrame(rows)
