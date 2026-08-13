"""Leakage-safe time-dependent fingerprint expansion."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TEMPORAL_COLUMNS = (
    "offense_ppa", "offense_success_rate", "offense_explosiveness",
    "defense_ppa", "defense_success_rate", "defense_explosiveness",
    "statOff_yards_per_rush_attempt", "statOff_yards_per_pass",
    "statOff_third_down_rate", "statDef_passes_intercepted",
    "statGen_turnovers", "statGen_penalties", "games_played",
)


def build_temporal_fingerprints(frame: pd.DataFrame, *, columns=None, half_life=3.0, trend_lag=3) -> pd.DataFrame:
    """Add lagged, exponentially decayed, and trend states by team-season.

    Every derived value uses ``shift(1)`` before smoothing/differencing. Thus a
    row cannot consume its own state or anything after its season/week cutoff.
    """
    required = {"keys_team", "keys_season", "keys_week"}
    if not required.issubset(frame):
        raise ValueError(f"Temporal fingerprints require {sorted(required)}")
    out = frame.copy()
    selected = [c for c in (columns or DEFAULT_TEMPORAL_COLUMNS) if c in out and pd.api.types.is_numeric_dtype(out[c])]
    ordered = out.sort_values(["keys_team", "keys_season", "keys_week", "keys_game_id" if "keys_game_id" in out else "keys_week"])
    groups = ordered.groupby(["keys_team", "keys_season"], sort=False)
    alpha = 1.0 - np.exp(np.log(0.5) / float(half_life))
    derived = {}
    for column in selected:
        lagged = groups[column].shift(1)
        derived[f"time_adj_lag1__{column}"] = lagged
        derived[f"time_adj_ewm__{column}"] = lagged.groupby(
            [ordered["keys_team"], ordered["keys_season"]], sort=False
        ).transform(lambda value: value.ewm(alpha=alpha, adjust=False, min_periods=1).mean())
        derived[f"time_adj_trend{int(trend_lag)}__{column}"] = lagged - groups[column].shift(1 + int(trend_lag))
    enriched = pd.concat([ordered, pd.DataFrame(derived, index=ordered.index)], axis=1)
    return enriched.sort_index()
