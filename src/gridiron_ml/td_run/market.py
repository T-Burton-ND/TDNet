"""src.gridiron_ml.td_run.market.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Evaluate model outputs, compare predictions to market baselines, and build reporting artifacts.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VegasConvention:
    """Represent the VegasConvention component and its local behavior."""
    spread_column: str = "market_spread_close"
    open_spread_column: str = "market_spread_open"
    total_column: str = "market_over_under"
    home_margin_multiplier: float = -1.0

    def spread_to_home_margin(self, spread):
        """Run the spread_to_home_margin step and return its normalized result."""
        return pd.to_numeric(spread, errors="coerce") * float(self.home_margin_multiplier)


DEFAULT_VEGAS_CONVENTION = VegasConvention()


def normalize_vegas_frame(frame, convention=None):
    """Run the normalize_vegas_frame step and return its normalized result."""
    convention = convention or DEFAULT_VEGAS_CONVENTION
    out = frame.copy()
    if convention.spread_column in out.columns:
        out["market_home_margin_close"] = convention.spread_to_home_margin(out[convention.spread_column])
    if convention.open_spread_column in out.columns:
        out["market_home_margin_open"] = convention.spread_to_home_margin(out[convention.open_spread_column])
    if convention.total_column in out.columns:
        out["market_total"] = pd.to_numeric(out[convention.total_column], errors="coerce")
    return out


def market_home_margin(frame, convention=None):
    """Run the market_home_margin step and return its normalized result."""
    convention = convention or DEFAULT_VEGAS_CONVENTION
    if "market_home_margin_close" in frame.columns:
        return pd.to_numeric(frame["market_home_margin_close"], errors="coerce")
    if convention.spread_column in frame.columns:
        return convention.spread_to_home_margin(frame[convention.spread_column])
    if "market_home_margin_open" in frame.columns:
        return pd.to_numeric(frame["market_home_margin_open"], errors="coerce")
    if convention.open_spread_column in frame.columns:
        return convention.spread_to_home_margin(frame[convention.open_spread_column])
    return pd.Series(np.nan, index=frame.index, dtype=float)
