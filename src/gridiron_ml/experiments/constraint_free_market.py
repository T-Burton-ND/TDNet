"""Explicit provenance and fail-closed 2026 market cutoff validation for F19."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .constraint_free import SNAPSHOT_MARKET_COLUMNS


@dataclass(frozen=True)
class MarketProvenance:
    name: str
    historical_raw_field: str
    prospective_raw_field: str
    historical_transformation: str
    prospective_transformation: str
    historical_source: str = "CFBD /lines archive, quote timestamp unavailable"
    prospective_source: str = "archived TDNet weekly pregame snapshot"
    cutoff_rule: str = "snapshot captured before declared prediction cutoff and target kickoff"
    historical_quote_time_verified: bool = False
    prospective_allowed: bool = True


MARKET_PROVENANCE = (
    MarketProvenance("market_home_spread", "lines[].spread", "market_spread_close",
                     "consensus if present, else provider median", "archived provider mean"),
    MarketProvenance("market_total", "lines[].overUnder", "market_over_under",
                     "consensus if present, else provider median", "archived provider mean"),
    MarketProvenance("market_open_spread", "lines[].spreadOpen", "market_spread_open",
                     "open from a provider with current/open pair", "archived provider mean"),
    MarketProvenance("market_spread_move", "lines[].spread - lines[].spreadOpen",
                     "market_spread_close - market_spread_open",
                     "same-provider current minus open", "difference of archived summary means"),
    MarketProvenance("market_home_implied_no_vig", "paired homeMoneyline/awayMoneyline",
                     "market_win_probability",
                     "same-provider moneyline probabilities normalized to sum one",
                     "archived mean of paired-provider no-vig probabilities"),
)
if tuple(item.name for item in MARKET_PROVENANCE) != SNAPSHOT_MARKET_COLUMNS:
    raise ValueError("F19 prospective market feature contract and provenance differ")


def provenance_manifest() -> list[dict]:
    return [asdict(item) for item in MARKET_PROVENANCE]


def validate_pregame_market_state(state: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Return availability by game; reject any market value without a valid cutoff.

    ``target`` must declare an intended prediction cutoff independent of the
    market record. The F18/F19 research what-if may declare the target kickoff
    as its latest possible cutoff, but must label that convention in its report.
    """
    required_target = {"target_game_id", "prediction_cutoff_utc", "target_start_utc"}
    required_state = {"target_game_id", "snapshot_timestamp_utc", *SNAPSHOT_MARKET_COLUMNS}
    if required_target - set(target) or required_state - set(state):
        raise ValueError("Market provenance frame lacks required columns")
    if target.target_game_id.duplicated().any() or state.target_game_id.duplicated().any():
        raise ValueError("Duplicate target or market game identity")
    merged = target[list(required_target)].merge(
        state[list(required_state)], on="target_game_id", how="left",
        validate="one_to_one", indicator=True,
    )
    if not merged._merge.eq("both").all():
        raise ValueError("A target game lacks a market availability row")
    cutoff = pd.to_datetime(merged.prediction_cutoff_utc, utc=True, errors="raise")
    kickoff = pd.to_datetime(merged.target_start_utc, utc=True, errors="raise")
    captured = pd.to_datetime(merged.snapshot_timestamp_utc, utc=True, errors="coerce")
    if cutoff.isna().any() or kickoff.isna().any() or (cutoff > kickoff).any():
        raise ValueError("Invalid declared prediction cutoff")
    market = merged[list(SNAPSHOT_MARKET_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    available = market.notna().any(axis=1)
    invalid = available & (captured.isna() | captured.ge(cutoff) | captured.ge(kickoff))
    if invalid.any():
        raise ValueError(f"Market snapshot violates cutoff for {int(invalid.sum())} game(s)")
    return pd.Series(available.to_numpy(bool), index=merged.target_game_id.to_numpy(), name="f19_available")
