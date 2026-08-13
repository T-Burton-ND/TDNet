"""Consistent team labels for weekly and postgame publication figures."""

from __future__ import annotations

import pandas as pd


def ap_rank_for_side(game: pd.Series, side: str):
    """Return the available AP rank for ``home`` or ``away``.

    Canonical all-game tables use ``ap_rank_home``/``ap_rank_away`` while the
    Top-25 matchup subset historically used ``home_rank``/``away_rank``.
    Supporting both keeps every publication view on the same label contract.
    """
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'")
    for column in (f"ap_rank_{side}", f"{side}_rank"):
        value = game.get(column)
        if pd.notna(value):
            return int(value)
    return None


def format_team_with_ap_rank(game: pd.Series, side: str) -> str:
    """Format a team as ``#N Team`` when an AP rank is available."""
    team = str(game[f"{side}_team"])
    rank = ap_rank_for_side(game, side)
    return f"#{rank} {team}" if rank is not None else team
