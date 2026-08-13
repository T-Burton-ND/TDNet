"""Shared feature, key, label, and leakage-sensitive column contracts."""

from __future__ import annotations

from collections.abc import Iterable

KEY_PREFIX = "keys_"
MARKET_PREFIX = "market_"
LABEL_PREFIX = "y_"
GAME_PREFIX = "game_"
COACH_PREFIX = "coach_"
COACH_SEASON_PREFIX = "coach_season_"

DEFAULT_TRAINING_TARGET = "y_next_margin"
SAME_WEEK_TARGET = "y_margin_this_week"
HAS_NEXT_GAME_COLUMN = "y_has_next_game"

KEY_COLUMNS = (
    "keys_season",
    "keys_team",
    "keys_week",
)
FINGERPRINT_KEY_COLUMNS = KEY_COLUMNS
GAME_KEY_COLUMNS = (
    "keys_season",
    "keys_week",
    "keys_game_id",
    "keys_team",
    "keys_opponent",
)
MARKET_CONTEXT_KEY_COLUMNS = (
    "keys_season",
    "keys_week",
    "keys_game_id",
    "keys_team",
    "keys_opponent",
    "next_week",
    "next_game_id",
    "next_opponent",
    "next_game_is_home",
)
LABEL_COLUMNS = (
    "keys_season",
    "keys_team",
    "keys_week",
    SAME_WEEK_TARGET,
    DEFAULT_TRAINING_TARGET,
    HAS_NEXT_GAME_COLUMN,
)
LABEL_VALUE_COLUMNS = (
    SAME_WEEK_TARGET,
    DEFAULT_TRAINING_TARGET,
    HAS_NEXT_GAME_COLUMN,
)
TARGET_COLUMNS = (
    "target_points_for",
    "target_points_against",
    "target_team_margin",
)
TARGET_AVG_COLUMNS = (
    "target_points_for_avg",
    "target_points_against_avg",
)
MARKET_COLUMNS = (
    "market_over_under",
    "market_spread_close",
    "market_spread_open",
    "market_win_probability",
)
NEXT_GAME_COLUMNS = (
    "next_game_id",
    "next_opponent",
    "next_game_is_home",
    "next_game_home_away",
    "next_week",
)
BLOCKED_COACH_FEATURES = frozenset(
    {
        "coach_career_mean_postseason_rank_points",
    }
)
POSTGAME_FEATURE_MARKERS = (
    "offense_",
    "defense_",
    "statOff_",
    "statDef_",
    "statGen_",
    "statSpe_",
    *TARGET_AVG_COLUMNS,
    "games_played",
)


def is_key_column(col: object) -> bool:
    return str(col).startswith(KEY_PREFIX)


def is_market_column(col: object) -> bool:
    name = str(col)
    return name.startswith(MARKET_PREFIX) or "_market_" in name


def is_label_column(col: object) -> bool:
    name = str(col)
    return name in {SAME_WEEK_TARGET, DEFAULT_TRAINING_TARGET, HAS_NEXT_GAME_COLUMN} or name.startswith(LABEL_PREFIX)


def is_coach_column(col: object) -> bool:
    return str(col).startswith(COACH_PREFIX)


def is_blocked_coach_column(col: object) -> bool:
    name = str(col)
    lower = name.lower()
    return (
        name in BLOCKED_COACH_FEATURES
        or name.startswith(COACH_SEASON_PREFIX)
        or (name.startswith(COACH_PREFIX) and "postseason" in lower)
    )


def is_feature_column(col: object) -> bool:
    name = str(col)
    return not (
        is_key_column(name)
        or is_market_column(name)
        or is_label_column(name)
        or name.startswith(GAME_PREFIX)
    )


def market_feature_columns(columns: Iterable[object] | None) -> list[str]:
    if columns is None:
        return []
    return [str(col) for col in columns if is_market_column(col)]


def blocked_coach_columns(columns: Iterable[object] | None) -> list[str]:
    if columns is None:
        return []
    return [str(col) for col in columns if is_blocked_coach_column(col)]


def blocked_training_columns(columns: Iterable[object] | None) -> list[str]:
    if columns is None:
        return []
    return [str(col) for col in columns if is_market_column(col) or is_blocked_coach_column(col)]
