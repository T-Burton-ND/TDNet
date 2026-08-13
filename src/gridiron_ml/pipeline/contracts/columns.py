"""Required dataframe column groups used by validators and builders."""

from __future__ import annotations

from gridiron_ml.pipeline.contracts.features import (
    DEFAULT_TRAINING_TARGET,
    FINGERPRINT_KEY_COLUMNS,
    HAS_NEXT_GAME_COLUMN,
)

REQUIRED_TEAM_GAME_COLUMNS = (
    "keys_season",
    "keys_week",
    "keys_team",
    "keys_opponent",
    "game_is_home",
    "target_points_for",
    "target_points_against",
    "target_team_margin",
)
REQUIRED_FINGERPRINT_COLUMNS = (
    *FINGERPRINT_KEY_COLUMNS,
    DEFAULT_TRAINING_TARGET,
    HAS_NEXT_GAME_COLUMN,
)
REQUIRED_LABEL_COLUMNS = REQUIRED_FINGERPRINT_COLUMNS
REQUIRED_MATCHUP_COLUMNS = ()
REQUIRED_PREDICTION_ROW_COLUMNS = (
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
)
