import pandas as pd
import pytest

from gridiron_ml.pipeline.prediction_rows import PredictionRowsBuilder
from gridiron_ml.pipeline.schemas import (
    SchemaValidationError,
    validate_fingerprint_frame,
    validate_label_frame,
    validate_matchup_frame,
    validate_prediction_rows,
    validate_team_game_table,
)


def test_core_schema_validators_accept_minimal_valid_frames():
    validate_team_game_table(
        pd.DataFrame(
            {
                "keys_season": [2024],
                "keys_week": [1],
                "keys_game_id": [10],
                "keys_team": ["Home"],
                "keys_opponent": ["Away"],
                "game_is_home": [True],
                "target_points_for": [28],
                "target_points_against": [14],
                "target_team_margin": [14],
            }
        )
    )
    labels = pd.DataFrame(
        {
            "keys_season": [2024],
            "keys_team": ["Home"],
            "keys_week": [0],
            "y_next_margin": [14.0],
            "y_has_next_game": [True],
        }
    )
    validate_fingerprint_frame(labels.copy())
    validate_label_frame(labels.copy())
    validate_matchup_frame(pd.DataFrame({"net_offense_ppa": [0.1], "diff_defense_ppa": [-0.2]}))


def test_matchup_schema_rejects_market_features():
    with pytest.raises(SchemaValidationError, match="market"):
        validate_matchup_frame(pd.DataFrame({"diff_market_spread_close": [7.5]}))


def test_schedule_only_rows_are_prediction_rows_not_training_rows():
    schedule = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_id": [1001],
            "home_team": ["Home"],
            "away_team": ["Away"],
            "home_points": [pd.NA],
            "away_points": [pd.NA],
        }
    )

    validate_prediction_rows(schedule)
    future_rows = PredictionRowsBuilder().from_schedule(schedule).to_frame()

    assert future_rows.loc[0, "home_team"] == "Home"
    with pytest.raises(SchemaValidationError):
        validate_team_game_table(schedule)


def test_prediction_rows_reject_filled_training_labels():
    schedule = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_id": [1001],
            "home_team": ["Home"],
            "away_team": ["Away"],
            "target_team_margin": [7.0],
        }
    )

    with pytest.raises(SchemaValidationError, match="training labels"):
        validate_prediction_rows(schedule)
