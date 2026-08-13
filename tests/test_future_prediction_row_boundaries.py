import pandas as pd
import pytest

from gridiron_ml.pipeline.prediction_rows import PredictionRowsBuilder
from gridiron_ml.pipeline.schemas import (
    SchemaValidationError,
    validate_prediction_rows,
    validate_team_game_table,
)


def _schedule_only_rows():
    return pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_id": [401856634],
            "home_team": ["Alabama"],
            "away_team": ["East Carolina"],
            "target_team_margin": [pd.NA],
            "home_points": [pd.NA],
            "away_points": [pd.NA],
        }
    )


def test_schedule_only_rows_validate_as_prediction_rows():
    rows = _schedule_only_rows()

    validate_prediction_rows(rows)
    wrapped = PredictionRowsBuilder().from_schedule(rows).to_frame()

    assert wrapped.loc[0, "season"] == 2026
    assert pd.isna(wrapped.loc[0, "target_team_margin"])


def test_schedule_only_rows_are_not_completed_training_rows():
    rows = _schedule_only_rows()

    with pytest.raises(SchemaValidationError):
        validate_team_game_table(rows)


def test_prediction_rows_allow_null_targets_but_reject_filled_targets():
    rows = _schedule_only_rows()
    validate_prediction_rows(rows)

    filled = rows.copy()
    filled["target_team_margin"] = [14.0]

    with pytest.raises(SchemaValidationError, match="training labels"):
        validate_prediction_rows(filled)


def test_completed_training_rows_require_non_null_targets():
    incomplete_training = pd.DataFrame(
        {
            "keys_season": [2026],
            "keys_week": [1],
            "keys_game_id": [401856634],
            "keys_team": ["Alabama"],
            "keys_opponent": ["East Carolina"],
            "game_is_home": [True],
            "target_points_for": [pd.NA],
            "target_points_against": [pd.NA],
            "target_team_margin": [pd.NA],
        }
    )

    with pytest.raises(SchemaValidationError, match="completed-game targets"):
        validate_team_game_table(incomplete_training)
