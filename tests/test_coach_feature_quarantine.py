import pandas as pd
import pytest

from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.fingerprints.features import split_frame
from gridiron_ml.models import TDLinear
from gridiron_ml.pipeline.schemas import SchemaValidationError, validate_training_feature_frame


def test_training_features_reject_coach_postseason_columns():
    X = pd.DataFrame({"coach_career_mean_postseason_rank": [1.0, 2.0]})

    with pytest.raises(SchemaValidationError, match="stale"):
        validate_training_feature_frame(X)


def test_training_features_reject_coach_season_columns():
    X = pd.DataFrame({"coach_season_wins": [10.0, 4.0]})

    with pytest.raises(SchemaValidationError, match="stale"):
        validate_training_feature_frame(X)


def test_training_features_reject_specific_postseason_rank_points_column():
    X = pd.DataFrame({"coach_career_mean_postseason_rank_points": [25.0, 0.0]})

    with pytest.raises(SchemaValidationError, match="coach_career_mean_postseason_rank_points"):
        validate_training_feature_frame(X)


def test_prior_season_safe_coach_columns_are_allowed():
    X = pd.DataFrame(
        {
            "coach_career_seasons": [4.0, 8.0],
            "coach_career_mean_sp_offense": [32.0, 28.0],
            "coach_career_mean_sp_defense": [18.0, 22.0],
        }
    )

    validate_training_feature_frame(X)


def test_split_frame_and_direct_model_training_quarantine_leaky_coach_features():
    frame = pd.DataFrame(
        {
            "keys_season": [2024, 2024],
            "keys_team": ["Home", "Away"],
            "keys_week": [1, 1],
            "coach_career_mean_postseason_rank_points": [25.0, 0.0],
            "y_next_margin": [7.0, -7.0],
            "y_has_next_game": [True, True],
        }
    )

    with pytest.raises(ValueError, match="stale"):
        split_frame(frame)

    with pytest.raises(ValueError, match="stale"):
        TDLinear({"model_type": "ridge"}).train(
            pd.DataFrame({"coach_season_games": [1.0, 2.0, 3.0, 4.0]}),
            pd.Series([1.0, -1.0, 2.0, -2.0]),
        )


def test_fingerprints_training_block_rejects_stale_persisted_coach_artifact(tmp_path):
    fp_dir = tmp_path / "data" / "fingerprints" / "v0"
    fp_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "keys_season": [2024, 2024],
            "keys_team": ["Home", "Away"],
            "keys_week": [0, 0],
            "offense_ppa": [0.2, -0.1],
            "coach_career_mean_postseason_rank_points": [25.0, 0.0],
            "next_game_id": [1, 1],
            "next_game_is_home": [True, False],
            "y_next_margin": [7.0, -7.0],
            "y_has_next_game": [True, True],
        }
    ).to_parquet(fp_dir / "canonical_fingerprint.parquet", index=False)

    with pytest.raises(ValueError, match="stale"):
        Fingerprints(version=0, root=tmp_path).training_block([2024])
