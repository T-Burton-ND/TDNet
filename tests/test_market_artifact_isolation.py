import pandas as pd
import pytest

from gridiron_ml.fingerprints.features import FeatureSpec, split_frame
from gridiron_ml.models import TDLinear
from gridiron_ml.pipeline.schemas import (
    SchemaValidationError,
    validate_fingerprint_feature_artifact,
    validate_matchup_frame,
)


def _frame_with_market():
    return pd.DataFrame(
        {
            "keys_season": [2024, 2024],
            "keys_team": ["Home", "Away"],
            "keys_week": [1, 1],
            "offense_ppa": [0.3, -0.2],
            "market_spread_close": [-6.5, 6.5],
            "market_over_under": [51.5, 51.5],
            "y_next_margin": [10.0, -10.0],
            "y_has_next_game": [True, True],
        }
    )


def test_default_split_keeps_market_context_out_of_x():
    X, y, meta_df, market_df = split_frame(_frame_with_market())

    assert list(X.columns) == ["offense_ppa"]
    assert y.tolist() == [10.0, -10.0]
    assert "market_spread_close" not in meta_df.columns
    assert "market_spread_close" in market_df.columns


def test_market_features_require_loud_opt_in():
    frame = _frame_with_market()

    with pytest.raises(ValueError, match="Market/Vegas"):
        split_frame(frame, FeatureSpec(include_market=True))

    X, _, _, _ = split_frame(
        frame,
        FeatureSpec(include_market=True, allow_market_features_for_training=True),
    )

    assert "market_spread_close" in X.columns
    validate_matchup_frame(X, allow_market_features_for_training=True)


def test_training_rejects_market_x_without_loud_opt_in():
    X = pd.DataFrame(
        {
            "net_offense_ppa": [0.1, 0.2, 0.3, 0.4],
            "market_spread_close": [3.0, -4.0, 7.5, -1.5],
        }
    )
    y = pd.Series([1.0, -2.0, 3.0, -4.0])

    with pytest.raises(ValueError, match="eval-only"):
        TDLinear({"model_type": "ridge"}).train(X, y)

    model = TDLinear(
        {
            "model_type": "ridge",
            "allow_market_features_for_training": True,
        }
    ).train(X, y)
    assert model.is_trained_


def test_fingerprint_feature_artifact_rejects_market_columns():
    artifact = pd.DataFrame(
        {
            "keys_season": [2024],
            "keys_team": ["Home"],
            "keys_week": [1],
            "offense_ppa": [0.1],
            "market_spread_close": [-7.0],
        }
    )

    with pytest.raises(SchemaValidationError, match="market"):
        validate_fingerprint_feature_artifact(artifact)
