import pandas as pd
import pytest

from gridiron_ml.td_run import TDEval
from gridiron_ml.fingerprints.features import DEFAULT_FEATURE_SPEC, FeatureSpec, split_frame
from gridiron_ml.models import TDLinear
from gridiron_ml.pipeline.validation.leakage import (
    assert_default_target_is_next_margin,
    assert_disjoint_years,
    assert_no_same_week_postgame_target,
)


def _fingerprint_frame():
    return pd.DataFrame(
        {
            "keys_season": [2024, 2024],
            "keys_team": ["Home", "Away"],
            "keys_week": [1, 1],
            "offense_ppa": [0.2, -0.1],
            "defense_ppa": [-0.3, 0.4],
            "market_spread_close": [-7.5, 7.5],
            "market_over_under": [48.0, 48.0],
            "y_next_margin": [14.0, -14.0],
            "y_margin_this_week": [10.0, -10.0],
            "y_has_next_game": [True, True],
        }
    )


def test_default_training_target_is_next_margin():
    assert DEFAULT_FEATURE_SPEC.target_column == "y_next_margin"
    assert_default_target_is_next_margin(DEFAULT_FEATURE_SPEC.target_column)
    with pytest.raises(ValueError, match="y_next_margin"):
        assert_default_target_is_next_margin("y_margin_this_week")


def test_default_split_excludes_market_features_but_keeps_market_context():
    X, y, meta_df, market_df = split_frame(_fingerprint_frame())

    assert "market_spread_close" not in X.columns
    assert "market_over_under" not in X.columns
    assert "market_spread_close" in market_df.columns
    assert "y_next_margin" not in meta_df.columns
    assert y.tolist() == [14.0, -14.0]


def test_include_market_requires_loud_training_opt_in():
    frame = _fingerprint_frame()

    with pytest.raises(ValueError, match="Market/Vegas/betting-derived"):
        split_frame(frame, FeatureSpec(include_market=True))

    X, _, _, _ = split_frame(
        frame,
        FeatureSpec(
            include_market=True,
            allow_market_features_for_training=True,
        ),
    )

    assert "market_spread_close" in X.columns


def test_model_training_blocks_market_features_without_explicit_opt_in():
    X = pd.DataFrame(
        {
            "net_offense_ppa": [0.1, 0.2, 0.3, 0.4],
            "home_market_spread_close": [3.0, -4.0, 7.5, -1.5],
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


def test_same_week_margin_target_rejects_postgame_features():
    with pytest.raises(ValueError, match="same-row completed-game target"):
        assert_no_same_week_postgame_target(
            "y_margin_this_week",
            ["offense_ppa", "games_played"],
        )


def test_year_splits_must_be_disjoint():
    assert_disjoint_years([2021, 2022], [2023])
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_years([2021, 2022], [2022, 2023])


def test_evaluator_train_checks_year_overlap_before_loading_data():
    evaluator = TDEval(
        {"eval": {"train_years": [2023], "test_years": [2023]}},
        fingerprints=object(),
        matchup_builder=object(),
        model=object(),
    )

    with pytest.raises(ValueError, match="train and val years overlap"):
        evaluator.train()
