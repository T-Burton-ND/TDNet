import numpy as np
import pandas as pd

from gridiron_ml.models import TDStat, TDStatPercentile, TDStatRobust, TDStatWeighted


def test_tdstat_percentile_trains_and_predicts_with_outlier():
    X, y = _stat_frame()
    X.loc[len(X) - 1, "statOff_first_downs"] = 500.0
    model = TDStatPercentile({"params": {"percentile_ridge_alpha": 1.0}})

    model.train(X, y)
    pred = model.predict(X)

    assert model.model_type == "percentile"
    assert model.normalization_ == "percentile"
    assert pred["pred_margin"].notna().all()
    assert pred["pred_proba_home_win"].between(0.0, 1.0).all()


def test_tdstat_robust_uses_mad_or_iqr_scaling():
    X, y = _stat_frame()
    model = TDStatRobust({"params": {"robust_scale_method": "iqr", "robust_clip": 3.0}})

    model.train(X, y)
    transformed = model._transform_features(X.loc[:, model.selected_feature_names_])

    assert model.model_type == "robust"
    assert model.normalization_ == "robust"
    assert np.nanmax(np.abs(transformed)) <= 3.0 + 1e-9


def test_tdstat_weighted_applies_configurable_family_weights():
    X, y = _stat_frame()
    model = TDStatWeighted(
        {
            "params": {
                "family_weights": {
                    "offense": 2.0,
                    "defense": 0.5,
                    "special_teams": 0.25,
                }
            }
        }
    )

    model.train(X, y)

    assert model.model_type == "weighted"
    assert model.feature_weights_["statOff_first_downs"] == 2.0
    assert model.feature_weights_["statDef_sacks"] == 0.5
    assert model.feature_weights_["statSpe_kicking_points"] == 0.25


def test_tdstat_base_accepts_variant_model_types():
    for model_type in ["percentile", "robust", "weighted"]:
        model = TDStat({"params": {"model_type": model_type}})
        assert model.model_type == model_type


def test_tdstat_defaults_to_winner_upset_objective():
    model = TDStat()

    assert model.objective == "winner_upset"
    assert model.loss_function == "WinnerUpsetAccuracy"


def test_tdstat_hybrid_winner_accuracy_alias_uses_winner_upset_objective():
    model = TDStat({"params": {"objective": "hybrid_winner_accuracy"}})

    assert model.objective == "winner_upset"
    assert model.loss_function == "WinnerUpsetAccuracy"


def test_tdstat_can_target_winner_upset_objective_and_caps_margin():
    X, y = _stat_frame()
    market = pd.DataFrame({"market_spread": [-3.5] * len(y)})
    model = TDStat(
        {
            "params": {
                "objective": "winner_upset",
                "upset_weight": 4.0,
                "prediction_margin_cap": 30.0,
            }
        }
    )

    model.train(X, y, market_train=market)
    pred = model.predict(X * 1000.0)

    assert model.loss_function == "WinnerUpsetAccuracy"
    assert "winner_accuracy" in model.training_history_.columns
    assert pred["pred_margin"].abs().max() <= 30.0


def test_tdstat_percentile_flips_lower_is_better_features():
    X, y = _stat_frame()
    model = TDStatPercentile()

    model.train(X, y)
    transformed = pd.DataFrame(
        model._transform_features(X.loc[:, model.selected_feature_names_]),
        columns=model.selected_feature_names_,
    )

    assert model.directions_["statGen_turnovers"] == -1.0
    assert (
        transformed.loc[0, "statGen_turnovers"]
        > transformed.loc[2, "statGen_turnovers"]
    )


def _stat_frame():
    rows = 18
    idx = np.arange(rows, dtype=float)
    X = pd.DataFrame(
        {
            "statOff_first_downs": 18.0 + idx,
            "statOff_yards_per_pass": 5.5 + idx * 0.08,
            "statDef_sacks": 1.0 + (idx % 4),
            "statGen_turnovers": (idx % 3),
            "statSpe_kicking_points": 4.0 + (idx % 5),
            "offense_ppa": -0.2 + idx * 0.03,
            "defense_ppa": 0.4 - idx * 0.02,
        }
    )
    y = pd.Series(-8.0 + idx * 1.2)
    return X, y
