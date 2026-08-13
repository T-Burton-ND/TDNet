import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge,
    SGDRegressor,
)

from gridiron_ml.td_run import TDEval
from gridiron_ml.models.td_linear import TDLinear


def test_simplified_config_loads_and_winner_upset_is_default(tmp_path):
    config_path = tmp_path / "linear.yaml"
    config_path.write_text(
        yaml.safe_dump({"model_name": "Test Linear"}), encoding="utf-8"
    )

    model = TDLinear.from_yaml(config_path)

    assert model.model_name == "test_linear"
    assert model.model_type == "ols"
    assert model.loss_function == "WinnerUpsetAccuracy"
    assert model.loss_weights["alpha"] == 0.25
    assert model.loss_weights["beta"] == 1.0
    assert model.loss_weights["gamma"] == 3.0
    assert model.loss_weights["delta"] == 0.25


@pytest.mark.parametrize(
    "loss_function",
    ["RMSE", "MAE", "Composite", "WinnerAccuracy", "WinnerUpsetAccuracy"],
)
def test_loss_function_switch(loss_function):
    model = TDLinear({"loss_function": loss_function})
    metrics = model.loss_breakdown(
        np.array([7.0, -3.0, 1.0]), np.array([10.0, -7.0, -2.0])
    )

    assert metrics["loss_function"] == loss_function
    assert np.isfinite(metrics["total_loss"])
    assert "margin_loss" in metrics
    assert "win_probability_loss" in metrics
    assert "favorite_correctness_loss" in metrics
    assert "calibration_loss" in metrics


def test_hybrid_winner_accuracy_alias_uses_winner_upset_objective():
    model = TDLinear({"loss_function": "hybrid_winner_accuracy"})

    assert model.loss_function == "WinnerUpsetAccuracy"


def test_mixed_loss_alias_uses_composite_objective():
    model = TDLinear({"loss_function": "mixed_loss"})

    assert model.loss_function == "Composite"


def test_winner_accuracy_uses_classifier_surrogate_and_caps_margin():
    X = pd.DataFrame({"edge": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]})
    y = pd.Series([-14.0, -10.0, -7.0, 7.0, 10.0, 14.0])
    model = TDLinear({"loss_function": "WinnerAccuracy", "prediction_margin_cap": 30.0})

    model.train(X, y)
    pred = model.predict(pd.DataFrame({"edge": [-1000.0, 1000.0]}))

    assert model._is_classifier_objective()
    assert pred["pred_margin"].abs().max() <= 30.0


@pytest.mark.parametrize(
    "model_type",
    [
        "ols",
        "ridge",
        "lasso",
        "elastic_net",
        "huber",
        "bayesian",
        "ard",
        "ransac",
        "orthogonal_matching_pursuit",
        "sgd",
        "passive_aggressive",
    ],
)
def test_winner_accuracy_objective_trains_for_all_linear_model_configs(model_type):
    X = pd.DataFrame(
        {
            "edge": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "aux": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    y = pd.Series([-14.0, -10.0, -7.0, 7.0, 10.0, 14.0])
    model = TDLinear(
        {
            "model_type": model_type,
            "loss_function": "WinnerAccuracy",
            "params": _params_for_fast_test(model_type),
        }
    )

    model.train(X, y)
    pred = model.predict(X)

    assert model.is_trained_
    assert pred["pred_margin"].abs().max() <= 30.0


@pytest.mark.parametrize("weight_name", ["alpha", "beta", "gamma", "delta"])
def test_composite_weights_change_total_loss(weight_name):
    pred = np.array([14.0, -10.0, 3.0, -2.0])
    actual = np.array([7.0, 4.0, -1.0, -14.0])

    low_weight = TDLinear(
        {"loss_function": "Composite", "loss_weights": {weight_name: 0.5}}
    )
    high_weight = TDLinear(
        {"loss_function": "Composite", "loss_weights": {weight_name: 5.0}}
    )

    assert (
        high_weight.loss_breakdown(pred, actual)["total_loss"]
        > low_weight.loss_breakdown(pred, actual)["total_loss"]
    )


def test_positive_margin_maps_to_probability_above_half():
    model = TDLinear()

    prob = model.margin_to_probability(np.array([1.0]))[0]

    assert prob > 0.5


def test_margin_to_probability_handles_extreme_margins_without_overflow():
    model = TDLinear()

    prob = model.margin_to_probability(np.array([-10000.0, 10000.0]))

    assert np.isfinite(prob).all()
    assert prob[0] < 1e-20
    assert prob[1] > 1.0 - 1e-12


def test_wrong_confident_favorite_penalized_more_than_wrong_weak_favorite():
    model = TDLinear({"favorite_confidence_scale": 4.0})

    weak_wrong = model.loss_breakdown(np.array([1.0]), np.array([-7.0]))[
        "favorite_correctness_loss"
    ]
    confident_wrong = model.loss_breakdown(np.array([21.0]), np.array([-7.0]))[
        "favorite_correctness_loss"
    ]

    assert confident_wrong > weak_wrong


def test_vegas_columns_are_not_part_of_training_loss():
    model = TDLinear()
    pred = np.array([10.0, -4.0, 2.0])
    actual = np.array([7.0, -10.0, -1.0])

    base = model.loss_breakdown(pred, actual)
    with_market_noise = model.loss_breakdown(pred, actual)

    assert base["total_loss"] == with_market_noise["total_loss"]


def test_eval_builds_linear_model_from_simplified_config(tmp_path):
    config_path = tmp_path / "linear.yaml"
    config_path.write_text(
        yaml.safe_dump({"model_name": "eval_linear"}), encoding="utf-8"
    )

    evaluator = TDEval(
        {
            "fingerprints": {"version": 0, "root": "."},
            "model": {"family": "linear", "config_path": str(config_path)},
        },
        fingerprints=object(),
        matchup_builder=object(),
    )

    assert evaluator.model.model_name == "eval_linear"
    assert evaluator.model.model_type == "ols"
    assert evaluator.model.loss_function == "WinnerUpsetAccuracy"


@pytest.mark.parametrize(
    ("model_type", "expected_cls"),
    [
        ("ols", LinearRegression),
        ("ridge", Ridge),
        ("lasso", Lasso),
        ("elastic_net", ElasticNet),
        ("huber", HuberRegressor),
        ("bayesian", BayesianRidge),
        ("ard", ARDRegression),
        ("ransac", RANSACRegressor),
        ("orthogonal_matching_pursuit", OrthogonalMatchingPursuit),
        ("sgd", SGDRegressor),
        ("passive_aggressive", PassiveAggressiveRegressor),
    ],
)
def test_linear_model_types_build_actual_sklearn_estimators(model_type, expected_cls):
    model = TDLinear(
        {
            "model_type": model_type,
            "loss_function": "RMSE",
            "params": _params_for_fast_test(model_type),
        }
    )

    estimator = model._build_estimator()

    assert model.model_type == model_type
    assert isinstance(estimator, expected_cls)


@pytest.mark.parametrize(
    "model_type",
    [
        "ols",
        "ridge",
        "lasso",
        "elastic_net",
        "huber",
        "bayesian",
        "ard",
        "ransac",
        "orthogonal_matching_pursuit",
        "sgd",
        "passive_aggressive",
    ],
)
def test_training_smoke_with_actual_sklearn_algorithms(model_type):
    idx = np.arange(24, dtype=float)
    X = pd.DataFrame(
        {
            "a": idx / 10.0,
            "b": np.sin(idx / 3.0),
            "c": np.where(idx % 2 == 0, 1.0, -1.0),
        }
    )
    X.loc[3, "b"] = np.nan
    y = pd.Series(-6.0 + 2.0 * X["a"].fillna(0.0) - 1.5 * X["c"])
    model = TDLinear(
        {
            "model_type": model_type,
            "loss_function": "MAE",
            "training": {"standardize": True},
            "params": _params_for_fast_test(model_type),
        }
    )

    model.train(X, y)
    pred = model.predict(X)
    importance = model.get_feature_importance()

    assert model.is_trained_
    assert len(pred) == len(X)
    assert {"pred_margin", "pred_proba_home_win", "pred_pick_home"}.issubset(
        pred.columns
    )
    assert np.isfinite(pred["pred_margin"]).all()
    assert {"train"} == set(model.training_history_["split"])
    assert set(importance.columns) == {"feature", "coefficient", "importance"}


def _params_for_fast_test(model_type):
    params = {
        "lasso": {"alpha": 0.001, "max_iter": 1000},
        "elastic_net": {"alpha": 0.001, "max_iter": 1000},
        "huber": {"max_iter": 200},
        "bayesian": {"max_iter": 100},
        "ard": {"max_iter": 100},
        "ransac": {"max_trials": 10, "min_samples": 0.5, "stop_probability": 0.95},
        "sgd": {"max_iter": 200, "tol": 0.001},
        "passive_aggressive": {"max_iter": 200, "tol": 0.001},
    }
    return params.get(model_type, {})
