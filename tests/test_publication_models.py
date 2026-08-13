from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridiron_ml.models import build_model_from_config, validate_model_contract


@pytest.fixture
def regression_data():
    x = np.linspace(-2.0, 2.0, 48)
    frame = pd.DataFrame({"offense": x, "defense": np.sin(np.arange(48))})
    target = 7.0 * frame["offense"] - 2.0 * frame["defense"]
    return frame, target


@pytest.mark.parametrize(
    "config",
    [
        {"family": "naive", "model_type": "constant_margin"},
        {"family": "linear", "model_type": "ridge", "loss_function": "RMSE", "params": {"alpha": 1.0}},
        {"family": "spline", "model_type": "spline_ridge", "loss_function": "RMSE", "params": {"alpha": 1.0}, "spline": {"n_knots": 3, "degree": 2}},
        {"family": "tree", "model_type": "random_forest", "loss_function": "RMSE", "params": {"n_estimators": 8, "n_jobs": 1}},
        {"family": "boosted", "model_type": "hist_gradient_boosted", "loss_function": "RMSE", "params": {"max_iter": 8, "min_samples_leaf": 5}},
    ],
)
def test_unified_model_contract_and_round_trip(config, regression_data, tmp_path):
    features, target = regression_data
    model = build_model_from_config(config)
    validate_model_contract(model)
    model.fit(features.iloc[:36], target.iloc[:36], X_val=features.iloc[36:], y_val=target.iloc[36:])
    prediction = model.predict(features.iloc[36:])
    assert list(prediction) == ["pred_margin", "pred_proba_home_win", "pred_pick_home"]
    assert prediction.notna().all().all()
    assert np.isfinite(prediction[["pred_margin", "pred_proba_home_win"]].to_numpy(dtype=float)).all()
    assert np.all((model.predict_proba(features.iloc[36:]) >= 0) & (model.predict_proba(features.iloc[36:]) <= 1))
    checkpoint = model.save(tmp_path / "model.pkl")
    loaded = type(model).load(checkpoint)
    np.testing.assert_allclose(model.predict_margin(features.iloc[36:]), loaded.predict_margin(features.iloc[36:]))


def test_neural_family_is_registered_without_requiring_training():
    model = build_model_from_config({"family": "neural", "hidden_layers": [8], "max_epochs": 1})
    validate_model_contract(model)
    assert model.model_family == "neural"


def test_batch_normalized_neural_model_avoids_singleton_final_batch():
    pytest.importorskip("torch")
    rows = 129
    features = pd.DataFrame({"offense": np.linspace(-2, 2, rows), "defense": np.linspace(1, -1, rows)})
    target = 4 * features["offense"] - features["defense"]
    model = build_model_from_config({
        "family": "neural",
        "hidden_layers": [8],
        "normalization": "batch_norm",
        "batch_size": 128,
        "max_epochs": 1,
        "quantile_transform": False,
        "torch_threads": 1,
    })
    model.fit(features, target)
    assert np.isfinite(model.predict_margin(features.iloc[:3])).all()


def test_structured_mlp_accepts_existing_matchup_prefix_style():
    model = build_model_from_config({"family": "structured_neural", "hidden_layers": [8], "max_epochs": 1})
    raw = pd.DataFrame({"home_offense": [1.0, 2.0], "away_offense": [0.5, 1.5], "net_context": [0.1, -0.1]})
    expanded = model._prepare_input_frame(raw, fitting=True)
    assert {"home__offense", "away__offense", "diff__offense", "product__offense"}.issubset(expanded)


def test_ransac_recovery_supports_regularized_base_estimator(regression_data):
    features, target = regression_data
    model = build_model_from_config({
        "family": "linear", "model_type": "ransac", "loss_function": "RMSE",
        "params": {"base_estimator": "ridge", "min_samples": 0.9, "max_trials": 5,
                   "estimator_params": {"alpha": 0.1}},
    })
    model.fit(features, target)
    assert np.isfinite(model.predict_margin(features)).all()
