import numpy as np
import pandas as pd
import pytest

from gridiron_ml.models import TDLinear, TDStat, TDTree


def opponent_adjusted_features():
    return pd.DataFrame(
        {
            "home_opp_adj_success_rate": [0.15, 0.18, 0.11, 0.21, 0.24, 0.13],
            "away_opp_adj_success_rate": [0.12, 0.16, 0.13, 0.18, 0.19, 0.15],
            "net_opp_adj_success_rate": [0.03, 0.02, -0.02, 0.03, 0.05, -0.02],
            "home_opponent_adjusted_explosiveness": [1.1, 1.4, 0.9, 1.7, 1.9, 0.8],
            "away_opponent_adjusted_explosiveness": [0.8, 1.0, 1.2, 1.3, 1.1, 1.0],
            "net_opponent_adjusted_explosiveness": [0.3, 0.4, -0.3, 0.4, 0.8, -0.2],
        }
    )


@pytest.mark.parametrize(
    "model_factory, selected_attr",
    [
        (
            lambda: TDLinear({"model_type": "ridge", "loss_function": "Composite"}),
            "feature_names_",
        ),
        (
            lambda: TDTree(
                {
                    "model_type": "random_forest",
                    "loss_function": "Composite",
                    "params": {
                        "n_estimators": 6,
                        "min_samples_leaf": 1,
                        "random_state": 7,
                        "n_jobs": 1,
                    },
                }
            ),
            "feature_names_",
        ),
        (
            lambda: TDStat(
                {
                    "params": {
                        "model_type": "z_index",
                        "feature_include_patterns": [
                            "*opp_adj*",
                            "*opponent_adjusted*",
                        ],
                    }
                }
            ),
            "selected_feature_names_",
        ),
    ],
)
def test_model_families_accept_opponent_adjusted_features(model_factory, selected_attr):
    X = opponent_adjusted_features()
    y = pd.Series([-10.0, -3.0, -1.0, 7.0, 14.0, 21.0])
    model = model_factory()

    model.train(X, y)
    pred = model.predict(
        X.assign(unused_future_adjusted_feature=np.linspace(0.0, 1.0, len(X)))
    )

    selected = set(getattr(model, selected_attr))
    assert "home_opp_adj_success_rate" in selected
    assert "net_opponent_adjusted_explosiveness" in selected
    assert len(pred) == len(X)
    assert np.isfinite(pred["pred_margin"]).all()
