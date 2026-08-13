import json

import numpy as np
import pandas as pd

from gridiron_ml.models import TDKNN, build_model_from_config


def _data(n=36):
    x = np.linspace(-2.0, 2.0, n)
    X = pd.DataFrame({"offense": x, "defense": np.sin(x)})
    X.loc[3, "defense"] = np.nan
    y = pd.Series(12.0 * x + np.cos(x), name="margin")
    meta = pd.DataFrame(
        {
            "keys_game_id": np.arange(n),
            "keys_season": 2010 + (np.arange(n) // 12),
            "keys_week": (np.arange(n) % 12) + 1,
            "keys_team_home": ["home"] * n,
            "keys_team_away": ["away"] * n,
        }
    )
    return X, y, meta


def test_knn_emits_calibrated_predictions_and_neighbor_audit(tmp_path):
    X, y, meta = _data()
    model = build_model_from_config(
        {
            "family": "knn",
            "model_type": "distance",
            "model_name": "knn_distance",
            "params": {"n_neighbors": 5, "weights": "distance", "metric": "manhattan"},
        }
    )
    assert isinstance(model, TDKNN)
    model.train(
        X.iloc[:24], y.iloc[:24], X_val=X.iloc[24:], y_val=y.iloc[24:],
        meta_train=meta.iloc[:24], meta_val=meta.iloc[24:],
    )
    predictions = model.predict(X.iloc[24:28], meta_df=meta.iloc[24:28])

    assert predictions["pred_proba_home_win"].between(0.0, 1.0).all()
    assert predictions["selected_k"].eq(5).all()
    assert predictions["knn_metric"].eq("manhattan").all()
    assert predictions["neighbor_game_ids"].map(json.loads).map(len).eq(5).all()
    assert len(model.neighbor_audit_) == 4 * 5
    assert {"neighbor_distance", "neighbor_weight", "actual_margin"}.issubset(model.neighbor_audit_)


def test_knn_clamps_k_when_training_set_is_smaller():
    X, y, meta = _data(6)
    model = build_model_from_config(
        {"family": "knn", "model_type": "uniform", "params": {"n_neighbors": 50}}
    )
    model.train(X, y, meta_train=meta)
    assert model.effective_n_neighbors_ == 6
    assert model.predict(X.iloc[:1])["selected_k"].iloc[0] == 6
