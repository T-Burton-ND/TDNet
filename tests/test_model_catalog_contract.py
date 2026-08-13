from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridiron_ml.models import build_model_from_config, load_model_checkpoint
from gridiron_ml.models import TDTree
from gridiron_ml.td_run import DEFAULT_MODEL_SPECS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "spec",
    DEFAULT_MODEL_SPECS,
    ids=[f"{spec.family}:{spec.name}" for spec in DEFAULT_MODEL_SPECS],
)
def test_default_model_catalog_trains_predicts_and_round_trips(spec, tmp_path):
    X, y = _model_catalog_frame()
    model = build_model_from_config(
        {
            "family": spec.family,
            "config_path": str(PROJECT_ROOT / spec.config_path),
        }
    )
    _make_catalog_smoke_fast(model)

    model.train(X, y, X_val=X.tail(12), y_val=y.tail(12))
    checkpoint = model.save(tmp_path / f"{spec.family}_{spec.name}.pkl")
    loaded = load_model_checkpoint(checkpoint)
    pred = loaded.predict(
        X.head(8),
        meta_df=pd.DataFrame({"game_id": range(8)}),
        market_df=pd.DataFrame({"market_spread_close": np.linspace(-7.0, 7.0, 8)}),
    )

    assert loaded.is_trained_
    assert {"pred_margin", "pred_proba_home_win", "pred_pick_home"}.issubset(
        pred.columns
    )
    assert pred["pred_margin"].notna().all()
    assert pred["pred_proba_home_win"].between(0.0, 1.0).all()
    assert pred["pred_pick_home"].isin([0, 1]).all()
    assert "game_id" in pred.columns
    assert "market_spread_close" in pred.columns


def _model_catalog_frame():
    rows = 72
    rng = np.random.default_rng(20260623)
    idx = np.linspace(-1.0, 1.0, rows)

    X = pd.DataFrame(
        {
            "home_statOff_success_rate": 0.44 + 0.08 * idx,
            "away_statOff_success_rate": 0.43 - 0.05 * idx,
            "net_statOff_success_rate": 0.13 * idx,
            "home_statDef_havoc": 0.16 + 0.04 * np.sin(np.arange(rows) / 4.0),
            "away_statDef_havoc": 0.15 + 0.03 * np.cos(np.arange(rows) / 5.0),
            "net_statDef_havoc": 0.04 * np.sin(np.arange(rows) / 3.0),
            "home_statGen_turnovers": (np.arange(rows) % 4).astype(float),
            "away_statGen_turnovers": ((np.arange(rows) + 1) % 4).astype(float),
            "net_statGen_turnovers": -1.0 + (np.arange(rows) % 3).astype(float),
            "home_statSpe_kicking_points": 5.0 + (np.arange(rows) % 6),
            "away_statSpe_kicking_points": 4.0 + ((np.arange(rows) + 2) % 5),
            "net_statSpe_kicking_points": -2.0 + (np.arange(rows) % 5),
            "home_offense_ppa": 0.1 + 0.25 * idx,
            "away_defense_ppa": -0.05 - 0.18 * idx,
            "net_adjusted_efficiency": 0.3 * idx,
        }
    )
    X += rng.normal(0.0, 0.015, size=X.shape)
    X.loc[::11, "home_statDef_havoc"] = np.nan
    X.loc[5::13, "net_adjusted_efficiency"] = np.nan

    y = pd.Series(
        18.0 * X["net_statOff_success_rate"].fillna(0.0)
        + 9.0 * X["net_adjusted_efficiency"].fillna(0.0)
        - 1.7 * X["net_statGen_turnovers"].fillna(0.0)
        + rng.normal(0.0, 2.5, rows),
        name="y_next_margin",
    )
    return X, y


def _make_catalog_smoke_fast(model):
    if not isinstance(model, TDTree):
        return
    model.params["n_estimators"] = min(int(model.params.get("n_estimators", 8)), 8)
    model.params["n_jobs"] = 1
