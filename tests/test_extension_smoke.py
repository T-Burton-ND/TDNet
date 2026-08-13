import numpy as np
import pandas as pd

from gridiron_ml.td_run import TDEval
from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.fingerprints.builders import BaseFingerprintBuilder, register_fingerprint_builder
from gridiron_ml.td_run.market import market_home_margin, normalize_vegas_frame
from gridiron_ml.models import register_model_family


class DummyModel:
    model_family = "dummy"
    model_type = "smoke"

    def __init__(self, config=None):
        self.config = dict(config or {})

    def predict(self, X, meta_df=None, market_df=None):
        out = pd.DataFrame(
            {
                "pred_margin": np.zeros(len(X), dtype=float),
                "pred_proba_home_win": np.full(len(X), 0.5, dtype=float),
                "pred_pick_home": np.zeros(len(X), dtype=int),
            }
        )
        if meta_df is not None:
            out = pd.concat([meta_df.reset_index(drop=True), out], axis=1)
        if market_df is not None:
            keep = [c for c in market_df.columns if c not in out.columns]
            out = pd.concat([out, market_df.reset_index(drop=True).loc[:, keep]], axis=1)
        return out

    def total_rank(self, X, meta_df=None):
        return self.predict(X, meta_df=meta_df).rename(columns={"pred_margin": "score"})

    def top25(self, X, meta_df=None):
        return self.total_rank(X, meta_df=meta_df).head(25)


class DummyFingerprintBuilder(BaseFingerprintBuilder):
    def _build_from_team_game_tables(self, overwrite=False):
        self.fp_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            {
                "keys_season": [2099],
                "keys_team": ["A"],
                "keys_week": [0],
                "offense_ppa": [0.0],
            }
        )
        frame.to_parquet(self.canonical_path, index=False)
        return self.canonical_path


def test_evaluator_uses_registered_model_family():
    register_model_family("dummy_smoke", DummyModel)

    evaluator = TDEval(
        {"model": {"family": "dummy_smoke", "custom_setting": 7}},
        fingerprints=object(),
        matchup_builder=object(),
    )

    assert isinstance(evaluator.model, DummyModel)
    assert evaluator.model.config["custom_setting"] == 7


def test_fingerprints_uses_registered_builder_version(tmp_path):
    register_fingerprint_builder(99, DummyFingerprintBuilder)

    fp = Fingerprints(version=99, root=tmp_path)
    path = fp.build(overwrite=True)

    assert path == tmp_path / "data" / "fingerprints" / "v99" / "canonical_fingerprint.parquet"
    assert path.exists()
    assert fp.frame().loc[0, "keys_team"] == "A"


def test_default_fingerprint_split_keeps_market_eval_only(tmp_path):
    fp = Fingerprints(version=0, root=tmp_path)
    frame = pd.DataFrame(
        {
            "keys_season": [2024, 2024],
            "keys_team": ["Home", "Away"],
            "keys_week": [1, 1],
            "keys_game_id": [10, 10],
            "keys_opponent": ["Away", "Home"],
            "game_is_home": [True, False],
            "offense_ppa": [0.1, -0.1],
            "defense_ppa": [-0.2, 0.2],
            "market_spread_close": [-7.5, 7.5],
            "market_over_under": [48.5, 48.5],
            "y_next_margin": [10.0, -10.0],
            "y_has_next_game": [True, True],
        }
    )

    X, y, meta_df, market_df = fp.split_frame(frame)

    assert "offense_ppa" in X.columns
    assert "defense_ppa" in X.columns
    assert "market_spread_close" not in X.columns
    assert "market_over_under" not in X.columns
    assert "y_next_margin" not in X.columns
    assert "market_spread_close" in market_df.columns
    assert "y_next_margin" not in meta_df.columns
    assert y.tolist() == [10.0, -10.0]


def test_matchup_eval_metrics_include_market_and_model_buckets():
    register_model_family("dummy_metric_smoke", DummyModel)
    evaluator = TDEval(
        {"model": {"family": "dummy_metric_smoke"}},
        fingerprints=object(),
        matchup_builder=object(),
    )
    result_df = pd.DataFrame(
        {
            "pred_margin": [1.0, 8.0, -15.0, -24.0],
            "y": [3.0, -7.0, -21.0, 10.0],
            "market_spread_close": [-2.5, -6.0, 14.0, 21.0],
        }
    )

    metrics = evaluator._metrics_from_result(result_df)

    assert "market_rmse" in metrics
    assert "favorite_0_3_count" in metrics
    assert "favorite_7_14_accuracy" in metrics
    assert metrics["ats_accuracy"] == 0.25
    assert metrics["ats_n"] == 4
    assert metrics["n_rows"] == 4


def test_vegas_spread_normalizes_to_home_margin_convention():
    frame = pd.DataFrame({"market_spread_close": [-7.5, 3.0]})

    normalized = normalize_vegas_frame(frame)

    assert normalized["market_home_margin_close"].tolist() == [7.5, -3.0]
    assert market_home_margin(frame).tolist() == [7.5, -3.0]


def test_eval_market_metrics_use_normalized_home_margin():
    register_model_family("dummy_market_smoke", DummyModel)
    evaluator = TDEval(
        {"model": {"family": "dummy_market_smoke"}},
        fingerprints=object(),
        matchup_builder=object(),
    )
    result_df = pd.DataFrame(
        {
            "pred_margin": [7.0, -3.0],
            "y": [10.0, -6.0],
            "market_spread_close": [-8.0, 2.0],
        }
    )

    metrics = evaluator._metrics_from_result(result_df)

    assert metrics["market_mae"] == 3.0
    assert np.isclose(metrics["market_rmse"], np.sqrt(10.0))
