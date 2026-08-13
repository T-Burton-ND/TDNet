"""Transparent M0 constant and home-team baselines."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import yaml

from .names import normalize_identifier, normalize_model_family


class TDNaive:
    """Naive model family with the same outputs as TDLinear/TDTree/TDStat."""

    def __init__(self, config=None):
        self.config = self._load_config(config)
        self.params = dict(self.config.get("params", {}))
        self.model_family = normalize_model_family("naive")
        self.model_type = normalize_identifier(
            self.config.get("model_type", self.config.get("variant", "majority"))
        )
        if self.model_type not in {"majority", "constant_margin", "home_team"}:
            raise ValueError(f"Unsupported TDNaive model_type='{self.model_type}'.")
        self.model_name = normalize_identifier(
            self.config.get("model_name", f"naive_{self.model_type}")
        )
        self.loss_function = str(self.config.get("loss_function", "RMSE"))
        self.margin_temperature = float(self.config.get("margin_temperature", 14.0))
        self.seed = int(self.config.get("seed", 42))
        self.feature_names_ = []
        self.constant_margin_ = 0.0
        self.home_probability_ = 0.5
        self.training_history_ = pd.DataFrame()
        self.is_trained_ = False

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        market_train=None,
        market_val=None,
        sample_weight=None,
    ):
        X = self._coerce_features(X_train)
        y = pd.to_numeric(pd.Series(y_train), errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("TDNaive targets must be finite.")
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.feature_names_ = list(X.columns)
        self.constant_margin_ = float(np.average(y, weights=weights))
        self.home_probability_ = float(np.average(y > 0, weights=weights))
        if self.model_type == "home_team":
            self.constant_margin_ = max(0.01, self.constant_margin_)
            self.home_probability_ = 1.0 - 1e-8
        self.training_history_ = pd.DataFrame(
            [{"split": "train", "n_rows": len(y), "mean_margin": self.constant_margin_}]
        )
        self.is_trained_ = True
        return self

    def fit(self, X_train, y_train, sample_weight=None, X_val=None, y_val=None, **kwargs):
        return self.train(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weight=sample_weight,
            **kwargs,
        )

    def predict_margin(self, X):
        self._assert_trained()
        rows = len(self._coerce_features(X, align=True))
        return np.full(rows, self.constant_margin_, dtype=float)

    def predict_proba(self, X):
        self._assert_trained()
        rows = len(self._coerce_features(X, align=True))
        home = np.full(rows, np.clip(self.home_probability_, 1e-8, 1.0 - 1e-8))
        return np.column_stack([1.0 - home, home])

    def predict(self, X, meta_df=None, market_df=None):
        margin = self.predict_margin(X)
        home = self.predict_proba(X)[:, 1]
        out = pd.DataFrame(
            {
                "pred_margin": margin,
                "pred_proba_home_win": home,
                "pred_pick_home": home >= 0.5,
            }
        )
        return self._attach(out, meta_df, market_df)

    def predict_block(self, season_week_df, meta_df=None, market_df=None):
        return self.predict(season_week_df, meta_df=meta_df, market_df=market_df)

    def rank(self, X, meta_df=None):
        """Return a deterministic ballot even for constant baselines."""
        ranked = self.predict(X, meta_df=meta_df).rename(columns={"pred_margin": "score"})
        return ranked.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)

    def total_rank(self, X, meta_df=None):
        return self.rank(X, meta_df=meta_df)

    def top25(self, X, meta_df=None):
        return self.total_rank(X, meta_df=meta_df).head(25).reset_index(drop=True)

    def save(self, path):
        if not self.is_trained_:
            raise RuntimeError("Cannot save an untrained TDNaive model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        return path

    @classmethod
    def load(cls, path, config=None):
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}.")
        if config is not None:
            model.config = cls._load_config(config)
        return model

    @classmethod
    def from_yaml(cls, path):
        return cls(path)

    def get_metadata(self):
        return {
            "model_family": self.model_family,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "feature_count": len(self.feature_names_),
            "constant_margin": self.constant_margin_,
            "home_probability": self.home_probability_,
            "seed": self.seed,
            "is_fitted": self.is_trained_,
            "config": self.config,
        }

    def get_feature_importance(self):
        return pd.DataFrame(columns=["feature", "importance"])

    def _coerce_features(self, X, align=False):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if align:
            missing = [c for c in self.feature_names_ if c not in frame.columns]
            if missing:
                raise ValueError(f"Missing TDNaive features: {missing[:8]}")
            frame = frame.loc[:, self.feature_names_]
        return frame.reset_index(drop=True)

    def _attach(self, out, meta_df, market_df):
        for extra in [meta_df, market_df]:
            if extra is None:
                continue
            frame = pd.DataFrame(extra).reset_index(drop=True)
            if len(frame) != len(out):
                raise ValueError("Prediction context must align row-wise.")
            keep = [c for c in frame.columns if c not in out.columns]
            out = pd.concat([frame.loc[:, keep], out], axis=1)
        return out

    def _assert_trained(self):
        if not self.is_trained_:
            raise RuntimeError("TDNaive is not trained.")

    @staticmethod
    def _load_config(config):
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        with Path(config).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
