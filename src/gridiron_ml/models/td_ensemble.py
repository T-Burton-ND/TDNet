"""Frozen historical ensemble model family."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .names import normalize_identifier, normalize_model_family


class TDEnsemble:
    """M7 nonnegative weighted probability or margin ensemble."""

    def __init__(self, config=None, members=None):
        self.config = dict(config or {})
        self.model_family = normalize_model_family("ensemble")
        self.model_type = normalize_identifier(
            self.config.get("model_type", self.config.get("variant", "mean_probability"))
        )
        self.model_name = normalize_identifier(
            self.config.get("model_name", self.model_type)
        )
        self.members = list(members or [])
        self.weights_ = None
        self.feature_names_ = []
        self.is_trained_ = False

    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None, **kwargs):
        if not self.members or any(not getattr(model, "is_trained_", False) for model in self.members):
            raise ValueError("TDEnsemble members must be supplied and already trained.")
        frame = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
        self.feature_names_ = list(frame.columns)
        configured = self.config.get("weights")
        if configured is None:
            self.weights_ = np.full(len(self.members), 1.0 / len(self.members))
        else:
            weights = np.asarray(configured, dtype=float)
            if len(weights) != len(self.members) or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("Ensemble weights must be nonnegative and match members.")
            self.weights_ = weights / weights.sum()
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
        margins = np.column_stack([model.predict_margin(X) for model in self.members])
        if self.model_type == "median_margin":
            return np.median(margins, axis=1)
        return np.average(margins, axis=1, weights=self.weights_)

    def predict_proba(self, X):
        self._assert_trained()
        probabilities = np.column_stack(
            [model.predict_proba(X)[:, 1] for model in self.members]
        )
        home = np.average(probabilities, axis=1, weights=self.weights_)
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
        for extra in [meta_df, market_df]:
            if extra is not None:
                extra = pd.DataFrame(extra).reset_index(drop=True)
                keep = [c for c in extra.columns if c not in out.columns]
                out = pd.concat([extra.loc[:, keep], out], axis=1)
        return out

    def rank(self, X, meta_df=None):
        """Rank every team for the weekly poll using the ensemble margin."""
        ranked = self.predict(X, meta_df=meta_df).rename(columns={"pred_margin": "score"})
        return ranked.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)

    def total_rank(self, X, meta_df=None):
        return self.rank(X, meta_df=meta_df)

    def top25(self, X, meta_df=None):
        return self.total_rank(X, meta_df=meta_df).head(25).reset_index(drop=True)

    def save(self, path):
        self._assert_trained()
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
        return model

    def get_metadata(self):
        return {
            "model_family": self.model_family,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "weights": self.weights_.tolist() if self.weights_ is not None else None,
            "members": [model.get_metadata() for model in self.members],
            "is_fitted": self.is_trained_,
            "config": self.config,
        }

    def _assert_trained(self):
        if not self.is_trained_:
            raise RuntimeError("TDEnsemble is not trained.")
