"""Models consuming leakage-safe time-dependent TD fingerprints."""

from __future__ import annotations

from pathlib import Path
import pickle

from .names import normalize_identifier, normalize_model_family
from .td_linear import TDLinear
from .td_tree import TDTree


class TDTemporal:
    """Four estimators built for lag, decayed-state, and trend fingerprints."""

    TYPES = {"decay_ridge", "trend_elastic_net", "temporal_random_forest", "temporal_hist_gradient_boosted"}

    def __init__(self, config=None):
        loaded = TDLinear._load_config(config)
        self.config = loaded
        self.model_family = normalize_model_family("temporal")
        self.model_type = normalize_identifier(loaded.get("model_type", "decay_ridge"))
        if self.model_type not in self.TYPES:
            raise ValueError(f"Unsupported TDTemporal model_type='{self.model_type}'.")
        self.model_name = normalize_identifier(loaded.get("model_name", self.model_type))
        core = dict(loaded)
        core["loss_function"] = "MAE"
        mapping = {
            "decay_ridge": (TDLinear, "ridge"),
            "trend_elastic_net": (TDLinear, "elastic_net"),
            "temporal_random_forest": (TDTree, "random_forest"),
            "temporal_hist_gradient_boosted": (TDTree, "hist_gradient_boosted"),
        }
        cls, model_type = mapping[self.model_type]
        core["model_type"] = model_type
        self.delegate_ = cls(core)
        self.loss_function = str(loaded.get("loss_function", "MAE"))
        self.margin_temperature = float(loaded.get("margin_temperature", self.delegate_.margin_temperature))
        self.training_history_ = self.delegate_.training_history_
        self.feature_names_ = []
        self.is_trained_ = False

    def train(self, X_train, y_train, **kwargs):
        self.delegate_.train(X_train, y_train, **kwargs)
        self.training_history_ = self.delegate_.training_history_
        self.feature_names_ = self.delegate_.feature_names_
        self.is_trained_ = True
        return self

    def fit(self, X_train, y_train, sample_weight=None, X_val=None, y_val=None, **kwargs):
        return self.train(X_train, y_train, X_val=X_val, y_val=y_val, sample_weight=sample_weight, **kwargs)

    def predict(self, X, meta_df=None, market_df=None):
        return self.delegate_.predict(X, meta_df=meta_df, market_df=market_df)

    def predict_margin(self, X):
        return self.delegate_.predict_margin(X)

    def predict_proba(self, X):
        return self.delegate_.predict_proba(X)

    def predict_block(self, X, meta_df=None, market_df=None):
        return self.predict(X, meta_df=meta_df, market_df=market_df)

    def total_rank(self, X, meta_df=None):
        """Expose the shared poll-ranking surface through the delegate."""
        return self.delegate_.total_rank(X, meta_df=meta_df)

    def top25(self, X, meta_df=None):
        return self.delegate_.top25(X, meta_df=meta_df)

    def get_feature_importance(self):
        return self.delegate_.get_feature_importance()

    def loss_breakdown(self, pred_margin, y_true):
        return self.delegate_.loss_breakdown(pred_margin, y_true)

    def margin_to_probability(self, margin):
        return self.delegate_.margin_to_probability(margin)

    def get_metadata(self):
        return {"model_family": self.model_family, "model_type": self.model_type,
                "model_name": self.model_name, "is_fitted": self.is_trained_,
                "temporal_fingerprint": self.config.get("temporal_fingerprint", {}),
                "delegate": self.delegate_.get_metadata(), "config": self.config}

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle: pickle.dump(self, handle)
        return path

    @classmethod
    def load(cls, path, config=None):
        with Path(path).open("rb") as handle: model = pickle.load(handle)
        if not isinstance(model, cls): raise TypeError(f"Expected {cls.__name__}.")
        return model

    @classmethod
    def from_yaml(cls, path):
        return cls(path)
