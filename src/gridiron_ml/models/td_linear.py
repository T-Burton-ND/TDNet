"""src.gridiron_ml.models.td_linear.

Usage:
    Import ``TDLinear`` directly or build it through ``gridiron_ml.models`` with
    a YAML config under ``configs/models/linear``.

Logic flow:
    1. Normalize the requested sklearn linear algorithm and config.
    2. Median-impute and optionally standardize matchup fingerprint features.
    3. Fit the selected sklearn regressor to home-margin targets.
    4. Emit TDNet-standard margin, win-probability, and pick columns.

Responsibility:
    Wrap actual sklearn linear-model algorithms behind the shared TDNet model
    interface used by TDEval, notebooks, checkpoints, and TD Sim.
"""

from inspect import signature
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    SGDClassifier,
)
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridiron_ml.pipeline.schemas import validate_training_feature_frame
from gridiron_ml.pipeline.validation.leakage import training_allows_market_features

from .features import ModelFeatureAdapter
from .names import normalize_identifier, normalize_model_family


DEFAULT_LOSS_WEIGHTS = {
    "alpha": 0.25,
    "beta": 1.0,
    "gamma": 3.0,
    "delta": 0.25,
}

FAVORITE_BUCKETS = [
    ("0_3", 0.0, 3.0),
    ("3_7", 3.0, 7.0),
    ("7_14", 7.0, 14.0),
    ("14_21", 14.0, 21.0),
    ("21_plus", 21.0, np.inf),
]

LINEAR_DEFAULTS = {
    "ols": {},
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 0.01, "max_iter": 10000, "tol": 1e-4, "selection": "cyclic"},
    "elastic_net": {
        "alpha": 0.01,
        "l1_ratio": 0.5,
        "max_iter": 10000,
        "tol": 1e-4,
        "selection": "cyclic",
    },
    "huber": {"epsilon": 1.35, "alpha": 0.0001, "max_iter": 1000, "tol": 1e-5},
    "bayesian": {"max_iter": 300, "tol": 1e-3},
    "ard": {"max_iter": 300, "tol": 1e-3},
    "ransac": {"min_samples": 0.5, "max_trials": 100, "stop_probability": 0.99},
    "orthogonal_matching_pursuit": {},
    "sgd": {
        "loss": "squared_error",
        "penalty": "l2",
        "alpha": 0.0001,
        "max_iter": 2000,
        "tol": 1e-3,
        "learning_rate": "invscaling",
        "eta0": 0.01,
    },
    "passive_aggressive": {
        "C": 1.0,
        "max_iter": 2000,
        "tol": 1e-3,
        "loss": "epsilon_insensitive",
    },
}

RANDOM_STATE_MODELS = {
    "ridge",
    "lasso",
    "elastic_net",
    "ransac",
    "sgd",
    "passive_aggressive",
}


class TDLinear:
    """Train and serve sklearn linear-regression algorithms for TDNet."""

    def __init__(self, config=None):
        """Initialize a TDLinear model from a dict or YAML-loaded config."""
        self.config = self._load_config(config)
        self.model_family = normalize_model_family("linear")
        self.model_type = self._normalize_model_type(
            self.config.get("model_type", self.config.get("variant", "ols"))
        )
        self.model_name = normalize_identifier(
            self.config.get("model_name", self.config.get("name", self.model_type))
        )
        self.loss_function = self._normalize_loss(
            self.config.get("loss_function", "WinnerUpsetAccuracy")
        )

        self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
        self.loss_weights.update(dict(self.config.get("loss_weights", {})))

        self.margin_temperature = float(self.config.get("margin_temperature", 14.0))
        self.prediction_margin_cap = self._normalize_margin_cap(
            self.config.get("prediction_margin_cap", 30.0)
        )
        self.favorite_confidence_scale = float(
            self.config.get("favorite_confidence_scale", 4.0)
        )
        self.huber_delta = float(self.config.get("huber_delta", 7.0))
        self.seed = int(
            self.config.get(
                "seed", self.config.get("params", {}).get("random_state", 42)
            )
        )

        self.training = dict(self.config.get("training", {}))
        self.standardize = bool(self.training.get("standardize", True))
        self.use_sample_weights = bool(self.training.get("use_sample_weights", False))
        self.sample_weight_column = self.training.get("sample_weight_column")
        self.allow_market_features_for_training = training_allows_market_features(
            self.config
        )
        self.feature_adapter = ModelFeatureAdapter()
        self.params = self._model_params()

        self.feature_names_ = []
        self.medians_ = pd.Series(dtype=float)
        self.means_ = pd.Series(dtype=float)
        self.stds_ = pd.Series(dtype=float)
        self.weights_ = np.array([], dtype=float)
        self.intercept_ = 0.0
        self.feature_importances_ = pd.DataFrame()
        self.training_history_ = pd.DataFrame()
        self.is_trained_ = False
        self.pipeline_ = None
        self.model_ = None

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
        """Fit the configured sklearn estimator on numeric matchup features."""
        X_train = self._coerce_feature_df(X_train)
        y_train = self._coerce_target(y_train)
        X_val = self._coerce_feature_df(X_val, allow_none=True)
        y_val = self._coerce_target(y_val, allow_none=True)
        market_train = self._coerce_market_df(market_train)
        market_val = self._coerce_market_df(market_val)
        self._assert_training_features_are_safe(X_train, X_val)

        objective_weight = self._objective_sample_weight(y_train, market_train)
        if sample_weight is None:
            sample_weight = objective_weight
        elif objective_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float) * objective_weight
        if (
            self.use_sample_weights
            and self.sample_weight_column
            and self.sample_weight_column in X_train.columns
        ):
            configured_weight = (
                pd.to_numeric(X_train.pop(self.sample_weight_column), errors="coerce")
                .fillna(1.0)
                .to_numpy(dtype=float)
            )
            sample_weight = (
                configured_weight
                if sample_weight is None
                else sample_weight * configured_weight
            )

        self.feature_names_ = list(X_train.columns)
        self.medians_ = X_train.median(numeric_only=True).fillna(0.0)
        X_filled = X_train.fillna(self.medians_)
        self.means_ = X_filled.mean(numeric_only=True).fillna(0.0)
        self.stds_ = X_filled.std(numeric_only=True).replace(0.0, 1.0).fillna(1.0)

        pipeline = self._build_pipeline()
        fit_kwargs = self._fit_kwargs(pipeline, sample_weight)
        y_fit = self._training_target(y_train)
        pipeline.fit(X_train.loc[:, self.feature_names_], y_fit, **fit_kwargs)

        self.pipeline_ = pipeline
        self.model_ = pipeline
        self.is_trained_ = True
        self._capture_linear_coefficients()
        self.feature_importances_ = self.get_feature_importance()

        rows = [
            self._history_row(
                split="train",
                y_true=y_train,
                y_pred=self._predict_margin_array(X_train),
            )
        ]
        if X_val is not None and y_val is not None and len(X_val) > 0:
            rows.append(
                self._history_row(
                    split="val",
                    y_true=y_val,
                    y_pred=self._predict_margin_array(X_val),
                )
            )
        self.training_history_ = pd.DataFrame(rows)
        return self

    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        X_val=None,
        y_val=None,
        market_train=None,
        market_val=None,
    ):
        """Fit through the publication-standard interface.

        ``train`` remains the canonical legacy entry point; this alias keeps
        existing callers and checkpoints compatible while allowing every model
        family to share one experiment runner.
        """
        return self.train(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            market_train=market_train,
            market_val=market_val,
            sample_weight=sample_weight,
        )

    def predict(self, X, meta_df=None, market_df=None):
        """Predict TDNet-standard margin, probability, and pick columns."""
        X = self._coerce_feature_df(X)
        margin = self._predict_margin_array(X)
        proba = self.margin_to_probability(margin)

        pred_df = pd.DataFrame(
            {
                "pred_margin": margin,
                "pred_proba_home_win": proba,
                "pred_pick_home": (margin > 0).astype(int),
            }
        )
        return self._attach_context(pred_df, meta_df=meta_df, market_df=market_df)

    def predict_proba(self, X):
        """Return two-column away/home probabilities for sklearn compatibility."""
        margin = self._predict_margin_array(self._coerce_feature_df(X))
        home = np.clip(self.margin_to_probability(margin), 1e-8, 1.0 - 1e-8)
        return np.column_stack([1.0 - home, home])

    def predict_margin(self, X):
        """Return signed home margins under the unified model contract."""
        return self._predict_margin_array(self._coerce_feature_df(X))

    def predict_block(self, season_week_df, meta_df=None, market_df=None):
        """Predict a season/week block using the shared TDNet interface."""
        return self.predict(season_week_df, meta_df=meta_df, market_df=market_df)

    def rank(self, X, meta_df=None):
        """Rank rows by predicted margin score descending."""
        rank_df = self.predict(X, meta_df=meta_df)
        rank_df = rank_df.rename(columns={"pred_margin": "score"})
        return rank_df.sort_values("score", ascending=False).reset_index(drop=True)

    def total_rank(self, X, meta_df=None):
        """Alias for rank used by poll-building code."""
        return self.rank(X, meta_df=meta_df)

    def top25(self, X, meta_df=None):
        """Return the top 25 rows under the model's ranking score."""
        return self.total_rank(X, meta_df=meta_df).head(25).reset_index(drop=True)

    def save(self, path):
        """Pickle the trained model wrapper to a checkpoint path."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            pickle.dump(self, f)
        return save_path

    @classmethod
    def load(cls, path, config=None):
        """Load a pickled TDLinear checkpoint."""
        with Path(path).open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(
                f"Expected a saved {cls.__name__} instance, got {type(model).__name__}."
            )
        if config is not None:
            model.config = model._load_config(config)
        return model

    @classmethod
    def from_yaml(cls, config_path):
        """Build a TDLinear model from a YAML config path."""
        return cls(config=cls._load_config(config_path))

    def loss_breakdown(self, pred_margin, y_true):
        """Compute common TDNet regression, winner, and calibration metrics."""
        pred = np.asarray(pred_margin, dtype=float).reshape(-1)
        y = np.asarray(pd.Series(y_true), dtype=float).reshape(-1)
        valid = np.isfinite(pred) & np.isfinite(y)
        pred = pred[valid]
        y = y[valid]
        if len(pred) == 0:
            raise ValueError(
                "loss_breakdown requires at least one finite prediction and target."
            )

        actual = (y > 0).astype(float)
        prob = self.margin_to_probability(pred)
        eps = 1e-8

        err = pred - y
        abs_err = np.abs(err)
        margin_loss = float(
            np.mean(
                np.where(
                    abs_err <= self.huber_delta,
                    0.5 * err**2 / self.huber_delta,
                    abs_err - 0.5 * self.huber_delta,
                )
            )
        )
        win_probability_loss = float(
            np.mean(
                -(
                    actual * np.log(np.clip(prob, eps, 1.0))
                    + (1.0 - actual) * np.log(np.clip(1.0 - prob, eps, 1.0))
                )
            )
        )
        confidence = np.abs(prob - 0.5) * 2.0
        favorite_weight = 1.0 + self.favorite_confidence_scale * confidence
        row_bce = self._row_bce(prob, actual)
        favorite_correctness_loss = float(np.mean(row_bce * favorite_weight))
        calibration_loss = float(np.mean((prob - actual) ** 2))

        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(abs_err))
        winner_accuracy = float(((pred > 0) == (y > 0)).mean())
        brier_score = calibration_loss

        selected = self.loss_function
        if selected == "RMSE":
            total_loss = rmse
        elif selected == "MAE":
            total_loss = mae
        elif selected == "WinnerAccuracy":
            total_loss = 1.0 - winner_accuracy
        elif selected == "WinnerUpsetAccuracy":
            total_loss = 1.0 - winner_accuracy
        else:
            total_loss = (
                float(self.loss_weights["alpha"]) * margin_loss
                + float(self.loss_weights["beta"]) * win_probability_loss
                + float(self.loss_weights["gamma"]) * favorite_correctness_loss
                + float(self.loss_weights["delta"]) * calibration_loss
            )

        out = {
            "loss_function": selected,
            "total_loss": float(total_loss),
            "margin_loss": margin_loss,
            "win_probability_loss": win_probability_loss,
            "favorite_correctness_loss": favorite_correctness_loss,
            "calibration_loss": calibration_loss,
            "mae": mae,
            "rmse": rmse,
            "winner_accuracy": winner_accuracy,
            "brier_score": brier_score,
        }
        out.update(self.favorite_bucket_metrics(pred, y))
        return out

    def favorite_bucket_metrics(self, pred_margin, y_true):
        """Summarize winner accuracy by absolute predicted-margin bucket."""
        pred = np.asarray(pred_margin, dtype=float).reshape(-1)
        y = np.asarray(pd.Series(y_true), dtype=float).reshape(-1)
        abs_margin = np.abs(pred)
        correct = ((pred > 0) == (y > 0)).astype(float)

        out = {}
        for label, low, high in FAVORITE_BUCKETS:
            if np.isinf(high):
                mask = abs_margin >= low
            else:
                mask = (abs_margin >= low) & (abs_margin < high)
            out[f"favorite_{label}_count"] = int(mask.sum())
            out[f"favorite_{label}_accuracy"] = (
                float(correct[mask].mean()) if mask.any() else np.nan
            )
        return out

    def margin_to_probability(self, margin):
        """Convert predicted margin to home-win probability with a logistic link."""
        margin = np.asarray(margin, dtype=float)
        logits = np.clip(margin / max(self.margin_temperature, 1e-8), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def get_feature_importance(self):
        """Return absolute linear coefficients as feature-importance rows."""
        estimator = self._coefficient_estimator()
        if estimator is None or not hasattr(estimator, "coef_"):
            return pd.DataFrame(columns=["feature", "coefficient", "importance"])
        coef = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        if len(coef) != len(self.feature_names_):
            return pd.DataFrame(columns=["feature", "coefficient", "importance"])
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "coefficient": coef,
                    "importance": np.abs(coef),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_metadata(self):
        """Return checkpoint-safe model identity and fitted-state metadata."""
        return {
            "model_family": self.model_family,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "loss_function": self.loss_function,
            "seed": self.seed,
            "feature_count": len(self.feature_names_),
            "feature_names": list(self.feature_names_),
            "is_fitted": bool(self.is_trained_),
            "params": dict(self.params),
            "config": dict(self.config),
        }

    def _build_pipeline(self):
        """Build the imputer/scaler/estimator sklearn pipeline."""
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if self.standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("estimator", self._build_estimator()))
        return Pipeline(steps=steps)

    def _build_estimator(self):
        """Instantiate the selected sklearn linear estimator."""
        params = dict(self.params)
        if self.model_type in RANDOM_STATE_MODELS:
            params.setdefault("random_state", self.seed)

        if self._is_classifier_objective():
            return self._build_classifier_estimator(params)

        if self.model_type == "ols":
            return LinearRegression(**params)
        if self.model_type == "ridge":
            return Ridge(**params)
        if self.model_type == "lasso":
            return Lasso(**params)
        if self.model_type == "elastic_net":
            return ElasticNet(**params)
        if self.model_type == "huber":
            return HuberRegressor(**params)
        if self.model_type == "bayesian":
            return BayesianRidge(**params)
        if self.model_type == "ard":
            return ARDRegression(**params)
        if self.model_type == "ransac":
            estimator_params = dict(params.pop("estimator_params", {}) or {})
            base_estimator = normalize_identifier(params.pop("base_estimator", "linear"))
            if base_estimator in {"ridge", "ridge_regression"}:
                estimator = Ridge(**estimator_params)
            elif base_estimator in {"linear", "ols", "linear_regression"}:
                estimator = LinearRegression(**estimator_params)
            else:
                raise ValueError(
                    f"Unsupported RANSAC base_estimator='{base_estimator}'."
                )
            return RANSACRegressor(estimator=estimator, **params)
        if self.model_type == "orthogonal_matching_pursuit":
            return OrthogonalMatchingPursuit(**params)
        if self.model_type == "sgd":
            return SGDRegressor(**params)
        if self.model_type == "passive_aggressive":
            return PassiveAggressiveRegressor(**params)
        raise ValueError(f"Unhandled TDLinear model_type='{self.model_type}'")

    def _build_classifier_estimator(self, params):
        """Instantiate a classification surrogate for winner-focused objectives."""
        params = dict(params)
        params.setdefault("random_state", self.seed)
        if self.model_type == "sgd":
            allowed = {
                "penalty",
                "alpha",
                "max_iter",
                "tol",
                "random_state",
                "fit_intercept",
                "class_weight",
                "n_jobs",
                "shuffle",
                "learning_rate",
                "eta0",
            }
            params = {k: v for k, v in params.items() if k in allowed}
            return SGDClassifier(
                loss=str(self.config.get("classifier_loss", "log_loss")),
                penalty=str(params.pop("penalty", "l2")),
                alpha=float(params.pop("alpha", 0.0001)),
                max_iter=int(params.pop("max_iter", 2000)),
                tol=params.pop("tol", 1e-3),
                random_state=int(params.pop("random_state", self.seed)),
                fit_intercept=bool(params.pop("fit_intercept", True)),
                **params,
            )
        if self.model_type == "passive_aggressive":
            allowed = {
                "C",
                "max_iter",
                "tol",
                "random_state",
                "fit_intercept",
                "class_weight",
                "n_jobs",
                "shuffle",
                "average",
            }
            params = {k: v for k, v in params.items() if k in allowed}
            return PassiveAggressiveClassifier(
                C=float(params.pop("C", 1.0)),
                max_iter=int(params.pop("max_iter", 2000)),
                tol=params.pop("tol", 1e-3),
                random_state=int(params.pop("random_state", self.seed)),
                fit_intercept=bool(params.pop("fit_intercept", True)),
                **params,
            )

        alpha = float(params.pop("alpha", 1.0) or 1.0)
        l1_ratio = float(params.pop("l1_ratio", 0.5) or 0.5)
        allowed = {
            "C",
            "max_iter",
            "tol",
            "fit_intercept",
            "random_state",
            "class_weight",
            "n_jobs",
        }
        params = {k: v for k, v in params.items() if k in allowed}
        logistic_params = {
            "C": float(params.pop("C", 1.0 / max(alpha, 1e-8))),
            "max_iter": int(params.pop("max_iter", 2000)),
            "tol": float(params.pop("tol", 1e-4)),
            "fit_intercept": bool(params.pop("fit_intercept", True)),
            "random_state": int(params.pop("random_state", self.seed)),
        }
        if self.model_type == "lasso":
            logistic_params.update({"penalty": "l1", "solver": "liblinear"})
        elif self.model_type == "elastic_net":
            logistic_params.update(
                {
                    "penalty": "elasticnet",
                    "solver": "saga",
                    "l1_ratio": l1_ratio,
                }
            )
        else:
            logistic_params.update({"penalty": "l2", "solver": "lbfgs"})
        return LogisticRegression(**logistic_params)

    def _model_params(self):
        """Merge default, nested, variant-specific, and explicit estimator params."""
        params = dict(LINEAR_DEFAULTS[self.model_type])
        tdlinear_cfg = dict(self.config.get("tdlinear", {}) or {})
        params.update(dict(tdlinear_cfg.get(self.model_type, {}) or {}))
        params.update(dict(self.config.get(self.model_type, {}) or {}))
        params.update(dict(self.config.get("params", {}) or {}))

        for key in ["model_type", "variant", "seed"]:
            params.pop(key, None)
        return {key: value for key, value in params.items() if value is not None}

    def _fit_kwargs(self, pipeline, sample_weight):
        """Return sklearn pipeline fit kwargs for supported sample weights."""
        if sample_weight is None:
            return {}
        estimator = pipeline.named_steps["estimator"]
        if "sample_weight" in signature(estimator.fit).parameters:
            return {"estimator__sample_weight": sample_weight}
        return {}

    def _history_row(self, split, y_true, y_pred):
        """Build one train/validation history row from shared metrics."""
        metrics = self.loss_breakdown(y_pred, y_true)
        row = {
            "split": split,
            "model_type": self.model_type,
            "n_features": int(len(self.feature_names_)),
            "optimized_loss": metrics["total_loss"],
        }
        row.update(metrics)
        return row

    def _predict_margin_array(self, X):
        """Return raw margin predictions from the fitted sklearn pipeline."""
        if not self.is_trained_:
            raise RuntimeError("Model is not trained. Call train() first.")
        return self._cap_margin(self._predict_margin_with_pipeline(X, self.pipeline_))

    def _predict_margin_with_pipeline(self, X, pipeline):
        """Align input columns and call the fitted sklearn pipeline."""
        feature_names = list(getattr(self, "feature_names_", X.columns))
        aligned = self._feature_adapter().align_frame(X, feature_names)
        if self._is_classifier_objective():
            if hasattr(pipeline, "predict_proba"):
                proba = np.asarray(pipeline.predict_proba(aligned), dtype=float)
                if proba.ndim == 2 and proba.shape[1] > 1:
                    p_home = proba[:, 1]
                else:
                    p_home = proba.reshape(-1)
                p_home = np.clip(p_home, 1e-8, 1.0 - 1e-8)
                return np.log(p_home / (1.0 - p_home)) * self.margin_temperature
            if hasattr(pipeline, "decision_function"):
                return (
                    np.asarray(
                        pipeline.decision_function(aligned), dtype=float
                    ).reshape(-1)
                    * self.margin_temperature
                )
        return np.asarray(pipeline.predict(aligned), dtype=float).reshape(-1)

    def _capture_linear_coefficients(self):
        """Store estimator coefficients and intercept when the algorithm exposes them."""
        estimator = self._coefficient_estimator()
        if estimator is None or not hasattr(estimator, "coef_"):
            self.weights_ = np.array([], dtype=float)
            self.intercept_ = 0.0
            return
        self.weights_ = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        self.intercept_ = float(
            np.asarray(getattr(estimator, "intercept_", 0.0), dtype=float).reshape(-1)[
                0
            ]
        )

    def _coefficient_estimator(self):
        """Return the fitted estimator that owns linear coefficients, if available."""
        estimator = self._estimator()
        if estimator is None:
            return None
        if self.model_type == "ransac":
            return getattr(estimator, "estimator_", None)
        return estimator

    def _estimator(self):
        """Return the final estimator from the fitted pipeline."""
        pipeline = getattr(self, "pipeline_", None)
        if pipeline is None:
            return None
        if hasattr(pipeline, "named_steps") and "estimator" in pipeline.named_steps:
            return pipeline.named_steps["estimator"]
        return pipeline

    def _transform_features(self, X):
        """Apply TDLinear's fitted median fill and optional standardization."""
        X = self._feature_adapter().align_frame(X, self.feature_names_)
        X = X.fillna(self.medians_)
        if self.standardize:
            X = (X - self.means_) / self.stds_
        return X.to_numpy(dtype=np.float32)

    def _attach_context(self, pred_df, meta_df=None, market_df=None):
        """Attach aligned metadata and market columns to prediction output."""
        out = pred_df.reset_index(drop=True)
        if meta_df is not None:
            meta_df = meta_df.reset_index(drop=True)
            if len(meta_df) != len(out):
                raise ValueError("meta_df must align row-wise with prediction output.")
            out = pd.concat([meta_df, out], axis=1)
        if market_df is not None:
            market_df = market_df.reset_index(drop=True)
            if len(market_df) != len(out):
                raise ValueError(
                    "market_df must align row-wise with prediction output."
                )
            keep = [c for c in market_df.columns if c not in out.columns]
            out = pd.concat([out, market_df.loc[:, keep]], axis=1)
        return out.reset_index(drop=True)

    def _coerce_feature_df(self, X, allow_none=False):
        """Coerce model features to a numeric pandas dataframe."""
        return self._feature_adapter().coerce_frame(X, allow_none=allow_none)

    def _feature_adapter(self):
        """Return the feature adapter, creating it for older checkpoints."""

        adapter = getattr(self, "feature_adapter", None)
        if adapter is None:
            adapter = ModelFeatureAdapter()
            self.feature_adapter = adapter
        return adapter

    def _coerce_target(self, y, allow_none=False):
        """Coerce target values to a one-dimensional numeric pandas series."""
        if y is None:
            if allow_none:
                return None
            raise ValueError("Target values are required.")
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Target dataframe must contain exactly one column.")
            y = y.iloc[:, 0]
        return pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce")

    def _coerce_market_df(self, market_df):
        if market_df is None:
            return None
        return pd.DataFrame(market_df).reset_index(drop=True)

    def _assert_training_features_are_safe(self, X_train, X_val=None):
        validate_training_feature_frame(
            X_train,
            allow_market_features_for_training=self.allow_market_features_for_training,
        )
        if X_val is not None:
            validate_training_feature_frame(
                X_val,
                allow_market_features_for_training=self.allow_market_features_for_training,
            )

    def _training_target(self, y):
        """Return estimator target values for margin or winner objectives."""
        y = self._coerce_target(y)
        if self._is_classifier_objective():
            return (y.to_numpy(dtype=float) > 0.0).astype(int)
        return y.to_numpy(dtype=float)

    def _is_classifier_objective(self):
        return self.loss_function in {"WinnerAccuracy", "WinnerUpsetAccuracy"}

    def _objective_sample_weight(self, y, market_df=None):
        """Return row weights for winner/upset objectives."""
        if (
            self.loss_function != "WinnerUpsetAccuracy"
            or market_df is None
            or market_df.empty
        ):
            return None
        from gridiron_ml.td_run.market import (
            DEFAULT_VEGAS_CONVENTION,
            market_home_margin,
            normalize_vegas_frame,
        )

        y = self._coerce_target(y)
        market = normalize_vegas_frame(market_df.copy(), DEFAULT_VEGAS_CONVENTION)
        market_margin = market_home_margin(market, DEFAULT_VEGAS_CONVENTION)
        valid = y.notna() & market_margin.notna()
        weights = np.ones(len(y), dtype=float)
        if valid.any():
            actual_home = y.loc[valid].to_numpy(dtype=float) > 0.0
            market_home = market_margin.loc[valid].to_numpy(dtype=float) > 0.0
            weights[np.flatnonzero(valid.to_numpy())] = np.where(
                actual_home != market_home,
                float(
                    self.config.get(
                        "upset_weight", self.config.get("winner_upset_weight", 3.0)
                    )
                ),
                1.0,
            )
        return weights

    def _cap_margin(self, margin):
        margin = np.asarray(margin, dtype=float).reshape(-1)
        if self.prediction_margin_cap is None:
            return margin
        cap = float(self.prediction_margin_cap)
        return np.clip(margin, -cap, cap)

    @staticmethod
    def _normalize_margin_cap(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {
            "",
            "none",
            "null",
            "false",
        }:
            return None
        cap = abs(float(value))
        return None if cap <= 0.0 else cap

    @staticmethod
    def _row_bce(prob, actual):
        """Compute binary cross entropy per row from probabilities."""
        eps = 1e-8
        return -(
            actual * np.log(np.clip(prob, eps, 1.0))
            + (1.0 - actual) * np.log(np.clip(1.0 - prob, eps, 1.0))
        )

    @staticmethod
    def _normalize_loss(value):
        """Normalize TDNet evaluation-loss labels."""
        aliases = {
            "rmse": "RMSE",
            "mae": "MAE",
            "composite": "Composite",
            "mixed": "Composite",
            "mixed_loss": "Composite",
            "winner": "WinnerAccuracy",
            "winneraccuracy": "WinnerAccuracy",
            "winner_accuracy": "WinnerAccuracy",
            "accuracy": "WinnerAccuracy",
            "classification": "WinnerAccuracy",
            "winner_upset": "WinnerUpsetAccuracy",
            "winnerupsetaccuracy": "WinnerUpsetAccuracy",
            "winner_upset_accuracy": "WinnerUpsetAccuracy",
            "hybrid": "WinnerUpsetAccuracy",
            "hybrid_winner": "WinnerUpsetAccuracy",
            "hybrid_winner_accuracy": "WinnerUpsetAccuracy",
            "hybridwinneraccuracy": "WinnerUpsetAccuracy",
            "upset": "WinnerUpsetAccuracy",
            "upset_accuracy": "WinnerUpsetAccuracy",
        }
        key = str(value).strip().lower()
        if key not in aliases:
            raise ValueError(
                "loss_function must be one of: RMSE, MAE, Composite, WinnerAccuracy, WinnerUpsetAccuracy."
            )
        return aliases[key]

    @staticmethod
    def _normalize_model_type(value):
        """Normalize supported sklearn linear estimator aliases."""
        model_type = str(value).strip().lower().replace("-", "_")
        aliases = {
            "linear": "ols",
            "linear_regression": "ols",
            "ordinary_least_squares": "ols",
            "ols": "ols",
            "ridge": "ridge",
            "ridge_regression": "ridge",
            "lasso": "lasso",
            "elastic": "elastic_net",
            "elasticnet": "elastic_net",
            "elastic_net": "elastic_net",
            "huber": "huber",
            "huber_regressor": "huber",
            "bayesian": "bayesian",
            "bayesian_ridge": "bayesian",
            "ard": "ard",
            "ard_regression": "ard",
            "ransac": "ransac",
            "ransac_regressor": "ransac",
            "omp": "orthogonal_matching_pursuit",
            "orthogonal_matching_pursuit": "orthogonal_matching_pursuit",
            "sgd": "sgd",
            "sgd_regressor": "sgd",
            "passive_aggressive": "passive_aggressive",
            "passive_aggressive_regressor": "passive_aggressive",
        }
        if model_type not in aliases:
            supported = ", ".join(sorted(set(aliases.values())))
            raise ValueError(
                f"Unsupported TDLinear model_type='{value}'. Supported: {supported}"
            )
        return aliases[model_type]

    @staticmethod
    def _load_config(config):
        """Load a TDLinear config from None, dict, or YAML path."""
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        path = Path(config)
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
