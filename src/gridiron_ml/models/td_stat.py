"""src.gridiron_ml.models.td_stat.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Define model wrappers and checkpoint helpers behind the shared TDNet interface.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import yaml

from gridiron_ml.pipeline.schemas import validate_training_feature_frame
from gridiron_ml.pipeline.validation.leakage import training_allows_market_features

from .features import ModelFeatureAdapter
from .names import normalize_identifier, normalize_model_family


class TDStat:
    """Represent the TDStat component and its local behavior."""

    def __init__(self, config=None):
        """Internal helper for the init__ step."""
        self.config = self._load_config(config)
        self.params = dict(self.config.get("params", {}))
        self.model_family = normalize_model_family("stat")
        self.model_type = self._normalize_model_type(
            self.params.get("model_type", "z_index")
        )
        self.model_name = normalize_identifier(
            self.config.get("model_name", f"stat_{self.model_type}")
        )
        self.objective = self._normalize_objective(
            self.params.get(
                "objective",
                self.config.get("loss_function", "winner_upset"),
            )
        )
        self.loss_function = (
            "WinnerUpsetAccuracy"
            if self.objective == "winner_upset"
            else "WinnerAccuracy" if self.objective == "winner" else "RMSE"
        )
        self.center_target = bool(self.params.get("center_target", True))
        self.feature_prefixes = tuple(
            self.params.get(
                "feature_prefixes",
                ["statOff_", "statDef_", "statGen_", "statSpe_"],
            )
        )
        self.feature_include_patterns = tuple(
            self.params.get(
                "feature_include_patterns",
                self.params.get("include_feature_patterns", []),
            )
            or []
        )
        self.feature_exclude_patterns = tuple(
            self.params.get(
                "feature_exclude_patterns",
                self.params.get("exclude_feature_patterns", []),
            )
            or []
        )
        self.require_stat_features = bool(
            self.params.get("require_stat_features", False)
        )
        self.feature_adapter = ModelFeatureAdapter(
            feature_prefixes=self.feature_prefixes,
            include_patterns=self.feature_include_patterns,
            exclude_patterns=self.feature_exclude_patterns,
            require_selected=self.require_stat_features,
        )
        self.margin_scale = self.params.get("margin_scale")
        self.prediction_margin_cap = self._normalize_margin_cap(
            self.params.get(
                "prediction_margin_cap", self.config.get("prediction_margin_cap", 30.0)
            )
        )
        self.min_margin_scale = float(self.params.get("min_margin_scale", 7.0))
        self.seed = int(self.params.get("seed", 42))
        self.robust_scale_method = (
            str(self.params.get("robust_scale_method", "mad")).strip().lower()
        )
        self.robust_clip = self.params.get("robust_clip", 5.0)
        self.percentile_center = bool(self.params.get("percentile_center", True))
        self.weighted_scaling = (
            str(self.params.get("weighted_scaling", "robust")).strip().lower()
        )
        self.winner_target_margin = float(
            self.params.get(
                "winner_target_margin", self.params.get("classification_margin", 14.0)
            )
        )
        self.upset_weight = float(
            self.params.get("upset_weight", self.config.get("upset_weight", 3.0))
        )
        self.allow_market_features_for_training = training_allows_market_features(
            self.config
        )

        self.feature_names_ = []
        self.selected_feature_names_ = []
        self.medians_ = pd.Series(dtype=float)
        self.means_ = pd.Series(dtype=float)
        self.stds_ = pd.Series(dtype=float)
        self.directions_ = pd.Series(dtype=float)
        self.percentile_reference_ = {}
        self.normalization_ = "zscore"
        self.feature_weights_ = pd.Series(dtype=float)
        self.intercept_ = 0.0
        self.beta_ = np.array([], dtype=float)
        self.beta_std_ = None
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
        """Run the train step and return its normalized result."""
        X_train = self._coerce_feature_df(X_train)
        y_train = self._coerce_target(y_train)
        X_val = self._coerce_feature_df(X_val, allow_none=True)
        y_val = self._coerce_target(y_val, allow_none=True)
        market_train = self._coerce_market_df(market_train)
        market_val = self._coerce_market_df(market_val)
        self._assert_training_features_are_safe(X_train, X_val)

        self.feature_names_ = list(X_train.columns)
        self.selected_feature_names_ = self._select_feature_names(self.feature_names_)

        X_train_sel = X_train.loc[:, self.selected_feature_names_].copy()
        self.medians_, self.means_, self.stds_ = self._fit_standardizer(X_train_sel)
        X_train_z = self._transform_features(X_train_sel)

        y_fit = self._training_target(y_train)
        objective_weight = self._objective_sample_weight(y_train, market_train)
        if sample_weight is None:
            sample_weight = objective_weight
        elif objective_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float) * objective_weight
        model = self._fit_model(X_train_z, y_fit, sample_weight=sample_weight)
        self.intercept_ = float(model["intercept"])
        self.beta_ = np.asarray(model["beta"], dtype=float)
        self.beta_std_ = (
            np.asarray(model["beta_std"], dtype=float)
            if model.get("beta_std") is not None
            else None
        )
        self.is_trained_ = True

        train_pred = self._predict_margin_array(X_train)
        if self.margin_scale is None:
            resid_scale = float(np.nanstd(y_train.to_numpy(dtype=float) - train_pred))
            self.margin_scale = max(resid_scale, self.min_margin_scale)
        else:
            self.margin_scale = float(self.margin_scale)

        rows = [
            {
                "split": "train",
                "model_type": self.model_type,
                "n_features": int(len(self.selected_feature_names_)),
                "loss_function": self.loss_function,
                "rmse": self._rmse(y_train, train_pred),
                "mae": self._mae(y_train, train_pred),
                "winner_accuracy": self._winner_accuracy(y_train, train_pred),
                "upset_accuracy": self._upset_accuracy(
                    y_train, train_pred, market_train
                ),
                "bias": self._bias(y_train, train_pred),
            }
        ]

        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_pred = self._predict_margin_array(X_val)
            rows.append(
                {
                    "split": "val",
                    "model_type": self.model_type,
                    "n_features": int(len(self.selected_feature_names_)),
                    "loss_function": self.loss_function,
                    "rmse": self._rmse(y_val, val_pred),
                    "mae": self._mae(y_val, val_pred),
                    "winner_accuracy": self._winner_accuracy(y_val, val_pred),
                    "upset_accuracy": self._upset_accuracy(y_val, val_pred, market_val),
                    "bias": self._bias(y_val, val_pred),
                }
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
        """Fit through the unified publication/legacy model interface."""
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
        """Run the predict step and return its normalized result."""
        X = self._coerce_feature_df(X)
        margin = self._predict_margin_array(X)
        proba = np.clip(self._margin_to_proba(margin), 1e-8, 1.0 - 1e-8)

        pred_df = pd.DataFrame(
            {
                "pred_margin": margin,
                "pred_proba_home_win": proba,
                "pred_pick_home": (margin > 0).astype(int),
            }
        )
        return self._attach_context(pred_df, meta_df=meta_df, market_df=market_df)

    def predict_proba(self, X):
        """Return two-column away/home win probabilities."""
        X = self._coerce_feature_df(X)
        home = np.clip(self._margin_to_proba(self._predict_margin_array(X)), 1e-8, 1.0 - 1e-8)
        return np.column_stack([1.0 - home, home])

    def predict_margin(self, X):
        """Return signed home margins under the unified model contract."""
        return self._predict_margin_array(self._coerce_feature_df(X))

    def get_metadata(self):
        """Return checkpoint-safe model identity and fitted-state metadata."""
        return {
            "model_family": self.model_family,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "objective": self.objective,
            "loss_function": self.loss_function,
            "seed": self.seed,
            "feature_count": len(self.feature_names_),
            "selected_feature_count": len(self.selected_feature_names_),
            "feature_names": list(self.feature_names_),
            "selected_feature_names": list(self.selected_feature_names_),
            "is_fitted": bool(self.is_trained_),
            "params": dict(self.params),
            "config": dict(self.config),
        }

    def predict_block(self, season_week_df, meta_df=None, market_df=None):
        """Run the predict_block step and return its normalized result."""
        return self.predict(season_week_df, meta_df=meta_df, market_df=market_df)

    def rank(self, X, meta_df=None):
        """Run the rank step and return its normalized result."""
        rank_df = self.predict(X, meta_df=meta_df)
        rank_df = rank_df.rename(columns={"pred_margin": "score"})
        return rank_df.sort_values("score", ascending=False).reset_index(drop=True)

    def total_rank(self, X, meta_df=None):
        """Run the total_rank step and return its normalized result."""
        return self.rank(X, meta_df=meta_df)

    def top25(self, X, meta_df=None):
        """Run the top25 step and return its normalized result."""
        return self.total_rank(X, meta_df=meta_df).head(25).reset_index(drop=True)

    def save(self, path=None):
        """Run the save step and return its normalized result."""
        save_path = Path(
            path or self.config.get("model_path", "models/stat/models/tdstat_model.pkl")
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            pickle.dump(self, f)
        return save_path

    @classmethod
    def load(cls, path, config=None):
        """Run the load step and return its normalized result."""
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a saved {cls.__name__} instance, got {type(obj).__name__}."
            )
        model = obj
        if config is not None:
            model.config = model._load_config(config)
            model.params = dict(model.config.get("params", model.params))
        return model

    @classmethod
    def from_yaml(cls, config_path):
        """Run the from_yaml step and return its normalized result."""
        config = cls._load_config(config_path)
        model_path = config.get("model_path")
        if (
            model_path
            and Path(model_path).exists()
            and not bool(config.get("overwrite", False))
        ):
            return cls.load(model_path, config=config)
        return cls(config=config)

    @staticmethod
    def _load_config(config):
        """Internal helper for the load_config step."""
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        path = Path(config)
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if "model_path" in loaded:
            loaded["model_path"] = str((path.parent / loaded["model_path"]).resolve())
        return loaded

    @staticmethod
    def _normalize_model_type(value):
        """Internal helper for the normalize_model_type step."""
        model_type = str(value).strip().lower()
        aliases = {
            "zscore": "z_index",
            "z_index": "z_index",
            "ridge": "ridge",
            "ols": "ols",
            "corr_linear": "corr_linear",
            "bayes_bootstrap_ridge": "bayes_bootstrap_ridge",
            "percentile": "percentile",
            "rank": "percentile",
            "rank_percentile": "percentile",
            "robust": "robust",
            "robust_z": "robust",
            "robust_z_index": "robust",
            "weighted": "weighted",
            "weighted_blend": "weighted",
        }
        if model_type not in aliases:
            supported = ", ".join(sorted(aliases))
            raise ValueError(
                f"Unsupported model_type='{model_type}'. Supported: {supported}"
            )
        return aliases[model_type]

    @staticmethod
    def _normalize_objective(value):
        aliases = {
            "margin": "margin",
            "rmse": "margin",
            "mae": "margin",
            "winner": "winner",
            "winner_accuracy": "winner",
            "accuracy": "winner",
            "classification": "winner",
            "winner_upset": "winner_upset",
            "winner_upset_accuracy": "winner_upset",
            "hybrid": "winner_upset",
            "hybrid_winner": "winner_upset",
            "hybrid_winner_accuracy": "winner_upset",
            "hybridwinneraccuracy": "winner_upset",
            "upset": "winner_upset",
            "upset_accuracy": "winner_upset",
        }
        key = str(value).strip().lower()
        if key not in aliases:
            raise ValueError(
                "TDStat objective must be one of: margin, winner, winner_upset."
            )
        return aliases[key]

    def _fit_model(self, X, y, sample_weight=None):
        """Internal helper for the fit_model step."""
        if self.model_type == "corr_linear":
            return self._fit_weights_correlation(X, y, sample_weight=sample_weight)
        if self.model_type == "ols":
            return self._fit_weights_ols(X, y, sample_weight=sample_weight)
        if self.model_type in {"ridge", "z_index", "percentile", "robust", "weighted"}:
            return self._fit_weights_ridge(X, y, sample_weight=sample_weight)
        if self.model_type == "bayes_bootstrap_ridge":
            return self._fit_weights_bayesian_bootstrap_ridge(
                X, y, sample_weight=sample_weight
            )
        raise ValueError(f"Unhandled model_type='{self.model_type}'")

    def _fit_weights_correlation(self, X, y, sample_weight=None):
        """Internal helper for the fit_weights_correlation step."""
        weights = self._normalize_sample_weight(sample_weight, len(y))
        y_mean = self._weighted_mean(y, weights)
        y_base = y - y_mean if self.center_target else y
        y_std = float(np.sqrt(self._weighted_mean(y_base**2, weights)))
        X_centered = X - np.average(X, axis=0, weights=weights)
        x_std = np.sqrt(np.average(X_centered**2, axis=0, weights=weights))
        cov = np.average(X_centered * y_base[:, None], axis=0, weights=weights)
        denom = x_std * max(y_std, 1e-12)
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-12)

        corr_power = float(self.params.get("corr_power", 1.0))
        weight_scale = float(self.params.get("weight_scale", 1.0))
        l1_shrink = float(self.params.get("l1_shrink", 0.0))
        beta = np.sign(corr) * (np.abs(corr) ** corr_power) * y_std * weight_scale
        if l1_shrink > 0:
            beta = np.sign(beta) * np.maximum(np.abs(beta) - l1_shrink, 0.0)
        return {
            "intercept": float(y_mean) if self.center_target else 0.0,
            "beta": beta,
        }

    def _fit_weights_ols(self, X, y, sample_weight=None):
        """Internal helper for the fit_weights_ols step."""
        weights = self._normalize_sample_weight(sample_weight, len(y))
        y_mean = self._weighted_mean(y, weights)
        y_base = y - y_mean if self.center_target else y
        sqrt_w = np.sqrt(weights)
        beta = np.linalg.pinv(X * sqrt_w[:, None]) @ (y_base * sqrt_w)
        return {
            "intercept": float(y_mean) if self.center_target else 0.0,
            "beta": beta,
        }

    def _fit_weights_ridge(self, X, y, sample_weight=None):
        """Internal helper for the fit_weights_ridge step."""
        weights = self._normalize_sample_weight(sample_weight, len(y))
        y_mean = self._weighted_mean(y, weights)
        y_base = y - y_mean if self.center_target else y
        alpha = float(
            self.params.get(
                "z_ridge_alpha" if self.model_type == "z_index" else "ridge_alpha",
                self.params.get(
                    f"{self.model_type}_ridge_alpha",
                    self.params.get(
                        "alpha",
                        (
                            10.0
                            if self.model_type
                            in {"z_index", "percentile", "robust", "weighted"}
                            else 8.0
                        ),
                    ),
                ),
            )
        )
        p = X.shape[1]
        beta = np.linalg.solve(
            X.T @ (X * weights[:, None]) + alpha * np.eye(p), X.T @ (y_base * weights)
        )
        return {
            "intercept": float(y_mean) if self.center_target else 0.0,
            "beta": beta,
        }

    def _fit_weights_bayesian_bootstrap_ridge(self, X, y, sample_weight=None):
        """Internal helper for the fit_weights_bayesian_bootstrap_ridge step."""
        weights = self._normalize_sample_weight(sample_weight, len(y))
        y_mean = self._weighted_mean(y, weights)
        y_base = y - y_mean if self.center_target else y
        alpha = float(
            self.params.get("bootstrap_ridge_alpha", self.params.get("alpha", 8.0))
        )
        n_boot = int(
            self.params.get("n_boot", self.params.get("bootstrap_rounds", 300))
        )
        aggregate = str(self.params.get("aggregation", "mean")).strip().lower()
        rng = np.random.default_rng(self.seed)

        n, p = X.shape
        betas = np.zeros((n_boot, p), dtype=float)
        probs = weights / weights.sum()
        for idx in range(n_boot):
            sample_idx = rng.choice(np.arange(n), size=n, replace=True, p=probs)
            Xb = X[sample_idx]
            yb = y_base[sample_idx]
            betas[idx] = np.linalg.solve(Xb.T @ Xb + alpha * np.eye(p), Xb.T @ yb)

        beta = (
            np.median(betas, axis=0)
            if aggregate == "median"
            else np.mean(betas, axis=0)
        )
        beta_std = betas.std(axis=0, ddof=1) if n_boot > 1 else np.zeros(p, dtype=float)
        return {
            "intercept": float(y_mean) if self.center_target else 0.0,
            "beta": beta,
            "beta_std": beta_std,
        }

    def _predict_margin_array(self, X):
        """Internal helper for the predict_margin_array step."""
        if not self.is_trained_:
            raise RuntimeError("Model is not trained. Call train() first.")
        X = self._feature_adapter().align_frame(X, self.feature_names_)
        X_sel = X.loc[:, self.selected_feature_names_].copy()
        X_z = self._transform_features(X_sel)
        return self._cap_margin(self.intercept_ + (X_z @ self.beta_))

    def _fit_standardizer(self, X):
        """Internal helper for the fit_standardizer step."""
        medians = X.median(numeric_only=True).fillna(0.0)
        X_filled = X.fillna(medians)
        self.normalization_ = self._normalization_for_model()
        self.directions_ = pd.Series(
            {col: self._feature_direction(col) for col in X_filled.columns},
            dtype=float,
        )
        self.percentile_reference_ = {}
        self.feature_weights_ = pd.Series(1.0, index=X_filled.columns, dtype=float)

        if self.normalization_ == "percentile":
            for col in X_filled.columns:
                values = (
                    pd.to_numeric(X_filled[col], errors="coerce")
                    .dropna()
                    .sort_values()
                    .to_numpy(dtype=float)
                )
                self.percentile_reference_[col] = values
            means = pd.Series(0.0, index=X_filled.columns, dtype=float)
            stds = pd.Series(1.0, index=X_filled.columns, dtype=float)
            return medians, means, stds

        if self.normalization_ == "robust":
            means = medians.copy()
            stds = self._robust_scale(X_filled, center=medians)
        else:
            means = X_filled.mean(numeric_only=True).fillna(0.0)
            stds = X_filled.std(numeric_only=True).replace(0.0, 1.0).fillna(1.0)

        if self.model_type == "weighted":
            self.feature_weights_ = (
                pd.Series(
                    {col: self._feature_weight(col) for col in X_filled.columns},
                    dtype=float,
                )
                .reindex(X_filled.columns)
                .fillna(1.0)
            )
        return medians, means, stds

    def _transform_features(self, X):
        """Internal helper for the transform_features step."""
        X_filled = X.fillna(self.medians_)
        if self.normalization_ == "percentile":
            X_norm = self._transform_percentiles(X_filled)
        else:
            X_norm = (X_filled - self.means_) / self.stds_
            if self.normalization_ == "robust" and self.robust_clip is not None:
                clip_value = float(self.robust_clip)
                X_norm = X_norm.clip(lower=-clip_value, upper=clip_value)
        X_norm = X_norm.mul(
            self.feature_weights_.reindex(X_norm.columns).fillna(1.0), axis=1
        )
        return X_norm.to_numpy(dtype=float)

    def _normalization_for_model(self):
        """Choose the feature normalization strategy for the configured stat variant."""
        if self.model_type == "percentile":
            return "percentile"
        if self.model_type == "robust":
            return "robust"
        if self.model_type == "weighted":
            if self.weighted_scaling in {"robust", "mad", "iqr"}:
                return "robust"
            if self.weighted_scaling in {"percentile", "rank"}:
                return "percentile"
            return "zscore"
        return "zscore"

    def _robust_scale(self, X, center):
        """Compute robust per-feature scales from MAD or IQR."""
        method = self.robust_scale_method
        if self.model_type == "weighted" and self.weighted_scaling in {"mad", "iqr"}:
            method = self.weighted_scaling
        if method == "iqr":
            q75 = X.quantile(0.75, numeric_only=True)
            q25 = X.quantile(0.25, numeric_only=True)
            scale = (q75 - q25) / 1.349
        else:
            mad = X.sub(center, axis=1).abs().median(numeric_only=True)
            scale = mad * 1.4826
        return (
            scale.replace(0.0, np.nan)
            .fillna(X.std(numeric_only=True))
            .replace(0.0, 1.0)
            .fillna(1.0)
        )

    def _transform_percentiles(self, X):
        """Convert feature values into direction-aware percentile scores."""
        out = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        for col in X.columns:
            ref = np.asarray(self.percentile_reference_.get(col, []), dtype=float)
            values = (
                pd.to_numeric(X[col], errors="coerce")
                .fillna(self.medians_.get(col, 0.0))
                .to_numpy(dtype=float)
            )
            if ref.size == 0:
                pct = np.full(len(values), 0.5, dtype=float)
            else:
                left = np.searchsorted(ref, values, side="left")
                right = np.searchsorted(ref, values, side="right")
                pct = (left + right) / (2.0 * max(ref.size, 1))
            if float(self.directions_.get(col, 1.0)) < 0:
                pct = 1.0 - pct
            out[col] = (pct - 0.5) * 2.0 if self.percentile_center else pct
        return out

    def _feature_direction(self, name):
        """Infer whether higher values are better for a feature."""
        directions = dict(self.params.get("feature_directions", {}) or {})
        if name in directions:
            return self._direction_value(directions[name])

        text = str(name).lower()
        if "_vs_" in text:
            if text.startswith("away_"):
                return -1.0
            if text.startswith(("home_", "net_")):
                return 1.0

        lower_patterns = [
            str(item).lower()
            for item in self.params.get(
                "lower_is_better_patterns",
                [
                    "defense_",
                    "allowed",
                    "points_against",
                    "yards_against",
                    "yards_allowed",
                    "target_points_against",
                    "penalties",
                    "turnovers",
                    "fumbles_lost",
                    "interceptions_thrown",
                ],
            )
        ]
        higher_patterns = [
            str(item).lower()
            for item in self.params.get(
                "higher_is_better_patterns",
                [
                    "statdef_",
                    "havoc",
                    "sacks",
                    "tackles_for_loss",
                    "passes_intercepted",
                    "interception_tds",
                    "defensive_tds",
                    "fumbles_recovered",
                    "kicking_points",
                    "_tds",
                ],
            )
        ]
        if any(pattern in text for pattern in lower_patterns):
            return -1.0
        if any(pattern in text for pattern in higher_patterns):
            return 1.0
        return 1.0

    def _direction_value(self, value):
        """Normalize a configured feature direction into +1 or -1."""
        if isinstance(value, (int, float)):
            return 1.0 if float(value) >= 0 else -1.0
        text = str(value).strip().lower()
        return (
            -1.0
            if text in {"low", "lower", "lower_is_better", "negative", "-1"}
            else 1.0
        )

    def _feature_weight(self, name):
        """Infer a configurable family weight for a feature."""
        weights = self._default_family_weights()
        weights.update(
            {
                str(k): float(v)
                for k, v in dict(self.params.get("family_weights", {}) or {}).items()
            }
        )
        text = str(name).lower()
        weight = 1.0

        prefix_key = self._feature_family_key(text)
        if prefix_key in weights:
            weight *= float(weights[prefix_key])

        for key, patterns in self._modifier_patterns().items():
            if any(pattern in text for pattern in patterns):
                weight *= float(weights.get(key, 1.0))
        return float(weight)

    def _feature_family_key(self, text):
        """Map a feature name to a broad configurable family key."""
        text = str(text).lower()
        for prefix in ("home_", "away_", "net_", "raw_"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        if text.startswith(("offense_", "statoff_")):
            return "offense"
        if text.startswith(("defense_", "statdef_")):
            return "defense"
        if text.startswith("statspe_"):
            return "special_teams"
        if text.startswith("statgen_"):
            return "general"
        if text.startswith("roster_"):
            return "roster"
        if text.startswith("coach_"):
            return "coach"
        if text.startswith("travel_"):
            return "travel"
        return "other"

    def _default_family_weights(self):
        """Return default hand-tuned family weights for weighted TDStat."""
        return {
            "offense": 1.15,
            "defense": 1.15,
            "special_teams": 0.70,
            "general": 0.85,
            "roster": 0.60,
            "coach": 0.35,
            "travel": 0.25,
            "other": 1.0,
            "efficiency": 1.15,
            "explosiveness": 1.00,
            "field_position": 0.80,
            "finishing_drives": 1.00,
            "turnovers": 1.10,
            "penalties": 0.60,
            "talent": 0.75,
        }

    def _modifier_patterns(self):
        """Return feature-name patterns for weighted TDStat modifier groups."""
        return {
            "efficiency": ["ppa", "success_rate", "power_success"],
            "explosiveness": ["explosiveness", "yards_per"],
            "field_position": ["field_position", "starting_field"],
            "finishing_drives": ["finishing", "points_per_opportunity"],
            "turnovers": ["turnover", "interception", "fumble"],
            "penalties": ["penalty", "penalties"],
            "talent": ["talent", "recruit", "return_"],
        }

    def _select_feature_names(self, names):
        """Internal helper for the select_feature_names step."""
        return self._feature_adapter().select_feature_names(names)

    def _attach_context(self, pred_df, meta_df=None, market_df=None):
        """Internal helper for the attach_context step."""
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
        """Internal helper for the coerce_feature_df step."""
        return self._feature_adapter().coerce_frame(X, allow_none=allow_none)

    def _feature_adapter(self):
        """Return the feature adapter, creating it for older checkpoints."""

        adapter = getattr(self, "feature_adapter", None)
        if adapter is None:
            adapter = ModelFeatureAdapter(
                feature_prefixes=tuple(getattr(self, "feature_prefixes", ())),
                include_patterns=tuple(
                    getattr(self, "feature_include_patterns", ()) or ()
                ),
                exclude_patterns=tuple(
                    getattr(self, "feature_exclude_patterns", ()) or ()
                ),
                require_selected=bool(getattr(self, "require_stat_features", False)),
            )
            self.feature_adapter = adapter
        return adapter

    def _coerce_target(self, y, allow_none=False):
        """Internal helper for the coerce_target step."""
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
        y = self._coerce_target(y).to_numpy(dtype=float)
        if self.objective in {"winner", "winner_upset"}:
            return np.where(
                y > 0.0, self.winner_target_margin, -self.winner_target_margin
            )
        return y

    def _objective_sample_weight(self, y, market_df=None):
        if self.objective != "winner_upset" or market_df is None or market_df.empty:
            return None
        from gridiron_ml.td_run.market import (
            DEFAULT_VEGAS_CONVENTION,
            market_home_margin,
            normalize_vegas_frame,
        )

        y = self._coerce_target(y)
        market = normalize_vegas_frame(market_df.copy(), DEFAULT_VEGAS_CONVENTION)
        market_margin = market_home_margin(market, DEFAULT_VEGAS_CONVENTION)
        weights = np.ones(len(y), dtype=float)
        valid = y.notna() & market_margin.notna()
        if valid.any():
            actual_home = y.loc[valid].to_numpy(dtype=float) > 0.0
            market_home = market_margin.loc[valid].to_numpy(dtype=float) > 0.0
            weights[np.flatnonzero(valid.to_numpy())] = np.where(
                actual_home != market_home, self.upset_weight, 1.0
            )
        return weights

    def _normalize_sample_weight(self, sample_weight, n):
        if sample_weight is None:
            return np.ones(int(n), dtype=float)
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(weights) != int(n):
            raise ValueError("sample_weight must align with y.")
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
        return weights

    @staticmethod
    def _weighted_mean(values, weights):
        values = np.asarray(values, dtype=float).reshape(-1)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if not valid.any():
            return 0.0
        return float(np.average(values[valid], weights=weights[valid]))

    @staticmethod
    def _winner_accuracy(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if not valid.any():
            return np.nan
        return float(((y_true[valid] > 0.0) == (y_pred[valid] > 0.0)).mean())

    def _upset_accuracy(self, y_true, y_pred, market_df=None):
        if market_df is None or market_df.empty:
            return np.nan
        from gridiron_ml.td_run.market import (
            DEFAULT_VEGAS_CONVENTION,
            market_home_margin,
            normalize_vegas_frame,
        )

        y_true = pd.to_numeric(
            pd.Series(y_true).reset_index(drop=True), errors="coerce"
        )
        y_pred = pd.to_numeric(
            pd.Series(y_pred).reset_index(drop=True), errors="coerce"
        )
        market = normalize_vegas_frame(market_df.copy(), DEFAULT_VEGAS_CONVENTION)
        market_margin = market_home_margin(market, DEFAULT_VEGAS_CONVENTION)
        valid = y_true.notna() & y_pred.notna() & market_margin.notna()
        if not valid.any():
            return np.nan
        actual_home = y_true.loc[valid].to_numpy(dtype=float) > 0.0
        pred_home = y_pred.loc[valid].to_numpy(dtype=float) > 0.0
        market_home = market_margin.loc[valid].to_numpy(dtype=float) > 0.0
        upset = actual_home != market_home
        if not upset.any():
            return np.nan
        return float((pred_home[upset] == actual_home[upset]).mean())

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

    def _margin_to_proba(self, margin):
        """Internal helper for the margin_to_proba step."""
        return 1.0 / (
            1.0
            + np.exp(
                -np.asarray(margin, dtype=float) / max(float(self.margin_scale), 1e-8)
            )
        )

    @staticmethod
    def _rmse(y_true, y_pred):
        """Internal helper for the rmse step."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))

    @staticmethod
    def _mae(y_true, y_pred):
        """Internal helper for the mae step."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.nanmean(np.abs(y_true - y_pred)))

    @staticmethod
    def _bias(y_true, y_pred):
        """Internal helper for the bias step."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.nanmean(y_pred - y_true))


class TDStatPercentile(TDStat):
    """TDStat variant that uses direction-aware percentile feature scaling."""

    def __init__(self, config=None):
        """Initialize a percentile-based TDStat model."""
        super().__init__(self._with_model_type(config, "percentile", "stat_percentile"))

    @staticmethod
    def _with_model_type(config, model_type, model_name):
        """Inject a stat model type into an optional config object."""
        loaded = TDStat._load_config(config)
        loaded.setdefault("model_name", model_name)
        params = dict(loaded.get("params", {}) or {})
        params["model_type"] = model_type
        loaded["params"] = params
        return loaded


class TDStatRobust(TDStat):
    """TDStat variant that uses median/MAD or median/IQR robust scaling."""

    def __init__(self, config=None):
        """Initialize a robust z-score TDStat model."""
        super().__init__(
            TDStatPercentile._with_model_type(config, "robust", "stat_robust")
        )


class TDStatWeighted(TDStat):
    """TDStat variant that applies configurable football-family feature weights."""

    def __init__(self, config=None):
        """Initialize a weighted statistical TDStat model."""
        super().__init__(
            TDStatPercentile._with_model_type(config, "weighted", "stat_weighted")
        )
