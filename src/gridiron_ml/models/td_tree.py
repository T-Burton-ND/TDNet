"""src.gridiron_ml.models.td_tree.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Define model wrappers and checkpoint helpers behind the shared TDNet interface.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridiron_ml.pipeline.validation.leakage import training_allows_market_features

from .names import normalize_identifier, normalize_model_family
from .td_linear import DEFAULT_LOSS_WEIGHTS, TDLinear


TREE_DEFAULTS = {
    "random_forest": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
    },
    "extra_trees": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": False,
        "n_jobs": -1,
    },
    "gradient_boosted": {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_samples_leaf": 10,
        "subsample": 0.8,
    },
    "hist_gradient_boosted": {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "early_stopping": True,
    },
}


class TDTree(TDLinear):
    """
    Tree-based TDNet margin model family.

    TDTree is a drop-in peer to TDLinear for matchup fingerprint inputs. It
    predicts the same target convention used by TDLinear and TD Eval:
    positive ``pred_margin`` means the home team is projected to win by that
    many points, and negative ``pred_margin`` means the away team is projected
    to win.

    The family currently supports three sklearn regressors:
    ``random_forest`` uses RandomForestRegressor, ``extra_trees`` uses
    ExtraTreesRegressor, and ``gradient_boosted`` uses GradientBoostingRegressor.
    GradientBoostingRegressor is used instead of HistGradientBoostingRegressor
    because it exposes ``feature_importances_`` like the forest variants, which
    keeps TDTree diagnostics consistent across variants.
    """

    def __init__(self, config=None):
        """Internal helper for the init__ step."""
        self.config = self._load_config(config)
        self.model_family = normalize_model_family("tree")
        self.model_type = self._normalize_model_type(
            self.config.get(
                "model_type",
                self.config.get(
                    "variant",
                    self.config.get("tdtree", {}).get(
                        "variant",
                        self.config.get("params", {}).get(
                            "model_type", "random_forest"
                        ),
                    ),
                ),
            )
        )
        self.model_variant = self.model_type
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
            self.config.get("seed", self.config.get("params", {}).get("seed", 42))
        )

        self.training = dict(self.config.get("training", {}))
        self.standardize = bool(self.training.get("standardize", False))
        self.use_sample_weights = bool(self.training.get("use_sample_weights", False))
        self.sample_weight_column = self.training.get("sample_weight_column")
        self.allow_market_features_for_training = training_allows_market_features(
            self.config
        )

        self.params = self._model_params()
        self.feature_names_ = []
        self.medians_ = pd.Series(dtype=float)
        self.means_ = pd.Series(dtype=float)
        self.stds_ = pd.Series(dtype=float)
        self.feature_importances_ = pd.DataFrame()
        self.training_history_ = pd.DataFrame()
        self.is_trained_ = False
        self.pipeline_ = None

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
        """
        Fit the selected tree regressor on matchup fingerprint features.

        The input contract matches TDLinear: callers pass a numeric feature
        dataframe and a one-dimensional margin target. Vegas and market columns
        are not consumed here; they remain attached later as evaluation context.
        Missing feature values are median-imputed so all supported sklearn tree
        estimators receive the same final matrix shape at train and prediction
        time.
        """
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
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["estimator__sample_weight"] = sample_weight
        pipeline.fit(
            X_train.loc[:, self.feature_names_],
            self._training_target(y_train),
            **fit_kwargs,
        )

        self.pipeline_ = pipeline
        self.model_ = pipeline
        self.is_trained_ = True
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

    def get_feature_importance(self):
        """
        Return model-native feature importances as a dataframe.

        All current TDTree variants expose sklearn's ``feature_importances_``.
        The dataframe is sorted descending and uses the trained matchup feature
        names, making it suitable for downstream artifact saving or notebook
        inspection.
        """
        estimator = self._estimator()
        if estimator is None or not hasattr(estimator, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])
        importance = np.asarray(estimator.feature_importances_, dtype=float).reshape(-1)
        if len(importance) != len(self.feature_names_):
            return pd.DataFrame(columns=["feature", "importance"])
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _build_pipeline(self):
        """Internal helper for the build_pipeline step."""
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if self.standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("estimator", self._build_estimator()))
        return Pipeline(steps=steps)

    def _build_estimator(self):
        """Internal helper for the build_estimator step."""
        params = dict(self.params)
        params["random_state"] = int(params.get("random_state", self.seed))
        if self.model_type == "random_forest":
            if self._is_classifier_objective():
                return RandomForestClassifier(**params)
            return RandomForestRegressor(**params)
        if self.model_type == "extra_trees":
            if self._is_classifier_objective():
                return ExtraTreesClassifier(**params)
            return ExtraTreesRegressor(**params)
        if self.model_type == "gradient_boosted":
            params.pop("n_jobs", None)
            if self._is_classifier_objective():
                return GradientBoostingClassifier(**params)
            return GradientBoostingRegressor(**params)
        if self.model_type == "hist_gradient_boosted":
            params.pop("n_jobs", None)
            if self._is_classifier_objective():
                return HistGradientBoostingClassifier(**params)
            return HistGradientBoostingRegressor(**params)
        raise ValueError(f"Unhandled TDTree model_type='{self.model_type}'")

    def _model_params(self):
        """Internal helper for the model_params step."""
        params = dict(TREE_DEFAULTS[self.model_type])
        params["random_state"] = self.seed

        tdtree_cfg = dict(self.config.get("tdtree", {}))
        variant_cfg = {}
        variant_cfg.update(dict(tdtree_cfg.get(self.model_type, {})))
        variant_cfg.update(dict(self.config.get(self.model_type, {})))
        params.update(variant_cfg)
        params.update(dict(self.config.get("params", {})))

        for key in ["model_type", "variant", "seed"]:
            params.pop(key, None)
        return params

    @staticmethod
    def _normalize_model_type(value):
        """Internal helper for the normalize_model_type step."""
        model_type = str(value).strip().lower().replace("-", "_")
        aliases = {
            "rf": "random_forest",
            "random_forest": "random_forest",
            "randomforest": "random_forest",
            "extra_trees": "extra_trees",
            "extratrees": "extra_trees",
            "et": "extra_trees",
            "gb": "gradient_boosted",
            "gbm": "gradient_boosted",
            "gbt": "gradient_boosted",
            "gradient_boosted": "gradient_boosted",
            "gradient_boosting": "gradient_boosted",
            "hist_gb": "hist_gradient_boosted",
            "hist_gbm": "hist_gradient_boosted",
            "hist_gradient_boosted": "hist_gradient_boosted",
            "hist_gradient_boosting": "hist_gradient_boosted",
        }
        if model_type not in aliases:
            supported = ", ".join(sorted(set(aliases.values())))
            raise ValueError(
                f"Unsupported TDTree model_type='{value}'. Supported: {supported}"
            )
        return aliases[model_type]
