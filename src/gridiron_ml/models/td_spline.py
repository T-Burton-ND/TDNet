"""Controlled nonlinear M2 spline model family."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .names import normalize_identifier, normalize_model_family
from .td_linear import TDLinear


class TDSpline(TDLinear):
    """Spline expansion followed by the existing regularized TDLinear core."""

    def __init__(self, config=None):
        loaded = self._load_config(config)
        requested = normalize_identifier(
            loaded.get("model_type", loaded.get("variant", "spline_ridge"))
        )
        if requested not in {"spline_ridge", "spline_logistic", "spline"}:
            raise ValueError(f"Unsupported TDSpline model_type='{requested}'.")
        linear_config = dict(loaded)
        linear_config["model_type"] = "ridge"
        linear_config.setdefault("model_name", requested)
        super().__init__(linear_config)
        self.config = loaded
        self.model_family = normalize_model_family("spline")
        self.model_type = requested
        self.model_name = normalize_identifier(loaded.get("model_name", requested))
        self.spline_config = dict(loaded.get("spline", {}))

    def _build_pipeline(self):
        """Build fold-only imputation, scaling, spline expansion, and estimator."""
        # TDLinear's estimator dispatch expects a core linear type.
        spline_type = self.model_type
        self.model_type = "ridge"
        try:
            estimator = self._build_estimator()
        finally:
            self.model_type = spline_type
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "spline",
                    SplineTransformer(
                        n_knots=int(self.spline_config.get("n_knots", 4)),
                        degree=int(self.spline_config.get("degree", 2)),
                        include_bias=False,
                    ),
                ),
                ("estimator", estimator),
            ]
        )

    def get_feature_importance(self):
        """Spline coefficients are exported separately from raw feature names."""
        estimator = self._estimator()
        if estimator is None or not hasattr(estimator, "coef_"):
            return pd.DataFrame(columns=["feature", "coefficient", "importance"])
        coefficients = pd.Series(estimator.coef_.reshape(-1), dtype=float)
        return pd.DataFrame(
            {
                "feature": [f"spline_basis_{i:05d}" for i in range(len(coefficients))],
                "coefficient": coefficients,
                "importance": coefficients.abs(),
            }
        ).sort_values("importance", ascending=False, ignore_index=True)
