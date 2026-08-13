"""Kernel regression models behind the shared TDNet interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .names import normalize_identifier, normalize_model_family
from .td_linear import TDLinear


class TDKernel(TDLinear):
    """Nonlinear similarity models for margin prediction and win probabilities."""

    TYPES = {"rbf_kernel_ridge", "rbf_svr", "gaussian_process", "nystroem_ridge"}

    def __init__(self, config=None):
        loaded = self._load_config(config)
        requested = normalize_identifier(loaded.get("model_type", "rbf_kernel_ridge"))
        if requested not in self.TYPES:
            raise ValueError(f"Unsupported TDKernel model_type='{requested}'.")
        core = dict(loaded)
        core["model_type"] = "ridge"
        # Kernel families always learn a continuous margin. Winner searches
        # select the same model by winner/Brier metrics rather than changing it
        # into a binary classifier.
        core["loss_function"] = loaded.get("loss_function", "MAE")
        super().__init__(core)
        self.config = loaded
        self.model_family = normalize_model_family("kernel")
        self.model_type = requested
        self.model_name = normalize_identifier(loaded.get("model_name", requested))
        self.params = dict(loaded.get("params", {}))
        self.max_train_samples = int(loaded.get("max_train_samples", 1000))

    def train(self, X_train, y_train, **kwargs):
        if self.model_type == "gaussian_process" and len(X_train) > self.max_train_samples:
            rng = np.random.default_rng(self.seed)
            chosen = np.sort(rng.choice(len(X_train), self.max_train_samples, replace=False))
            X_train = X_train.iloc[chosen].reset_index(drop=True)
            y_train = np.asarray(y_train)[chosen]
            if kwargs.get("sample_weight") is not None:
                kwargs["sample_weight"] = np.asarray(kwargs["sample_weight"])[chosen]
        return super().train(X_train, y_train, **kwargs)

    def _is_classifier_objective(self):
        return False

    def _build_pipeline(self):
        params = dict(self.params)
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                 ("scaler", StandardScaler())]
        if self.model_type == "rbf_kernel_ridge":
            estimator = KernelRidge(kernel="rbf", **params)
        elif self.model_type == "rbf_svr":
            estimator = SVR(kernel="rbf", **params)
        elif self.model_type == "gaussian_process":
            amplitude = float(params.pop("amplitude", 1.0))
            length_scale = float(params.pop("length_scale", 1.0))
            noise = float(params.pop("noise_level", 0.1))
            kernel = ConstantKernel(amplitude, constant_value_bounds="fixed") * RBF(
                length_scale, length_scale_bounds="fixed"
            ) + WhiteKernel(noise, noise_level_bounds="fixed")
            estimator = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=self.seed, **params)
        else:
            components = int(params.pop("n_components", 300))
            gamma = params.pop("gamma", None)
            alpha = float(params.pop("alpha", 1.0))
            steps.append(("kernel_map", Nystroem(kernel="rbf", gamma=gamma, n_components=components, random_state=self.seed)))
            estimator = Ridge(alpha=alpha, **params)
        steps.append(("estimator", estimator))
        return Pipeline(steps)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.update({"model_family": "kernel", "model_type": self.model_type,
                         "max_train_samples": self.max_train_samples})
        return metadata

    def get_feature_importance(self):
        # Kernel basis weights are not raw-feature coefficients. Permutation or
        # SHAP importance is computed by the diagnostics pipeline instead.
        return pd.DataFrame(columns=["feature", "importance"])
