"""Additional fold-fitted reducers for matched F18/F19 historical screening."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FastICA, NMF, PCA, SparsePCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler


CANDIDATES = (
    "pca975", "pca99", "pca_fixed64", "pca_fixed128",
    "sparse_pca32", "fastica64", "nmf64", "pls32",
    "kernel_nystroem64", "autoencoder32", "mi128",
    "raw_missing_scaled", "quantile_pca90", "power_pca90",
    "select128_scaled",
)


class PLSProjector(BaseEstimator, TransformerMixin):
    def __init__(self, components: int = 32):
        self.components = components

    def fit(self, X, y):
        data = np.asarray(X, dtype=float)
        self.model_ = PLSRegression(n_components=min(self.components, data.shape[1], len(data) - 1),
                                    scale=False)
        self.model_.fit(data, np.asarray(y, dtype=float))
        return self

    def transform(self, X):
        return self.model_.transform(np.asarray(X, dtype=float))


class AutoencoderProjector(BaseEstimator, TransformerMixin):
    """One-hidden-layer reconstruction net; expose its fold-fitted bottleneck."""

    def __init__(self, components: int = 32, seed: int = 1729):
        self.components = components
        self.seed = seed

    def fit(self, X, y=None):
        data = np.asarray(X, dtype=float)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(self.components,), activation="tanh", solver="adam",
            alpha=0.001, batch_size=256, learning_rate_init=0.001,
            max_iter=30, early_stopping=True, validation_fraction=0.10,
            n_iter_no_change=5, random_state=self.seed,
        )
        self.model_.fit(data, data)
        return self

    def transform(self, X):
        data = np.asarray(X, dtype=float)
        return np.tanh(data @ self.model_.coefs_[0] + self.model_.intercepts_[0])


def extra_representation(kind: str, n_features: int, seed: int = 1729):
    """Return a pipeline whose empirical stages fit only on a training fold."""
    if kind not in CANDIDATES:
        raise ValueError(f"Unknown extra reducer: {kind}")
    impute = ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    scale = ("scale", StandardScaler())
    if kind.startswith("pca"):
        if kind == "pca975":
            components = 0.975
        elif kind == "pca99":
            components = 0.99
        else:
            components = min(int(kind.removeprefix("pca_fixed")), n_features)
        return Pipeline([impute, scale,
                         ("pca", PCA(n_components=components, svd_solver="full"))])
    if kind == "sparse_pca32":
        return Pipeline([impute, scale,
                         ("sparse", SparsePCA(n_components=min(32, n_features),
                                              alpha=1.0, ridge_alpha=0.01,
                                              max_iter=50, method="cd",
                                              random_state=seed, n_jobs=1))])
    if kind == "fastica64":
        return Pipeline([impute, scale,
                         ("ica", FastICA(n_components=min(64, n_features),
                                         whiten="unit-variance", max_iter=400,
                                         tol=0.001, random_state=seed))])
    if kind == "nmf64":
        return Pipeline([impute, ("nonnegative", MinMaxScaler(clip=True)),
                         ("nmf", NMF(n_components=min(64, n_features),
                                     init="nndsvda", max_iter=150, random_state=seed))])
    if kind == "pls32":
        return Pipeline([impute, scale, ("pls", PLSProjector(32))])
    if kind == "kernel_nystroem64":
        return Pipeline([impute, scale,
                         ("kernel", Nystroem(kernel="rbf", gamma=1 / max(n_features, 1),
                                              n_components=128, random_state=seed)),
                         ("pca", PCA(n_components=64, svd_solver="full"))])
    if kind == "autoencoder32":
        return Pipeline([impute, scale, ("autoencoder", AutoencoderProjector(32, seed))])
    if kind == "mi128":
        return Pipeline([impute,
                         ("select", SelectKBest(
                             score_func=lambda X, y: mutual_info_regression(
                                 X, y, random_state=seed, n_jobs=1),
                             k=min(128, n_features)))])
    if kind == "raw_missing_scaled":
        return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True,
                                                   keep_empty_features=True)), scale])
    if kind == "quantile_pca90":
        return Pipeline([impute,
                         ("quantile", QuantileTransformer(
                             n_quantiles=1000, output_distribution="normal", random_state=seed)),
                         ("pca", PCA(n_components=0.90, svd_solver="full"))])
    if kind == "power_pca90":
        return Pipeline([impute, ("power", PowerTransformer(method="yeo-johnson")),
                         ("pca", PCA(n_components=0.90, svd_solver="full"))])
    return Pipeline([impute,
                     ("select", SelectKBest(f_regression, k=min(128, n_features))),
                     scale])
