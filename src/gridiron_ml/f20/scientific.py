"""Frozen F20 probability panel for the six scientific architectures."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

MODEL_IDS = ("M1", "M2", "M3", "M4", "M5", "M10")


def estimator(model_id: str):
    """Fixed model families; no outcome-driven search on 2025 or 2026 games."""
    impute = lambda: SimpleImputer(strategy="median", keep_empty_features=True)
    if model_id == "M1":
        return make_pipeline(impute(), StandardScaler(), LogisticRegression(C=100, max_iter=2000))
    if model_id == "M2":
        return make_pipeline(impute(), StandardScaler(), SplineTransformer(n_knots=4, degree=2),
                             LogisticRegression(C=1, max_iter=2000))
    if model_id == "M3":
        return make_pipeline(impute(), RandomForestClassifier(
            n_estimators=200, min_samples_leaf=8, max_features="sqrt", random_state=1701, n_jobs=1))
    if model_id == "M4":
        return make_pipeline(impute(), HistGradientBoostingClassifier(
            max_iter=200, max_leaf_nodes=15, l2_regularization=1, random_state=1701))
    if model_id == "M5":
        return make_pipeline(impute(), StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=.01, early_stopping=True,
            random_state=1701, max_iter=400))
    if model_id == "M10":
        return make_pipeline(impute(), StandardScaler(), KNeighborsClassifier(
            n_neighbors=75, weights="distance"))
    raise ValueError(f"Unknown scientific model: {model_id}")


def predict_panel(models: dict, features):
    if set(models) != set(MODEL_IDS):
        raise ValueError("F20 scientific panel must contain M1–M5 and M10")
    scores = {model_id: np.asarray(models[model_id].predict_proba(features)[:, 1], dtype=float)
              for model_id in MODEL_IDS}
    scores["F20"] = np.mean(np.stack([scores[model_id] for model_id in MODEL_IDS]), axis=0)
    if any(not np.isfinite(value).all() or not ((0 <= value) & (value <= 1)).all()
           for value in scores.values()):
        raise ValueError("Invalid F20 probability")
    return scores
