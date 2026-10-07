"""Hierarchical representation genome for the matched F18/F19 GA and BO tracks."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from .constraint_free import FeatureRecord


MODES = ("raw", "pca", "family_pca", "select", "hybrid")
VARIANCES = (0.80, 0.90, 0.95, 0.975)
SELECT_COUNTS = (64, 128, 256)
CORRELATIONS = (0.0, 0.90, 0.98)
FAMILIES = ("adjusted", "coaching", "historical", "microstructure", "roster", "trend", "units")


def normalize_genome(genome: dict) -> dict:
    required = {"mode", "variance", "select_count", "correlation", "nonlinear",
                "family_mask", "market_on", "robust", "hp_index", "target_mode"}
    if set(genome) != required:
        raise ValueError("Genome fields changed")
    result = dict(genome)
    if result["mode"] not in MODES or result["variance"] not in VARIANCES:
        raise ValueError("Unknown representation mode or variance")
    if result["select_count"] not in SELECT_COUNTS or result["correlation"] not in CORRELATIONS:
        raise ValueError("Unknown selection or correlation setting")
    if not isinstance(result["family_mask"], int) or result["family_mask"] < 1 or result["family_mask"] >= 2 ** len(FAMILIES):
        raise ValueError("At least one valid football family is required")
    if result["hp_index"] not in (0, 1, 2):
        raise ValueError("Unknown architecture-compatible hyperparameter choice")
    if result["target_mode"] not in ("direct", "residual"):
        raise ValueError("Unknown target mode")
    for key in ("nonlinear", "market_on", "robust"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be Boolean")
    return result


def genome_hash(genome: dict) -> str:
    value = json.dumps(normalize_genome(genome), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def random_genome(rng: np.random.Generator) -> dict:
    return {
        "mode": str(rng.choice(MODES)),
        "variance": float(rng.choice(VARIANCES)),
        "select_count": int(rng.choice(SELECT_COUNTS)),
        "correlation": float(rng.choice(CORRELATIONS)),
        "nonlinear": bool(rng.integers(2)),
        "family_mask": int(rng.integers(1, 2 ** len(FAMILIES))),
        "market_on": bool(rng.integers(2)),
        "robust": bool(rng.integers(2)),
        "hp_index": int(rng.integers(3)),
        "target_mode": str(rng.choice(("direct", "residual"))),
    }


def encode_genome(genome: dict) -> np.ndarray:
    g = normalize_genome(genome)
    return np.array([
        MODES.index(g["mode"]) / (len(MODES) - 1),
        VARIANCES.index(g["variance"]) / (len(VARIANCES) - 1),
        SELECT_COUNTS.index(g["select_count"]) / (len(SELECT_COUNTS) - 1),
        CORRELATIONS.index(g["correlation"]) / (len(CORRELATIONS) - 1),
        float(g["nonlinear"]), float(g["market_on"]), float(g["robust"]),
        g["hp_index"] / 2,
        float(g["target_mode"] == "residual"),
        *[(g["family_mask"] >> i) & 1 for i in range(len(FAMILIES))],
    ], dtype=float)


class GenomeRepresentation(BaseEstimator, TransformerMixin):
    """Fit all empirical operations on the supplied historical training fold."""

    def __init__(self, genome: dict, records: list[FeatureRecord], tier: str):
        self.genome = normalize_genome(genome)
        self.records = records
        self.tier = tier

    def fit(self, X, y):
        data = np.asarray(X, dtype=float)
        if data.ndim != 2 or data.shape[1] != len(self.records):
            raise ValueError("Genome input differs from feature provenance")
        allowed = {family for i, family in enumerate(FAMILIES)
                   if self.genome["family_mask"] & (1 << i)}
        self.columns_ = np.array([
            i for i, r in enumerate(self.records)
            if (r.market_derived and self.tier == "F19" and self.genome["market_on"])
            or (not r.market_derived and r.family in allowed)
        ], dtype=int)
        if len(self.columns_) == 0:
            raise ValueError("Genome selected zero source features")
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        filled = self.imputer_.fit_transform(data[:, self.columns_])
        if self.genome["nonlinear"]:
            filled = np.sign(filled) * np.log1p(np.abs(filled))
        correlation = self.genome["correlation"]
        if correlation:
            corr = np.corrcoef(filled, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            remove = np.any(np.triu(np.abs(corr) > correlation, k=1), axis=0)
            self.kept_ = np.flatnonzero(~remove)
        else:
            self.kept_ = np.arange(filled.shape[1])
        selected = filled[:, self.kept_]
        self.mode_ = self.genome["mode"]
        self.scaler_ = None
        self.pca_ = None
        self.family_blocks_ = []
        self.selector_ = None
        if self.mode_ in ("pca", "hybrid"):
            self.scaler_ = RobustScaler() if self.genome["robust"] else StandardScaler()
            scaled = self.scaler_.fit_transform(selected)
            self.pca_ = PCA(n_components=self.genome["variance"], svd_solver="full")
            self.pca_.fit(scaled)
        if self.mode_ == "family_pca":
            selected_records = [self.records[self.columns_[i]] for i in self.kept_]
            for family in sorted({r.family for r in selected_records}):
                indices = np.array([i for i, r in enumerate(selected_records) if r.family == family])
                scaler = RobustScaler() if self.genome["robust"] else StandardScaler()
                scaled = scaler.fit_transform(selected[:, indices])
                pca = PCA(n_components=self.genome["variance"], svd_solver="full")
                pca.fit(scaled)
                self.family_blocks_.append((family, indices, scaler, pca))
        if self.mode_ in ("select", "hybrid"):
            self.selector_ = SelectKBest(f_regression, k=min(self.genome["select_count"], selected.shape[1]))
            self.selector_.fit(selected, y)
        self.n_features_in_ = data.shape[1]
        return self

    def transform(self, X):
        data = np.asarray(X, dtype=float)
        if data.ndim != 2 or data.shape[1] != self.n_features_in_:
            raise ValueError("Genome transform input differs from fitting columns")
        filled = self.imputer_.transform(data[:, self.columns_])
        if self.genome["nonlinear"]:
            filled = np.sign(filled) * np.log1p(np.abs(filled))
        selected = filled[:, self.kept_]
        if self.mode_ == "raw":
            return selected
        if self.mode_ == "pca":
            return self.pca_.transform(self.scaler_.transform(selected))
        if self.mode_ == "family_pca":
            return np.concatenate([pca.transform(scaler.transform(selected[:, indices]))
                                   for _, indices, scaler, pca in self.family_blocks_], axis=1)
        if self.mode_ == "select":
            return self.selector_.transform(selected)
        return np.concatenate([
            self.selector_.transform(selected),
            self.pca_.transform(self.scaler_.transform(selected)),
        ], axis=1)
