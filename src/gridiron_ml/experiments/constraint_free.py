"""Leakage-safe representation and provenance contract for paired F18/F19 search.

The 2026 season is deliberately absent from this module's fitting interface.
All empirical transformations are fitted by ``fit`` on each historical training
fold; the caller may only pass future rows to ``transform``.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.random_projection import GaussianRandomProjection


TIERS = {
    "F18": "F18_constraint_free_market_free",
    "F19": "F19_constraint_free_market",
}
ROSTER = ("M1", "M2", "M3", "M4", "M5", "M10")
MARKET_COLUMNS = (
    "market_home_spread", "market_total", "market_open_spread",
    "market_open_total", "market_spread_move", "market_total_move",
    "market_home_moneyline", "market_away_moneyline",
    "market_home_implied_no_vig", "market_spread_quote_sd",
    "market_total_quote_sd", "market_quote_count",
)
# Only these archived summary fields can also be populated from a verified
# pre-kickoff 2026 snapshot. The historical quote archive itself is untimed.
SNAPSHOT_MARKET_COLUMNS = (
    "market_home_spread", "market_total", "market_open_spread",
    "market_spread_move", "market_home_implied_no_vig",
)
MARKET_DENY = re.compile(
    r"(^|[_\W])(market|vegas|betting|bookmaker|sportsbook|spread|moneyline|"
    r"over_under|implied_prob|quote_count|quote_sd|line_move)([_\W]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FeatureRecord:
    name: str
    family: str
    market_derived: bool = False
    source: str = ""
    timestamp_semantics: str = ""
    prospective_allowed: bool = True


def assert_feature_contract(names: list[str], metadata: list[FeatureRecord], tier: str) -> None:
    """Fail closed on untagged market data, duplicate names, or F18 contamination."""
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier}")
    if len(names) != len(set(names)) or [r.name for r in metadata] != names:
        raise ValueError("Feature names and ordered metadata must match uniquely")
    for record in metadata:
        looks_market = bool(MARKET_DENY.search(record.name))
        if looks_market and not record.market_derived:
            raise ValueError(f"Untagged market-like feature: {record.name}")
        if tier == "F18" and (record.market_derived or looks_market):
            raise ValueError(f"F18 market contamination: {record.name}")
        if record.market_derived:
            if record.name not in MARKET_COLUMNS:
                raise ValueError(f"Unregistered market feature: {record.name}")
            if not record.source or not record.timestamp_semantics:
                raise ValueError(f"Incomplete market provenance: {record.name}")
            if record.prospective_allowed and record.name not in SNAPSHOT_MARKET_COLUMNS:
                raise ValueError(f"Unavailable prospective market field: {record.name}")
    if tier == "F19" and not any(r.market_derived for r in metadata):
        raise ValueError("F19 must explicitly include a tagged market feature")


def feature_family(name: str) -> str:
    if name in MARKET_COLUMNS:
        return "market"
    value = name.removeprefix("matchup__").lower()
    if any(s in value for s in ("recruit", "transfer", "roster", "talent", "returning")):
        return "roster"
    if any(s in value for s in ("coach", "staff", "tenure")):
        return "coaching"
    if any(s in value for s in ("recent", "trend", "change", "rolling")):
        return "trend"
    if any(s in value for s in ("play", "drive", "rush", "pass", "reception", "down")):
        return "microstructure"
    if any(s in value for s in ("offense", "defense", "unit", "special_team")):
        return "units"
    if any(s in value for s in ("adjust", "opp", "resid", "context")):
        return "adjusted"
    return "historical"


def metadata_for_matrix(names: list[str], tier: str) -> list[FeatureRecord]:
    records = []
    for name in names:
        is_market = name in MARKET_COLUMNS
        records.append(FeatureRecord(
            name=name, family=feature_family(name), market_derived=is_market,
            source="CFBD archived lines / pregame publication snapshot" if is_market else "F16 checked source matrix",
            timestamp_semantics=("historical quotes untimed; 2026 snapshot captured before kickoff"
                                 if is_market else "source available before target kickoff"),
            prospective_allowed=name in SNAPSHOT_MARKET_COLUMNS if is_market else True,
        ))
    assert_feature_contract(names, records, tier)
    return records


class FamilyPCA(BaseEstimator, TransformerMixin):
    """Fit separate imputer, scaler, and PCA for each declared family."""

    def __init__(self, families: tuple[str, ...], variance: float = 0.90):
        self.families = families
        self.variance = variance

    def fit(self, X, y=None):
        data = np.asarray(X, dtype=float)
        if data.ndim != 2 or data.shape[1] != len(self.families):
            raise ValueError("Family metadata does not match the feature matrix")
        self.blocks_ = []
        self.n_features_in_ = data.shape[1]
        for family in sorted(set(self.families)):
            columns = np.flatnonzero(np.asarray(self.families) == family)
            block = Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=self.variance, svd_solver="full")),
            ])
            block.fit(data[:, columns])
            self.blocks_.append((family, columns, block))
        return self

    def transform(self, X):
        data = np.asarray(X, dtype=float)
        if data.ndim != 2 or data.shape[1] != self.n_features_in_:
            raise ValueError("FamilyPCA received different feature columns")
        return np.concatenate([block.transform(data[:, columns])
                               for _, columns, block in self.blocks_], axis=1)


def representation(kind: str, records: list[FeatureRecord], seed: int = 1701):
    """Return a fold-fitted representation. Both tiers use the same menu."""
    n = len(records)
    if kind == "raw":
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))])
    if kind.startswith("pca"):
        variance = float(kind.removeprefix("pca")) / 100
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=variance, svd_solver="full")),
        ])
    if kind == "robust_pca90":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", RobustScaler()),
            ("pca", PCA(n_components=0.90, svd_solver="full")),
        ])
    if kind == "family_pca90":
        return FamilyPCA(tuple(r.family for r in records), variance=0.90)
    if kind == "select128":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("selector", SelectKBest(score_func=f_regression, k=min(128, n))),
        ])
    if kind == "projection128":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("projection", GaussianRandomProjection(n_components=min(128, n), random_state=seed)),
        ])
    raise ValueError(f"Unknown representation {kind}")
