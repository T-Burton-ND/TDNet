"""Validation helpers for TDNet training and evaluation safety checks."""

from .leakage import (
    assert_default_target_is_next_margin,
    assert_disjoint_years,
    assert_no_market_features,
    assert_no_same_week_postgame_target,
    market_feature_columns,
    training_allows_market_features,
)

__all__ = [
    "assert_default_target_is_next_margin",
    "assert_disjoint_years",
    "assert_no_market_features",
    "assert_no_same_week_postgame_target",
    "market_feature_columns",
    "training_allows_market_features",
]
