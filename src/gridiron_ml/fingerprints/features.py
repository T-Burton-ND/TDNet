"""src.gridiron_ml.fingerprints.features.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Build, load, and split time-dependent team fingerprints.
"""

from dataclasses import dataclass

import pandas as pd

from gridiron_ml.pipeline.contracts.features import (
    DEFAULT_TRAINING_TARGET,
    MARKET_CONTEXT_KEY_COLUMNS,
    is_feature_column,
    is_market_column,
)
from gridiron_ml.fingerprints.feature_map import map_from_dataframe
from gridiron_ml.pipeline.validation.leakage import (
    assert_no_leaky_coach_features,
    assert_no_market_features,
    assert_no_same_week_postgame_target,
)


@dataclass(frozen=True)
class FeatureSpec:
    """Feature-selection contract for TDNet fingerprint frames.

    Market/Vegas columns stay out of training features by default. Setting
    include_market=True requires the loud opt-in flag below because those columns
    are intended for evaluation context unless an experiment explicitly says
    otherwise.
    """
    include_market: bool = False
    include_travel: bool = True
    include_meta: bool = False
    include_stat_off: bool = True
    include_stat_def: bool = True
    include_stat_gen: bool = True
    include_stat_spe: bool = True
    include_labels: bool = False
    target_column: str = DEFAULT_TRAINING_TARGET
    allow_market_features_for_training: bool = False


DEFAULT_FEATURE_SPEC = FeatureSpec()


def split_frame(frame, spec=None):
    """Split a fingerprint frame into model features, target, metadata, and market context.

    v0 features describe state after ``keys_week`` has completed. Market columns
    are returned as separate context so evaluation can compare against Vegas
    without silently training on betting-derived features.
    """
    spec = spec or DEFAULT_FEATURE_SPEC
    if frame is None or frame.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    fm = map_from_dataframe(frame, keep_unknown=True)
    feature_cols = [
        c
        for c in fm.feature_columns(
            include_market=spec.include_market,
            include_travel=spec.include_travel,
            include_meta=spec.include_meta,
            include_stat_off=spec.include_stat_off,
            include_stat_def=spec.include_stat_def,
            include_stat_gen=spec.include_stat_gen,
            include_stat_spe=spec.include_stat_spe,
            include_labels=spec.include_labels,
        )
        if c in frame.columns and (pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]))
    ]
    feature_cols.extend(
        [
            c
            for c in frame.columns
            if c not in feature_cols
            and is_adjusted_feature_column(c)
            and (pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]))
        ]
    )

    if not feature_cols:
        feature_cols = [
            c
            for c in frame.columns
            if is_feature_column(c)
            and (pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]))
        ]

    assert_no_market_features(
        feature_cols,
        allow_market_features_for_training=spec.allow_market_features_for_training,
    )
    assert_no_leaky_coach_features(feature_cols)
    assert_no_same_week_postgame_target(spec.target_column, feature_cols)

    x_df = frame.loc[:, feature_cols].copy()
    for col in x_df.columns:
        if pd.api.types.is_bool_dtype(x_df[col]):
            x_df[col] = x_df[col].astype(float)
        elif not pd.api.types.is_numeric_dtype(x_df[col]):
            x_df[col] = pd.to_numeric(x_df[col], errors="coerce")
    x_df = x_df.astype(float).reset_index(drop=True)

    y = pd.Series(dtype=float)
    if spec.target_column in frame.columns:
        y = pd.to_numeric(frame[spec.target_column], errors="coerce").reset_index(drop=True)

    market_cols = [c for c in frame.columns if is_market_column(c)]
    key_cols = [c for c in MARKET_CONTEXT_KEY_COLUMNS if c in frame.columns]
    market_df = frame.loc[:, list(dict.fromkeys(key_cols + market_cols))].copy().reset_index(drop=True)

    meta_exclude = set(feature_cols) | set(market_cols)
    if spec.target_column in frame.columns:
        meta_exclude.add(spec.target_column)
    meta_df = frame.loc[:, [c for c in frame.columns if c not in meta_exclude]].copy().reset_index(drop=True)

    return x_df, y, meta_df, market_df


def is_adjusted_feature_column(col):
    """Return whether a registry predictor needs explicit split inclusion.

    Most raw predictors are discovered through the legacy prefix map. This
    helper carries newer engineered families and the three reviewed predictors
    whose legacy prefixes otherwise classify them as metadata/targets.
    """

    name = str(col).lower()
    return (
        is_feature_column(col)
        and (
            name in {
                "games_played",
                "target_points_for_avg",
                "target_points_against_avg",
            }
            or
            name.startswith("opp_adj_")
            or name.startswith("opponent_adj_")
            or name.startswith("opponent_adjusted_")
            or name.startswith("adjusted_")
            or name.startswith("time_adj_")
            or name.startswith("graph_")
        )
    )
