"""Leakage-safety checks for TDNet feature and training paths."""

from __future__ import annotations

from collections.abc import Iterable

from gridiron_ml.pipeline.contracts.features import (
    BLOCKED_COACH_FEATURES,
    DEFAULT_TRAINING_TARGET,
    POSTGAME_FEATURE_MARKERS,
    SAME_WEEK_TARGET,
    blocked_coach_columns,
    market_feature_columns,
)

MARKET_FEATURE_ERROR = (
    "Market/Vegas/betting-derived columns are eval-only by default and must not "
    "be included in model training features. Remove market_* columns from X, or "
    "set allow_market_features_for_training=True only for an explicit market-feature experiment."
)

STALE_ARTIFACT_ERROR = (
    "Training features contain coach columns that are not allowed because they "
    "may include current-season or postseason information. This usually means "
    "a persisted fingerprint artifact is stale and should be rebuilt or cleaned."
)


def assert_no_market_features(
    feature_columns: Iterable[object] | None,
    *,
    allow_market_features_for_training: bool = False,
) -> None:
    """Raise when model feature columns contain market/Vegas-derived signals."""
    market_cols = market_feature_columns(feature_columns)
    if market_cols and not bool(allow_market_features_for_training):
        preview = ", ".join(market_cols[:8])
        extra = "" if len(market_cols) <= 8 else f", ... (+{len(market_cols) - 8} more)"
        raise ValueError(f"{MARKET_FEATURE_ERROR} Offending columns: {preview}{extra}")


def leaky_coach_feature_columns(feature_columns: Iterable[object] | None) -> list[str]:
    """Return coach features that are unsafe for model training."""
    return blocked_coach_columns(feature_columns)


def assert_no_leaky_coach_features(feature_columns: Iterable[object] | None) -> None:
    """Raise when feature columns contain blocked coach-derived signals."""
    leaky_cols = leaky_coach_feature_columns(feature_columns)
    if leaky_cols:
        preview = ", ".join(leaky_cols[:8])
        extra = "" if len(leaky_cols) <= 8 else f", ... (+{len(leaky_cols) - 8} more)"
        raise ValueError(f"{STALE_ARTIFACT_ERROR} Offending columns: {preview}{extra}")


def assert_disjoint_years(
    train_years: Iterable[object] | None,
    val_years: Iterable[object] | None,
    test_years: Iterable[object] | None = None,
) -> None:
    """Raise if train/validation/test season sets overlap."""
    splits = {
        "train": _year_set(train_years),
        "val": _year_set(val_years),
        "test": _year_set(test_years),
    }
    split_names = list(splits)
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1 :]:
            overlap = splits[left_name] & splits[right_name]
            if overlap:
                years = ", ".join(str(year) for year in sorted(overlap))
                raise ValueError(
                    f"Training/evaluation leakage risk: {left_name} and {right_name} years overlap: {years}."
                )


def assert_default_target_is_next_margin(target_column: str | None) -> None:
    """Raise unless the configured default training target is the next-game margin."""
    if str(target_column) != DEFAULT_TRAINING_TARGET:
        raise ValueError(
            f"Default TDNet training target must be {DEFAULT_TRAINING_TARGET}. "
            f"{SAME_WEEK_TARGET} is a same-row completed-game label and is unsafe with postgame features."
        )


def assert_no_same_week_postgame_target(
    target_column: str | None,
    feature_columns: Iterable[object] | None,
) -> None:
    """Raise when y_margin_this_week is paired with same-row postgame features."""
    if str(target_column) != SAME_WEEK_TARGET:
        return
    columns = [] if feature_columns is None else feature_columns
    postgame_cols = [
        str(col)
        for col in columns
        if any(_has_feature_marker(str(col), marker) for marker in POSTGAME_FEATURE_MARKERS)
    ]
    if postgame_cols:
        preview = ", ".join(postgame_cols[:8])
        extra = "" if len(postgame_cols) <= 8 else f", ... (+{len(postgame_cols) - 8} more)"
        raise ValueError(
            f"{SAME_WEEK_TARGET} is a same-row completed-game target and is unsafe "
            f"with postgame/current-week feature columns. Use {DEFAULT_TRAINING_TARGET}, or build "
            f"a truly pregame-safe feature frame. Offending columns: {preview}{extra}"
        )


def training_allows_market_features(config: dict | None) -> bool:
    """Read the loud market-feature opt-in flag from common config locations."""
    if not isinstance(config, dict):
        return False
    candidates = [
        config,
        config.get("training", {}),
        config.get("model", {}),
        config.get("eval", {}),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and "allow_market_features_for_training" in candidate:
            return bool(candidate["allow_market_features_for_training"])
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict):
        training_cfg = model_cfg.get("training", {})
        if (
            isinstance(training_cfg, dict)
            and "allow_market_features_for_training" in training_cfg
        ):
            return bool(training_cfg["allow_market_features_for_training"])
    return False


def _year_set(years: Iterable[object] | None) -> set[int]:
    if years is None:
        return set()
    if isinstance(years, (str, int, float)):
        return {int(years)}
    return {int(year) for year in years}


def _has_feature_marker(name: str, marker: str) -> bool:
    return name.startswith(marker) or f"_{marker}" in name
