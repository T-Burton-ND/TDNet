"""Small dataframe schema checks for TDNet pipeline contracts."""

from __future__ import annotations

import pandas as pd

from gridiron_ml.pipeline.contracts.columns import (
    REQUIRED_FINGERPRINT_COLUMNS,
    REQUIRED_LABEL_COLUMNS,
    REQUIRED_PREDICTION_ROW_COLUMNS,
    REQUIRED_TEAM_GAME_COLUMNS,
)
from gridiron_ml.pipeline.contracts.features import (
    DEFAULT_TRAINING_TARGET,
    SAME_WEEK_TARGET,
    TARGET_COLUMNS,
    blocked_coach_columns,
    market_feature_columns,
)
from gridiron_ml.pipeline.validation.leakage import (
    assert_no_leaky_coach_features,
    assert_no_market_features,
)


class SchemaValidationError(ValueError):
    """Raised when a TDNet dataframe does not match an expected row contract."""


def validate_team_game_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validate completed historical team-game rows used for training artifacts."""
    frame = _dataframe(df, "team_game_table")
    _require_columns(frame, REQUIRED_TEAM_GAME_COLUMNS, "team_game_table")
    _require_nonempty(frame, "team_game_table")
    _require_any_non_null(
        frame,
        TARGET_COLUMNS,
        "team_game_table",
        "completed-game targets",
    )
    unique_keys = [
        col
        for col in ["keys_season", "keys_week", "keys_game_id", "keys_team"]
        if col in frame.columns
    ]
    if "keys_game_id" in unique_keys:
        _require_unique(frame, unique_keys, "team_game_table")
    return frame


def validate_fingerprint_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate team-week fingerprint rows with explicit v0 next-game labels."""
    frame = _dataframe(df, "fingerprint_frame")
    _require_columns(
        frame,
        REQUIRED_FINGERPRINT_COLUMNS,
        "fingerprint_frame",
    )
    _require_unique(frame, ["keys_season", "keys_team", "keys_week"], "fingerprint_frame")
    return frame


def validate_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate labels where y_next_margin is the intended training target."""
    frame = _dataframe(df, "label_frame")
    _require_columns(
        frame,
        REQUIRED_LABEL_COLUMNS,
        "label_frame",
    )
    _require_unique(frame, ["keys_season", "keys_team", "keys_week"], "label_frame")
    return frame


def validate_fingerprint_feature_artifact(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a persisted fingerprint feature artifact before it is written."""
    frame = _dataframe(df, "fingerprint_feature_artifact")
    _require_columns(frame, ["keys_season", "keys_team", "keys_week"], "fingerprint_feature_artifact")
    _require_unique(frame, ["keys_season", "keys_team", "keys_week"], "fingerprint_feature_artifact")
    market_cols = market_feature_columns(frame.columns)
    if market_cols:
        raise SchemaValidationError(
            "fingerprint_feature_artifact must not contain market/Vegas-derived columns: "
            + ", ".join(market_cols[:8])
        )
    leaky_cols = blocked_coach_columns(frame.columns)
    if leaky_cols:
        raise SchemaValidationError(
            "fingerprint_feature_artifact contains blocked coach leakage columns. "
            "Rebuild or clean stale artifacts. Offending columns: "
            + ", ".join(leaky_cols[:8])
        )
    return frame


def _require_numeric_or_bool(df: pd.DataFrame, name: str) -> None:
    non_numeric = [
        col
        for col in df.columns
        if not (
            pd.api.types.is_numeric_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        )
    ]
    if non_numeric:
        raise SchemaValidationError(
            f"{name} feature columns must be numeric/bool. "
            f"Non-numeric columns: {non_numeric[:8]}"
        )


def validate_training_feature_frame(
    df: pd.DataFrame,
    *,
    allow_market_features_for_training: bool = False,
    name: str = "training_feature_frame",
) -> pd.DataFrame:
    """Validate model-ready training features for leakage-sensitive columns."""
    frame = _dataframe(df, name)
    _require_nonempty(frame, name)
    try:
        assert_no_market_features(
            frame.columns,
            allow_market_features_for_training=allow_market_features_for_training,
        )
        assert_no_leaky_coach_features(frame.columns)
    except ValueError as exc:
        raise SchemaValidationError(str(exc)) from exc
    _require_numeric_or_bool(frame, name)
    return frame


def validate_matchup_frame(
    df: pd.DataFrame,
    *,
    allow_market_features_for_training: bool = False,
) -> pd.DataFrame:
    """Validate numeric model matchup features without unsafe training columns."""
    frame = _dataframe(df, "matchup_frame")
    return validate_training_feature_frame(
        frame,
        allow_market_features_for_training=allow_market_features_for_training,
        name="matchup_frame",
    )


def validate_market_context_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a market context frame that is separate from model features."""
    frame = _dataframe(df, "market_context_frame")
    market_cols = [col for col in frame.columns if str(col).startswith("market_")]
    if not market_cols:
        return frame
    key_cols = [col for col in frame.columns if str(col).startswith("keys_") or str(col).startswith("next_")]
    _require_columns(frame, key_cols + market_cols, "market_context_frame")
    return frame


def validate_prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Validate future schedule rows for prediction, not historical training."""
    frame = _dataframe(df, "prediction_rows")
    _require_columns(
        frame,
        REQUIRED_PREDICTION_ROW_COLUMNS,
        "prediction_rows",
    )
    _require_nonempty(frame, "prediction_rows")
    _require_non_null(frame, ["season", "week", "home_team", "away_team"], "prediction_rows")
    label_cols = [
        col
        for col in [DEFAULT_TRAINING_TARGET, SAME_WEEK_TARGET, "target_team_margin"]
        if col in frame.columns and frame[col].notna().any()
    ]
    if label_cols:
        raise SchemaValidationError(
            "prediction_rows are schedule-only rows and must not carry filled training labels: "
            + ", ".join(label_cols)
        )
    return frame


PUBLIC_PREDICTION_REQUIRED_COLUMNS = (
    "prediction_id",
    "season",
    "week",
    "game_id",
    "game_start_time_utc",
    "home_team",
    "away_team",
    "neutral_site",
    "conference_game",
    "season_type",
    "created_at_utc",
    "prediction_deadline_utc",
    "model_family",
    "model_name",
    "objective",
    "checkpoint_sha256",
    "feature_manifest_sha256",
    "data_snapshot_sha256",
    "schedule_snapshot_sha256",
    "git_commit",
    "pipeline_version",
    "environment_lock_sha256",
    "pred_home_margin",
    "pred_home_win_probability",
    "pred_winner",
    "confidence",
    "pred_total",
    "model_rank_home",
    "model_rank_away",
    "top25_rank_home",
    "top25_rank_away",
    "vegas_spread_close_as_of_prediction",
    "vegas_total_close_as_of_prediction",
    "vegas_home_win_probability_as_of_prediction",
    "ap_rank_home",
    "ap_rank_away",
    "coaches_rank_home",
    "coaches_rank_away",
    "cfp_rank_home",
    "cfp_rank_away",
)


def validate_public_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validate immutable pre-kickoff model/game prediction assertions."""
    frame = _dataframe(df, "public_prediction_table")
    _require_columns(frame, PUBLIC_PREDICTION_REQUIRED_COLUMNS, "public_prediction_table")
    _require_nonempty(frame, "public_prediction_table")
    non_null = [
        column
        for column in PUBLIC_PREDICTION_REQUIRED_COLUMNS
        if column
        not in {
            "pred_total",
            "model_rank_home",
            "model_rank_away",
            "top25_rank_home",
            "top25_rank_away",
            "vegas_spread_close_as_of_prediction",
            "vegas_total_close_as_of_prediction",
            "vegas_home_win_probability_as_of_prediction",
            "ap_rank_home",
            "ap_rank_away",
            "coaches_rank_home",
            "coaches_rank_away",
            "cfp_rank_home",
            "cfp_rank_away",
        }
    ]
    _require_non_null(frame, non_null, "public_prediction_table")
    _require_unique(frame, ["prediction_id"], "public_prediction_table")
    _require_unique(frame, ["season", "week", "game_id", "model_name", "objective"], "public_prediction_table")
    created = pd.to_datetime(frame["created_at_utc"], utc=True, errors="coerce")
    deadline = pd.to_datetime(frame["prediction_deadline_utc"], utc=True, errors="coerce")
    kickoff = pd.to_datetime(frame["game_start_time_utc"], utc=True, errors="coerce")
    if created.isna().any() or deadline.isna().any() or kickoff.isna().any():
        raise SchemaValidationError("Public prediction timestamps must be valid UTC timestamps.")
    if (created >= kickoff).any():
        raise SchemaValidationError("Every public prediction must be created before kickoff.")
    if (deadline > kickoff).any():
        raise SchemaValidationError("Prediction deadlines cannot be after kickoff.")
    if (created > deadline).any():
        raise SchemaValidationError("Predictions cannot be created after their declared deadline.")
    probability = pd.to_numeric(frame["pred_home_win_probability"], errors="coerce")
    if probability.isna().any() or ((probability <= 0) | (probability >= 1)).any():
        raise SchemaValidationError("Win probabilities must be finite and strictly between 0 and 1.")
    expected_winner = frame["home_team"].where(probability >= 0.5, frame["away_team"])
    if not expected_winner.astype(str).eq(frame["pred_winner"].astype(str)).all():
        raise SchemaValidationError("pred_winner must agree with pred_home_win_probability.")
    if "kickoff_time_confirmed" in frame.columns and not frame["kickoff_time_confirmed"].fillna(False).astype(bool).all():
        raise SchemaValidationError("Public bundles cannot include unconfirmed kickoff times.")
    return frame


def validate_scored_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validate append-only scored rows without changing frozen predictions."""
    frame = _dataframe(df, "scored_prediction_table")
    _require_columns(
        frame,
        ["prediction_id", "game_id", "actual_home_margin", "actual_home_win", "scored_at_utc"],
        "scored_prediction_table",
    )
    _require_unique(frame, ["prediction_id"], "scored_prediction_table")
    return frame


def _dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise SchemaValidationError(f"{name} must be a pandas DataFrame.")
    return df


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SchemaValidationError(f"{name} is missing required columns: {missing}")


def _require_nonempty(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise SchemaValidationError(f"{name} must contain at least one row.")


def _require_non_null(df: pd.DataFrame, columns: list[str], name: str) -> None:
    bad = [col for col in columns if df[col].isna().any()]
    if bad:
        raise SchemaValidationError(f"{name} has null values in required columns: {bad}")


def _require_any_non_null(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
    label: str,
) -> None:
    if not df.loc[:, columns].notna().any().any():
        raise SchemaValidationError(f"{name} must contain at least one non-null {label} column.")


def _require_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    if df.duplicated(subset=keys).any():
        raise SchemaValidationError(f"{name} rows must be unique by keys: {keys}")
