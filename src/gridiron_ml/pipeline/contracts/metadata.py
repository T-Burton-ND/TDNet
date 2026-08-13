"""Shared metadata sidecar keys and values."""

from __future__ import annotations

METADATA_FINGERPRINT_VERSION = "fingerprint_version"
METADATA_ROW_SEMANTICS = "row_semantics"
METADATA_DEFAULT_TARGET = "default_target"
METADATA_UNSAFE_SAME_ROW_TARGET = "unsafe_same_row_target"
METADATA_MARKET_POLICY = "market_policy"
METADATA_ARTIFACT_KIND = "artifact_kind"
METADATA_SEASON = "season"

ROW_SEMANTICS_STATE_AFTER_WEEK = "state_after_week"
MARKET_POLICY_EVALUATION_ONLY = "evaluation_only"
ARTIFACT_KIND_HISTORICAL_FINGERPRINT = "historical_fingerprint"
ARTIFACT_KIND_PREDICTION_ROWS = "prediction_rows"
ARTIFACT_KIND_MARKET_CONTEXT = "market_context"


def fingerprint_metadata_payload(
    *,
    version: int,
    default_target: str,
    unsafe_same_row_target: str,
    artifact_kind: str = ARTIFACT_KIND_HISTORICAL_FINGERPRINT,
    season: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        METADATA_FINGERPRINT_VERSION: int(version),
        METADATA_ROW_SEMANTICS: ROW_SEMANTICS_STATE_AFTER_WEEK,
        METADATA_DEFAULT_TARGET: default_target,
        METADATA_UNSAFE_SAME_ROW_TARGET: unsafe_same_row_target,
        METADATA_MARKET_POLICY: MARKET_POLICY_EVALUATION_ONLY,
        METADATA_ARTIFACT_KIND: artifact_kind,
    }
    if season is not None:
        payload[METADATA_SEASON] = int(season)
    return payload
