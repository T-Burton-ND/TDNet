"""F19 snapshot provenance must fail closed at the declared cutoff."""
from __future__ import annotations

import pandas as pd
import pytest

from gridiron_ml.experiments.constraint_free import SNAPSHOT_MARKET_COLUMNS
from gridiron_ml.experiments.constraint_free_market import (
    provenance_manifest, validate_pregame_market_state,
)


def target_frame():
    return pd.DataFrame({
        "target_game_id": [1, 2],
        "prediction_cutoff_utc": ["2026-09-01T17:00:00Z"] * 2,
        "target_start_utc": ["2026-09-01T18:00:00Z"] * 2,
    })


def state_frame():
    frame = pd.DataFrame({"target_game_id": [1, 2],
                          "snapshot_timestamp_utc": ["2026-09-01T16:00:00Z", None]})
    for name in SNAPSHOT_MARKET_COLUMNS:
        frame[name] = [1.0, None]
    return frame


def test_market_provenance_lists_every_prospective_field():
    records = provenance_manifest()
    assert [r["name"] for r in records] == list(SNAPSHOT_MARKET_COLUMNS)
    assert all(not r["historical_quote_time_verified"] for r in records)
    assert all(r["prospective_raw_field"] and r["cutoff_rule"] for r in records)


def test_missing_market_is_unavailable_and_late_market_fails():
    state = state_frame()
    coverage = validate_pregame_market_state(state, target_frame())
    assert coverage.to_dict() == {1: True, 2: False}
    state.loc[0, "snapshot_timestamp_utc"] = "2026-09-01T17:30:00Z"
    with pytest.raises(ValueError, match="violates cutoff"):
        validate_pregame_market_state(state, target_frame())
    state.loc[0, "snapshot_timestamp_utc"] = None
    with pytest.raises(ValueError, match="violates cutoff"):
        validate_pregame_market_state(state, target_frame())
