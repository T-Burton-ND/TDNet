from pathlib import Path

import pandas as pd
import pytest

from gridiron_ml.publication.season_phases import (
    REQUIRED_PREDICTION_METADATA,
    load_phase_map,
    validate_prediction_phase_metadata,
)


PHASE_CONFIG = Path("configs/publication/season_phase_buckets.yaml")


def _prediction(week=4):
    mapping = load_phase_map(PHASE_CONFIG)
    row = {column: "value" for column in REQUIRED_PREDICTION_METADATA}
    row.update({"season": 2026, "official_week": week, "season_phase": mapping[week]})
    return pd.DataFrame([row])


def test_owner_approved_phase_mapping_is_fixed():
    mapping = load_phase_map(PHASE_CONFIG)
    assert {mapping[0], mapping[3], mapping[4], mapping[6], mapping[7], mapping[9], mapping[10]} == {
        "PHASE_1", "PHASE_2", "PHASE_3", "PHASE_4"
    }


def test_prediction_phase_metadata_fails_closed():
    mapping = load_phase_map(PHASE_CONFIG)
    validate_prediction_phase_metadata(_prediction(), mapping)
    invalid = _prediction()
    invalid.loc[0, "season_phase"] = "PHASE_1"
    with pytest.raises(ValueError, match="does not match"):
        validate_prediction_phase_metadata(invalid, mapping)
