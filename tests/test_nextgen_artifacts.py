"""Canonical feature writes and model inputs must cross temporal validation."""

from pathlib import Path

import pandas as pd
import pytest

from gridiron_ml.experiments.nextgen_artifacts import (
    NextgenFeatureBuilder, NextgenModelBoundary, canonical_family_path,
)


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "season": [2025], "season_type": ["regular"], "feature_kind": ["dynamic"],
        "target_game_id": [2], "target_start_utc": ["2025-09-06T12:00:00Z"],
        "feature_available_utc": ["2025-09-01T00:00:00Z"],
        "latest_source_game_id": [1],
        "latest_source_game_utc": ["2025-08-31T23:00:00Z"],
        "latest_source_season_type": ["regular"],
        "static_availability_documentation": [None], "prior_margin": [14.0],
    })


def test_builder_cannot_write_without_temporal_validation(tmp_path: Path):
    class Builder(NextgenFeatureBuilder):
        generation = "F09"
        family = "rushing_state"
        feature_columns = ("prior_margin",)

        def __init__(self, frame):
            self.frame = frame

        def build_frame(self):
            return self.frame

    path = canonical_family_path(tmp_path, "F09", "rushing_state")
    with pytest.raises(ValueError):
        Builder(valid_frame().assign(feature_available_utc="2025-09-06T12:00:00Z")).materialize(tmp_path)
    assert not path.exists()
    assert Builder(valid_frame()).materialize(tmp_path) == path
    assert path.exists()
    with pytest.raises(TypeError):
        class Bypass(NextgenFeatureBuilder):
            def build_frame(self):
                return valid_frame()

            def materialize(self, root):
                return root / "unchecked.parquet"


def test_fit_and_shap_revalidate_loaded_canonical_frame(tmp_path: Path):
    path = canonical_family_path(tmp_path, "F09", "rushing_state")
    path.parent.mkdir(parents=True)
    bad = valid_frame().assign(latest_source_game_utc="2025-09-06T12:00:00Z")
    bad.to_parquet(path)
    with pytest.raises(ValueError):
        NextgenModelBoundary.from_canonical(path, ["prior_margin"])
    with pytest.raises(ValueError):
        NextgenModelBoundary(bad, ["prior_margin"])
    valid_frame().to_parquet(path)
    boundary = NextgenModelBoundary.from_canonical(path, ["prior_margin"])
    assert len(boundary.matchup(valid_frame(), valid_frame())) == 1
    assert boundary.for_fit().prior_margin.tolist() == [14.0]
    assert boundary.for_shap().prior_margin.tolist() == [14.0]
