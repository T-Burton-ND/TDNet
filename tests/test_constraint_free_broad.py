"""Checks for the widened historical and 2026 pregame feature contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from constraint_free_2026_matrix import build, formula, target_schedule
from constraint_free_broad_universe import load_broad_historical
from constraint_free_search import DATA


@pytest.mark.parametrize("operation,inputs,expected", [
    ("mean", ["a", "b"], [3.0, np.nan, np.nan]),
    ("difference", ["a", "b"], [-2.0, np.nan, np.nan]),
    ("product", ["a", "b"], [8.0, np.nan, np.nan]),
    ("ratio", ["a", "b"], [0.5, np.nan, np.nan]),
])
def test_manifest_formulas_preserve_missingness(operation, inputs, expected):
    source = pd.DataFrame({"a": [2.0, 2.0, np.nan], "b": [4.0, 0.0, 4.0]})
    actual = formula({"name": "derived", "operation": operation,
                      "source_inputs": inputs}, source)
    if operation in ("difference", "product", "mean"):
        expected[1] = {"difference": 2.0, "product": 0.0, "mean": 1.0}[operation]
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_widened_historical_and_target_matrices_share_columns():
    import constraint_free_2026_matrix as matrix

    archive = matrix.ARCHIVE / "fingerprints"
    required = [
        *(archive / f"F12_F_{design}" / "feature_manifest.json" for design in "abc"),
        *(archive / f"F12_F_{design}" / "values.parquet" for design in "bc"),
        DATA / "prepare_receipt.json",
        matrix.LADDER, matrix.GAMES, matrix.MARKET, *matrix.STATE_PATHS.values(),
    ]
    if any(not path.is_file() for path in required):
        pytest.skip("TDNet private historical and 2026 input archives are unavailable")
    x18, hmeta, records18, evidence18 = load_broad_historical("F18")
    x19, hmeta19, records19, evidence19 = load_broad_historical("F19")
    target, target_meta, report = build()
    assert x18.shape == (7358, 467)
    assert x19.shape == (7358, 472)
    assert hmeta.target_game_id.equals(hmeta19.target_game_id)
    assert list(target.columns[1:]) == [record.name for record in records18]
    assert len(records19) == len(records18) + 5
    assert len(evidence18["extra_source_features"]) == 98
    assert evidence19["prospective_extra_features_approved"] is False
    assert len(target_meta) == 271
    assert report["no_2026_outcome_columns_read"]
    assert not any("margin" in column or "points" in column
                   for column in target_meta.columns)


def test_2026_schedule_reader_requests_no_score_columns(monkeypatch, tmp_path):
    import constraint_free_2026_matrix as module

    games_path = tmp_path / "games.parquet"
    pd.DataFrame({
        "id": np.arange(271), "season": 2026, "week": 1,
        "start_date": "2026-09-01T17:00:00Z", "home_team": "Home",
        "away_team": "Away", "season_type": "regular", "completed": True,
        "home_classification": "fbs", "away_classification": "fbs",
        "home_points": 100,
    }).to_parquet(games_path, index=False)
    monkeypatch.setattr(module, "GAMES", games_path)
    read = pd.read_parquet
    seen = []

    def checked_read(path, *args, **kwargs):
        seen.extend(kwargs.get("columns", []))
        return read(path, *args, **kwargs)

    monkeypatch.setattr(module.pd, "read_parquet", checked_read)
    assert len(target_schedule()) == 271
    assert seen
    assert not any("points" in name or "margin" in name or "score" in name
                   for name in seen)
