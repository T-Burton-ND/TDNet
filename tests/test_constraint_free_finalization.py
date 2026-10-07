"""Freeze gates that prevent one-year selection and premature 2026 scoring."""
from __future__ import annotations

import json

import pytest


def test_finalist_cannot_use_one_year_reducer_screen(tmp_path):
    from constraint_free_finalists import _fold_candidate

    row = {
        "status": "success", "tier": "F18", "architecture": "M1",
        "folds": [{"year": 2025, "mae": 12.0}],
        "mean_mae": 12.0, "mean_brier": 0.2, "worst_year_mae": 12.0,
        "model_fits": 1,
    }
    with pytest.raises(ValueError, match="common four-year folds"):
        _fold_candidate(row, track="extra_stage2", path=tmp_path / "screen.json",
                        manifest=tmp_path / "manifest.json")


def test_invalid_prediction_freeze_blocks_2026_outcome_read(tmp_path, monkeypatch):
    import constraint_free_score_2026 as scorer

    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"no_2026_outcomes_read": True}))
    receipt_dir = tmp_path / "predictions"
    receipt_dir.mkdir()
    (receipt_dir / "receipt.json").write_text(json.dumps({
        "freeze_manifest": str(freeze), "freeze_manifest_sha256": "incorrect",
        "no_2026_outcomes_read": True,
    }))
    monkeypatch.setattr(scorer, "FREEZE", freeze)

    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("2026 outcomes were read before the freeze check")

    monkeypatch.setattr(scorer.pd, "read_parquet", forbidden_read)
    with pytest.raises(ValueError, match="freeze identity changed"):
        scorer.score(receipt_dir, tmp_path / "scores")
