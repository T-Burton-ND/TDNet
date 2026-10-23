"""The weekly registry must not present partial research as a full roster."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from gridiron_ml.publication.weekly_rosters import write_weekly_roster_registry


def _cells(path, last_tier: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"game_id": game, "fingerprint": f"F{tier}",
         "model_name": f"scientific_F{tier}_{model}"}
        for game in (101, 102)
        for tier in range(last_tier + 1)
        for model in ("M1", "M2", "M3", "M4", "M5", "M10")
    ).to_csv(path, index=False)


def test_weekly_registry_has_three_honest_rosters(tmp_path):
    output = tmp_path / "week_09" / "pre_game"
    frozen = output / "frozen_bundle/public/prediction_manifest.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("{}")
    _cells(output / "scientific/scientific_all_game_predictions.csv", 6)
    _cells(output / "scientific/expanded_f0_f20/f0_f8_component/scientific_all_game_predictions.csv", 8)
    path = write_weekly_roster_registry(
        output=output, operations=tmp_path / "operations", season=2026, week=9,
        f20_research={"status": "succeeded", "result": {"run_report": "private.json"}},
    )
    rosters = json.loads(path.read_text())["rosters"]
    assert [row["id"] for row in rosters] == [
        "margin_wide_publication", "original_frozen_scientific_f0_f6",
        "expanded_scientific_f0_f20",
    ]
    assert rosters[1]["model_cells"] == 42
    assert rosters[2]["base_component"]["model_cells"] == 54
    assert len(rosters[2]["missing_margin_tiers"]) == 11
    assert rosters[2]["complete_consensus_available"] is False
    assert rosters[2]["f20"]["canonical_variant"] is None


def test_weekly_registry_rejects_missing_scientific_cell(tmp_path):
    output = tmp_path / "week_09" / "pre_game"
    frozen = output / "frozen_bundle/public/prediction_manifest.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("{}")
    original = output / "scientific/scientific_all_game_predictions.csv"
    _cells(original, 6)
    frame = pd.read_csv(original).iloc[1:]
    frame.to_csv(original, index=False)
    _cells(output / "scientific/expanded_f0_f20/f0_f8_component/scientific_all_game_predictions.csv", 8)
    with pytest.raises(ValueError, match="coverage"):
        write_weekly_roster_registry(
            output=output, operations=tmp_path / "operations", season=2026, week=9,
        )
