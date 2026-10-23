"""The weekly registry must not present partial research as a full roster."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from gridiron_ml.publication.weekly_rosters import (
    write_f20_prediction_component,
    write_weekly_roster_registry,
)


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
    assert rosters[2]["f20"]["canonical_variant"] == "F20"
    assert rosters[2]["f20"]["models"] == ["M1", "M2", "M3", "M4", "M5", "M10"]


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


def test_f20_component_requires_six_models_and_exact_frozen_slate(tmp_path):
    output = tmp_path / "week_09/pre_game"
    bundle = output / "frozen_bundle/public"
    bundle.mkdir(parents=True)
    pd.DataFrame({"season": [2026, 2026], "week": [9, 9], "game_id": [101, 102]}).to_parquet(
        bundle / "predictions.parquet", index=False)
    archive = tmp_path / "private"
    cutoff = "2026-10-30T03:59:00+00:00"
    for game_id in (101, 102):
        folder = archive / "forecasts" / str(game_id)
        folder.mkdir(parents=True)
        probabilities = {model: .5 for model in ("M1", "M2", "M3", "M4", "M5", "M10", "F20")}
        margins = {model: 3.0 for model in ("M1", "M2", "M3", "M4", "M5", "M10", "F20")}
        record_path = folder / "frozen.json"
        record_path.write_text(json.dumps({"season": 2026, "week": 9, "game_id": game_id,
                                           "issuance_class": "eligible_weekly",
                                           "canonical_variant": "F20", "weekly_cutoff_utc": cutoff,
                                           "home_win_probability": probabilities,
                                           "pred_home_margin": margins}))
        pointer = {"record_file": str(record_path), "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
                   "canonical_variant": "F20", "weekly_cutoff_utc": cutoff}
        (folder / "official.json").write_text(json.dumps(pointer))
    manifest = write_f20_prediction_component(output=output, archive=archive,
                                                season=2026, week=9, cutoff_utc=cutoff)
    assert json.loads(manifest.read_text())["model_cells"] == 6
    cells = pd.read_csv(manifest.parent / "f20_model_predictions.csv")
    assert len(cells) == 12 and set(cells.model_id) == {"M1", "M2", "M3", "M4", "M5", "M10"}
    assert cells.pred_home_margin.eq(3.0).all()
    assert write_f20_prediction_component(output=output, archive=archive,
                                           season=2026, week=9, cutoff_utc=cutoff) == manifest
    with pytest.raises(ValueError, match="frozen slate"):
        write_f20_prediction_component(output=output, archive=archive,
                                        season=2026, week=9, cutoff_utc="2026-10-30T04:00:00+00:00")
