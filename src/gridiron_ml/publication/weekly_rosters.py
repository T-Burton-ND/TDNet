"""Audit the three named weekly rosters without promoting partial research."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

MODEL_IDS = frozenset({"M1", "M2", "M3", "M4", "M5", "M10"})


def _scientific_cells(path: Path, tiers: set[str]) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=["game_id", "fingerprint", "model_name"])
    games = set(frame["game_id"])
    if not games or set(frame["fingerprint"]) != tiers:
        raise ValueError(f"Scientific tier coverage changed: {path}")
    cells = frame.groupby(["fingerprint", "model_name"])["game_id"].agg(["nunique", "size"])
    if len(cells) != 6 * len(tiers) or not cells["nunique"].eq(len(games)).all() or not cells["size"].eq(len(games)).all():
        raise ValueError(f"Scientific model/game coverage changed: {path}")
    if any(not MODEL_IDS.issuperset({str(name).rsplit("_", 1)[-1]}) for _, name in cells.index):
        raise ValueError(f"Scientific architecture changed: {path}")
    return len(games), len(cells)


def write_weekly_roster_registry(
    *, output: Path, operations: Path, season: int, week: int,
    f20_research: dict | None = None,
) -> Path:
    """Record exactly three roster identities and actual research readiness."""
    scientific = output / "scientific"
    original = scientific / "scientific_all_game_predictions.csv"
    expanded_base = scientific / "expanded_f0_f20" / "f0_f8_component" / "scientific_all_game_predictions.csv"
    original_games, original_cells = _scientific_cells(original, {f"F{i}" for i in range(7)})
    expanded_games, expanded_cells = _scientific_cells(expanded_base, {f"F{i}" for i in range(9)})
    if original_games != expanded_games:
        raise ValueError("Frozen and expanded scientific components have different game coverage")
    frozen_bundle = output / "frozen_bundle" / "public" / "prediction_manifest.json"
    if not frozen_bundle.is_file():
        raise FileNotFoundError(frozen_bundle)
    f20_status = None if f20_research is None else f20_research.get("status")
    registry = {
        "season": season,
        "week": week,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rosters": [
            {"id": "margin_wide_publication", "role": "canonical_publication",
             "status": "frozen", "bundle_manifest": str(frozen_bundle)},
            {"id": "original_frozen_scientific_f0_f6", "role": "scientific",
             "status": "complete", "games": original_games, "model_cells": original_cells,
             "predictions": str(original)},
            {"id": "expanded_scientific_f0_f20", "role": "scientific_research",
             "status": "incomplete_pending_f9_f19_and_f20_selection",
             "base_component": {"tiers": "F0-F8", "games": expanded_games,
                                "model_cells": expanded_cells, "predictions": str(expanded_base)},
             "missing_margin_tiers": [f"F{i}" for i in range(9, 20)],
             "f20": {"candidate": "F20-A", "candidate_status": f20_status,
                     "research_run_report": (
                         f20_research.get("result", {}).get("run_report")
                         if f20_research else None
                     ),
                     "canonical_variant": None,
                     "output_type": "home_win_probability_only",
                     "included_in_margin_consensus": False},
             "complete_consensus_available": False},
        ],
    }
    operations.mkdir(parents=True, exist_ok=True)
    path = operations / "weekly_rosters.json"
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return path
