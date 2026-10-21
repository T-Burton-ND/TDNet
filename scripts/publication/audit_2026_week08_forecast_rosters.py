#!/usr/bin/env python3
"""Verify Week 8 forecast coverage across the operational and research rosters."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "publication/2026/week_08/pre_game"
SOURCES = {
    "margin_wide": ("frozen_bundle/public/predictions.parquet", 33),
    "scientific_f0_f6": ("scientific/scientific_all_game_predictions.csv", 42),
    "scientific_f0_f8": ("scientific/full_f0_f8/scientific_all_game_predictions.csv", 54),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    proof_path = BASE / "metadata/cutoff_proof.json"
    proof = json.loads(proof_path.read_text())
    if proof["status"] != "pass" or proof["future_target_games"] != 53:
        raise ValueError("Week 8 cutoff proof is missing or incomplete")
    start = pd.Timestamp(proof["earliest_target_kickoff_utc"])
    results = {}
    game_sets = []
    for name, (relative, expected_cells) in SOURCES.items():
        path = BASE / relative
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        games = set(pd.to_numeric(frame.game_id, errors="raise").astype(int))
        models = set(frame.model_name.astype(str))
        if len(games) != 53 or len(models) != expected_cells or len(frame) != 53 * expected_cells:
            raise ValueError(f"Incomplete Week 8 roster: {name}")
        if frame.duplicated(["game_id", "model_name"]).any():
            raise ValueError(f"Duplicate Week 8 cell: {name}")
        if frame[["pred_home_margin", "pred_home_win_probability"]].isna().any().any():
            raise ValueError(f"Missing Week 8 prediction: {name}")
        if not frame.pred_home_win_probability.between(0, 1).all():
            raise ValueError(f"Invalid Week 8 probability: {name}")
        if name == "margin_wide":
            freeze = pd.to_datetime(frame.created_at_utc, utc=True)
            kickoff = pd.to_datetime(frame.game_start_time_utc, utc=True)
            if not ((freeze < kickoff) & (freeze < start)).all():
                raise ValueError("Operational freeze is not before kickoff")
        else:
            manifest = json.loads((path.parent / "manifest.json").read_text())
            if manifest["phase"] != "pre_game" or manifest["scientific_prediction_model_count"] != expected_cells:
                raise ValueError(f"Scientific manifest does not match {name}")
            if pd.Timestamp(manifest["created_at_eastern"]).tz_convert("UTC") >= start:
                raise ValueError(f"Scientific freeze after first remaining kickoff: {name}")
        game_sets.append(games)
        results[name] = {
            "status": "frozen_complete", "games": len(games), "model_cells": len(models),
            "model_game_rows": len(frame), "file": str(path.relative_to(ROOT)), "sha256": digest(path),
        }
    if not all(games == game_sets[0] for games in game_sets):
        raise ValueError("Week 8 rosters use different game IDs")
    expanded_dir = BASE / "scientific/full_f0_f19"
    expanded_manifest = expanded_dir / "manifest.json"
    if expanded_manifest.exists():
        manifest = json.loads(expanded_manifest.read_text())
        path = expanded_dir / "scientific_all_game_predictions.csv"
        frame = pd.read_csv(path)
        if (manifest["model_cells"] != 120 or len(frame) != 120 * 53
                or frame.duplicated(["game_id", "fingerprint", "model_id"]).any()
                or set(frame.game_id.astype(int)) != game_sets[0]
                or not manifest["no_target_outcomes_read"]):
            raise ValueError("Expanded Week 8 roster is not a complete, score-free 120-cell panel")
        results["scientific_f0_f19"] = {
            "status": "frozen_complete_research", "games": 53, "model_cells": 120,
            "model_game_rows": len(frame), "file": str(path.relative_to(ROOT)),
            "sha256": digest(path), "manifest_sha256": digest(expanded_manifest),
        }
    else:
        expanded_path = BASE / "expanded_scientific_status.json"
        expanded = json.loads(expanded_path.read_text())
        if expanded["current_week_games"] != 53:
            raise ValueError("Expanded status uses a different game cohort")
        results["scientific_f0_f19"] = {
            "status": expanded["status"], "games": 0, "model_cells": expanded["trained_model_cells"],
            "model_game_rows": 0, "status_file": str(expanded_path.relative_to(ROOT)),
            "status_sha256": digest(expanded_path),
        }
    output = {
        "season": 2026, "week": 8, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_games": 53, "same_game_cohort_verified": True,
        "first_remaining_kickoff_utc": proof["earliest_target_kickoff_utc"],
        "cutoff_proof_sha256": digest(proof_path), "rosters": results,
    }
    target = BASE / "forecast_roster_coverage.json"
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: row["status"] for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
