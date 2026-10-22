#!/usr/bin/env python3
"""Read-only source audit for the planned 120-cell historical search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "publication/2026/research/scientific_120_hps_v1/input_audit.json"
OLDER_FOLDS = Path("/groups/bsavoie2/tburton2/TDNet/publication_artifacts/scientific_roster_heatmaps/corrected_full_ladder_v1/summary/heatmaps/tables/selected_complete_fold_results.parquet")
OLDER_SOURCE = Path("/groups/bsavoie2/tburton2/TDNet/publication_artifacts/fingerprint_ladder_v3/canonical_fingerprint.parquet")
NEWER_PARENT = Path("/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/fingerprints/F12_F_a/values.parquet")


def inspect() -> dict:
    sources = {}
    for label, path in (("f0_f8_selected_fold_summaries", OLDER_FOLDS),
                        ("f0_f8_canonical_team_state", OLDER_SOURCE),
                        ("f09_f19_parent_team_games", NEWER_PARENT)):
        if path.is_file():
            file = pq.ParquetFile(path)
            sources[label] = {"path": str(path), "exists": True,
                              "rows": file.metadata.num_rows, "columns": file.schema.names}
        else:
            sources[label] = {"path": str(path), "exists": False}
    later = {}
    if NEWER_PARENT.is_file():
        frame = pd.read_parquet(NEWER_PARENT, columns=["season", "target_game_id", "is_home", "target_start_utc"])
        post_2025_rows = int(frame.season.gt(2025).sum())
        frame = frame.loc[frame.season.between(2013, 2025)]
        home = frame.loc[frame.is_home]
        later = {"target_seasons": sorted(home.season.unique().astype(int).tolist()),
                 "test_games_by_year": {str(y): int(home.season.eq(y).sum()) for y in range(2021, 2026)},
                 "unique_home_game_ids": not home.target_game_id.duplicated().any(),
                 "post_2025_rows": post_2025_rows}
    old = {}
    if OLDER_FOLDS.is_file():
        frame = pd.read_parquet(OLDER_FOLDS, columns=["objective", "feature_config", "model_level", "outer_fold", "status"])
        frame = frame.loc[(frame.objective == "margin") & frame.feature_config.isin([f"F{i}" for i in range(9)])
                          & frame.model_level.isin(["M1", "M2", "M3", "M4", "M5", "M10"])
                          & (frame.status == "success")]
        old = {"selected_summary_rows": len(frame),
               "fold_test_seasons": sorted((frame.outer_fold.astype(int) + 2015).unique().tolist()),
               "contains_per_game_features": False}
    blockers = [
        "No versioned F0-F8 per-game feature matrices aligned to the proposed 2021-2025 cohort have been bound to this study.",
        "Common game IDs and pregame feature availability have not been verified across all 20 fingerprints.",
        "Historical F7/F8/F17/F19 market quote timestamps remain unverified.",
        "Original incumbent settings for each of 120 cells have not been bound as a separate control manifest.",
    ]
    if not NEWER_PARENT.is_file():
        blockers.append("The F09-F19 parent historical team-game matrix is missing.")
    if not OLDER_FOLDS.is_file():
        blockers.append("The F0-F8 selected-fold summary archive is missing.")
    return {"status": "blocked_before_fit", "historical_fits_authorized": False,
            "proposed_test_seasons": list(range(2021, 2026)), "sources": sources,
            "f09_f19_parent_cohort": later, "f0_f8_prior_evaluation": old,
            "launch_blockers": blockers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
