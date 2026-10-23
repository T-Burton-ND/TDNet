#!/usr/bin/env python3
"""Audit historical Prediction Tracker coverage against CFBD FBS games."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fit_f20_baselines import ROOT, cohort

from gridiron_ml.f20.model import HISTORICAL_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=ROOT / "data/private/f20/historical")
    parser.add_argument("--schedule-dir", type=Path, default=ROOT / "data/raw/cfbd/v2/games")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/publication_2026/f20_source_inventory.json")
    args = parser.parse_args()
    _, audit = cohort(args.historical, args.schedule_dir, range(2013, 2026))
    for season in range(2013, 2026):
        source = args.historical / f"ncaa{season}.csv"
        frame = pd.read_csv(source)
        audit[str(season)]["source_column_nonmissing_rows"] = {
            name: int(frame[column].notna().sum()) if column in frame else 0
            for name, column in HISTORICAL_COLUMNS.items()}
    report = {"generated_at_utc": datetime.now(UTC).isoformat(),
              "source_id": "prediction_tracker_ncaaseason_csv",
              "archive_index": "https://www.thepredictiontracker.com/ncaaarchive.html",
              "provenance": "archived_pregame_unverified",
              "historical_publication_times_available": False,
              "licensing": "Public download links found; redistribution permission not established; raw files retained privately",
              "seasons": audit,
              "other_sources": {
                  "cfbd_fpi_sp": "Current-season ratings can change; historical response times not verified for F20",
                  "direct_espn_fei_sagarin_massey_teamrankings": "Not independently captured or terms-cleared; tracker snapshot columns used instead",
              }}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "reported", "seasons": len(audit), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
