#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json

def main():
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    directory = root / f"data/publication/{args.season}/weekly_operations/week_{args.week:02d}"
    report = json.loads((directory / "monday_inspection.json").read_text())
    if report.get("automated_status") != "pass":
        raise RuntimeError("Cannot approve a failed Monday inspection.")
    marker = {"approved_at_utc": datetime.now(timezone.utc).isoformat(), "inspection_created_at_utc": report["created_at_utc"]}
    (directory / "monday_review.approved").write_text(json.dumps(marker, indent=2) + "\n")
    print(directory / "monday_review.approved")

if __name__ == "__main__":
    main()
