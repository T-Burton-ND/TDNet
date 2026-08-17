#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
from argparse import ArgumentParser
from pathlib import Path
from gridiron_ml.publication.roster_poll import build_frozen_roster_poll

def main():
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", type=Path, default=root)
    args = parser.parse_args()
    result = build_frozen_roster_poll(
        args.inventory, season=args.season, week=args.week, output_dir=args.output_dir,
        project_root=args.project_root, logo_dir=args.project_root / "data/meta/logos/by_team",
    )
    print(result["poll"].to_string(index=False))

if __name__ == "__main__":
    main()
