#!/usr/bin/env python3
"""Extract one immutable AP Top 25 snapshot from the local CFBD archive."""
from argparse import ArgumentParser
from pathlib import Path
from gridiron_ml.publication import load_ap_top25

def main():
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "publication" in args.output.resolve().parts:
        raise ValueError("Raw AP/CFBD snapshot spreadsheets are private inputs and may not be written under publication/.")
    source = args.source or Path(f"data/raw/cfbd/v2/rankings/{args.season}.parquet")
    poll = load_ap_top25(source, season=args.season, week=args.week)
    if len(poll) != 25:
        raise ValueError(f"Expected 25 AP teams, found {len(poll)} in {source}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    poll.to_csv(args.output, index=False)
    print(args.output)

if __name__ == "__main__":
    main()
