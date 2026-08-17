#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Generate canonical weekly tables, figures, and blog markdown."""

from argparse import ArgumentParser
from pathlib import Path

from gridiron_ml.publication import build_weekly_blog_package


def main():
    parser = ArgumentParser()
    parser.add_argument("--project-root", default=project_root())
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--schedule-snapshot", required=True)
    parser.add_argument("--top25")
    parser.add_argument("--top25-label")
    parser.add_argument("--ap-top25", help="Official AP/CFBD ranking snapshot; drives ranked-game output.")
    parser.add_argument("--tdnet-top25", help="Optional TDNet poll retained as a separate comparison.")
    parser.add_argument("--preseason-rankings", help="Preseason-frozen historical-performance ranking sidecar.")
    parser.add_argument("--fingerprint-version", type=int, default=0)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = build_weekly_blog_package(
        project_root=args.project_root,
        season=args.season,
        week=args.week,
        model_inventory_path=args.model_inventory,
        schedule_snapshot_path=args.schedule_snapshot,
        top25_path=args.top25,
        top25_label=args.top25_label,
        ap_top25_path=args.ap_top25,
        tdnet_top25_path=args.tdnet_top25,
        fingerprint_version=args.fingerprint_version,
        output_root=args.output_root,
        preseason_ranking_path=args.preseason_rankings,
    )
    print(f"games={len(result['all_games'])}")
    print(f"top25_games={len(result['top25_games'])}")
    print(result["output_root"])


if __name__ == "__main__":
    main()
