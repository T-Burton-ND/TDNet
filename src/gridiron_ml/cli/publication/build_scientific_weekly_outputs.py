"""Generate the paper-only scientific roster package for one weekly phase."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from gridiron_ml.cli._paths import project_root
from gridiron_ml.publication.scientific_weekly import build_scientific_weekly_outputs

ROOT = project_root()


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--schedule-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True, help="Reader-facing week label.")
    parser.add_argument("--phase", choices=["pre_game", "post_game"], required=True)
    parser.add_argument("--poll-week", type=int)
    parser.add_argument("--prediction-week", type=int)
    parser.add_argument("--market-lines", type=Path)
    parser.add_argument("--reference-poll", type=Path)
    args = parser.parse_args()
    paths = build_scientific_weekly_outputs(
        project_root=ROOT,
        inventory_path=args.inventory,
        schedule_snapshot_path=args.schedule_snapshot,
        market_lines_path=args.market_lines,
        reference_poll_path=args.reference_poll,
        output_root=args.output_root,
        season=args.season,
        week=args.week,
        poll_week=args.poll_week,
        prediction_week=args.prediction_week,
        phase=args.phase,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
