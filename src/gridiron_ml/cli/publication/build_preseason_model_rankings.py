#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Build the preseason-frozen historical-performance ranking sidecar."""

from argparse import ArgumentParser
from pathlib import Path

from gridiron_ml.publication import build_preseason_performance_rankings


def main() -> None:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=root / "models/season_2026_full_roster/final_model_inventory.csv")
    parser.add_argument("--leaderboard", type=Path, default=root / "data/comparisons/season_2026_full_roster_vs_vegas/roster_leaderboard_across_seasons.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-baselines", action="store_true", help="Allow KNN/naive baseline rows to become Top-1/Top-3 selections.")
    args = parser.parse_args()
    output = args.output or args.inventory.parent / "preseason_model_rankings.csv"
    frame = build_preseason_performance_rankings(
        args.inventory, args.leaderboard, output, exclude_baselines=not args.include_baselines
    )
    print(f"wrote {len(frame)} ranking rows: {output}")
    print(frame[["preseason_performance_rank", "model_id", "objective", "historical_performance_score", "ranking_eligible"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
