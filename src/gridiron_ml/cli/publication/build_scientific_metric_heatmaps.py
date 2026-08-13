#!/usr/bin/env python3
"""Build six publication heatmaps from one or more complete HPS result tables."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from gridiron_ml.publication.scientific_metric_heatmaps import (
    aggregate_metric_grid,
    render_metric_heatmaps,
    selected_complete_trials,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-folds", type=int, default=10)
    parser.add_argument("--required-tiers", nargs="*", default=[])
    args = parser.parse_args()
    results = pd.concat([pd.read_parquet(path) for path in args.results], ignore_index=True)
    trials = selected_complete_trials(results, expected_folds=args.expected_folds)
    table = aggregate_metric_grid(trials)
    args.output_root.mkdir(parents=True, exist_ok=True)
    trials.to_parquet(args.output_root / "selected_complete_fold_results.parquet", index=False)
    table.to_parquet(args.output_root / "scientific_roster_metric_grid.parquet", index=False)
    table.to_csv(args.output_root / "scientific_roster_metric_grid.csv", index=False)
    paths = render_metric_heatmaps(table, args.output_root, required_tiers=args.required_tiers)
    print(f"cells={len(table)} figures={len(paths)} output={args.output_root}")


if __name__ == "__main__":
    main()
