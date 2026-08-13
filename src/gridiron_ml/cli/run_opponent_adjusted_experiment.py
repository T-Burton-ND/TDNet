#!/usr/bin/env python
"""Run the full opponent-adjusted fingerprint experiment sweep."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from pathlib import Path

from gridiron_ml.experiments.opponent_adjusted import run_opponent_adjusted_sweep


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate all TDNet models across opponent-adjusted fingerprints."
    )
    parser.add_argument(
        "--project-root",
        default=project_root(),
        type=Path,
        help="TDNet repository root.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        type=Path,
        help="Experiment output directory. Defaults under data/experiments.",
    )
    parser.add_argument(
        "--overwrite-fingerprints",
        action="store_true",
        help="Rebuild adjusted fingerprint parquet artifacts.",
    )
    parser.add_argument(
        "--clear-output-root",
        action="store_true",
        help="Delete the existing experiment output root before running.",
    )
    parser.add_argument(
        "--delete-checkpoints",
        action="store_true",
        help="Delete model checkpoint files after evaluation to save disk.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_opponent_adjusted_sweep(
        project_root=args.project_root,
        output_root=args.output_root,
        overwrite_fingerprints=args.overwrite_fingerprints,
        clear_output_root=args.clear_output_root,
        keep_checkpoints=not args.delete_checkpoints,
    )
    summary = result["summary"]
    failures = result["failures"]
    print(f"Output root: {result['output_root']}")
    print(f"Successful runs: {int(summary['status'].eq('success').sum())}")
    print(f"Failed runs: {len(failures)}")


if __name__ == "__main__":
    main()
