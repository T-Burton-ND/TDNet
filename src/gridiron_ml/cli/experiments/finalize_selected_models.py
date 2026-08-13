#!/usr/bin/env python3
"""Select finalist trial rows; refits are separate, explicit jobs."""

from argparse import ArgumentParser
from pathlib import Path

from gridiron_ml.experiments.publication import atomic_write_frame, read_frame, select_finalists


def main():
    parser = ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-per-cell", type=int, default=10)
    args = parser.parse_args()
    finalists = select_finalists(
        read_frame(args.results), max_per_cell=args.max_per_cell
    )
    path = Path(args.output_root) / "summary" / "tables" / "selected_finalists.parquet"
    atomic_write_frame(finalists, path)
    finalists.to_csv(path.with_suffix(".csv"), index=False)
    print(f"finalists={len(finalists)}")
    print(path)


if __name__ == "__main__":
    main()

