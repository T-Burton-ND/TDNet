#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Score a frozen weekly bundle into a separate output tree."""

from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from gridiron_ml.publication import score_prediction_bundle


def main():
    parser = ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    path = Path(args.results)
    results = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    tables = score_prediction_bundle(args.bundle, results, output_root=args.output_root)
    print(tables["scorecard"].to_string(index=False))


if __name__ == "__main__":
    main()
