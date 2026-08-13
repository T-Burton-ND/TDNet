#!/usr/bin/env python3
"""Rebuild an append-only season ledger from immutable weekly manifests."""

from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument("--weekly-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.weekly_root).glob("week_*/public/prediction_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "week_directory": path.parents[1].name,
                "created_at_utc": manifest.get("created_at_utc"),
                "manifest_sha256": manifest.get("manifest_sha256"),
                "git_commit": manifest.get("git_commit"),
                "prediction_rows": manifest.get("prediction_rows"),
                "game_count": manifest.get("game_count"),
                "model_count": manifest.get("model_count"),
                "manifest_path": str(path),
            }
        )
    ledger = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output, index=False)
    ledger.to_parquet(output.with_suffix(".parquet"), index=False)
    print(f"weeks={len(ledger)}")


if __name__ == "__main__":
    main()

