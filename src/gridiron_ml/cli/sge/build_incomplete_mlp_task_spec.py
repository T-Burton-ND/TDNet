#!/usr/bin/env python3
"""Build an SGE chunk spec for MLP chunks with missing or stale trials."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


def main() -> int:
    manifest = pd.read_parquet(Path(sys.argv[1]))
    incomplete = []
    for chunk_id, group in manifest.groupby("chunk_id", sort=True):
        needs_retry = False
        for row in group.to_dict("records"):
            run = Path(row["output_path"])
            result = run / "result.parquet"
            status = run / "status.json"
            if not result.exists():
                needs_retry = True
                break
            try:
                frame = pd.read_parquet(result)
                if len(frame) != 1 or str(frame.iloc[0].get("status")) not in {"success", "failed"}:
                    needs_retry = True
                    break
            except Exception:
                needs_retry = True
                break
            if not status.exists():
                needs_retry = True
                break
        if needs_retry:
            incomplete.append(int(chunk_id) + 1)
    if incomplete:
        print(",".join(map(str, incomplete)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
