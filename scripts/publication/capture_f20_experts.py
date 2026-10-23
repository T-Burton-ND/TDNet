#!/usr/bin/env python3
"""Capture a private, immutable, point-in-time F20 expert snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.f20.capture import SOURCE_URL, capture, receipt_filename


def download(retries: int = 3) -> tuple[bytes, datetime]:
    error = None
    for attempt in range(retries):
        try:
            request = Request(SOURCE_URL, headers={"User-Agent": "TDNet-F20-research/1.0 (point-in-time archival)"})
            with urlopen(request, timeout=20) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                content = response.read(2_000_001)
                timestamp = datetime.now(UTC)
                if len(content) > 2_000_000 or not content.strip():
                    raise ValueError("Tracker response is empty or unexpectedly large")
                return content, timestamp
        except (OSError, RuntimeError, ValueError) as exc:
            error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Tracker capture failed after {retries} attempts: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/private/f20/prospective")
    parser.add_argument("--schedule", type=Path, default=ROOT / "data/raw/cfbd/v2/games/2026.parquet")
    parser.add_argument("--import-html", type=Path, help="Import bytes captured earlier; uses file mtime as collection time")
    args = parser.parse_args()
    args.archive.mkdir(parents=True, exist_ok=True)
    try:
        if args.import_html:
            raw = args.import_html.read_bytes()
            collected_at = datetime.fromtimestamp(args.import_html.stat().st_mtime, UTC)
        else:
            raw, collected_at = download()
        schedule = pd.read_parquet(args.schedule)
        receipt = capture(raw, schedule, args.archive, collected_at=collected_at,
                          schedule_sha256=hashlib.sha256(args.schedule.read_bytes()).hexdigest(),
                          schedule_modified_at_utc=datetime.fromtimestamp(
                              args.schedule.stat().st_mtime, UTC).isoformat())
    except Exception as exc:
        # Capture failures are recorded independently from any forecast issuance.
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        failure = {"status": "failed", "source_url": SOURCE_URL,
                   "attempted_at_utc": datetime.now(UTC).isoformat(),
                   "error_type": type(exc).__name__, "error": str(exc)}
        path = args.archive / "failures"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{timestamp}.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    print(json.dumps({"status": "captured", "raw_sha256": receipt["raw_sha256"],
                      "collected_at_utc": receipt["collected_at_utc"],
                      "receipt_file": str(args.archive / "receipts" / receipt_filename(
                          receipt["raw_sha256"], receipt["schedule_sha256"],
                          receipt["schedule_modified_at_utc"])),
                      "matched_games": receipt["matched_games"],
                      "pregame_games": receipt["pregame_games"],
                      "unmatched": receipt["unmatched"],
                      "archive": str(args.archive)}, indent=2))


if __name__ == "__main__":
    main()
