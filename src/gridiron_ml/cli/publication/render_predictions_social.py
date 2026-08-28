"""Re-render TDNet Top 3 + Sickos prediction assets without running models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import subprocess

import pandas as pd

from gridiron_ml.publication.social_predictions import render_predictions_social
from gridiron_ml.publication.weekly import merge_cfbd_market_lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--poll", type=Path, help="Optional TDNet Top 25 CSV or parquet.")
    parser.add_argument(
        "--market-lines", type=Path,
        help="Optional raw CFBD /lines CSV or parquet, merged by game id using the provider average.",
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logo-dir", type=Path, default=Path("data/meta/logos/by_team"))
    parser.add_argument("--generated-at-utc")
    args = parser.parse_args()

    games = _read(args.games)
    if args.market_lines:
        games = merge_cfbd_market_lines(
            games, _read(args.market_lines), season=args.season, week=args.week
        )
    poll = _read(args.poll) if args.poll else None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    generated = args.generated_at_utc or datetime.now(timezone.utc).isoformat()
    source_hash = sha256(args.games.read_bytes())
    if args.market_lines:
        source_hash.update(args.market_lines.read_bytes())
    source_sha256 = source_hash.hexdigest()
    for variant in ("4x5", "16x9"):
        output = args.output_dir / f"week_{args.week:02d}_tdnet_predictions_{variant}.png"
        render_predictions_social(
            games, output, season=args.season, week=args.week, logo_dir=args.logo_dir,
            tdnet_poll=poll, variant=variant, generated_at_utc=generated,
            git_commit=commit, source_sha256=source_sha256,
        )
        print(output)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


if __name__ == "__main__":
    main()
