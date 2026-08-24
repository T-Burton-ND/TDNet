"""Re-render TDNet Top 10 social assets without running weekly predictions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import subprocess

import pandas as pd

from gridiron_ml.publication.social_top10 import render_top10_social


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logo-dir", type=Path, default=Path("data/meta/logos/by_team"))
    parser.add_argument("--reference-poll", type=Path,
                        help="Optional AP/reference CSV with rank and team columns.")
    parser.add_argument("--generated-at-utc",
                        help="Stable ISO timestamp for deterministic review renders.")
    args = parser.parse_args()

    poll = pd.read_parquet(args.poll) if args.poll.suffix == ".parquet" else pd.read_csv(args.poll)
    source_sha256 = sha256(args.poll.read_bytes()).hexdigest()
    if args.reference_poll:
        reference = pd.read_csv(args.reference_poll)
        team_column = next(c for c in ("team", "keys_team", "school") if c in reference)
        rank_column = next(c for c in ("rank", "poll_rank", "ap_rank") if c in reference)
        reference = reference.loc[:, [team_column, rank_column]].rename(
            columns={team_column: "keys_team", rank_column: "reference_rank"}
        )
        poll_team = next(c for c in ("team", "keys_team", "school") if c in poll)
        poll = poll.merge(reference, left_on=poll_team, right_on="keys_team", how="left",
                          suffixes=("", "_reference"))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    generated = args.generated_at_utc or datetime.now(timezone.utc).isoformat()
    for variant in ("4x5", "16x9"):
        output = args.output_dir / f"week_{args.week:02d}_tdnet_top10_social_{variant}.png"
        render_top10_social(
            poll, output, season=args.season, week=args.week, logo_dir=args.logo_dir,
            variant=variant, generated_at_utc=generated, git_commit=commit,
            source_sha256=source_sha256,
        )
        print(output)


if __name__ == "__main__":
    main()
