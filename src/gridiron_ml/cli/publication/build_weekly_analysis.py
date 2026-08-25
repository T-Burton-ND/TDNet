#!/usr/bin/env python3
"""Generate descriptive analysis beside immutable weekly release packages."""

from argparse import ArgumentParser
from pathlib import Path
import json

from gridiron_ml.cli._paths import project_root
from gridiron_ml.publication.output_layout import require_week_directory
from gridiron_ml.publication.weekly_analysis import build_weekly_analysis


def main() -> int:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--tdnet-poll", type=Path, required=True)
    parser.add_argument("--ap-poll", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument(
        "--feature-metadata", type=Path,
        default=root / "docs/publication_2026/FINGERPRINT_FEATURE_MATRIX.csv",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = require_week_directory(args.output_root, "analysis")
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty weekly analysis: {output}")
    result = build_weekly_analysis(
        games_path=args.games,
        tdnet_poll_path=args.tdnet_poll,
        ap_poll_path=args.ap_poll,
        fingerprint_path=args.fingerprint,
        feature_metadata_path=args.feature_metadata,
        output_root=output,
        season=args.season,
        week=args.week,
        project_root=root,
    )
    print(json.dumps({"output_root": str(output), "matchup_signals": len(result["matchup"]), "disparity_signals": len(result["disparity"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
