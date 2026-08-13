"""src.gridiron_ml.pipeline.build_team_game_table.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Transform cached raw data into canonical tables and downstream artifacts.
"""

# File: src/pipeline/build_team_game_tables.py

from __future__ import annotations

from pathlib import Path
import argparse

from .raw_weekly_builder import build_team_game_table_from_parquets
from .canonicalization import canonicalize_team_game_table_columns


def build_season(
    *,
    cache_dir: Path,
    out_dir: Path,
    year: int,
    division: str,
    week: int | None = None,
    output_format: str = "parquet",
) -> None:
    """
    Build and save the team-game table for a single season.
    """
    print("=" * 60)
    scope = f"year={year} | division={division}" if week is None else f"year={year} | week={week} | division={division}"
    print(f"Building team-game table | {scope}")
    print("=" * 60)

    df = build_team_game_table_from_parquets(
        cache_dir=cache_dir,
        current_year=year,
        division=division,
        week=week,
    )

    df_canonical = canonicalize_team_game_table_columns(df)

    suffix = "parquet" if output_format == "parquet" else "csv"
    if week is None:
        out_path = out_dir / f"team_game_table_{year}_{division}.{suffix}"
    else:
        out_path = out_dir / f"team_game_table_{year}_week_{int(week):02d}_{division}.{suffix}"

    if output_format == "parquet":
        df_canonical.to_parquet(out_path, index=False)
    else:
        df_canonical.to_csv(out_path, index=False)


    print(f"Wrote {len(df_canonical):,} rows × {df_canonical.shape[1]} cols")
    print(f"Saved to: {out_path}")
    print()


def main():
    """Run the main step and return its normalized result."""
    ap = argparse.ArgumentParser(
        description="Build team-game tables for multiple seasons."
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Root of CFBD parquet cache (e.g., data/raw/cfbd/v2)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/Users/tburton2/Desktop/Repos/TDNet/data/team_game_tables"
        ),
        help="Directory to write team-game tables (one per season)",
    )
    ap.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="First season to build (default: 2014)",
    )
    ap.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last season to build, inclusive (default: 2024)",
    )
    ap.add_argument(
        "--division",
        default="fbs",
        help="Division tag (used for filtering / filenames)",
    )
    ap.add_argument(
        "--week",
        type=int,
        default=None,
        help="Optional week to build from the cached season parquet set",
    )
    ap.add_argument(
        "--output-format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Artifact format for the built table",
    )

    args = ap.parse_args()

    cache_dir: Path = args.cache_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Team-game table batch build")
    print(f"Cache dir : {cache_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Years     : {args.start_year} → {args.end_year}")
    print(f"Division  : {args.division}")
    print(f"Week      : {args.week if args.week is not None else 'full season'}")
    print(f"Format    : {args.output_format}")
    print("=" * 60)
    print()

    for year in range(args.start_year, args.end_year + 1):
        try:
            build_season(
                cache_dir=cache_dir,
                out_dir=out_dir,
                year=year,
                division=args.division,
                week=args.week,
                output_format=args.output_format,
            )
        except Exception as e:
            print(f"❌ Failed to build season {year}")
            raise e

    print("=" * 60)
    print("Finished building all team-game tables.")
    print("=" * 60)


if __name__ == "__main__":
    main()
