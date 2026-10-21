#!/usr/bin/env python3
"""Freeze a current-week model input state from the preceding completed week."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def build(source: Path, games_path: Path, *, season: int, week: int,
          cutoff_utc: str, output: Path, proof: Path, lag_hours: int = 48) -> dict:
    cutoff = pd.Timestamp(cutoff_utc)
    if cutoff.tzinfo is None:
        raise ValueError("Cutoff must include a UTC timezone")
    cutoff = cutoff.tz_convert("UTC")
    frame = pd.read_parquet(source)
    games = pd.read_parquet(games_path)
    prior = frame.loc[
        pd.to_numeric(frame["keys_season"], errors="coerce").eq(season)
        & pd.to_numeric(frame["keys_week"], errors="coerce").eq(week - 1)
    ].copy()
    if prior.empty or prior["keys_team"].duplicated().any():
        raise ValueError("Preceding week must contain one state per team")
    target = games.loc[
        pd.to_numeric(games["season"], errors="coerce").eq(season)
        & pd.to_numeric(games["week"], errors="coerce").eq(week)
        & games["season_type"].astype(str).str.casefold().eq("regular")
        & (games["home_classification"].astype(str).str.casefold().eq("fbs")
           | games["away_classification"].astype(str).str.casefold().eq("fbs"))
    ].copy()
    kickoff = pd.to_datetime(target["start_date"], utc=True, errors="coerce")
    future = target.loc[kickoff.gt(cutoff)].copy()
    if future.empty or kickoff.isna().any():
        raise ValueError("Target slate has no future games or has unknown kickoffs")
    source_ids = pd.to_numeric(prior["keys_game_id"], errors="coerce").dropna().astype(int)
    if set(source_ids) & set(target["id"].astype(int)):
        raise ValueError("A target-week game appears in the preceding-week source state")
    lookup = games.drop_duplicates("id").set_index("id")
    unknown = set(source_ids) - set(lookup.index.astype(int))
    if unknown:
        raise ValueError(f"Unknown source game IDs: {sorted(unknown)[:5]}")
    source_games = lookup.loc[sorted(set(source_ids))]
    source_kickoff = pd.to_datetime(source_games["start_date"], utc=True, errors="coerce")
    completed = source_games["completed"].fillna(False).astype(bool)
    scored = source_games[["home_points", "away_points"]].notna().all(axis=1)
    if source_kickoff.isna().any() or not completed.all() or not scored.all():
        raise ValueError("Source state contains an unfinished or undated game")
    latest_available = source_kickoff.max() + pd.Timedelta(hours=lag_hours)
    if latest_available >= cutoff:
        raise ValueError("The source reporting lag overlaps the prediction cutoff")
    if output.exists() or proof.exists():
        raise FileExistsError("Refusing to overwrite a frozen cutoff state or proof")
    synthetic = prior.copy()
    synthetic["keys_week"] = week
    if "next_week" in synthetic:
        synthetic["next_week"] = week
    keep = ~(
        pd.to_numeric(frame["keys_season"], errors="coerce").eq(season)
        & pd.to_numeric(frame["keys_week"], errors="coerce").ge(week)
    )
    frozen = pd.concat([frame.loc[keep], synthetic], ignore_index=True, sort=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    frozen.to_parquet(output, index=False)
    report = {
        "status": "pass",
        "scope": "current-week state copied from the preceding completed week",
        "season": season,
        "target_week": week,
        "source_week": week - 1,
        "cutoff_utc": cutoff.isoformat(),
        "reporting_lag_hours": lag_hours,
        "latest_source_kickoff_utc": source_kickoff.max().isoformat(),
        "latest_source_available_utc": latest_available.isoformat(),
        "earliest_target_kickoff_utc": pd.to_datetime(future["start_date"], utc=True).min().isoformat(),
        "source_team_rows": len(prior),
        "unique_source_games": len(set(source_ids)),
        "future_target_games": len(future),
        "already_started_target_games": len(target) - len(future),
        "source_fingerprint_path": str(source),
        "source_fingerprint_sha256": digest(source),
        "games_path": str(games_path),
        "games_sha256": digest(games_path),
        "cutoff_fingerprint_path": str(output),
        "cutoff_fingerprint_sha256": digest(output),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limits": [
            "This proves source game kickoffs plus the declared lag precede the cutoff; it does not prove provider field-level publication times.",
            "The source cache was refreshed at the current run date, so later provider corrections to earlier games cannot be ruled out.",
        ],
    }
    proof.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--cutoff-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.games, season=args.season,
                           week=args.week, cutoff_utc=args.cutoff_utc,
                           output=args.output, proof=args.proof), indent=2))


if __name__ == "__main__":
    main()
