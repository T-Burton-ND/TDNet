#!/usr/bin/env python3
"""Prepare separate corrected-reference and F13–F17 A research inputs.

Run on the saved 2010–2025 archive. No API calls or saved F09–F12 writes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.experiments.nextgen_f09 import trailing_game_state, f09_formulas
from gridiron_ml.experiments.nextgen_microstructure import (
    play_flags, game_sufficient_statistics, drive_sufficient_statistics,
)
from gridiron_ml.experiments.nextgen_rounds_features import (
    F13_METRICS, F14_METRICS, F15_METRICS, actor_game_statistics,
    context_plays, context_totals, past_only_context_baselines,
    play_actor_lookup, residual_game_statistics, sequence_game_statistics,
    trailing_research_state,
)
from gridiron_ml.experiments.nextgen_rounds_market import load_market_archive
from gridiron_ml.pipeline.fetch.nextgen_acquisition import load_authoritative_schedule, sha256_file


def _records(snapshot: list[dict], endpoint: str) -> list[dict]:
    return sorted([r for r in snapshot if r.get("endpoint") == endpoint
                   and r.get("status") in {"success_complete", "skipped_existing_complete"}
                   and r.get("year") is not None and 2010 <= int(r["year"]) <= 2025],
                  key=lambda r: (int(r["year"]), int(r.get("week") or 0), str(r.get("request_id"))))


def _read_checked(record: dict, *, columns=None) -> pd.DataFrame:
    path = Path(record["cache_path"])
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"Archive source hash mismatch: {record['request_id']}")
    return pd.read_parquet(path, columns=columns)


def _save(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def _group_game_stats(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["game_id", "team"])
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate game/team sufficient statistics")
    return result


def prepare(archive: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = archive / "results/preflight/archive_review_20260930/ledger_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    inventory = json.loads((ROOT / "configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json").read_text())
    schedule, schedule_hash = load_authoritative_schedule(archive, inventory)
    target_parent = pd.read_parquet(archive / "fingerprints/F12_F_a/values.parquet",
                                    columns=["target_game_id", "team", "season", "target_start_utc"])
    target_parent = target_parent.loc[target_parent.season.between(2013, 2025)].copy()
    eligible = set(schedule.id.astype(int))
    plays = _records(snapshot, "/plays")
    drives = {int(r["year"]): r for r in _records(snapshot, "/drives")}
    if len(plays) != 245 or len(drives) != 16:
        raise ValueError("Expected complete verified 245-play/16-drive source set")
    if len({(int(r["year"]), int(r["week"])) for r in plays}) != len(plays):
        raise ValueError("Overlapping play source partitions")
    print(f"Pass 1: {len(plays)} verified play partitions", flush=True)
    context_parts, f09_parts, lookup_parts = [], [], []
    current_year, yearly_drives = None, None
    source_counts = Counter()
    for index, record in enumerate(plays, 1):
        year = int(record["year"])
        if year != current_year:
            if lookup_parts:
                _save(pd.concat(lookup_parts, ignore_index=True),
                      output / "play_lookup" / f"{current_year}.parquet")
                lookup_parts.clear()
            current_year = year
            yearly_drives = _read_checked(drives[year])
            yearly_drives = yearly_drives.loc[yearly_drives.game_id.isin(eligible)]
        raw = _read_checked(record)
        raw = raw.loc[raw.game_id.isin(eligible)].copy()
        flagged = play_flags(raw)
        context_parts.append(context_totals(context_plays(flagged), year))
        lookup_parts.append(play_actor_lookup(flagged))
        f09 = game_sufficient_statistics(raw)
        drive = drive_sufficient_statistics(
            yearly_drives.loc[yearly_drives.game_id.isin(raw.game_id)], raw)
        f09_parts.append(f09.merge(drive, on=["game_id", "team"], how="left", validate="one_to_one"))
        source_counts[year] += len(raw)
        if index % 30 == 0:
            print(f"  pass 1 {index}/{len(plays)}", flush=True)
    if lookup_parts:
        _save(pd.concat(lookup_parts, ignore_index=True),
              output / "play_lookup" / f"{current_year}.parquet")
    context = pd.concat(context_parts, ignore_index=True)
    context = context.groupby(["season", "context_id"], as_index=False)[
        ["n", "sum_yards", "sum_success"]].sum()
    _save(context, output / "context_training_totals.parquet")
    baselines = past_only_context_baselines(context)
    f09_stats = _group_game_stats(f09_parts)
    _save(f09_stats, output / "corrected_f09_game_statistics.parquet")

    print("Pass 2: context residuals and possession transitions", flush=True)
    residual_parts, sequence_parts = [], []
    sequence_audit = Counter()
    for index, record in enumerate(plays, 1):
        raw = pd.read_parquet(record["cache_path"])
        raw = raw.loc[raw.game_id.isin(eligible)]
        flagged = play_flags(raw)
        residual_parts.append(residual_game_statistics(flagged, baselines[int(record["year"])]))
        sequence, report = sequence_game_statistics(flagged)
        sequence_parts.append(sequence)
        sequence_audit.update(report)
        if index % 30 == 0:
            print(f"  pass 2 {index}/{len(plays)}", flush=True)
    residuals = _group_game_stats([p for p in residual_parts if not p.empty])
    sequences = _group_game_stats([p for p in sequence_parts if not p.empty])

    print("Pass 3: verified game-level actor roles", flush=True)
    actor_records = [r for r in _records(snapshot, "/plays/stats")
                     if r.get("game_id") is not None and 2013 <= int(r["year"]) <= 2025]
    if len({int(r["game_id"]) for r in actor_records}) != len(actor_records):
        raise ValueError("Duplicate direct/derived complete actor game source")
    by_year = defaultdict(list)
    for record in actor_records:
        by_year[int(record["year"])].append(record)
    actor_parts = []
    actor_audit = Counter()
    for year in sorted(by_year):
        lookup = pd.read_parquet(output / "play_lookup" / f"{year}.parquet")
        lookup_by_game = {int(g): frame for g, frame in lookup.groupby("game_id")}
        for index, record in enumerate(by_year[year], 1):
            game = int(record["game_id"])
            if game not in lookup_by_game:
                actor_audit["source_games_without_plays"] += 1
                continue
            actor = _read_checked(record)
            stats, report = actor_game_statistics(actor, lookup_by_game[game])
            if not stats.empty:
                actor_parts.append(stats)
            actor_audit.update(report)
            if index % 300 == 0:
                print(f"  actors {year}: {index}/{len(by_year[year])}", flush=True)
        del lookup, lookup_by_game
    actors = _group_game_stats(actor_parts)

    stats = f09_stats[["game_id", "team"]].merge(residuals, on=["game_id", "team"],
                                                   how="left", validate="one_to_one")
    stats = stats.merge(sequences, on=["game_id", "team"], how="left", validate="one_to_one")
    stats = stats.merge(actors, on=["game_id", "team"], how="left", validate="one_to_one")
    for side in ("offense", "defense"):
        for metric in (*F13_METRICS, *F14_METRICS):
            for suffix in ("__sum", "__n"):
                col = f"{side}_{metric}{suffix}"
                if col not in stats:
                    stats[col] = np.nan
    for metric in F15_METRICS:
        for suffix in ("__sum", "__n"):
            col = metric + suffix
            if col not in stats:
                stats[col] = np.nan
    _save(stats, output / "research_game_statistics.parquet")
    state = trailing_research_state(stats, schedule, target_parent)
    _save(state, output / "f13_f16_target_state.parquet")

    corrected, corrected_coverage = trailing_game_state(f09_stats, schedule)
    formulas = f09_formulas("a")
    f09_features = corrected[["target_game_id", "team"]].copy()
    for formula in formulas:
        f09_features[formula.name] = formula.evaluate(corrected)
    selected = target_parent[["target_game_id", "team"]].merge(
        f09_features, on=["target_game_id", "team"], how="left", validate="one_to_one")
    if selected[[f.name for f in formulas]].isna().all(axis=1).any():
        raise ValueError("Corrected F09 state missing for an F12 target team")
    _save(selected, output / "corrected_f09_a_target_features.parquet")

    market, market_report = load_market_archive(snapshot_path, set(target_parent.target_game_id.astype(int)))
    _save(market, output / "f17_market_game_features.parquet")
    report = {"status": "prepared_untrained", "scope": "F13_F16_A_and_F17_market_research",
              "source_archive": str(archive), "schedule_sha256": schedule_hash,
              "ledger_snapshot_sha256": sha256_file(snapshot_path),
              "f12_parent_sha256": sha256_file(archive / "fingerprints/F12_F_a/values.parquet"),
              "plays_by_season": dict(source_counts), "context_bins": 72,
              "context_baseline": "earlier_seasons_only_shrink_50_to_play_class",
              "sequence_audit": dict(sequence_audit), "actor_audit": dict(actor_audit),
              "f09_corrected_coverage": corrected_coverage,
              "target_team_rows": len(target_parent), "research_state_rows": len(state),
              "market": market_report,
              "outputs": {name: sha256_file(output / name) for name in (
                  "context_training_totals.parquet", "corrected_f09_game_statistics.parquet",
                  "research_game_statistics.parquet", "f13_f16_target_state.parquet",
                  "corrected_f09_a_target_features.parquet", "f17_market_game_features.parquet")}}
    temp = output / "prepare_receipt.tmp.json"
    temp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    temp.replace(output / "prepare_receipt.json")
    print(json.dumps({"status": report["status"], "target_team_rows": len(target_parent),
                      "research_state_rows": len(state), "market_games": len(market)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path,
                        default=Path("/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/nextgen_rounds_2026")
    args = parser.parse_args()
    prepare(args.archive, args.output)


if __name__ == "__main__":
    main()
