"""Bind canonical post-week F6 states to fresh next-game schedule identities.

The publication source is never modified. Its next-game labels and market
columns are not trusted for outcomes or admitted to the predictor matrix.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pandas as pd

from .nextgen_contract import assert_design_years, assert_safe_inputs
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (
    atomic_json, load_authoritative_schedule, sha256_file,
)

ROOT = Path(__file__).resolve().parents[3]
CANONICAL_SOURCE = Path(
    "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
    "fingerprint_ladder_v3/canonical_fingerprint.parquet")


def align_baseline(source: pd.DataFrame, schedule: pd.DataFrame,
                   features: list[str], *, reporting_lag_hours: int = 48):
    """Retain exact source values while verifying pre-target weekly cutoffs.

    Graph features use the entire completed source week. Thus availability is
    conservatively the latest league-wide kickoff in that week plus 48 hours,
    not just the team's most recent kickoff. This is a reconstructed reporting
    lag assumption, not an archived provider publication timestamp.
    """
    assert_design_years(source.keys_season.unique())
    assert_design_years(schedule.season.unique())
    assert_safe_inputs(features)
    if len(features) != len(set(features)) or not set(features) <= set(source):
        raise ValueError("Baseline source feature list is missing or duplicated")
    if reporting_lag_hours < 24:
        raise ValueError("Reporting lag must allow completed game statistics")
    games = schedule.copy()
    if not games.id.is_unique:
        raise ValueError("Duplicate authoritative game ID")
    games["kickoff"] = pd.to_datetime(games.start_date, utc=True)
    games = games.loc[games.season_type.eq("regular") & games.completed.eq(True)]
    by_id = games.set_index("id")
    weekly = games.groupby(["season", "week"]).kickoff.max()
    first = {}
    for game in games.itertuples():
        for team in (game.home_team, game.away_team):
            key = (int(game.season), str(team))
            first[key] = min(first.get(key, game.kickoff), game.kickoff)
    rejected = Counter()
    rows = []
    # Week-0 and bye rows are deliberate state snapshots, not extra games.
    candidates = source.loc[source.next_game_id.notna()].copy()
    candidates = candidates.sort_values(["keys_season", "keys_team", "keys_week"])
    for _, row in candidates.iterrows():
        target_id = int(row.next_game_id)
        team, season, week = str(row.keys_team), int(row.keys_season), int(row.keys_week)
        if target_id not in by_id.index:
            rejected["target_absent_from_regular_completed_schedule"] += 1
            continue
        target = by_id.loc[target_id]
        if str(target.home_classification).lower() != "fbs" or str(target.away_classification).lower() != "fbs":
            rejected["target_not_fbs_vs_fbs"] += 1
            continue
        if team not in (target.home_team, target.away_team) or season != int(target.season):
            raise ValueError("Canonical target identity disagrees with fresh schedule")
        source_id, source_time = None, None
        if week == 0:
            # Annual preseason values are inherited unchanged from canonical F6.
            # They require a separate source-semantics audit before training.
            available = first[(season, team)] - pd.Timedelta(days=1)
            kind = "static_week0"
        else:
            if pd.isna(row.keys_game_id) or int(row.keys_game_id) not in by_id.index:
                rejected["source_absent_from_regular_completed_schedule"] += 1
                continue
            source_id = int(row.keys_game_id)
            prior = by_id.loc[source_id]
            if team not in (prior.home_team, prior.away_team) or int(prior.season) != season:
                raise ValueError("Canonical source identity disagrees with fresh schedule")
            source_time = prior.kickoff
            if int(prior.week) > week or week >= int(target.week):
                rejected["source_week_not_strictly_before_target"] += 1
                continue
            # Include earlier weeks too, guarding rescheduled out-of-order games.
            ends = weekly.loc[season]
            history_end = ends.loc[ends.index <= week].max()
            available = history_end + pd.Timedelta(hours=reporting_lag_hours)
            kind = "dynamic"
        if available >= target.kickoff or (source_time is not None and source_time >= available):
            rejected["reporting_cutoff_not_before_target"] += 1
            continue
        home = team == target.home_team
        pf = float(target.home_points if home else target.away_points)
        pa = float(target.away_points if home else target.home_points)
        record = {name: row[name] for name in features}
        record.update(season=season, week=int(target.week), team=team,
                      target_game_id=target_id, target_start_utc=target.kickoff,
                      feature_available_utc=available, feature_kind=kind,
                      latest_source_game_id=source_id, latest_source_game_utc=source_time,
                      source_state_week=week, is_home=home,
                      next_game_margin=pf-pa, next_game_win=float(pf > pa),
                      next_game_points_for=pf, next_game_points_against=pa)
        rows.append(record)
    if not rows:
        raise ValueError("No canonical next-game states passed alignment")
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("source_state_week").drop_duplicates(["target_game_id", "team"], keep="last")
    counts = frame.groupby("target_game_id").team.transform("size")
    rejected["unpaired_target_team_row"] = int(counts.ne(2).sum())
    frame = frame.loc[counts.eq(2)].sort_values(["season", "target_game_id", "team"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No paired FBS targets passed alignment")
    report = {"source_rows": len(source), "target_team_rows": len(frame),
              "target_games": frame.target_game_id.nunique(), "rejected_rows": dict(rejected),
              "reporting_lag_hours": reporting_lag_hours,
              "availability_evidence": "reconstructed_week_end_plus_lag_not_archived_publication_timestamp",
              "source_semantics_audit_required_before_training": True,
              "games_by_season": frame.groupby("season").target_game_id.nunique().to_dict()}
    return frame, report


def prepare_baseline(root: Path):
    config = json.loads((ROOT / "configs/experiments/nextgen_fingerprints_v1.json").read_text())
    manifest_path = ROOT / config["source_f6_manifest"]
    manifest = json.loads(manifest_path.read_text())
    features = manifest["feature_names"]
    assert len(features) == config["source_f6_feature_count"] == 227
    assert manifest["schema_hash"] == config["source_f6_schema_hash"]
    from .nextgen_source_policy import baseline_features
    features = baseline_features(features, config)
    inventory = json.loads((ROOT / "configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json").read_text())
    schedule, schedule_hash = load_authoritative_schedule(root, inventory)
    metadata = ["keys_season", "keys_team", "keys_week", "keys_game_id", "next_game_id"]
    source = pd.read_parquet(CANONICAL_SOURCE, columns=metadata + features,
                             filters=[("keys_season", ">=", 2010), ("keys_season", "<=", 2025)])
    frame, report = align_baseline(source, schedule, features)
    output = root / "canonical/f06_aligned.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(output)
    report.update(source_path=str(CANONICAL_SOURCE), source_sha256=sha256_file(CANONICAL_SOURCE),
                  data_sha256=sha256_file(output), schedule_sha256=schedule_hash,
                  manifest_sha256=sha256_file(manifest_path), feature_names=features,
                  source_schema_hash=manifest["schema_hash"], max_design_year=int(frame.season.max()))
    report['baseline_source_policy'] = config['baseline_source_policy']
    atomic_json(output.with_suffix(".provenance.json"), report)
    return report


if __name__ == "__main__":
    config = json.loads((ROOT / "configs/experiments/nextgen_fingerprints_v1.json").read_text())
    report = prepare_baseline(Path(config["artifact_root"]))
    print(json.dumps({k: v for k, v in report.items() if k != "feature_names"}, indent=2))
