#!/usr/bin/env python3
"""Reconstruct score-free Week 6–7 F09–F16 inputs after the games.

These are retrospective inputs from the October 21 source cache. They cannot
establish what provider fields were available at the original kickoffs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/publication")]
from build_2026_f09_whatif import f09_state, read_schedule, source_statistics  # noqa: E402
from build_2026_f10_whatif_features import build as build_f10  # noqa: E402
from build_2026_f11_whatif import f11_state  # noqa: E402
from build_2026_f12_whatif import (  # noqa: E402
    _bind_current_ppa, _ppa_state, _special_state, observations,
)
from build_2026_f13_f16_whatif import current_research_stats  # noqa: E402

OUT = ROOT / "data/what_if_2026_fingerprints/week06_07_retrospective_states"
FROZEN = ROOT / "publication/2026/week_06/pre_game/frozen_bundle/public/predictions.parquet"
WEEK7 = ROOT / "data/publication/2026/catchup_2026_10_21/week_07_replay_results.csv"
GAME_SOURCE = ROOT / "data/raw/cfbd/v2/games/2026.parquet"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_games() -> pd.DataFrame:
    week6 = pd.read_parquet(FROZEN, columns=["game_id"]).drop_duplicates("game_id")
    week7 = pd.read_csv(WEEK7, usecols=["game_id"])
    target_ids = set(week6.game_id) | set(week7.game_id)
    if len(week6) != 58 or len(week7) != 59 or len(target_ids) != 117:
        raise ValueError("Week 6–7 target IDs differ from published cohorts")
    columns = ["id", "season", "week", "start_date", "home_team", "away_team",
               "season_type", "completed", "home_classification", "away_classification"]
    games = pd.read_parquet(GAME_SOURCE, columns=columns)
    games = games.loc[games.id.isin(target_ids)].copy().sort_values("id")
    if len(games) != 117 or set(games.id) != target_ids:
        raise ValueError("Week 6–7 target schedule differs from published cohorts")
    if set(games.loc[games.week.eq(6), "id"]) != set(week6.game_id) or set(games.loc[games.week.eq(7), "id"]) != set(week7.game_id):
        raise ValueError("Week 6–7 targets have changed week")
    if not (games.home_classification.astype(str).str.lower().eq("fbs")
            & games.away_classification.astype(str).str.lower().eq("fbs")).all():
        raise ValueError("Week 6–7 target is not FBS against FBS")
    return games


def save_state(name: str, state: pd.DataFrame, targets: pd.DataFrame) -> dict:
    keys = ["target_game_id", "team"]
    if state.duplicated(keys).any() or len(state) != 234 or set(state.target_game_id) != set(targets.id):
        raise ValueError(f"Incomplete {name} Week 6–7 state")
    expected = {(int(game.id), team) for game in targets.itertuples()
                for team in (game.home_team, game.away_team)}
    if set(zip(state.target_game_id.astype(int), state.team)) != expected:
        raise ValueError(f"Incorrect {name} target teams")
    starts = pd.to_datetime(state.target_game_id.map(targets.set_index("id").start_date), utc=True)
    for column in ("latest_source_available_utc", "latest_source_available_utc_st",
                   "latest_source_available_utc_ppa", "staff_available_utc",
                   "feature_available_utc"):
        if column in state:
            available = pd.to_datetime(state[column], utc=True, errors="coerce")
            if (available.notna() & ~(available < starts)).any():
                raise ValueError(f"{name} has a source at or after target kickoff: {column}")
    for column in ("latest_source_game_id", "latest_source_game_id_st",
                   "latest_source_game_id_ppa"):
        if column in state and state[column].eq(state.target_game_id).any():
            raise ValueError(f"{name} includes its own target: {column}")
    path = OUT / f"{name}_target_state.parquet"
    state.to_parquet(path, index=False, compression="zstd")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "team_rows": len(state)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = target_games()
    schedule = read_schedule()
    target_path = OUT / "target_schedule_without_scores.parquet"
    targets.to_parquet(target_path, index=False, compression="zstd")
    outputs = {}
    print("Building F09 source statistics and retrospective Week 6–7 state", flush=True)
    stats = source_statistics(schedule)
    f09 = f09_state(stats, schedule, targets)
    outputs["F09"] = save_state("f09", f09, targets)
    print("Building F10 player and recruitment state", flush=True)
    f10, f10_receipt = build_f10(targets)
    outputs["F10"] = save_state("f10", f10, targets)
    print("Building F11 prior-staff state", flush=True)
    f11 = f11_state(targets)
    outputs["F11"] = save_state("f11", f11, targets)
    print("Building F12 special-teams and player PPA state", flush=True)
    source_schedule = schedule.loc[schedule.season_type.astype(str).str.lower().eq("regular")
                                   & schedule.completed.fillna(False).astype(bool)].copy()
    source_schedule["kickoff"] = pd.to_datetime(source_schedule.start_date, utc=True)
    target_kickoffs = targets.copy()
    target_kickoffs["kickoff"] = pd.to_datetime(target_kickoffs.start_date, utc=True)
    obs = observations(source_schedule)
    special = _special_state(obs, source_schedule, target_kickoffs)
    ppa, ppa_audit = _bind_current_ppa(source_schedule)
    player = _ppa_state(ppa, target_kickoffs)
    f12 = special.merge(player, on=["target_game_id", "team", "target_start_utc"],
                        validate="one_to_one", suffixes=("_st", "_ppa"))
    outputs["F12"] = save_state("f12", f12, targets)
    print("Building F13–F16 play-context research state", flush=True)
    research_out = OUT / "research"
    research = current_research_stats(schedule, targets, research_out)
    outputs["research"] = save_state("research", research, targets)
    manifest = {
        "season": 2026, "target_weeks": [6, 7], "target_games": 117, "target_team_rows": 234,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_schedule_without_scores": {"path": str(target_path.relative_to(ROOT)),
                                           "sha256": digest(target_path)},
        "week06_frozen_forecast_sha256": digest(FROZEN), "week07_replay_results_sha256": digest(WEEK7),
        "raw_games_sha256": digest(GAME_SOURCE), "source_reporting_lag_hours": 48,
        "target_schedule_projection_excludes_outcome_columns": True,
        "target_outcomes_used_as_features": False,
        "full_raw_schedule_read_by_source_builders": True,
        "provider_field_level_availability_verified": False,
        "provenance_class": "retrospective_reconstruction_after_target_games",
        "f10_receipt": f10_receipt, "ppa_audit": ppa_audit, "states": outputs,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({key: value["team_rows"] for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
