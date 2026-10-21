#!/usr/bin/env python3
"""Build score-free Week 8 F09–F16 team states for the 53 frozen target games."""
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

OUT = ROOT / "data/what_if_2026_fingerprints/week08_expanded_states"
FROZEN = ROOT / "publication/2026/week_08/pre_game/frozen_bundle/public/predictions.parquet"
GAME_SOURCE = ROOT / "data/raw/cfbd/v2/games/2026.parquet"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_games() -> pd.DataFrame:
    frozen = pd.read_parquet(FROZEN, columns=["game_id", "game_start_time_utc"])
    frozen = frozen.drop_duplicates("game_id")
    columns = ["id", "season", "week", "start_date", "home_team", "away_team",
               "season_type", "completed", "home_classification", "away_classification"]
    games = pd.read_parquet(GAME_SOURCE, columns=columns)
    games = games.loc[games.id.isin(frozen.game_id)].copy().sort_values("id")
    if len(games) != 53 or set(games.id) != set(frozen.game_id):
        raise ValueError("Week 8 target schedule differs from frozen margin cohort")
    if not games.week.eq(8).all() or games.completed.fillna(False).astype(bool).any():
        raise ValueError("Week 8 target games are not all unplayed")
    if not (games.home_classification.astype(str).str.lower().eq("fbs")
            & games.away_classification.astype(str).str.lower().eq("fbs")).all():
        raise ValueError("Week 8 target is not FBS against FBS")
    if not (pd.to_datetime(games.start_date, utc=True)
            == pd.to_datetime(games.id.map(frozen.set_index("game_id").game_start_time_utc), utc=True)).all():
        raise ValueError("Frozen and current target kickoff differ")
    return games


def save_state(name: str, state: pd.DataFrame, targets: pd.DataFrame) -> dict:
    keys = ["target_game_id", "team"]
    if state.duplicated(keys).any() or len(state) != 106 or set(state.target_game_id) != set(targets.id):
        raise ValueError(f"Incomplete {name} Week 8 state")
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
    print("Building F09 source statistics and Week 8 state", flush=True)
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
        "season": 2026, "target_week": 8, "target_games": 53, "target_team_rows": 106,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_schedule_without_scores": {"path": str(target_path.relative_to(ROOT)),
                                           "sha256": digest(target_path)},
        "frozen_forecast_sha256": digest(FROZEN), "raw_games_sha256": digest(GAME_SOURCE),
        "source_weeks_acquired": [6, 7], "source_reporting_lag_hours": 48,
        "no_target_outcome_columns_read": True,
        "provider_field_level_availability_verified": False,
        "f10_receipt": f10_receipt, "ppa_audit": ppa_audit, "states": outputs,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({key: value["team_rows"] for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
