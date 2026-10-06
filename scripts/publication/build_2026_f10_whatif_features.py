#!/usr/bin/env python3
"""Build cutoff-safe 2026 F10-A player/recruit features for the what-if."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/publication")]
from build_2026_f09_whatif import read_schedule  # noqa: E402
from gridiron_ml.experiments.nextgen_players import (  # noqa: E402
    USAGE_METRICS, flatten_player_boxes, stable_id, usage_concentration,
)
from gridiron_ml.experiments.nextgen_recruiting import RECRUIT_UNITS  # noqa: E402
from gridiron_ml.experiments.nextgen_units import POSITION_UNIT  # noqa: E402

DATA = ROOT / "data/what_if_2026_fingerprints"
ARCHIVE = Path("/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/canonical")
OUT = DATA / "f10_features"
TARGETS = ROOT / "data/raw/cfbd/v2/games/2026.parquet"


def player_history(schedule: pd.DataFrame) -> pd.DataFrame:
    files = sorted((ARCHIVE / "player_observations").glob("*.parquet"))
    if len(files) < 200:
        raise ValueError(f"Incomplete historical player observation archive: {len(files)} partitions")
    old = pd.concat([pd.read_parquet(p, columns=["game_id", "team", "athlete_id", "metric", "value"])
                     for p in files], ignore_index=True)
    old = old.loc[old.metric.isin(USAGE_METRICS)]
    current = []
    for path in sorted((DATA / "raw_cache/games_players").glob("*.parquet")):
        raw = pd.read_parquet(path)
        obs, _ = flatten_player_boxes(raw)
        if not obs.empty:
            current.append(obs.loc[obs.metric.isin(USAGE_METRICS)])
    if not current:
        raise ValueError("No current-season player box observations were acquired")
    source = pd.concat([old, *current], ignore_index=True)
    source["athlete_id"] = source.athlete_id.map(stable_id)
    sched = schedule.loc[schedule.season_type.astype(str).str.lower().eq("regular")
                         & schedule.completed.fillna(False).astype(bool),
                         ["id", "season", "kickoff", "home_team", "away_team",
                          "home_classification", "away_classification"]]
    source = source.merge(sched, left_on="game_id", right_on="id", how="left", validate="many_to_one")
    if source.season.isna().any():
        raise ValueError("Player source includes games without completed regular-season authority")
    aliases = {"Savannah State": "Savannah St", "Saint Francis": "St. Francis (PA)"}
    source["team"] = source.team.replace(aliases)
    valid_participant = source.team.eq(source.home_team) | source.team.eq(source.away_team)
    fbs_names = set(pd.read_parquet(ROOT / "data/raw/cfbd/v2/teams_fbs/2026.parquet")
                    .loc[lambda d: d.classification.astype(str).str.lower().eq("fbs"), "school"].astype(str))
    invalid_fbs = ~valid_participant & source.team.isin(fbs_names)
    if (~valid_participant & invalid_fbs).any():
        raise ValueError("FBS player observation team does not match the scheduled game")
    source = source.loc[valid_participant].copy()
    source["available"] = source.kickoff + pd.Timedelta(hours=48)
    return source[["game_id", "team", "athlete_id", "metric", "value", "season", "kickoff", "available"]]


def recruiting_source() -> pd.DataFrame:
    parts = []
    for year in range(2022, 2025):
        path = ROOT / f"data/raw/cfbd/v2/recruiting_players/{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_parquet(path))
    current = pd.read_parquet(DATA / "raw_cache/recruiting_players/year_2025.parquet")
    current = current.rename(columns={"athleteId": "athlete_id", "recruitType": "recruit_type",
                                      "committedTo": "committed_to", "stateProvince": "state_province"})
    parts.append(current)
    recruits = pd.concat(parts, ignore_index=True, sort=False)
    recruits = recruits.loc[recruits.recruit_type.eq("HighSchool")].copy()
    recruits["year"] = pd.to_numeric(recruits.year, errors="coerce")
    if recruits.year.max() != 2025 or recruits.year.min() > 2022:
        raise ValueError("The 2022–2025 recruiting cohort is incomplete")
    return recruits


def _recruit_features(recruits: pd.DataFrame, team: str) -> dict:
    cohort = recruits.loc[recruits.year.between(2022, 2025) & recruits.committed_to.eq(team)].copy()
    cohort["unit"] = cohort.position.astype(str).str.upper().map(RECRUIT_UNITS)
    cohort["rating"] = pd.to_numeric(cohort.rating, errors="coerce")
    cohort["stars"] = pd.to_numeric(cohort.stars, errors="coerce")
    cohort.loc[~cohort.rating.between(0, 1), "rating"] = np.nan
    cohort.loc[~cohort.stars.between(0, 5), "stars"] = np.nan
    result = {}
    for unit in ("ol", "qb", "rb", "wrte", "front", "lb", "secondary", "special_teams"):
        selected = cohort.loc[cohort.unit.eq(unit)]
        ratings, stars = selected.rating.dropna(), selected.stars.dropna()
        result[f"recruit_history_{unit}_rating"] = float(ratings.mean()) if len(ratings) >= 3 else np.nan
        result[f"recruit_history_{unit}_bluechip_share"] = float(stars.ge(4).mean()) if len(stars) >= 3 else np.nan
    result["recruitment_available_utc"] = pd.Timestamp("2026-01-01", tz="UTC")
    result["recruitment_max_class"] = int(cohort.year.max()) if not cohort.empty else np.nan
    return result


def _experience(prior: pd.DataFrame, appearances: dict, cutoff: pd.Timestamp) -> dict:
    result = {}
    for metric in USAGE_METRICS:
        rows = prior.loc[prior.metric.eq(metric)]
        total = rows.value.sum(min_count=1)
        valid = (not rows.empty and rows.athlete_id.notna().all() and rows.value.notna().all()
                 and np.isfinite(rows.value).all() and rows.value.ge(0).all() and total > 0)
        value = np.nan
        if valid:
            weights = rows.groupby("athlete_id").value.sum()
            weighted = 0.0
            for pid, weight in weights.items():
                if weight <= 0:
                    continue
                dates = appearances.get(pid)
                n = int(np.searchsorted(dates, cutoff.value, side="left")) if dates is not None else 0
                if not n:
                    valid = False
                    break
                weighted += float(weight) * n
            if valid:
                value = weighted / float(weights.sum())
        result[metric + "_weighted_observed_games"] = value
    return result


def _appearance_index(history: pd.DataFrame) -> dict[str, np.ndarray]:
    observed = history.loc[history.athlete_id.notna() & np.isfinite(history.value),
                          ["athlete_id", "game_id", "available"]].drop_duplicates()
    return {pid: part.available.sort_values().map(lambda ts: pd.Timestamp(ts).value).to_numpy(dtype=np.int64)
            for pid, part in observed.groupby("athlete_id")}


def _continuity(history: pd.DataFrame, team: str, kickoff: pd.Timestamp) -> dict:
    source = history.loc[history.team.eq(team) & history.available.lt(kickoff)
                         & history.season.isin([2025, 2026])].copy()
    current, previous = source.loc[source.season.eq(2026)], source.loc[source.season.eq(2025)]
    result = {}
    for metric in USAGE_METRICS:
        now, before = current.loc[current.metric.eq(metric)], previous.loc[previous.metric.eq(metric)]
        valid = all(not x.empty and x.athlete_id.notna().all() and x.value.notna().all()
                    and np.isfinite(x.value).all() and x.value.ge(0).all() and x.value.sum() > 0
                    for x in (now, before))
        now_share = old_share = np.nan
        if valid:
            n, p = now.groupby("athlete_id").value.sum(), before.groupby("athlete_id").value.sum()
            overlap = set(n.index[n.gt(0)]) & set(p.index[p.gt(0)])
            now_share = float(n.reindex(list(overlap)).sum() / n.sum())
            old_share = float(p.reindex(list(overlap)).sum() / p.sum())
        result[metric + "_observed_returning_usage_share"] = now_share
        result[metric + "_prior_production_share_of_observed_returners"] = old_share
    return result


def build() -> tuple[pd.DataFrame, dict]:
    schedule = read_schedule()
    history = player_history(schedule)
    appearances = _appearance_index(history)
    recruits = recruiting_source()
    games = pd.read_parquet(TARGETS)
    games = games.loc[games.season_type.astype(str).str.lower().eq("regular")
                      & games.completed.fillna(False).astype(bool)
                      & games.home_classification.astype(str).str.lower().eq("fbs")
                      & games.away_classification.astype(str).str.lower().eq("fbs")].copy()
    games["kickoff"] = pd.to_datetime(games.start_date, utc=True)
    rows = []
    for game in games.itertuples(index=False):
        for team in (game.home_team, game.away_team):
            team_history = history.loc[history.team.eq(team)]
            prior = team_history.loc[team_history.available.lt(game.kickoff)]
            observed_games = (prior[["game_id", "kickoff", "available"]].drop_duplicates()
                              .sort_values(["available", "game_id"]).tail(12))
            if observed_games.empty:
                raise ValueError(f"No pregame player history for {team} in target {game.id}")
            used = prior.loc[prior.game_id.isin(observed_games.game_id)]
            latest = observed_games.iloc[-1]
            values = {**usage_concentration(used),
                      **_experience(used, appearances, latest.available + pd.Timedelta(nanoseconds=1)),
                      **_continuity(team_history, team, game.kickoff),
                      **_recruit_features(recruits, team)}
            values.update(target_game_id=int(game.id), team=team,
                          latest_source_game_id=int(latest.game_id),
                          latest_source_available_utc=latest.available,
                          target_start_utc=game.kickoff, source_game_count=len(observed_games))
            rows.append(values)
    state = pd.DataFrame(rows)
    if state.duplicated(["target_game_id", "team"]).any() or len(state) != 542:
        raise ValueError(f"Unexpected F10 target state coverage: {len(state)} rows")
    if not (pd.to_datetime(state.latest_source_available_utc, utc=True)
            < pd.to_datetime(state.target_start_utc, utc=True)).all():
        raise ValueError("F10 game-derived features are not pregame available")
    if state.latest_source_game_id.eq(state.target_game_id).any():
        raise ValueError("F10 target game contributed to its own features")
    if state.recruitment_max_class.dropna().gt(2025).any():
        raise ValueError("F10 recruitment source includes a post-2025 class")
    return state, {"history_rows": len(history), "target_game_count": int(games.id.nunique()),
                   "target_team_rows": len(state), "reporting_lag_hours": 48,
                   "recruitment_classes": sorted(recruits.year.dropna().astype(int).unique().tolist()),
                   "recruitment_snapshot_cutoff_utc": "2026-01-01T00:00:00+00:00",
                   "target_game_feature_leak_check": "passed"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state, receipt = build()
    state.to_parquet(OUT / "f10_target_state.parquet", index=False, compression="zstd")
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
