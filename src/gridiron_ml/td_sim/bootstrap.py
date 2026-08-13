"""src.gridiron_ml.td_sim.bootstrap.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Run recursive season simulations with evolving synthetic fingerprints.
"""

from __future__ import annotations

import os
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_ml.pipeline.contracts.artifacts import team_week_fingerprints_path, team_week_labels_path
from gridiron_ml.pipeline.contracts.features import (
    DEFAULT_TRAINING_TARGET,
    HAS_NEXT_GAME_COLUMN,
    LABEL_VALUE_COLUMNS,
    MARKET_COLUMNS,
    NEXT_GAME_COLUMNS,
    SAME_WEEK_TARGET,
    TARGET_COLUMNS,
    is_market_column,
)
from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import CFBDClient, fetch_games, fetch_teams_fbs, fetch_venues, write_parquet
from gridiron_ml.pipeline.schemas import validate_prediction_rows


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GAME_ID_COLS = {"keys_game_id", "keys_opponent", "keys_game_date", "game_is_home", "game_home_away"}
TARGET_COLS = set(TARGET_COLUMNS)
NEXT_GAME_COLS = set(NEXT_GAME_COLUMNS)


def ensure_schedule_team_game_table(
    *,
    season: int,
    raw_cache_dir: str | Path | None = None,
    team_game_tables_dir: str | Path | None = None,
    division: str = "fbs",
    output_format: str = "parquet",
    api_key_env: str = "CFBD_API_KEY",
    refresh_raw: bool = False,
) -> Path:
    """Build a schedule-only canonical team-game table from CFBD games data.

    The full historical team-game pipeline expects completed-game stat sources.
    Upcoming seasons often only have schedule metadata, so this builder writes a
    narrow but canonical two-row-per-game table that TD Sim's schedule loader can
    consume without requiring game stats.
    """

    season = int(season)
    output_format = str(output_format).strip().lower()
    suffix = "parquet" if output_format == "parquet" else "csv"
    raw_cache = _resolve(raw_cache_dir or PROJECT_ROOT / "data" / "raw" / "cfbd" / "v2")
    tables_dir = _resolve(team_game_tables_dir or PROJECT_ROOT / "data" / "team_game_tables")
    out_path = tables_dir / f"team_game_table_{season}_{division}.{suffix}"
    if out_path.exists():
        return out_path

    _ensure_raw_schedule_cache(
        season=season,
        raw_cache_dir=raw_cache,
        division=division,
        api_key_env=api_key_env,
        refresh=refresh_raw,
    )
    games_path = raw_cache / "games" / f"{season}.parquet"
    if not games_path.exists():
        raise FileNotFoundError(
            f"Missing CFBD games cache for {season}: {games_path}. "
            f"Set {api_key_env} or fetch games before running TD Sim."
        )

    games = pd.read_parquet(games_path)
    table = schedule_team_game_table_from_cfbd_games(games, season=season, raw_cache_dir=raw_cache, division=division)
    validate_prediction_rows(_prediction_rows_from_schedule_table(table))
    tables_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "parquet":
        table.to_parquet(out_path, index=False)
    else:
        table.to_csv(out_path, index=False)
    return out_path


def schedule_team_game_table_from_cfbd_games(
    games: pd.DataFrame,
    *,
    season: int,
    raw_cache_dir: str | Path | None = None,
    division: str = "fbs",
) -> pd.DataFrame:
    """Run the schedule_team_game_table_from_cfbd_games step and return its normalized result."""
    games = games.copy()
    if games.empty:
        raise ValueError(f"Cannot build schedule-only team-game table for {season} from an empty games frame.")

    if "season" in games.columns:
        games = games.loc[pd.to_numeric(games["season"], errors="coerce") == int(season)].copy()
    if "season_type" in games.columns:
        games = games.loc[games["season_type"].astype(str).str.lower().eq("regular")].copy()

    fbs_teams = _fbs_teams(raw_cache_dir=raw_cache_dir, season=season) if str(division).lower() == "fbs" else set()
    if fbs_teams:
        games = games.loc[games["home_team"].astype(str).isin(fbs_teams) | games["away_team"].astype(str).isin(fbs_teams)].copy()

    rows = []
    for _, game in games.iterrows():
        home_team = _clean_team(game.get("home_team"))
        away_team = _clean_team(game.get("away_team"))
        if not home_team or not away_team:
            continue
        game_id = game.get("id", game.get("game_id", pd.NA))
        week = pd.to_numeric(pd.Series([game.get("week")]), errors="coerce").fillna(0).iloc[0]
        home_points = pd.to_numeric(pd.Series([game.get("home_points")]), errors="coerce").iloc[0]
        away_points = pd.to_numeric(pd.Series([game.get("away_points")]), errors="coerce").iloc[0]
        common = {
            "keys_season": int(season),
            "keys_week": int(week),
            "keys_game_id": game_id,
            "keys_season_type": game.get("season_type", "regular"),
            "keys_game_date": game.get("start_date", pd.NA),
            "game_neutral_site": bool(game.get("neutral_site", False)) if pd.notna(game.get("neutral_site", pd.NA)) else False,
            "game_conference_game": game.get("conference_game", pd.NA),
            "venue_name": game.get("venue", pd.NA),
        }
        rows.append(
            {
                **common,
                "keys_team": home_team,
                "keys_opponent": away_team,
                "keys_conference": game.get("home_conference", pd.NA),
                "game_is_home": True,
                "game_home_away": "home",
                "target_points_for": home_points,
                "target_points_against": away_points,
                "target_team_margin": home_points - away_points if pd.notna(home_points) and pd.notna(away_points) else np.nan,
            }
        )
        rows.append(
            {
                **common,
                "keys_team": away_team,
                "keys_opponent": home_team,
                "keys_conference": game.get("away_conference", pd.NA),
                "game_is_home": False,
                "game_home_away": "away",
                "target_points_for": away_points,
                "target_points_against": home_points,
                "target_team_margin": away_points - home_points if pd.notna(home_points) and pd.notna(away_points) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No usable games were found while building the {season} schedule table.")
    return out.sort_values(["keys_season", "keys_week", "keys_game_id", "game_is_home"], ascending=[True, True, True, False]).reset_index(drop=True)


def append_bootstrap_week0(
    fingerprint_frame: pd.DataFrame,
    *,
    season: int,
    teams: list[str] | set[str] | tuple[str, ...],
    seasons_back: int = 3,
    recency_halflife: float = 1.5,
    persist: bool = False,
    root: str | Path | None = None,
    version: int = 0,
) -> pd.DataFrame:
    """Append synthetic target-season Week 0 fingerprints when they are absent."""

    frame = fingerprint_frame.copy()
    if frame.empty:
        return frame
    mask = (pd.to_numeric(frame["keys_season"], errors="coerce") == int(season)) & (
        pd.to_numeric(frame["keys_week"], errors="coerce") == 0
    )
    if mask.any():
        return frame

    week0 = bootstrap_week0_fingerprints(
        frame,
        season=season,
        teams=teams,
        seasons_back=seasons_back,
        recency_halflife=recency_halflife,
    )
    if persist:
        persist_bootstrap_week0(week0, root=root, version=version, season=season)
    week0_for_concat = week0.dropna(axis=1, how="all")
    return pd.concat([frame, week0_for_concat], ignore_index=True, sort=False).reindex(columns=frame.columns)


def bootstrap_week0_fingerprints(
    fingerprint_frame: pd.DataFrame,
    *,
    season: int,
    teams: list[str] | set[str] | tuple[str, ...],
    seasons_back: int = 3,
    recency_halflife: float = 1.5,
) -> pd.DataFrame:
    """Run the bootstrap_week0_fingerprints step and return its normalized result."""
    frame = fingerprint_frame.copy()
    teams = sorted({str(team) for team in teams if str(team) and str(team).lower() != "nan"})
    if not teams:
        raise ValueError("Cannot bootstrap Week 0 fingerprints without scheduled teams.")

    frame["keys_season"] = pd.to_numeric(frame["keys_season"], errors="coerce")
    frame["keys_week"] = pd.to_numeric(frame["keys_week"], errors="coerce")
    hist = frame.loc[frame["keys_season"] < int(season)].copy()
    if seasons_back > 0:
        recent = hist.loc[hist["keys_season"] >= int(season) - int(seasons_back)].copy()
        if not recent.empty:
            hist = recent
    if hist.empty:
        raise ValueError(f"No historical fingerprints are available to bootstrap {season} Week 0.")

    final_rows = _final_team_season_rows(hist)
    numeric_cols = [c for c in final_rows.columns if pd.api.types.is_numeric_dtype(final_rows[c])]
    feature_numeric_cols = [
        c
        for c in numeric_cols
        if c not in {"keys_season", "keys_week", "keys_game_id"}
        and not c.startswith("y_")
        and not is_market_column(c)
    ]
    global_prior = _weighted_mean(final_rows, feature_numeric_cols, target_season=season, recency_halflife=recency_halflife)
    rows = []
    for team in teams:
        team_hist = final_rows.loc[final_rows["keys_team"].astype(str) == team].copy()
        if team_hist.empty:
            row = global_prior.copy()
        else:
            row = _weighted_mean(team_hist, feature_numeric_cols, target_season=season, recency_halflife=recency_halflife)
            row = row.fillna(global_prior)
        out = pd.Series(index=frame.columns, dtype="object")
        out.loc[:] = pd.NA
        for col in feature_numeric_cols:
            if col in out.index:
                out[col] = row.get(col, global_prior.get(col, np.nan))
        out["keys_season"] = int(season)
        out["keys_team"] = team
        out["keys_week"] = 0
        if "keys_season_type" in out.index:
            out["keys_season_type"] = "regular"
        if "games_played" in out.index:
            out["games_played"] = 0.0
        for col in GAME_ID_COLS | TARGET_COLS | NEXT_GAME_COLS | set(MARKET_COLUMNS) | set(LABEL_VALUE_COLUMNS):
            if col in out.index:
                out[col] = False if col == HAS_NEXT_GAME_COLUMN else pd.NA
        if "travel_tz_diff" in out.index:
            out["travel_tz_diff"] = 0.0
        if "travel_distance_diff" in out.index:
            out["travel_distance_diff"] = 0.0
        if "fp_version" in out.index:
            out["fp_version"] = pd.to_numeric(frame["fp_version"], errors="coerce").dropna().max()
        if "fp_subversion" in out.index:
            out["fp_subversion"] = 1
        if "fp_build_timestamp" in out.index:
            out["fp_build_timestamp"] = pd.Timestamp.now(tz=timezone.utc).isoformat()
        rows.append(out)

    week0 = pd.DataFrame(rows, columns=frame.columns)
    for col in feature_numeric_cols + ["keys_season", "keys_week", "games_played", "fp_version", "fp_subversion"]:
        if col in week0.columns:
            week0[col] = pd.to_numeric(week0[col], errors="coerce")
    return week0.reset_index(drop=True)


def persist_bootstrap_week0(week0: pd.DataFrame, *, root: str | Path | None, version: int, season: int) -> dict[str, Path]:
    """Run the persist_bootstrap_week0 step and return its normalized result."""
    root_path = _resolve(root or PROJECT_ROOT)
    season_dir = root_path / "data" / "fingerprints" / f"v{int(version)}" / str(int(season))
    season_dir.mkdir(parents=True, exist_ok=True)
    fp_path = team_week_fingerprints_path(root_path / "data" / "fingerprints" / f"v{int(version)}", season)
    label_path = team_week_labels_path(root_path / "data" / "fingerprints" / f"v{int(version)}", season)

    label_cols = [c for c in LABEL_COLUMNS if c in week0.columns]
    labels = week0.loc[:, label_cols].copy() if label_cols else pd.DataFrame()
    fingerprints = week0.drop(columns=[c for c in [SAME_WEEK_TARGET, DEFAULT_TRAINING_TARGET, HAS_NEXT_GAME_COLUMN] if c in week0.columns])

    fingerprints.to_parquet(fp_path, index=False)
    labels.to_parquet(label_path, index=False)
    return {"fingerprints": fp_path, "labels": label_path}


def _final_team_season_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Internal helper for the final_team_season_rows step."""
    work = frame.copy()
    sort_cols = ["keys_season", "keys_team", "keys_week"]
    if "games_played" in work.columns:
        work["__games_played"] = pd.to_numeric(work["games_played"], errors="coerce").fillna(0.0)
        sort_cols.append("__games_played")
    work = work.sort_values(sort_cols)
    out = work.groupby(["keys_season", "keys_team"], sort=False, observed=True).tail(1).copy()
    return out.drop(columns=["__games_played"], errors="ignore")


def _weighted_mean(frame: pd.DataFrame, cols: list[str], *, target_season: int, recency_halflife: float) -> pd.Series:
    """Internal helper for the weighted_mean step."""
    values = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    age = (int(target_season) - pd.to_numeric(frame["keys_season"], errors="coerce")).clip(lower=1)
    if recency_halflife > 0:
        weights = np.power(0.5, (age - 1) / float(recency_halflife))
    else:
        weights = pd.Series(1.0, index=frame.index)
    weights = pd.Series(weights, index=frame.index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    denominator = values.notna().astype(float).mul(weights, axis=0).sum(axis=0)
    return values.mul(weights, axis=0).sum(axis=0, min_count=1) / denominator.replace(0.0, np.nan)


def _ensure_raw_schedule_cache(
    *,
    season: int,
    raw_cache_dir: Path,
    division: str,
    api_key_env: str,
    refresh: bool,
) -> None:
    """Internal helper for the ensure_raw_schedule_cache step."""
    tasks = {
        "games": lambda client: fetch_games(client, season, division),
        "teams_fbs": lambda client: fetch_teams_fbs(client, season),
        "venue": lambda client: fetch_venues(client, season),
    }
    missing = [name for name in tasks if refresh or not (raw_cache_dir / name / f"{season}.parquet").exists()]
    if not missing:
        return
    if not os.environ.get(api_key_env):
        return
    client = CFBDClient(api_key_env=api_key_env)
    for name in missing:
        out_path = raw_cache_dir / name / f"{season}.parquet"
        frame = tasks[name](client)
        write_parquet(frame, out_path, snake=True)


def _fbs_teams(*, raw_cache_dir: str | Path | None, season: int) -> set[str]:
    """Internal helper for the fbs_teams step."""
    if raw_cache_dir is None:
        return set()
    path = _resolve(raw_cache_dir) / "teams_fbs" / f"{int(season)}.parquet"
    if not path.exists():
        return set()
    teams = pd.read_parquet(path)
    if "school" not in teams.columns:
        return set()
    return set(teams["school"].dropna().astype(str))


def _clean_team(value) -> str:
    """Internal helper for the clean_team step."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _prediction_rows_from_schedule_table(table: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical two-row schedule table into home-side prediction rows."""
    frame = table.copy()
    home = frame.loc[frame["game_is_home"].astype(bool)].copy()
    return pd.DataFrame(
        {
            "season": pd.to_numeric(home["keys_season"], errors="coerce"),
            "week": pd.to_numeric(home["keys_week"], errors="coerce"),
            "game_id": home["keys_game_id"],
            "home_team": home["keys_team"].astype(str),
            "away_team": home["keys_opponent"].astype(str),
            "target_team_margin": pd.to_numeric(home.get("target_team_margin"), errors="coerce"),
        }
    )


def _resolve(path: str | Path) -> Path:
    """Internal helper for the resolve step."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
