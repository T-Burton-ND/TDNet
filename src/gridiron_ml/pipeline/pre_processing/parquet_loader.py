# File: src/gridironml/fingerprints/parquet_loader.py
"""Utilities for loading and flattening cached parquet files.

Usage:
    Import `load_and_flatten_parquet` from raw table builders when cached CFBD
    endpoint files include nested dict-like columns.

Logic flow:
    1. Read parquet, optionally pushing a week filter into Arrow.
    2. Detect dict-like or JSON-like columns and flatten them with prefixes.
    3. Return a scalar-column DataFrame with key-column diagnostics.

This module provides a generic helper to:
- Load a parquet file into a pandas DataFrame.
- Detect any dict-like / JSON-like columns.
- Flatten those columns into scalar columns with prefixed names.
- Optionally sanity-check that key columns (season/week/team/etc.) exist.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _read_parquet_filtered(
    parquet_path: str,
    *,
    week: int | None = None,
) -> pd.DataFrame:
    """Read a parquet file and optionally push a `week` predicate into Arrow."""
    read_kwargs: Dict[str, Any] = {}
    if week is not None:
        try:
            schema = pq.read_schema(parquet_path).names
        except Exception:
            schema = []
        if "week" in schema:
            read_kwargs["filters"] = [("week", "==", int(week))]
    return pd.read_parquet(parquet_path, **read_kwargs)

def load_and_flatten_parquet(
    parquet_path: str,
    index_cols: Iterable[str] = ("season", "week", "team"),
    week: int | None = None,
) -> pd.DataFrame:
    """
    Load a parquet file, flatten dict-like columns, and return a DataFrame.

    Parameters
    ----------
    parquet_path : str
        Path to the parquet file to load.
    index_cols : Iterable[str], optional
        Columns that should identify rows (e.g., season/week/team).
        These are not modified, but their presence and missingness
        will be logged for sanity checking.

    Returns
    -------
    pd.DataFrame
        Flattened DataFrame where:
        - All original scalar columns are preserved.
        - Any dict-like / JSON-like columns are expanded into multiple
          columns named "<original>_<key>".
        - The index_cols (if present) are left untouched.
    """
    # --- Load parquet into a DataFrame ---
    print("[parquet_loader] Loading parquet: %s", parquet_path)
    df = _read_parquet_filtered(parquet_path, week=week)
    print(
        "[parquet_loader] Loaded DataFrame with %d rows and %d columns",
        len(df),
        df.shape[1],
    )

    # --- Sanity-check key/index columns ---
    index_cols = list(index_cols)
    for col in index_cols:
        if col not in df.columns:
            print(
                "[parquet_loader] Expected index column '%s' is missing in %s",
                col,
                parquet_path,
            )
        else:
            null_frac = df[col].isna().mean()
            if null_frac > 0.0:
                print(
                    "[parquet_loader] Index column '%s' has %.2f%% missing values",
                    col,
                    null_frac * 100.0,
                )

    # --- Flatten any dict/JSON-like columns ---
    df = _flatten_dict_columns(df)

    print(
        "[parquet_loader] Final flattened DataFrame from %s: %d rows, %d columns",
        parquet_path,
        len(df),
        df.shape[1],
    )
    return df


def _as_list(obj) -> List[Any]:
    """
    Try very hard to interpret `obj` as a list of items.

    Shared helper (also used by rankings loader).

    Handles:
    - list/tuple         -> list(...)
    - numpy.ndarray      -> list(...)
    - dict               -> [dict]
    - string             -> try json.loads, then ast.literal_eval
    - generic Iterable   -> list(iterable)
    - NaN / None         -> []
    """
    if obj is None:
        return []
    if isinstance(obj, float) and math.isnan(obj):
        return []

    if isinstance(obj, (list, tuple, np.ndarray)):
        return list(obj)

    if isinstance(obj, dict):
        return [obj]

    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return []
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(s)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return list(parsed)
                if isinstance(parsed, dict):
                    return [parsed]
            except Exception:
                continue
        return []

    try:
        if isinstance(obj, Iterable):
            return list(obj)
    except Exception:
        pass

    return []


def load_coaches_parquet(parquet_path: str, target_season: int | None = None, cutoff_mode: str = "lt") -> pd.DataFrame:

    """
    Multi-year, career-aware loader for CFBD `coaches` parquets.

    By default, aggregates use seasons strictly before target_season. This keeps
    coaching features available before kickoff and avoids blending current-season
    outcomes into training rows.

    Behavior:
    ---------
    1. Uses *all* coaches/*.parquet files in the same directory as
       `parquet_path` to build prior-season aggregates per coach_full_name:
         - coach_career_seasons (number of CFBD seasons)
         - coach_career_total_games / wins / losses / ties
         - coach_career_mean_sp_offense / sp_defense / sp_overall / srs
         - coach_career_mean_preseason_rank / preseason_rank_points

       Current-season coach_season_* aggregates and postseason/final ranking
       aggregates are intentionally excluded from the default feature surface.

    2. For the specific file at `parquet_path` (e.g. 2024.parquet),
       flattens its `seasons` array to get the mapping:
         (season, team) -> coach_full_name (+ hire_date)

    3. Merges the career + per-season aggregates onto the (season, team)
       mapping, returning one row per (season, team) with coach features.

    Returns
    -------
    pd.DataFrame with keys:
        season, team
    and columns:
        coach_first_name, coach_last_name, coach_full_name, coach_hire_date,
        coach_career_seasons,
        coach_career_total_games, coach_career_total_wins,
        coach_career_total_losses, coach_career_total_ties,
        coach_career_mean_sp_offense, coach_career_mean_sp_defense,
        coach_career_mean_sp_overall, coach_career_mean_srs,
        coach_career_mean_preseason_rank,
        coach_career_mean_preseason_rank_points
    """
    path = Path(parquet_path)
    coaches_dir = path.parent

    print("[parquet_loader] Loading coaches (multi-year) from dir: %s", coaches_dir)

    # -------------------------------------------------
    # Helper: rank -> points (1 → 25, 25 → 1, else 0)
    # -------------------------------------------------
    def _rank_to_points(r: Any) -> int:
        """Internal helper for the rank_to_points step."""
        try:
            if pd.isna(r):
                return 0
            r_int = int(r)
            if r_int < 1 or r_int > 25:
                return 0
            return 26 - r_int
        except Exception:
            return 0

    # -------------------------------------------------
    # 1) Build historical (all years) rows
    # -------------------------------------------------
    all_records: List[Dict[str, Any]] = []

    for f in sorted(coaches_dir.glob("*.parquet")):
        try:
            df_f = pd.read_parquet(f)
        except Exception as e:
            print(
                "[parquet_loader] Failed to read coaches parquet %s: %s", f, e
            )
            continue

        for _, row in df_f.iterrows():
            first_name = row.get("first_name")
            last_name = row.get("last_name")
            hire_date = row.get("hire_date")
            full_name = f"{first_name} {last_name}".strip()

            seasons_list = _as_list(row.get("seasons"))
            if not seasons_list:
                continue

            for s in seasons_list:
                if not isinstance(s, dict):
                    continue

                year = s.get("year") or s.get("season")
                school = s.get("school") or s.get("team")

                if year is None or school is None:
                    continue

                rec: Dict[str, Any] = {
                    "coach_first_name": first_name,
                    "coach_last_name": last_name,
                    "coach_full_name": full_name,
                    "coach_hire_date": hire_date,
                    "hist_year": year,
                    "hist_team": school,
                    "hist_games": s.get("games"),
                    "hist_wins": s.get("wins"),
                    "hist_losses": s.get("losses"),
                    "hist_ties": s.get("ties"),
                    "hist_sp_offense": s.get("spOffense"),
                    "hist_sp_defense": s.get("spDefense"),
                    "hist_sp_overall": s.get("spOverall"),
                    "hist_srs": s.get("srs"),
                    "hist_preseason_rank": s.get("preseasonRank"),
                    "hist_postseason_rank": s.get("postseasonRank"),
                }
                all_records.append(rec)

    if not all_records:
        print(
            "[parquet_loader] No historical coach records found under %s",
            coaches_dir,
        )
        return pd.DataFrame()

    hist_df = pd.DataFrame.from_records(all_records)
    if cutoff_mode not in {"lte", "lt"}:
        raise ValueError("cutoff_mode must be 'lte' or 'lt'")
    if target_season is not None:
        hist_df["hist_year"] = pd.to_numeric(hist_df["hist_year"], errors="coerce")
        if cutoff_mode == "lte":
            hist_df = hist_df[hist_df["hist_year"] <= target_season].copy()
        else:
            hist_df = hist_df[hist_df["hist_year"] < target_season].copy()

    print(
        "[parquet_loader] Historical coaches (all seasons/files): %d rows, %d cols",
        len(hist_df),
        hist_df.shape[1],
    )

    # --- NEW: rank → points columns ---
    hist_df["hist_preseason_rank_points"] = hist_df["hist_preseason_rank"].apply(
        _rank_to_points
    )
    hist_df["hist_postseason_rank_points"] = hist_df["hist_postseason_rank"].apply(
        _rank_to_points
    )

    # -------------------------------------------------
    # 2) Career aggregates per coach_full_name
    # -------------------------------------------------
    career_df = (
        hist_df.groupby("coach_full_name")
        .agg(
            coach_first_name=("coach_first_name", "first"),
            coach_last_name=("coach_last_name", "first"),
            coach_hire_date=("coach_hire_date", "first"),
            coach_career_seasons=("hist_year", "nunique"),
            coach_career_total_games=("hist_games", "sum"),
            coach_career_total_wins=("hist_wins", "sum"),
            coach_career_total_losses=("hist_losses", "sum"),
            coach_career_total_ties=("hist_ties", "sum"),
            coach_career_mean_sp_offense=("hist_sp_offense", "mean"),
            coach_career_mean_sp_defense=("hist_sp_defense", "mean"),
            coach_career_mean_sp_overall=("hist_sp_overall", "mean"),
            coach_career_mean_srs=("hist_srs", "mean"),
            coach_career_mean_preseason_rank=("hist_preseason_rank", "mean"),
            coach_career_mean_preseason_rank_points=(
                "hist_preseason_rank_points",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        "[parquet_loader] Career coach aggregates: %d rows, %d cols",
        len(career_df),
        career_df.shape[1],
    )

    # -------------------------------------------------
    # 3) Mapping for the specific season file at `parquet_path`
    #    (season, team) -> coach_full_name
    # -------------------------------------------------
    try:
        df_current = pd.read_parquet(parquet_path)
    except Exception as e:
        print(
            "[parquet_loader] Failed to read current coaches parquet %s: %s",
            parquet_path,
            e,
        )
        return pd.DataFrame()

    # Infer current season from filename, if possible
    try:
        current_season = int(path.stem)
    except ValueError:
        current_season = None

    mapping_records: List[Dict[str, Any]] = []

    for _, row in df_current.iterrows():
        first_name = row.get("first_name")
        last_name = row.get("last_name")
        hire_date = row.get("hire_date")
        full_name = f"{first_name} {last_name}".strip()

        seasons_list = _as_list(row.get("seasons"))
        if not seasons_list:
            continue

        for s in seasons_list:
            if not isinstance(s, dict):
                continue

            year = s.get("year") or s.get("season")
            school = s.get("school") or s.get("team")

            # We only want mapping for the *current* season/team
            if current_season is not None and year != current_season:
                continue

            if year is None or school is None:
                continue

            mapping_records.append(
                {
                    "season": year,
                    "team": school,
                    "coach_full_name": full_name,
                    "coach_hire_date": hire_date,
                }
            )

    if not mapping_records:
        print(
            "[parquet_loader] No (season, team) coach mapping built from %s",
            parquet_path,
        )
        return pd.DataFrame()

    mapping_df = pd.DataFrame.from_records(mapping_records)
    mapping_df = mapping_df.drop_duplicates(subset=["season", "team"], keep="first")

    print(
        "[parquet_loader] Current-season coach mapping (season, team): %d rows, %d cols",
        len(mapping_df),
        mapping_df.shape[1],
    )

    # -------------------------------------------------
    # 4) Join mapping with prior-season career aggregates
    # -------------------------------------------------
    out = mapping_df.merge(career_df, on="coach_full_name", how="left")

    print(
        "[parquet_loader] Final coaches table (season/team with coach stats): %d rows, %d cols",
        len(out),
        out.shape[1],
    )

    return out





def _as_list(obj) -> List[Any]:
    """
    Try very hard to interpret `obj` as a list of items.

    Handles:
    - list/tuple         -> list(...)
    - dict               -> [dict]
    - string             -> try json.loads, then ast.literal_eval
    - generic Iterable   -> list(iterable)
    - NaN / None         -> []
    """
    if obj is None:
        return []
    if isinstance(obj, float) and math.isnan(obj):
        return []

    if isinstance(obj, (list, tuple)):
        return list(obj)

    if isinstance(obj, dict):
        return [obj]

    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return []
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(s)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                if isinstance(parsed, dict):
                    return [parsed]
            except Exception:
                continue
        return []

    try:
        if isinstance(obj, Iterable):
            return list(obj)
    except Exception:
        pass

    return []


def _normalize_stats_field(stats_val: Any) -> List[Dict[str, Any]]:
    """
    Normalize the 'stats' field from CFBD game_team_stats into a list of dicts.

    CFBD returns this as a list, NumPy array, single dict, or sometimes null.
    We want a clean List[dict] with each dict having at least 'category' and 'stat'.
    """
    # Handle None / NaN
    if stats_val is None:
        return []

    if isinstance(stats_val, float) and math.isnan(stats_val):
        return []

    # NumPy array of dicts
    if isinstance(stats_val, np.ndarray):
        return [x for x in stats_val if isinstance(x, dict)]

    # Plain Python list/tuple
    if isinstance(stats_val, (list, tuple)):
        return [x for x in stats_val if isinstance(x, dict)]

    # Single dict
    if isinstance(stats_val, dict):
        return [stats_val]

    # Anything else -> ignore
    return []

def _select_primary_line(lines_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Choose a single 'primary' line dict from a list of provider lines.

    Heuristic:
    - Prefer providers whose name contains 'consensus' or 'vegas'
    - Otherwise, fall back to the first dict.
    """
    if not lines_list:
        return {}

    # Normalize provider names
    best = None
    for d in lines_list:
        if not isinstance(d, dict):
            continue
        provider = str(d.get("provider", "")).lower()
        if "consensus" in provider or "vegas" in provider:
            best = d
            break

    if best is None:
        # Fallback: first dict
        for d in lines_list:
            if isinstance(d, dict):
                best = d
                break

    return best or {}

def load_lines_parquet(parquet_path: str, week: int | None = None) -> pd.DataFrame:
    """
    Specialized loader for CFBD 'lines' parquet.

    Produces one row per (game_id, team) with:
    - season, week, season_type, game_id, start_date
    - team, team_id, opponent, opponent_id, is_home
    - home_score, away_score, margin
    - line_spread_raw        (home-based spread from CFBD)
    - line_over_under
    - line_home_moneyline
    - line_away_moneyline
    - line_team_spread       (spread from the TEAM's perspective)
    """
    print("[parquet_loader] Loading lines parquet: %s", parquet_path)
    df_raw = _read_parquet_filtered(parquet_path, week=week)
    print(
        "[parquet_loader] Raw lines: %d rows, %d cols",
        len(df_raw),
        df_raw.shape[1],
    )

    if df_raw.empty:
        print("[parquet_loader] lines parquet is empty: %s", parquet_path)
        return df_raw

    records: List[Dict[str, Any]] = []

    for _, row in df_raw.iterrows():
        game_id = row.get("id") or row.get("game_id")
        season = row.get("season") or row.get("year")
        week = row.get("week")
        season_type = row.get("season_type")
        start_date = row.get("start_date")

        home_team = row.get("home_team")
        away_team = row.get("away_team")
        home_team_id = row.get("home_team_id")
        away_team_id = row.get("away_team_id")
        home_score = row.get("home_score")
        away_score = row.get("away_score")

        # Basic sanity: skip if we don't have both teams
        if not home_team or not away_team:
            continue

        margin_home = None
        try:
            if home_score is not None and away_score is not None:
                margin_home = float(home_score) - float(away_score)
        except Exception:
            margin_home = None

        # Pick a single representative line dict
        lines_list = _normalize_lines_field(row.get("lines"))
        primary = _select_primary_line(lines_list)

        spread = primary.get("spread")
        over_under = primary.get("overUnder")
        home_ml = primary.get("homeMoneyline")
        away_ml = primary.get("awayMoneyline")

        # Home team record
        rec_home: Dict[str, Any] = {
            "season": season,
            "week": week,
            "season_type": season_type,
            "game_id": game_id,
            "start_date": start_date,
            "team": home_team,
            "team_id": home_team_id,
            "opponent": away_team,
            "opponent_id": away_team_id,
            "is_home": True,
            "home_score": home_score,
            "away_score": away_score,
            "margin": margin_home,
            "line_spread_raw": spread,           # CFBD's home-based spread
            "line_over_under": over_under,
            "line_home_moneyline": home_ml,
            "line_away_moneyline": away_ml,
            "line_team_spread": spread,          # From HOME team perspective
        }
        records.append(rec_home)

        # Away team record
        margin_away = None
        if margin_home is not None:
            margin_away = -margin_home

        rec_away: Dict[str, Any] = {
            "season": season,
            "week": week,
            "season_type": season_type,
            "game_id": game_id,
            "start_date": start_date,
            "team": away_team,
            "team_id": away_team_id,
            "opponent": home_team,
            "opponent_id": home_team_id,
            "is_home": False,
            "home_score": home_score,
            "away_score": away_score,
            "margin": margin_away,
            "line_spread_raw": spread,           # still the HOME spread
            "line_over_under": over_under,
            "line_home_moneyline": home_ml,
            "line_away_moneyline": away_ml,
            "line_team_spread": -spread if spread is not None else None,
        }
        records.append(rec_away)

    if not records:
        print(
            "[parquet_loader] No per-team records built from lines: %s",
            parquet_path,
        )
        return pd.DataFrame()

    out = pd.DataFrame.from_records(records)
    print(
        "[parquet_loader] Flattened lines: %d rows, %d cols",
        len(out),
        out.shape[1],
    )

    # Sanity check
    for key in ("season", "week", "team", "game_id"):
        if key not in out.columns:
            print(
                "[parquet_loader] Expected key column '%s' missing in flattened lines",
                key,
            )
        else:
            null_frac = out[key].isna().mean()
            if null_frac > 0.0:
                print(
                    "[parquet_loader] Key column '%s' has %.2f%% missing in flattened lines",
                    key,
                    null_frac * 100.0,
                )

    return out

def load_pregame_wp_parquet(parquet_path: str, week: int | None = None) -> pd.DataFrame:
    """
    Specialized loader for CFBD 'pregame_wp' parquet.

    Produces one row per (game_id, team) with:
    - season, week, season_type, game_id
    - team, opponent, is_home
    - spread_raw               (home-based spread from CFBD)
    - team_spread              (spread from the TEAM's perspective)
    - home_win_probability
    - team_win_probability     (from TEAM's perspective)
    """
    print("[parquet_loader] Loading pregame_wp parquet: %s", parquet_path)
    df_raw = _read_parquet_filtered(parquet_path, week=week)
    print(
        "[parquet_loader] Raw pregame_wp: %d rows, %d cols",
        len(df_raw),
        df_raw.shape[1],
    )

    if df_raw.empty:
        print("[parquet_loader] pregame_wp parquet is empty: %s", parquet_path)
        return df_raw

    records: List[Dict[str, Any]] = []

    for _, row in df_raw.iterrows():
        season = row.get("season") or row.get("year")
        week = row.get("week")
        season_type = row.get("season_type")
        game_id = row.get("game_id")

        home_team = row.get("home_team")
        away_team = row.get("away_team")
        spread = row.get("spread")                  # home-based spread
        home_wp = row.get("home_win_probability")

        if not home_team or not away_team:
            continue

        # Home perspective
        rec_home: Dict[str, Any] = {
            "season": season,
            "week": week,
            "season_type": season_type,
            "game_id": game_id,
            "team": home_team,
            "opponent": away_team,
            "is_home": True,
            "pregame_spread_raw": spread,
            "pregame_team_spread": spread,          # home perspective
            "home_win_probability": home_wp,
            "team_win_probability": home_wp,
        }
        records.append(rec_home)

        # Away perspective
        away_wp = None
        try:
            if home_wp is not None:
                away_wp = 1.0 - float(home_wp)
        except Exception:
            away_wp = None

        rec_away: Dict[str, Any] = {
            "season": season,
            "week": week,
            "season_type": season_type,
            "game_id": game_id,
            "team": away_team,
            "opponent": home_team,
            "is_home": False,
            "pregame_spread_raw": spread,
            "pregame_team_spread": -spread if spread is not None else None,
            "home_win_probability": home_wp,
            "team_win_probability": away_wp,
        }
        records.append(rec_away)

    if not records:
        print(
            "[parquet_loader] No per-team records built from pregame_wp: %s",
            parquet_path,
        )
        return pd.DataFrame()

    out = pd.DataFrame.from_records(records)
    print(
        "[parquet_loader] Flattened pregame_wp: %d rows, %d cols",
        len(out),
        out.shape[1],
    )

    for key in ("season", "week", "team", "game_id"):
        if key not in out.columns:
            print(
                "[parquet_loader] Expected key column '%s' missing in flattened pregame_wp",
                key,
            )
        else:
            null_frac = out[key].isna().mean()
            if null_frac > 0.0:
                print(
                    "[parquet_loader] Key column '%s' has %.2f%% missing in flattened pregame_wp",
                    key,
                    null_frac * 100.0,
                )

    return out




def _normalize_lines_field(lines_val: Any) -> List[Dict[str, Any]]:
    """
    Normalize the 'lines' field from CFBD /lines into a list of dicts.

    CFBD returns this as a list, NumPy array, or sometimes null.
    We want a clean List[dict] for downstream selection.
    """
    if lines_val is None:
        return []

    if isinstance(lines_val, float) and math.isnan(lines_val):
        return []

    if isinstance(lines_val, np.ndarray):
        return [x for x in lines_val if isinstance(x, dict)]

    if isinstance(lines_val, (list, tuple)):
        return [x for x in lines_val if isinstance(x, dict)]

    if isinstance(lines_val, dict):
        return [lines_val]

    return []


def load_game_team_stats_parquet(parquet_path: str, week: int | None = None) -> pd.DataFrame:
    """
    Specialized loader for CFBD game_team_stats parquet.

    This flattens:
    - One row per (game_id, team) side.
    - Extracts season, week, team, opponent, is_home, points.
    - Expands the 'stats' field (array/list of {category, stat}) into
      wide columns named 'stat_<category>'.

    Parameters
    ----------
    parquet_path : str
        Path to the game_team_stats parquet file for a single season.

    Returns
    -------
    pd.DataFrame
        One row per team-game with columns:
        - season, week, game_id, team, opponent, is_home, points
        - stat_* columns for each CFBD stat category encountered.
    """
    print("[parquet_loader] Loading game_team_stats parquet: %s", parquet_path)
    df_raw = _read_parquet_filtered(parquet_path, week=week)
    print(
        "[parquet_loader] Raw game_team_stats: %d rows, %d cols",
        len(df_raw),
        df_raw.shape[1],
    )

    if df_raw.empty:
        print("[parquet_loader] game_team_stats parquet is empty: %s", parquet_path)
        return df_raw

    records: List[Dict[str, Any]] = []

    for _, row in df_raw.iterrows():
        game_id = row.get("id") or row.get("game_id")
        season = row.get("season") or row.get("year")
        week = row.get("week")

        teams_val = row.get("teams")
        if not isinstance(teams_val, (list, tuple, np.ndarray)):
            # Nothing sensible to do if 'teams' isn't iterable
            continue

        # Each entry in teams is a dict like:
        # {'conference': ..., 'homeAway': 'home'/'away', 'points': 40,
        #  'stats': [ {...}, {...}, ... ], 'team': 'Butler', 'teamId': 2086}
        for t in teams_val:
            if not isinstance(t, dict):
                continue

            team_name = t.get("team")
            if not team_name:
                continue

            # Base record for this team/game
            rec: Dict[str, Any] = {
                "season": season,
                "week": week,
                "game_id": game_id,
                "team": team_name,
                "team_id": t.get("teamId"),
                "conference": t.get("conference"),
                "home_away": t.get("homeAway"),
                "points": t.get("points"),
            }

            # Flatten the per-team 'stats' list into wide stat_* columns.
            stats_list = _normalize_stats_field(t.get("stats"))
            for stat in stats_list:
                if not isinstance(stat, dict):
                    continue
                cat = stat.get("category")
                val = stat.get("stat")
                if cat is None:
                    continue
                col_name = f"stat_{str(cat)}"
                rec[col_name] = val

            records.append(rec)

    if not records:
        print(
            "[parquet_loader] No per-team records built from game_team_stats: %s",
            parquet_path,
        )
        return pd.DataFrame()

    out = pd.DataFrame.from_records(records)
    print(
        "[parquet_loader] Flattened game_team_stats: %d rows, %d cols",
        len(out),
        out.shape[1],
    )

    # Basic sanity check on key columns
    for key in ("season", "week", "team", "game_id"):
        if key not in out.columns:
            print(
                "[parquet_loader] Expected key column '%s' missing in flattened game_team_stats",
                key,
            )
        else:
            null_frac = out[key].isna().mean()
            if null_frac > 0.0:
                print(
                    "[parquet_loader] Key column '%s' has %.2f%% missing values in flattened game_team_stats",
                    key,
                    null_frac * 100.0,
                )

    return out


def _flatten_dict_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and flatten dict-like object columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that may contain dict-like columns.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame where any dict-like columns have been
        expanded into multiple scalar columns and the original
        dict-like columns removed.
    """
    # --- Find candidate columns: object dtype is where dicts/JSON usually live ---
    candidate_cols: List[str] = [
        c for c in df.columns if df[c].dtype == "object"
    ]
    if not candidate_cols:
        print("[parquet_loader] No object-dtype columns found; nothing to flatten.")
        return df

    print(
        "[parquet_loader] Inspecting %d object-dtype columns for dict-like data",
        len(candidate_cols),
    )

    dict_like_cols: List[str] = []
    for col in candidate_cols:
        if _is_dict_like_series(df[col]):
            dict_like_cols.append(col)

    if not dict_like_cols:
        print("[parquet_loader] No dict-like columns detected; skipping flattening.")
        return df

    print(
        "[parquet_loader] Detected %d dict-like columns to flatten: %s",
        len(dict_like_cols),
        ", ".join(dict_like_cols),
    )

    # --- Expand each dict-like column into multiple scalar columns ---
    for col in dict_like_cols:
        print("[parquet_loader] Flattening column '%s'", col)
        expanded = _expand_dict_column(df[col], prefix=col)
        # Drop the original column and concatenate the expanded columns
        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    return df


def _is_dict_like_series(s: pd.Series, sample_size: int = 20) -> bool:
    """
    Heuristically determine whether a Series contains dict-like data.

    Parameters
    ----------
    s : pd.Series
        Series to inspect.
    sample_size : int, optional
        Number of non-null entries to sample for type checking.

    Returns
    -------
    bool
        True if the sample suggests that the Series mostly contains
        dicts or JSON strings that decode to dicts.
    """
    # --- Take a small sample of non-null values for inspection ---
    sample = s.dropna().head(sample_size)
    if sample.empty:
        return False

    dict_count = 0
    for val in sample:
        # Direct dict objects
        if isinstance(val, dict):
            dict_count += 1
            continue

        # Try to interpret strings as JSON dicts
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    dict_count += 1
            except Exception:
                # Not valid JSON, ignore
                continue

    # --- If at least half of the sample looks like dicts, treat as dict-like ---
    is_dict_like = dict_count >= max(1, len(sample) // 2)
    return is_dict_like


def _expand_dict_column(s: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Expand a dict-like Series into separate columns.

    Parameters
    ----------
    s : pd.Series
        Series containing dict or JSON-dict values.
    prefix : str
        Prefix used for naming new columns, usually the original column name.

    Returns
    -------
    pd.DataFrame
        DataFrame whose columns correspond to the union of dict keys across
        the Series. Column names are formatted as '<prefix>_<key>'.
    """
    # --- Normalize each entry to a dict ---
    dicts = []
    for val in s:
        if isinstance(val, dict):
            dicts.append(val)
        elif isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    dicts.append(parsed)
                else:
                    dicts.append({})
            except Exception:
                dicts.append({})
        else:
            dicts.append({})

    # --- Use json_normalize to turn list-of-dicts into a DataFrame ---
    expanded = pd.json_normalize(dicts)

    # --- Prefix column names to keep provenance and avoid collisions ---
    expanded.columns = [
        f"{prefix}_{str(c).replace('.', '_')}" for c in expanded.columns
    ]

    # --- Preserve original index so we can concat back to the parent DataFrame ---
    expanded.index = s.index

    print(
        "[parquet_loader] Expanded '%s' into %d new columns",
        prefix,
        expanded.shape[1],
    )
    return expanded
