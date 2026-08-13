"""src.gridiron_ml.pipeline.raw_weekly_builder.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Transform cached raw data into canonical tables and downstream artifacts.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import pandas as pd

from gridiron_ml.pipeline.pre_processing.cleaners import main_clean, prep_for_fingerprints
from gridiron_ml.pipeline.pre_processing.parquet_loader import (
    load_and_flatten_parquet,
    load_coaches_parquet,
    load_game_team_stats_parquet,
    load_lines_parquet,
    load_pregame_wp_parquet,
)

# -------------------------------------------------------------------
# Helpers for raw parquet → team-week DataFrame
# -------------------------------------------------------------------


Loader = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class CacheTableSpec:
    """Represent the CacheTableSpec component and its local behavior."""
    label: str
    subdir: str
    loader: Loader


def _generic_loader(index_cols: list[str]) -> Loader:
    """Internal helper for the generic_loader step."""
    return lambda parquet_path, *, week=None, year=None: load_and_flatten_parquet(
        parquet_path,
        index_cols=index_cols,
        week=week,
    )


CACHE_TABLE_SPECS = {
    spec.label: spec
    for spec in [
        CacheTableSpec("games", "games", _generic_loader(["season", "week"])),
        CacheTableSpec(
            "game_team_stats",
            "game_team_stats",
            lambda parquet_path, *, week=None, year=None: load_game_team_stats_parquet(parquet_path, week=week),
        ),
        CacheTableSpec("stats_advanced_game", "stats_advanced_game", _generic_loader(["season", "week", "team"])),
        CacheTableSpec("havoc_game", "havoc_game", _generic_loader(["season", "week", "team"])),
        CacheTableSpec(
            "lines",
            "lines",
            lambda parquet_path, *, week=None, year=None: load_lines_parquet(parquet_path, week=week),
        ),
        CacheTableSpec(
            "pregame_wp",
            "pregame_wp",
            lambda parquet_path, *, week=None, year=None: load_pregame_wp_parquet(parquet_path, week=week),
        ),
        CacheTableSpec("ppa_games", "ppa_games", _generic_loader(["season", "week", "team"])),
        CacheTableSpec("talent", "talent", _generic_loader(["year", "team"])),
        CacheTableSpec("returning", "returning", _generic_loader(["season", "team"])),
        CacheTableSpec("recruiting_teams", "recruiting_teams", _generic_loader(["year", "team"])),
        CacheTableSpec(
            "coaches",
            "coaches",
            lambda parquet_path, *, week=None, year=None: load_coaches_parquet(
                parquet_path,
                target_season=year,
                cutoff_mode="lt",
            ),
        ),
    ]
}

SERIAL_MERGE_SOURCES = [
    ("game_team_stats", "_gts"),
    ("stats_advanced_game", "_adv"),
    ("havoc_game", "_havoc"),
    ("lines", "_lines"),
    ("pregame_wp", "_wp"),
    ("ppa_games", "_ppa"),
]


def _load_parquet_safe(
    cache_dir: Path,
    label: str,
    year: int,
    *,
    week: int | None = None,
) -> pd.DataFrame:
    """Load one serialized CFBD cache table and normalize key dtypes."""
    if label not in CACHE_TABLE_SPECS:
        raise ValueError(f"Unsupported cache table label: {label}")

    spec = CACHE_TABLE_SPECS[label]
    path = cache_dir / spec.subdir / f"{year}.parquet"

    if not path.exists():
        print(f"[weekly_builder.raw] Missing parquet for '{label}': {path}")
        return pd.DataFrame()

    print(f"[weekly_builder.raw] Loading '{label}' from {path}")
    df = spec.loader(str(path), week=week, year=year)

    # 🔧 Normalize key dtypes (avoid int vs object merge issues)
    for col in ("season", "year", "week"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df




def _build_team_week_from_games(games: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (season, week, team, opponent, is_home, game_id) grid from CFBD games.

    Assumes columns are snake_case CFBD fields, e.g.:
    - season
    - week
    - id (game id)
    - home_team, away_team
    - home_points, away_points, etc.

    Returns one row per team per game (so 2 rows per game).
    """
    if games.empty:
        raise RuntimeError("[weekly_builder.raw] 'games' DataFrame is empty; cannot build team-week grid.")

    # Try to identify the game id column.
    game_id_col = None
    for cand in ("id", "game_id"):
        if cand in games.columns:
            game_id_col = cand
            break

    if game_id_col is None:
        print("[weekly_builder.raw] No game-id column ('id' or 'game_id') found in games; "
                    "team-week rows will not carry a game_id key.")
    else:
        print("[weekly_builder.raw] Using '%s' as game_id column.", game_id_col)

    base_cols = ["season", "week"]
    if game_id_col:
        base_cols.append(game_id_col)
    
    extra_cols = [c for c in ("venue_id", "home_id", "away_id") if c in games.columns]

    # Home side
    required_home = base_cols + ["home_team", "away_team"] + extra_cols
    missing_home = [c for c in required_home if c not in games.columns]
    if missing_home:
        raise RuntimeError(f"[weekly_builder.raw] Missing expected columns in games for home rows: {missing_home}")

    home = games[required_home].copy()
    home = home.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
    })
    home["is_home"] = True

    # Away side
    required_away = base_cols + ["home_team", "away_team"] + extra_cols
    missing_away = [c for c in required_away if c not in games.columns]
    if missing_away:
        raise RuntimeError(f"[weekly_builder.raw] Missing expected columns in games for away rows: {missing_away}")

    away = games[required_away].copy()
    away = away.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
    })
    away["is_home"] = False

    # Use a unified 'game_id' column name if possible
    if game_id_col:
        home = home.rename(columns={game_id_col: "game_id"})
        away = away.rename(columns={game_id_col: "game_id"})

    team_week = pd.concat([home, away], ignore_index=True)
    print("[weekly_builder.raw] Built team-week grid from games: shape=%s", team_week.shape)

    # Remove any obvious duplicates on season/week/team/game_id
    key_cols = ["season", "week", "team"]
    if "game_id" in team_week.columns:
        key_cols.append("game_id")
    dup_count = team_week.duplicated(subset=key_cols).sum()
    if dup_count > 0:
        print("[weekly_builder.raw] Found %d duplicate team-week rows on keys %s; dropping duplicates.",
                    dup_count, key_cols)
        team_week = team_week.drop_duplicates(subset=key_cols)

    return team_week


def _merge_on_team_week(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    source_name: str,
    prefer_game_id: bool = True,
    suffix: str = ""
) -> pd.DataFrame:
    """
    Merge additional per-game or per-team data onto the team-week grid.

    For per-game tables (e.g. game_team_stats, stats_advanced_game, havoc_game, lines),
    this tries to merge on (season, week, team, game_id) when available; otherwise
    falls back to (season, week, team).

    For season/team tables (e.g. ratings, talent, ppa_teams, teams_ats, returning),
    it merges on (season, team).

    Parameters
    ----------
    base : pd.DataFrame
        Team-week grid (must have at least season, week, team).
    extra : pd.DataFrame
        Data from a CFBD parquet to be merged.
    source_name : str
        Human-readable name for logging.
    prefer_game_id : bool, optional
        If True, use game_id-based keys when possible, by default True.
    suffix : str, optional
        Suffix for overlapping column names, by default "" (no suffix).

    Returns
    -------
    pd.DataFrame
        The merged DataFrame.
    """
    if extra is None or extra.empty:
        print("[weekly_builder.raw] Extra table '%s' is empty; skipping merge.", source_name)
        return base

    df = extra.copy()

    # Try to normalize key columns
    if "year" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"year": "season"})
    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})
    if "id" in df.columns and "game_id" not in df.columns and "game_id" in base.columns:
        # Use 'id' from CFBD as game_id when base has game_id
        df = df.rename(columns={"id": "game_id"})

    # 🔧 NEW: normalize key dtypes on both sides to avoid int vs object merge errors
    for col in ("season", "week", "game_id"):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Decide whether this is a season/team table or per-game/per-team table
    per_game_keys = ["season", "week", "team"]
    per_game_keys_with_gid = per_game_keys + ["game_id"]
    season_team_keys = ["season", "team"]

    merge_keys: List[str]

    if all(k in df.columns for k in per_game_keys) and all(k in base.columns for k in per_game_keys):
        if prefer_game_id and "game_id" in base.columns and "game_id" in df.columns:
            merge_keys = per_game_keys_with_gid
        else:
            merge_keys = per_game_keys
        print("[weekly_builder.raw] Merging '%s' as per-game data on keys %s", source_name, merge_keys)
    elif all(k in df.columns for k in season_team_keys) and all(k in base.columns for k in season_team_keys):
        merge_keys = season_team_keys
        print("[weekly_builder.raw] Merging '%s' as season/team data on keys %s", source_name, merge_keys)
    else:
        print("[weekly_builder.raw] Cannot merge '%s': missing required key columns. "
                    "Base keys=%s, extra columns=%s",
                    source_name, ["season", "week", "team", "game_id"], list(df.columns))
        return base

    before_rows = base.shape[0]
    merged = base.merge(df, how="left", on=merge_keys, suffixes=("", suffix))
    after_rows = merged.shape[0]

    if before_rows != after_rows:
        print("[weekly_builder.raw] Row count changed after merging '%s' (before=%d, after=%d). "
                    "Check join keys: %s",
                    source_name, before_rows, after_rows, merge_keys)
    else:
        print("[weekly_builder.raw] Successfully merged '%s'; shape now=%s",
                 source_name, merged.shape)

    return merged

def _load_fbs_teams(cache_dir: Path, season: int) -> set[str]:
    """
    Load the list of FBS team names for a given season from teams_fbs parquet.

    Returns a set of school names that should match the 'team' / 'opponent'
    columns in the weekly fingerprint.
    """
    path = cache_dir / "teams_fbs" / f"{season}.parquet"
    if not path.exists():
        print(
            "[weekly_builder.raw] Missing teams_fbs parquet for season %s: %s",
            season,
            path,
        )
        return set()

    df_fbs = pd.read_parquet(path)
    print(
        "[weekly_builder.raw] Loaded teams_fbs for %s: %d rows",
        season,
        len(df_fbs),
    )

    # CFBD usually uses 'school' here; fall back to 'team' if needed.
    if "school" in df_fbs.columns:
        names = df_fbs["school"].dropna().unique()
    elif "team" in df_fbs.columns:
        names = df_fbs["team"].dropna().unique()
    else:
        print(
            "[weekly_builder.raw] teams_fbs parquet missing 'school'/'team' columns: %s",
            path,
        )
        return set()

    fbs_set = {str(n) for n in names}
    print(
        "[weekly_builder.raw] FBS team list for %s has %d unique entries",
        season,
        len(fbs_set),
    )
    return fbs_set

def _load_team_locations(cache_dir: Path, season: int) -> pd.DataFrame:
    """
    Load FBS team home locations from teams_fbs parquet.

    Returns columns:
      team, loc_city, loc_state, loc_timezone, loc_latitude, loc_longitude, loc_elevation
    (season is NOT included; locations are effectively season-agnostic in teams_fbs.)
    """
    path = cache_dir / "teams_fbs" / f"{season}.parquet"
    if not path.exists():
        print("[weekly_builder.raw] Missing teams_fbs parquet for locations: %s", path)
        return pd.DataFrame()

    df = pd.read_parquet(path)
    print("[weekly_builder.raw] Loaded teams_fbs for locations: %d rows", len(df))

    # Normalize name
    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})

    # Flatten the 'location' dict column if present
    if "location" in df.columns:
        loc = pd.json_normalize(df["location"]).add_prefix("loc_")
        df = pd.concat([df.drop(columns=["location"]), loc], axis=1)

    cols_keep = [
        "team",
        "loc_city", "loc_state", "loc_timezone",
        "loc_latitude", "loc_longitude", "loc_elevation",
    ]
    cols_keep = [c for c in cols_keep if c in df.columns]

    return df[cols_keep].copy()

def add_venues(raw: pd.DataFrame, cache_dir: Path, current_year: int):
    """Run the add_venues step and return its normalized result."""
    venues_path = cache_dir / "venue" / f"{current_year}.parquet"
    if venues_path.exists():
        venues = pd.read_parquet(venues_path)
        print("[weekly_builder.raw] Loaded venues: %d rows", len(venues))

        venue_cols = [
            "id", "name", "city", "state", "timezone",
            "latitude", "longitude", "elevation",
        ]
        venue_cols = [c for c in venue_cols if c in venues.columns]
        venues = venues[venue_cols].rename(columns={"id": "venue_id"})

        raw = raw.merge(
            venues,
            how="left",
            on="venue_id",
            suffixes=("", "_venue"),
        )

        # Optionally make the names explicit
        raw = raw.rename(columns={
            "name": "venue_name",
            "city": "venue_city",
            "state": "venue_state",
            "timezone": "venue_timezone",
            "latitude": "venue_latitude",
            "longitude": "venue_longitude",
            "elevation": "venue_elevation",
        })
    else:
        print("[weekly_builder.raw] Venues parquet not found: %s", venues_path)
    return raw

def add_locations(raw: pd.DataFrame, cache_dir: Path, current_year: int):
    """Run the add_locations step and return its normalized result."""
    team_locs = _load_team_locations(cache_dir, current_year)
    if not team_locs.empty:
        # Attach location for the row's team
        team_locs_team = team_locs.rename(columns={
            "loc_city": "team_loc_city",
            "loc_state": "team_loc_state",
            "loc_timezone": "team_loc_timezone",
            "loc_latitude": "team_loc_latitude",
            "loc_longitude": "team_loc_longitude",
            "loc_elevation": "team_loc_elevation",
        })
        raw = raw.merge(
            team_locs_team,
            how="left",
            on="team",          # 🔁 only team
        )

        # Attach location for the opponent
        team_locs_opp = team_locs.rename(columns={
            "team": "opponent",
            "loc_city": "opp_loc_city",
            "loc_state": "opp_loc_state",
            "loc_timezone": "opp_loc_timezone",
            "loc_latitude": "opp_loc_latitude",
            "loc_longitude": "opp_loc_longitude",
            "loc_elevation": "opp_loc_elevation",
        })
        raw = raw.merge(
            team_locs_opp,
            how="left",
            on="opponent",      # 🔁 only opponent
        )
    else:
        print("[weekly_builder.raw] Team locations table is empty; skipping location merge.")

    return raw

def add_returning(raw: pd.DataFrame, returning: pd.DataFrame) -> pd.DataFrame:
    """
    Attach returning production/usage metrics for both team and opponent.

    Expects `returning` to come from `_load_parquet_safe(..., label="returning")`,
    typically with columns like:
        season or year, team or school, conference,
        total_p_p_a, total_passing_p_p_a, ..., rushing_usage

    Behavior:
      - Normalizes keys: year→season, school→team if needed.
      - Treats (season, team) as the key.
      - Identifies metric columns as everything except:
            season, team, conference
      - Adds, for the row's team:
            return_team_<metric>
        and for the row's opponent:
            return_opp_<metric>

    Does NOT:
      - add raw season/team/conference as feature columns
    """
    if returning is None or returning.empty:
        print("[weekly_builder.raw] Returning table is empty; skipping add_returning.")
        return raw

    df = returning.copy()

    # Normalize key names
    if "year" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"year": "season"})
    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})

    # Identify metric columns (everything except season/team/conference)
    key_drop = {"season", "team", "conference"}
    metric_cols = [c for c in df.columns if c not in key_drop]

    if not metric_cols:
        print(
            "[weekly_builder.raw] add_returning: no metric columns after dropping keys; cols=%s",
            list(df.columns),
        )
        return raw

    print(
        "[weekly_builder.raw] add_returning: metric columns: %s",
        ", ".join(metric_cols),
    )

    # --- TEAM returning metrics ---
    team_ret = df[["season", "team"] + metric_cols].copy()
    team_ret = team_ret.rename(
        columns={col: f"return_team_{col}" for col in metric_cols}
    )

    # Merge onto (season, team)
    raw = raw.merge(team_ret, how="left", on=["season", "team"])

    print(
        "[weekly_builder.raw] add_returning: merged team returning; shape=%s",
        raw.shape,
    )
    return raw


def add_talent(raw: pd.DataFrame, talent: pd.DataFrame) -> pd.DataFrame:
    """
    Attach team-level talent ratings for the row's team only.

    Expects `talent` to come from `_load_parquet_safe(...)` for label "talent",
    typically with columns like:
        year or season, team, talent, [maybe conference, etc.]

    Behavior:
      - Normalize keys: year→season, school→team if needed.
      - Keep ONLY (season, team, talent).
      - Merge onto `raw` on (season, team).
      - Does NOT add opponent talent or any extra columns.
    """
    if talent is None or talent.empty:
        print("[weekly_builder.raw] Talent table is empty; skipping add_talent.")
        return raw

    df = talent.copy()

    # Normalize key names if needed
    if "year" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"year": "season"})
    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})

    # Only keep season/team/talent
    keep_cols = [c for c in ("season", "team", "talent") if c in df.columns]
    if not all(k in keep_cols for k in ("season", "team")):
        print(
            "[weekly_builder.raw] add_talent: missing season/team in talent table; "
            "cols=%s",
            list(df.columns),
        )
        return raw

    df = df[keep_cols].drop_duplicates(subset=["season", "team"]).copy()

    # Merge onto (season, team) – team-only, no opponent
    merged = raw.merge(df, how="left", on=["season", "team"])

    print(
        "[weekly_builder.raw] add_talent: merged talent; shape %s -> %s",
        raw.shape,
        merged.shape,
    )
    return merged

def add_recruiting(raw: pd.DataFrame, recruiting: pd.DataFrame) -> pd.DataFrame:
    """
    Attach team-level recruiting rank/points for the row's team only.

    Expects `recruiting` to come from `_load_parquet_safe(..., label="recruiting_teams")`,
    typically with columns like:
        year or season, team or school, rank, points

    Behavior:
      - Normalizes keys: year→season, school→team if needed.
      - Treats (season, team) as the key.
      - Keeps only rank + points as metrics.
      - Adds:
            recruit_rank
            recruit_points
        for the row's team.

    Does NOT:
      - add raw season/team columns as extra features
      - add opponent recruiting metrics
    """
    if recruiting is None or recruiting.empty:
        print("[weekly_builder.raw] Recruiting table is empty; skipping add_recruiting.")
        return raw

    df = recruiting.copy()

    # Normalize key names
    if "year" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"year": "season"})
    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})

    # Ensure required columns exist
    missing_keys = [c for c in ("season", "team") if c not in df.columns]
    if missing_keys:
        print(
            "[weekly_builder.raw] add_recruiting: missing key columns %s; cols=%s",
            missing_keys,
            list(df.columns),
        )
        return raw

    missing_metrics = [c for c in ("rank", "points") if c not in df.columns]
    if missing_metrics:
        print(
            "[weekly_builder.raw] add_recruiting: missing metric columns %s; cols=%s",
            missing_metrics,
            list(df.columns),
        )
        return raw

    # Subset and dedupe
    df = df[["season", "team", "rank", "points"]].copy()
    df = df.drop_duplicates(subset=["season", "team"])

    # Prefix metrics
    df = df.rename(columns={
        "rank": "recruit_rank",
        "points": "recruit_points",
    })

    # Merge onto (season, team)
    merged = raw.merge(df, how="left", on=["season", "team"])

    print(
        "[weekly_builder.raw] add_recruiting: merged recruiting; shape %s -> %s",
        raw.shape,
        merged.shape,
    )
    return merged



def add_coaches(raw: pd.DataFrame, coaches: pd.DataFrame) -> pd.DataFrame:
    """Run the add_coaches step and return its normalized result."""
    if coaches is None or coaches.empty:
        print("[weekly_builder.raw] Coaches table empty; skipping coach merge.")
        return raw

    # ensure dtype alignment for join keys
    for col in ("season", "team"):
        if col in raw.columns and col in coaches.columns:
            coaches[col] = coaches[col].astype(raw[col].dtype)

    out = raw.merge(
        coaches,
        on=["season", "team"],
        how="left",
    )
    print(
        "[weekly_builder.raw] After add_coaches: shape=%s",
        out.shape,
    )
    return out


def build_team_year_summary(
    cache_dir: Path,
    year: int,
    division: str = "fbs",
) -> pd.DataFrame:
    """
    Build a per-team summary for a single season by averaging
    numeric stats over all that team's games.
    """
    team_games = build_team_game_table_from_parquets(
        cache_dir=cache_dir,
        out_dir=Path("."),
        current_year=year,
        division=division,
    )

    # Pick numeric stat columns only (avoid keys, times, IDs, etc.)
    numeric_cols = team_games.select_dtypes(include="number").columns.tolist()

    # Drop key-ish columns that shouldn't be averaged
# Columns that should NOT be averaged when building per-team year summaries / priors
    drop_cols = {
    # Keys / identifiers
    "season",
    "week",
    "game_id",
    "team_id",
    "opponent_id",

    # Location / geography (static, not something to average over games)
    "venue_latitude",
    "venue_longitude",
    "venue_elevation",
    "venue_tz_offset",
    "team_loc_latitude",
    "team_loc_longitude",
    "team_loc_elevation",
    "opp_loc_latitude",
    "opp_loc_longitude",
    "opp_loc_elevation",

    # Time zone offsets
    "team_tz_offset",
    "opp_tz_offset",

    # Calendar / clock info
    "game_date",
    "game_time",
    "game_day_of_week",

    # Optional future field you mentioned
    "kickoff_minutes_after_midnight",
    }

    numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    summary = (
        team_games
        .groupby(["season", "team"], as_index=False)[numeric_cols]
        .mean()
    )

    # Optionally save to disk: team_year_summary/<year>.parquet
    # summary.to_parquet(out_dir / "team_year_summary" / f"{year}.parquet")

    return summary

def table_health_report(df):
    """Run the table_health_report step and return its normalized result."""
    print("rows:", len(df))
    print("games:", df["game_id"].nunique(), " (expect rows = 2*games)")
    print("dup (game_id, team):", df.duplicated(["game_id", "team"]).sum())
    print()

    missing = df.isna().sum().sort_values(ascending=False).head(15)
    print("Top missing cols:")
    for col, n in missing.items():
        print(f"  {col}: {n}")
    print()

    core_missing = df[["points_for", "points_against", "team_margin"]].isna().any(axis=1).sum()
    print("Rows missing score/margin:", core_missing)




def build_team_game_table_from_parquets(
    cache_dir: Path,
    current_year: int,
    division: str = "fbs",
    week: int | None = None,
) -> pd.DataFrame:
    """
    Build a raw per-team-per-game table for a single season by ingesting
    CFBD Parquet files and merging them onto a (season, week, team, game_id) grid.

    Design goals:
    -------------
    - Exactly TWO rows per game: one for each team.
    - Clear home/away flag: `is_home` (bool).
    - Only **current game / current week** data:
        * box-score style stats (game_team_stats)
        * advanced stats per game (stats_advanced_game)
        * havoc per game (havoc_game)
        * betting lines (lines)
        * pregame win probabilities (pregame_wp)
        * per-game PPA (ppa_games)
        * venue + team/opponent locations (static but game-specific)
    - NO to-date, rolling, or prior-season/priors features. Those will be
      added later at the fingerprint-building stage.

    Parameters
    ----------
    cache_dir : Path
        CFBD cache root, e.g. data/raw/cfbd/v2.
    out_dir : Path
        Output directory (not used here, but reserved for future writes).
    current_year : int
        Season year to build (e.g. 2024).
    division : str, optional
        Division string used when fetching games. The games parquet should
        already be filtered by division; included here for future flexibility.

    Returns
    -------
    pd.DataFrame
        A wide, raw per-team-per-game DataFrame with columns like:

        Keys:
            season, week, game_id, team, opponent, is_home

        Game context:
            venue_id, home_id, away_id, [plus any venue/location columns]

        Result:
            home_score, away_score, margin (team perspective via lines/pre_wp),
            points (from game_team_stats, etc.)

        Stats:
            stat_* columns from game_team_stats
            advanced_* columns from stats_advanced_game (suffix "_adv")
            havoc_* columns from havoc_game (suffix "_havoc")
            ppa_* columns from ppa_games (suffix "_ppa")

        Betting:
            line_spread_raw, line_team_spread, line_over_under,
            line_home_moneyline, line_away_moneyline (from lines)

        Pregame:
            pregame_spread_raw, pregame_team_spread,
            home_win_probability, team_win_probability (from pregame_wp)

        Plus venue/location columns (e.g. venue_city, team_loc_state, etc.).
    """
    scope = f"year={current_year}, week={int(week)}, division={division}" if week is not None else f"year={current_year}, division={division}"
    print(f"[weekly_builder.raw] Building TEAM-GAME table for {scope}")

    # ----------------------------------------------------
    # 1) Base team-game grid from `games`
    # ----------------------------------------------------
    games = _load_parquet_safe(cache_dir, "games", current_year, week=week)
    if games.empty:
        raise RuntimeError(
            "[weekly_builder.raw] No games found for year=%d. Expected at %s",
            current_year,
            cache_dir / "games" / f"{current_year}.parquet",
        )

    if week is not None and "week" in games.columns:
        games = games.loc[pd.to_numeric(games["week"], errors="coerce") == int(week)].copy()
        print(f"[weekly_builder.raw] Filtered games to week={int(week)}; rows={len(games)}")
        if games.empty:
            raise RuntimeError(
                f"[weekly_builder.raw] No games found for year={current_year}, week={int(week)} "
                f"under {cache_dir / 'games' / f'{current_year}.parquet'}"
            )

    base = _build_team_week_from_games(games)  # 2 rows per game
    # base has: season, week, team, opponent, is_home, game_id, venue_id?, home_id?, away_id?

    # Filter to FBS games if we have the FBS list
    fbs_teams = _load_fbs_teams(cache_dir, current_year)
    if fbs_teams:
        before = len(base)
        mask = base["team"].isin(fbs_teams) | base["opponent"].isin(fbs_teams)
        base = base[mask].copy()
        print(
            "[weekly_builder.raw] Filtered base to games with ≥1 FBS team: %d -> %d rows",
            before,
            len(base),
        )
    else:
        print(
            "[weekly_builder.raw] FBS filter skipped because FBS team list is empty."
        )

    # ----------------------------------------------------
    # 2) Merge serialized per-game sources onto the base grid
    # ----------------------------------------------------
    raw = base

    for label, suffix in SERIAL_MERGE_SOURCES:
        table = _load_parquet_safe(cache_dir, label, current_year, week=week)
        raw = _merge_on_team_week(raw, table, label, prefer_game_id=True, suffix=suffix)

    # ----------------------------------------------------
    # 3) Merge serialized season/team enrichments
    # ----------------------------------------------------
    talent = _load_parquet_safe(cache_dir, "talent", current_year)
    returning = _load_parquet_safe(cache_dir, "returning", current_year)
    recruiting = _load_parquet_safe(cache_dir, "recruiting_teams", current_year)
    coaches = _load_parquet_safe(cache_dir, "coaches", current_year)

    raw = add_talent(raw, talent)
    raw = add_returning(raw, returning)
    raw = add_venues(raw, cache_dir, current_year)
    raw = add_locations(raw, cache_dir, current_year)
    raw = add_recruiting(raw, recruiting)
    raw = add_coaches(raw, coaches)

    print(
        "[weekly_builder.raw] Finished TEAM-GAME build for %d; final shape=%s",
        current_year,
        raw.shape,
    )

    # ----------------------------------------------------
    # 4) Clean Up Current Year Data
    # ----------------------------------------------------
    team_games = raw.copy()
    team_games = main_clean(team_games)  

    # ----------------------------------------------------
    # 5) Prep data for fingerprint building
    # ----------------------------------------------------
    team_games = prep_for_fingerprints(team_games)

    # display table health report
    table_health_report(team_games)
    
    return team_games
