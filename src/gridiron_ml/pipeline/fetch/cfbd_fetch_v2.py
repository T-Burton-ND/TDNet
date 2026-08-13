#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CFBD single-season bulk ingest helpers.

Usage:
    Run this module as a script with a fetch config, or import individual
    fetch_* helpers from pipeline code.

Logic flow:
    1. Resolve configuration, cache paths, and CFBD API credentials.
    2. Fetch each enabled endpoint once per season, with retry handling.
    3. Normalize column names and save endpoint parquet files for later stages.

Notes:
    Uses Bearer auth via $CFBD_API_KEY, writes one parquet per endpoint/year,
    and counts only real outbound API attempts.
"""

import os, time, argparse, re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import requests
import yaml
from tqdm import tqdm


BASE = "https://api.collegefootballdata.com"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "fetch" / "cfbd_single_year.yaml"

# --------------- Env expansion for ${env:VAR,"default"} ----------------
def expand_env_like(s: str) -> str:
    """Run the expand_env_like step and return its normalized result."""
    if not isinstance(s, str):
        return s
    def repl(m):
        """Run the repl step and return its normalized result."""
        var = m.group(1)
        default = m.group(2) or ""
        return os.environ.get(var, default)
    s2 = re.sub(r'\$\{env:([^,}]+)(?:,\s*"([^"]*)")?\}', repl, s)
    return os.path.expandvars(s2)

# ---------------- Optional camel->snake normalization ----------------
_camel_pat = re.compile(r'(?<!^)([A-Z])')
def camel_to_snake(s: str) -> str:
    """Run the camel_to_snake step and return its normalized result."""
    return _camel_pat.sub(r'_\1', s).lower()

def maybe_snake(df: pd.DataFrame, do_snake: bool) -> pd.DataFrame:
    """Run the maybe_snake step and return its normalized result."""
    if not do_snake or df is None or df.empty:
        return df
    return df.rename(columns={c: camel_to_snake(c) for c in df.columns})

# ---------------- Save with dirs + caching ----------------
def write_parquet(df: pd.DataFrame, path: Path, snake: bool):
    """Run the write_parquet step and return its normalized result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or len(df) == 0:
        df = pd.DataFrame()
    df = maybe_snake(df, snake)
    df.to_parquet(path, index=False)

# ---------------- Minimal client with call counter ----------------
class CFBDClient:
    """Represent the CFBDClient component and its local behavior."""
    def __init__(self, api_key_env: str = "CFBD_API_KEY", timeout: int = 60):
        """Internal helper for the init__ step."""
        key = os.environ.get(api_key_env)
        if not key:
            raise SystemExit("Set CFBD_API_KEY in environment.")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {key}"})
        self.s.timeout = timeout
        self.api_calls = 0  # counts only real network calls (no cache hits)

    def get_json(self, path: str, params: Dict[str, Any], max_retries: int = 5, backoff: float = 1.5):
        """Run the get_json step and return its normalized result."""
        url = f"{BASE}{path}"
        attempt = 0
        while True:
            # COUNT each outbound request attempt
            self.api_calls += 1
            r = self.s.get(url, params=params)
            if r.status_code == 404:
                return []
            if r.status_code in (429, 502, 503, 504):
                attempt += 1
                if attempt > max_retries:
                    r.raise_for_status()
                time.sleep(backoff ** attempt)
                continue
            if r.status_code == 400:
                # Often means "no data for this slice"
                return []
            r.raise_for_status()
            return r.json()

    def df_from(self, path: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Run the df_from step and return its normalized result."""
        data = self.get_json(path, params)
        return pd.DataFrame(data)

# ---------------- Simple year-wide fetchers (1 call each) ----------------
def fetch_ratings_sp(client: CFBDClient, year: int):
    """Run the fetch_ratings_sp step and return its normalized result."""
    return client.df_from("/ratings/sp", {"year": year})


def fetch_ratings_fpi(client: CFBDClient, year: int):
    """Run the fetch_ratings_fpi step and return its normalized result."""
    return client.df_from("/ratings/fpi", {"year": year})


def fetch_ratings_elo(client: CFBDClient, year: int):
    """Run the fetch_ratings_elo step and return its normalized result."""
    return client.df_from("/ratings/elo", {"year": year})


def fetch_stats_basic(client: CFBDClient, year: int):
    """Run the fetch_stats_basic step and return its normalized result."""
    return client.df_from("/stats/season", {"year": year})


def fetch_stats_adv(client: CFBDClient, year: int):
    """Run the fetch_stats_adv step and return its normalized result."""
    return client.df_from("/stats/season/advanced", {"year": year})


def fetch_games(client: CFBDClient, year: int, division: str):
    """Run the fetch_games step and return its normalized result."""
    return client.df_from("/games", {"year": year, "division": division, "seasonType": "both"})


def fetch_coaches(client: CFBDClient, year: int):
    """Run the fetch_coaches step and return its normalized result."""
    return client.df_from("/coaches", {"year": year})


def fetch_roster(client: CFBDClient, year: int):
    """Run the fetch_roster step and return its normalized result."""
    return client.df_from("/roster", {"year": year})


def fetch_recruit_players(client: CFBDClient, year: int):
    """Run the fetch_recruit_players step and return its normalized result."""
    return client.df_from("/recruiting/players", {"year": year})


def fetch_recruit_teams(client: CFBDClient, year: int):
    """Run the fetch_recruit_teams step and return its normalized result."""
    return client.df_from("/recruiting/teams", {"year": year})


def fetch_talent(client: CFBDClient, year: int):
    """Run the fetch_talent step and return its normalized result."""
    return client.df_from("/talent", {"year": year})


def fetch_teams_fbs(client: CFBDClient, year: int):
    """Run the fetch_teams_fbs step and return its normalized result."""
    return client.df_from("/teams/fbs", {"year": year})


def fetch_lines(client, year):
    # Lines can be book-specific; we fetch all and aggregate later
    """Run the fetch_lines step and return its normalized result."""
    return client.df_from("/lines", {"year": year, "seasonType": "both"})

def fetch_pregame_wp(client, year):
    """Run the fetch_pregame_wp step and return its normalized result."""
    return client.df_from("/metrics/wp/pregame", {"year": year})

def fetch_returning(client, year):
    """Run the fetch_returning step and return its normalized result."""
    return client.df_from("/player/returning", {"year": year})

def fetch_ratings_srs(client, year):
    """Run the fetch_ratings_srs step and return its normalized result."""
    return client.df_from("/ratings/srs", {"year": year})

def fetch_teams_ats(client, year):
    """Run the fetch_teams_ats step and return its normalized result."""
    return client.df_from("/teams/ats", {"year": year})

def fetch_ppa_teams(client, year):
    # team-level PPA (Predicted Points Added)
    """Run the fetch_ppa_teams step and return its normalized result."""
    return client.df_from("/ppa/teams", {"year": year, "seasonType": "both"})

def fetch_havoc_game(client, year):
    """Run the fetch_havoc_game step and return its normalized result."""
    return client.df_from("/stats/game/havoc", {"year": year, "seasonType": "both"})

def fetch_rankings(client, year):
    """Run the fetch_rankings step and return its normalized result."""
    return client.df_from("/rankings", {"year": year, "seasonType": "both"})

# --- add these helpers near your other fetch_* ---
# --- add these helpers ---
def fetch_stats_advanced_game(client: CFBDClient, year: int) -> pd.DataFrame:
    """
    Fetch advanced team statistics *by game* from CFBD and return as a DataFrame.

    This wraps the /stats/game/advanced endpoint and returns one row per
    team-game with advanced metrics (success rate, explosiveness, PPA, etc.).
    """
    df = client.df_from("/stats/game/advanced", {"year": year})
    # You can inspect and rename later as needed; for now just return raw.
    return df


def fetch_stats_basic_game(client: CFBDClient, year: int) -> pd.DataFrame:
    """
    Fetch basic team statistics *by game* from CFBD and return as a DataFrame.

    This wraps the `/games/teams` endpoint and loops over all plausible
    weeks in the season, concatenating results. Each row is a
    (season, week, team, game) stat line.

    Parameters
    ----------
    client : CFBDClient
        Lightweight HTTP client with Bearer auth (defined in this module).
    year : int
        Season year to fetch.

    Returns
    -------
    pd.DataFrame
        Flattened DataFrame of basic team game stats. Includes at least:
        - season (int)
        - week (int)
        - team (str)
        plus whatever box score stats CFBD provides.
    """
    all_frames: list[pd.DataFrame] = []
    max_weeks = 20  # covers regular season + bowls

    print(f"[cfbd] Fetching team game stats for year={year}")

    for week in range(1, max_weeks + 1):
        params = {
            "year": year,
            "week": week,
            "seasonType": "both",  # allowed param
            # IMPORTANT: do NOT include a 'division' param here; CFBD doesn't accept it on this endpoint
        }

        try:
            data = client.get_json("/games/teams", params)
        except Exception as e:
            print(f"[cfbd] ERROR fetching game_team_stats year={year} week={week}: {e}")
            continue

        if not data:
            # No games in this week (very normal for later weeks)
            continue

        df_w = pd.json_normalize(data)
        df_w["season"] = year
        df_w["week"] = week
        all_frames.append(df_w)

        print(f"[cfbd] week={week}: {len(df_w)} rows from /games/teams")

    if not all_frames:
        print(f"[cfbd] No team game stats found for year={year} (all weeks empty).")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    print(f"[cfbd] Combined team game stats for {year}: {len(df)} rows")

    # Standardize key columns for joining.
    if "school" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"school": "team"})

    for col in ("season", "week", "team"):
        if col not in df.columns:
            print(f"[cfbd] WARNING: missing column '{col}' in team game stats for year={year}")
        else:
            null_frac = df[col].isna().mean()
            if null_frac > 0:
                print(f"[cfbd] WARNING: null fraction for '{col}' in team game stats "
                      f"for year={year}: {null_frac:.3f}")

    return df

def fetch_ppa_games(client: CFBDClient, year: int) -> pd.DataFrame:
    """Run the fetch_ppa_games step and return its normalized result."""
    df = client.df_from("/ppa/games", {"year": year, "seasonType": "both"})
    return df

def fetch_venues(client: CFBDClient, year: int) -> pd.DataFrame:
    """Run the fetch_venues step and return its normalized result."""
    df = client.df_from("/venues", {"year": year, "seasonType": "both"})
    return df

def fetch_weather(client: CFBDClient, year: int) -> pd.DataFrame:
    """Run the fetch_weather step and return its normalized result."""
    json = client.get_json("/games/weather", {"year": year, "seasonType": "both"})
    df = pd.json_normalize(json)
    return df

# --- strengthen JSON parsing in CFBDClient.get_json ---
# (replace your current get_json with this version)
def get_json(self, path: str, params: Dict[str, Any], max_retries: int = 5, backoff: float = 1.5):
    """Run the get_json step and return its normalized result."""
    url = f"{BASE}{path}"
    attempt = 0
    while True:
        self.api_calls += 1
        r = self.s.get(url, params=params)
        if r.status_code == 404:
            return []
        if r.status_code in (429, 502, 503, 504):
            attempt += 1
            if attempt > max_retries:
                r.raise_for_status()
            time.sleep(backoff ** attempt)
            continue
        if r.status_code == 400:
            return []
        r.raise_for_status()
        # guard non-JSON bodies
        if "application/json" not in r.headers.get("Content-Type", ""):
            txt = (r.text or "")[:200].replace("\n", " ")
            raise RuntimeError(f"Non-JSON response from {path} params={params}: {txt!r}")
        return r.json()


CFBDClient.get_json = get_json  # monkey-patch if defined below class

# ---------------- Main ----------------
def main():
    """Run the main step and return its normalized result."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                    help="YAML config with: cache_dir, year, endpoints{}, division, sleep_seconds, snake_case, refresh")
    ap.add_argument("--refresh", action="store_true",
                    help="Override config to force re-fetch even if cache exists")
    ap.add_argument("--year", type=int,
                    help="Override the year in the YAML config")
    ap.add_argument("--output-root", type=Path,
                    help="Override the cache_dir in the YAML config")
    args = ap.parse_args()

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)

    cache_dir = Path(expand_env_like(raw_cfg["cache_dir"]))
    if args.output_root is not None:
        cache_dir = args.output_root.expanduser().resolve()
    year = int(args.year if args.year is not None else raw_cfg["year"])
    ep_cfg: Dict[str, Any] = raw_cfg.get("endpoints", {})
    sleep_sec = float(raw_cfg.get("sleep_seconds", 0.25))
    division = raw_cfg.get("division", "fbs")
    snake_case = bool(raw_cfg.get("snake_case", False))
    refresh = args.refresh or bool(raw_cfg.get("refresh", False))

    client = CFBDClient()

    # Build single-season, one-call tasks
    Task = Tuple[str, int, str, Path, callable]  # (name, year, subdir, out_path, fetch_fn)
    tasks: List[Task] = []

    def add(flag: str, name: str, subdir: str, fn_fetch):
        """Run the add step and return its normalized result."""
        if ep_cfg.get(flag, False):
            outp = cache_dir / subdir / f"{year}.parquet"
            tasks.append((name, year, subdir, outp, fn_fetch))
            
    add("teams_fbs",         "teams_fbs",         "teams_fbs",         lambda: fetch_teams_fbs(client, year))
    add("ratings_sp",        "ratings_sp",        "ratings_sp",        lambda: fetch_ratings_sp(client, year))
    add("ratings_fpi",       "ratings_fpi",       "ratings_fpi",       lambda: fetch_ratings_fpi(client, year))
    add("ratings_elo",       "ratings_elo",       "ratings_elo",       lambda: fetch_ratings_elo(client, year))
    add("stats_basic",       "stats_basic",       "stats_basic",       lambda: fetch_stats_basic(client, year))
    add("stats_advanced",    "stats_advanced",    "stats_advanced",    lambda: fetch_stats_adv(client, year))
    add("games",             "games",             "games",             lambda: fetch_games(client, year, division))
    add("coaches",           "coaches",           "coaches",           lambda: fetch_coaches(client, year))
    add("roster",            "roster",            "roster",            lambda: fetch_roster(client, year))
    add("recruiting_players","recruiting_players","recruiting_players",lambda: fetch_recruit_players(client, year))
    add("recruiting_teams",  "recruiting_teams",  "recruiting_teams",  lambda: fetch_recruit_teams(client, year))
    add("talent",            "talent",            "talent",            lambda: fetch_talent(client, year))
    add("lines",             "lines",             "lines",             lambda: fetch_lines(client, year))
    add("pregame_wp",        "pregame_wp",        "pregame_wp",        lambda: fetch_pregame_wp(client, year))
    add("returning",         "returning",         "returning",         lambda: fetch_returning(client, year))
    add("ratings_srs",       "ratings_srs",       "ratings_srs",       lambda: fetch_ratings_srs(client, year))
    add("teams_ats",         "teams_ats",         "teams_ats",         lambda: fetch_teams_ats(client, year))
    add("ppa_teams",         "ppa_teams",         "ppa_teams",         lambda: fetch_ppa_teams(client, year))
    add("havoc_game",        "havoc_game",        "havoc_game",        lambda: fetch_havoc_game(client, year))
    add("rankings",          "rankings",          "rankings",          lambda: fetch_rankings(client, year))

    add("stats_advanced_game", "stats_advanced_game", "stats_advanced_game", lambda: fetch_stats_advanced_game(client, year))
    add("game_team_stats",     "game_team_stats",     "game_team_stats", lambda: fetch_stats_basic_game(client, year))
    add("ppa_games",           "ppa_games",           "ppa_games", lambda: fetch_ppa_games(client, year))
    add("venue",               "venue",               "venue", lambda: fetch_venues(client, year))
    add("weather",             "weather",             "weather", lambda: fetch_weather(client, year))


    # Overall progress (counts endpoints)
    overall = tqdm(total=len(tasks), desc=f"CFBD {year} (endpoints)", unit="endpoint")

    ok, cached, fail = 0, 0, 0
    rows_written = 0

    for name, yr, subdir, outp, fn_fetch in tasks:
        # Per-endpoint progress bar (single call)
        with tqdm(total=1, desc=f"{name}", unit="call", leave=False) as pbar:
            try:
                if outp.exists() and not refresh:
                    cached += 1
                    status = "cached"
                    nrows = 0
                else:
                    df = fn_fetch()        # <-- exactly one API call inside
                    nrows = 0 if df is None else len(df)
                    write_parquet(df, outp, snake=snake_case)
                    status = "fetched"
                    ok += 1
                    rows_written += nrows
                    time.sleep(sleep_sec)

                pbar.update(1)
                pbar.set_postfix_str(f"{status}; rows={nrows}; out={outp.name}")
            except Exception as e:
                fail += 1
                pbar.set_postfix_str("error")
                tqdm.write(f"[ERROR] {name} {yr}: {e}")

        overall.update(1)

    overall.close()

    print("\n========== SUMMARY ==========")
    print(f"Year: {year}")
    print(f"Endpoints total: {len(tasks)}")
    print(f"OK: {ok} | Cached: {cached} | Failed: {fail}")
    print(f"Rows written (sum): {rows_written}")
    print(f"Raw cache dir: {str(cache_dir)}")
    print(f"Total API calls performed: {client.api_calls}")  # excludes cache hits
    print("=============================\n")

if __name__ == "__main__":
    main()
