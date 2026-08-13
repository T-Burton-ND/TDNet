"""src.gridiron_ml.pipeline.build_full_pipeline.

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

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import (
    CFBDClient,
    expand_env_like,
    fetch_coaches,
    fetch_games,
    fetch_havoc_game,
    fetch_lines,
    fetch_ppa_games,
    fetch_ppa_teams,
    fetch_pregame_wp,
    fetch_rankings,
    fetch_ratings_elo,
    fetch_ratings_fpi,
    fetch_ratings_sp,
    fetch_ratings_srs,
    fetch_recruit_players,
    fetch_recruit_teams,
    fetch_returning,
    fetch_roster,
    fetch_stats_adv,
    fetch_stats_advanced_game,
    fetch_stats_basic,
    fetch_stats_basic_game,
    fetch_talent,
    fetch_teams_ats,
    fetch_teams_fbs,
    fetch_venues,
    fetch_weather,
    write_parquet,
)
from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.pipeline.build_team_game_table import build_season as build_team_game_table_season


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RawFetchTaskSpec:
    """Represent the RawFetchTaskSpec component and its local behavior."""
    subdir: str
    fetcher: Callable[[CFBDClient, int, str], pd.DataFrame]


RAW_FETCH_TASKS: dict[str, RawFetchTaskSpec] = {
    "games": RawFetchTaskSpec("games", lambda client, year, division: fetch_games(client, year, division)),
    "game_team_stats": RawFetchTaskSpec(
        "game_team_stats",
        lambda client, year, division: fetch_stats_basic_game(client, year),
    ),
    "stats_advanced_game": RawFetchTaskSpec(
        "stats_advanced_game",
        lambda client, year, division: fetch_stats_advanced_game(client, year),
    ),
    "havoc_game": RawFetchTaskSpec("havoc_game", lambda client, year, division: fetch_havoc_game(client, year)),
    "lines": RawFetchTaskSpec("lines", lambda client, year, division: fetch_lines(client, year)),
    "pregame_wp": RawFetchTaskSpec("pregame_wp", lambda client, year, division: fetch_pregame_wp(client, year)),
    "ppa_games": RawFetchTaskSpec("ppa_games", lambda client, year, division: fetch_ppa_games(client, year)),
    "talent": RawFetchTaskSpec("talent", lambda client, year, division: fetch_talent(client, year)),
    "returning": RawFetchTaskSpec("returning", lambda client, year, division: fetch_returning(client, year)),
    "recruiting_teams": RawFetchTaskSpec(
        "recruiting_teams",
        lambda client, year, division: fetch_recruit_teams(client, year),
    ),
    "coaches": RawFetchTaskSpec("coaches", lambda client, year, division: fetch_coaches(client, year)),
    "teams_fbs": RawFetchTaskSpec("teams_fbs", lambda client, year, division: fetch_teams_fbs(client, year)),
    "venue": RawFetchTaskSpec("venue", lambda client, year, division: fetch_venues(client, year)),
    "ratings_sp": RawFetchTaskSpec("ratings_sp", lambda client, year, division: fetch_ratings_sp(client, year)),
    "ratings_fpi": RawFetchTaskSpec("ratings_fpi", lambda client, year, division: fetch_ratings_fpi(client, year)),
    "ratings_elo": RawFetchTaskSpec("ratings_elo", lambda client, year, division: fetch_ratings_elo(client, year)),
    "ratings_srs": RawFetchTaskSpec("ratings_srs", lambda client, year, division: fetch_ratings_srs(client, year)),
    "stats_basic": RawFetchTaskSpec("stats_basic", lambda client, year, division: fetch_stats_basic(client, year)),
    "stats_advanced": RawFetchTaskSpec("stats_advanced", lambda client, year, division: fetch_stats_adv(client, year)),
    "roster": RawFetchTaskSpec("roster", lambda client, year, division: fetch_roster(client, year)),
    "recruiting_players": RawFetchTaskSpec(
        "recruiting_players",
        lambda client, year, division: fetch_recruit_players(client, year),
    ),
    "teams_ats": RawFetchTaskSpec("teams_ats", lambda client, year, division: fetch_teams_ats(client, year)),
    "ppa_teams": RawFetchTaskSpec("ppa_teams", lambda client, year, division: fetch_ppa_teams(client, year)),
    "rankings": RawFetchTaskSpec("rankings", lambda client, year, division: fetch_rankings(client, year)),
    "weather": RawFetchTaskSpec("weather", lambda client, year, division: fetch_weather(client, year)),
}

PIPELINE_REQUIRED_RAW_ENDPOINTS = (
    "games",
    "game_team_stats",
    "stats_advanced_game",
    "havoc_game",
    "lines",
    "pregame_wp",
    "ppa_games",
    "talent",
    "returning",
    "recruiting_teams",
    "coaches",
    "teams_fbs",
    "venue",
)


def build_fingerprint_v0(*, root: Path, overwrite: bool, postseason: bool, team_game_tables_dir: Path) -> Path:
    """Run the build_fingerprint_v0 step and return its normalized result."""
    return Fingerprints(
        version=0,
        postseason=postseason,
        root=root,
        team_game_tables_dir=team_game_tables_dir,
    ).build(overwrite=overwrite)


def build_fingerprint_v1(*, root: Path, overwrite: bool, postseason: bool, team_game_tables_dir: Path) -> Path:
    """Run the build_fingerprint_v1 step and return its normalized result."""
    return Fingerprints(
        version=1,
        postseason=postseason,
        root=root,
        team_game_tables_dir=team_game_tables_dir,
    ).build(overwrite=overwrite)


def build_fingerprint_v2(*, root: Path, overwrite: bool, postseason: bool, team_game_tables_dir: Path) -> Path:
    """Run the build_fingerprint_v2 step and return its normalized result."""
    return Fingerprints(
        version=2,
        postseason=postseason,
        root=root,
        team_game_tables_dir=team_game_tables_dir,
    ).build(overwrite=overwrite)


FINGERPRINT_BUILDERS: dict[int, Callable[..., Path]] = {
    0: build_fingerprint_v0,
    1: build_fingerprint_v1,
    2: build_fingerprint_v2,
}


def resolve_path(value: str | Path | None, *, base: Path) -> Path:
    """Run the resolve_path step and return its normalized result."""
    if value is None:
        return base

    raw = expand_env_like(str(value))
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def resolve_years(config: dict[str, Any]) -> list[int]:
    """Run the resolve_years step and return its normalized result."""
    years_cfg = config.get("years")

    if isinstance(years_cfg, list):
        years = [int(year) for year in years_cfg]
    elif isinstance(years_cfg, dict):
        if years_cfg.get("values") is not None:
            years = [int(year) for year in years_cfg["values"]]
        else:
            start = years_cfg.get("start")
            end = years_cfg.get("end")
            if start is None or end is None:
                raise ValueError("`years` config must define either `values` or both `start` and `end`.")
            years = list(range(int(start), int(end) + 1))
    else:
        start = config.get("start_year")
        end = config.get("end_year")
        if start is None or end is None:
            raise ValueError("Config must define `years` or top-level `start_year`/`end_year`.")
        years = list(range(int(start), int(end) + 1))

    unique_years = sorted({int(year) for year in years})
    if not unique_years:
        raise ValueError("No seasons were resolved from the pipeline config.")
    return unique_years


def default_raw_endpoints() -> dict[str, bool]:
    """Run the default_raw_endpoints step and return its normalized result."""
    flags = {name: False for name in RAW_FETCH_TASKS}
    for name in PIPELINE_REQUIRED_RAW_ENDPOINTS:
        flags[name] = True
    return flags


def normalize_raw_endpoint_flags(endpoint_cfg: dict[str, Any] | None) -> dict[str, bool]:
    """Run the normalize_raw_endpoint_flags step and return its normalized result."""
    flags = default_raw_endpoints()
    if endpoint_cfg is None:
        return flags

    unknown = sorted(set(endpoint_cfg) - set(RAW_FETCH_TASKS))
    if unknown:
        raise ValueError(
            "Unsupported raw fetch endpoints in config: "
            + ", ".join(unknown)
            + "."
        )

    for name, enabled in endpoint_cfg.items():
        flags[name] = bool(enabled)
    return flags


def parse_fingerprint_version_key(value: int | str) -> int:
    """Run the parse_fingerprint_version_key step and return its normalized result."""
    if isinstance(value, int):
        return value

    text = str(value).strip().lower()
    if text.startswith("v"):
        text = text[1:]
    return int(text)


def enabled_fingerprint_versions(config: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """Run the enabled_fingerprint_versions step and return its normalized result."""
    versions_cfg = config.get("versions", {}) or {}
    enabled: list[tuple[int, dict[str, Any]]] = []

    for raw_key, raw_value in versions_cfg.items():
        version = parse_fingerprint_version_key(raw_key)
        cfg = dict(raw_value or {})
        if cfg.get("enabled", False):
            enabled.append((version, cfg))

    enabled.sort(key=lambda item: item[0])
    return enabled


def load_pipeline_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Run the load_pipeline_config step and return its normalized result."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config, path


def run_raw_fetch_stage(
    *,
    years: list[int],
    raw_cache_dir: Path,
    raw_fetch_cfg: dict[str, Any],
) -> dict[str, int]:
    """Run the run_raw_fetch_stage step and return its normalized result."""
    division = str(raw_fetch_cfg.get("division", "fbs"))
    refresh = bool(raw_fetch_cfg.get("refresh", False))
    snake_case = bool(raw_fetch_cfg.get("snake_case", True))
    sleep_seconds = float(raw_fetch_cfg.get("sleep_seconds", 0.25))
    api_key_env = str(raw_fetch_cfg.get("api_key_env", "CFBD_API_KEY"))
    timeout = int(raw_fetch_cfg.get("timeout", 60))
    endpoint_flags = normalize_raw_endpoint_flags(raw_fetch_cfg.get("endpoints"))
    enabled_endpoints = [name for name, enabled in endpoint_flags.items() if enabled]

    if not enabled_endpoints:
        print("[pipeline.raw] No endpoints enabled; skipping raw fetch stage.")
        return {"fetched": 0, "cached": 0, "years": len(years), "api_calls": 0}

    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    client = CFBDClient(api_key_env=api_key_env, timeout=timeout)
    fetched = 0
    cached = 0

    print(f"[pipeline.raw] Fetching {len(enabled_endpoints)} endpoints for seasons {years[0]}-{years[-1]}")
    for year in years:
        print(f"[pipeline.raw] Season {year}")
        for endpoint_name in enabled_endpoints:
            spec = RAW_FETCH_TASKS[endpoint_name]
            out_path = raw_cache_dir / spec.subdir / f"{year}.parquet"

            if out_path.exists() and not refresh:
                cached += 1
                print(f"[pipeline.raw] cached  {endpoint_name:<20} -> {out_path}")
                continue

            print(f"[pipeline.raw] fetching {endpoint_name:<20} -> {out_path}")
            frame = spec.fetcher(client, year, division)
            write_parquet(frame, out_path, snake=snake_case)
            fetched += 1

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return {"fetched": fetched, "cached": cached, "years": len(years), "api_calls": client.api_calls}


def run_team_game_table_stage(
    *,
    years: list[int],
    raw_cache_dir: Path,
    team_game_tables_dir: Path,
    table_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run the run_team_game_table_stage step and return its normalized result."""
    division = str(table_cfg.get("division", "fbs"))
    week = table_cfg.get("week")
    output_format = str(table_cfg.get("output_format", "parquet"))

    team_game_tables_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline.tables] Building team-game tables in {team_game_tables_dir}")
    for year in years:
        build_team_game_table_season(
            cache_dir=raw_cache_dir,
            out_dir=team_game_tables_dir,
            year=year,
            division=division,
            week=None if week is None else int(week),
            output_format=output_format,
        )

    return {"years": len(years), "week": week, "output_format": output_format}


def run_fingerprint_stage(
    *,
    root: Path,
    team_game_tables_dir: Path,
    fingerprints_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run the run_fingerprint_stage step and return its normalized result."""
    enabled_versions = enabled_fingerprint_versions(fingerprints_cfg)
    if not enabled_versions:
        print("[pipeline.fingerprints] No fingerprint versions enabled; skipping stage.")
        return []

    postseason = bool(fingerprints_cfg.get("postseason", False))
    global_overwrite = bool(fingerprints_cfg.get("overwrite", False))
    summaries: list[dict[str, Any]] = []

    print(
        "[pipeline.fingerprints] Building versions:",
        ", ".join(f"v{version}" for version, _ in enabled_versions),
    )
    for version, version_cfg in enabled_versions:
        builder = FINGERPRINT_BUILDERS.get(version)
        if builder is None:
            supported = ", ".join(f"v{supported_version}" for supported_version in sorted(FINGERPRINT_BUILDERS))
            raise ValueError(f"Fingerprint version v{version} is not wired in this runner. Supported versions: {supported}.")

        overwrite = bool(version_cfg.get("overwrite", global_overwrite))
        path = builder(
            root=root,
            overwrite=overwrite,
            postseason=postseason,
            team_game_tables_dir=team_game_tables_dir,
        )
        print(f"[pipeline.fingerprints] built v{version} -> {path}")
        summaries.append({"version": version, "path": str(path), "overwrite": overwrite})

    return summaries


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    """Run the run_pipeline step and return its normalized result."""
    config, config_path = load_pipeline_config(config_path)
    config_dir = config_path.parent

    root_value = config.get("root")
    root = resolve_path(root_value, base=config_dir) if root_value is not None else REPO_ROOT

    paths_cfg = config.get("paths", {}) or {}
    raw_cache_dir = resolve_path(paths_cfg.get("raw_cache_dir", "data/raw/cfbd/v2"), base=root)
    team_game_tables_dir = resolve_path(paths_cfg.get("team_game_tables_dir", "data/team_game_tables"), base=root)

    years = resolve_years(config)
    raw_fetch_cfg = dict(config.get("raw_fetch", {}) or {})
    table_cfg = dict(config.get("team_game_tables", {}) or {})
    fingerprints_cfg = dict(config.get("fingerprints", {}) or {})

    if (
        bool(fingerprints_cfg.get("enabled", True))
        and bool(table_cfg.get("enabled", True))
        and table_cfg.get("week") is not None
    ):
        raise ValueError(
            "Fingerprint builds require full-season team-game tables. "
            "Set `team_game_tables.week` to null when `fingerprints.enabled` is true."
        )

    summary: dict[str, Any] = {
        "root": str(root),
        "years": years,
        "raw_cache_dir": str(raw_cache_dir),
        "team_game_tables_dir": str(team_game_tables_dir),
    }

    print("=" * 72)
    print("TDNet Full Pipeline")
    print(f"Config    : {config_path}")
    print(f"Root      : {root}")
    print(f"Years     : {years[0]} -> {years[-1]} ({len(years)} seasons)")
    print(f"Raw cache : {raw_cache_dir}")
    print(f"Tables dir: {team_game_tables_dir}")
    print("=" * 72)

    if bool(raw_fetch_cfg.get("enabled", True)):
        summary["raw_fetch"] = run_raw_fetch_stage(
            years=years,
            raw_cache_dir=raw_cache_dir,
            raw_fetch_cfg=raw_fetch_cfg,
        )
    else:
        print("[pipeline.raw] Stage disabled.")

    if bool(table_cfg.get("enabled", True)):
        summary["team_game_tables"] = run_team_game_table_stage(
            years=years,
            raw_cache_dir=raw_cache_dir,
            team_game_tables_dir=team_game_tables_dir,
            table_cfg=table_cfg,
        )
    else:
        print("[pipeline.tables] Stage disabled.")

    if bool(fingerprints_cfg.get("enabled", True)):
        summary["fingerprints"] = run_fingerprint_stage(
            root=root,
            team_game_tables_dir=team_game_tables_dir,
            fingerprints_cfg=fingerprints_cfg,
        )
    else:
        print("[pipeline.fingerprints] Stage disabled.")

    print("=" * 72)
    print("Pipeline complete.")
    print("=" * 72)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the main step and return its normalized result."""
    parser = argparse.ArgumentParser(
        description="Run the TDNet raw-data -> team-game-table -> fingerprint pipeline from one YAML config.",
    )
    parser.add_argument("config", type=Path, help="Path to the pipeline YAML config.")
    args = parser.parse_args(argv)
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
