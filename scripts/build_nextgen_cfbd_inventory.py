#!/usr/bin/env python3
"""Audit CFBD OpenAPI routes into an explicit next-generation endpoint inventory.

This reads a previously downloaded public schema; it never sends an API request.
Update the explicit classification map when CFBD adds a route, then rerun it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CLASSES = {
    "current_feature_eligible": {
        "/games", "/games/teams", "/games/players", "/drives", "/plays",
        "/roster", "/talent", "/coaches", "/coaches/seasons", "/coaches/tenures",
        "/player/usage", "/player/returning", "/player/portal",
        "/recruiting/players", "/recruiting/teams", "/stats/player/season",
        "/stats/player/success/game", "/stats/game/advanced", "/stats/game/havoc",
        "/ppa/games", "/ppa/players/games", "/ppa/players/season",
    },
    "future_feature_candidate": {
        "/games/media", "/games/weather", "/records", "/calendar",
        "/plays/types", "/plays/stats", "/plays/stats/types",
        "/passing/plays", "/passing/players/season", "/passing/players/games",
        "/passing/teams/season", "/passing/teams/games",
        "/rushing/plays", "/rushing/players/season", "/rushing/players/games",
        "/rushing/teams/season", "/rushing/teams/games",
        "/teams", "/teams/fbs", "/conferences", "/conferences/changes",
        "/conferences/affiliations", "/venues", "/recruiting/groups",
        "/rankings", "/stats/categories", "/stats/player/success",
        "/draft/teams", "/draft/positions", "/draft/picks", "/metrics/fg/ep",
    },
    "evaluation_sidecar": {"/lines", "/teams/ats", "/metrics/wp/pregame"},
    "cache_only_temporal_risk": {
        "/teams/season/overview", "/teams/matchup", "/player/season/overview",
        "/ratings/core", "/ratings/sp", "/ratings/sp/conferences",
        "/ratings/srs", "/ratings/srs/expanded", "/ratings/elo", "/ratings/fpi",
        "/ppa/predicted", "/ppa/teams", "/stats/season",
        "/stats/season/advanced", "/wepa/team/season",
        "/wepa/players/passing", "/wepa/players/rushing", "/wepa/players/kicking",
        "/metrics/wp", "/game/box/advanced",
    },
    "skip_irrelevant": {
        "/playoffs/cfp", "/playoffs/cfp/participants", "/playoffs/cfp/games",
        "/scoreboard", "/live/plays", "/player/search", "/coaches/profile",
        "/info", "/info/usage",
    },
}

YEAR_WEEK = {"/plays", "/games/teams", "/games/players", "/stats/player/success/game",
             "/passing/plays", "/rushing/plays"}
GAME = {"/plays/stats", "/game/box/advanced"}
YEAR_TEAM = {"/teams/season/overview"}
YEAR_PLAYER = {"/player/season/overview"}
STATIC = {"/venues", "/plays/types", "/plays/stats/types", "/stats/categories",
          "/draft/teams", "/draft/positions", "/metrics/fg/ep"}
YEAR_RANGE = {"/recruiting/groups"}
ALIASES = {
    "/games": "games", "/games/teams": "game_team_stats", "/stats/game/advanced": "stats_advanced_game",
    "/stats/game/havoc": "havoc_game", "/ppa/games": "ppa_games", "/coaches": "coaches",
    "/roster": "roster", "/recruiting/players": "recruiting_players",
    "/recruiting/teams": "recruiting_teams", "/talent": "talent",
    "/player/returning": "returning", "/games/weather": "weather", "/venues": "venue",
    "/teams/fbs": "teams_fbs", "/lines": "lines", "/teams/ats": "teams_ats",
    "/metrics/wp/pregame": "pregame_wp", "/rankings": "rankings",
    "/ratings/sp": "ratings_sp", "/ratings/fpi": "ratings_fpi",
    "/ratings/elo": "ratings_elo", "/ratings/srs": "ratings_srs",
    "/ppa/teams": "ppa_teams", "/stats/season": "stats_basic",
    "/stats/season/advanced": "stats_advanced",
}
EARLIEST = {
    "/plays/stats": 2012, "/player/usage": 2013,
    "/player/returning": 2014, "/talent": 2015, "/player/portal": 2021,
    "/passing/plays": 2025, "/rushing/plays": 2025,
    "/passing/players/season": 2025, "/passing/players/games": 2025,
    "/passing/teams/season": 2025, "/passing/teams/games": 2025,
    "/rushing/players/season": 2025, "/rushing/players/games": 2025,
    "/rushing/teams/season": 2025, "/rushing/teams/games": 2025,
}
SEASON_AGGREGATES = {
    "/stats/player/season", "/stats/player/success", "/ppa/players/season",
    "/passing/players/season", "/passing/teams/season",
    "/rushing/players/season", "/rushing/teams/season",
    "/records", "/teams/season/overview", "/player/season/overview",
}
SPECIAL_PARAMS = {
    "/plays": {"seasonType": "regular", "classification": "fbs"},
    "/drives": {"seasonType": "regular", "classification": "fbs"},
    "/games": {"seasonType": "regular"},
    "/games/teams": {"seasonType": "regular", "classification": "fbs"},
    "/games/players": {"seasonType": "regular", "classification": "fbs"},
    "/passing/plays": {"seasonType": "regular", "classification": "fbs"},
    "/rushing/plays": {"seasonType": "regular", "classification": "fbs"},
    "/stats/player/success/game": {"seasonType": "regular"},
    "/rankings": {"seasonType": "regular"},
    "/stats/game/havoc": {"seasonType": "regular"},
    "/ppa/games": {"seasonType": "regular"},
}
SIDE_CAR_PARAMS = {"/lines": {"seasonType": "regular"},
                   "/metrics/wp/pregame": {"seasonType": "regular"}}


def classify(path: str) -> str:
    matches = [name for name, paths in CLASSES.items() if path in paths]
    if len(matches) != 1:
        raise ValueError(f"CFBD route {path} has {len(matches)} classes: {matches}")
    return matches[0]


def partition_for(path: str, query_names: set[str]) -> str:
    if path in YEAR_WEEK:
        return "year_week"
    if path in GAME:
        return "game"
    if path in YEAR_TEAM:
        return "year_team"
    if path in YEAR_PLAYER:
        return "year_player"
    if path in STATIC:
        return "static_once"
    if path in YEAR_RANGE:
        return "year_range"
    if "year" in query_names:
        return "year"
    return "manual_or_skip"


def stage_for(path: str, classification: str) -> str:
    if classification == "skip_irrelevant":
        return "skip"
    if path == "/plays/stats":
        return "E"
    if path in {"/plays", "/drives", "/games/teams", "/passing/plays", "/rushing/plays"}:
        return "C"
    if path.startswith(("/player/", "/stats/player/", "/ppa/players/", "/games/players",
                        "/passing/players/", "/rushing/players/", "/wepa/players/")):
        return "D"
    return "B"


def default_acquire(path: str, classification: str, partition: str) -> bool:
    if path == "/plays/stats" or partition in {"year_team", "year_player", "game", "manual_or_skip"}:
        return False
    return classification in {"current_feature_eligible", "future_feature_candidate",
                              "evaluation_sidecar", "cache_only_temporal_risk"}


def build_inventory(schema_path: Path) -> dict:
    raw = schema_path.read_bytes()
    schema = json.loads(raw)
    paths = {path: operation["get"] for path, operation in schema["paths"].items() if "get" in operation}
    classified = set().union(*CLASSES.values())
    if set(paths) != classified:
        raise ValueError(f"OpenAPI path mismatch; new={sorted(set(paths)-classified)}, removed={sorted(classified-set(paths))}")
    endpoints = []
    for path, operation in sorted(paths.items()):
        parameters = [p for p in operation.get("parameters", []) if p.get("in") == "query"]
        names = {p["name"] for p in parameters}
        classification = classify(path)
        partition = partition_for(path, names)
        params = SPECIAL_PARAMS.get(path, SIDE_CAR_PARAMS.get(path, {}))
        if not set(params) <= names:
            raise ValueError(f"Invalid planned query parameters for {path}: {params}")
        endpoints.append({
            "endpoint": path,
            "classification": classification,
            "stage": stage_for(path, classification),
            "partition_strategy": partition,
            "default_acquire": default_acquire(path, classification, partition),
            "required_query_parameters": sorted(p["name"] for p in parameters if p.get("required")),
            "legal_query_parameters": sorted(names),
            "fixed_parameters": params,
            "earliest_year": EARLIEST.get(path, 2010),
            "existing_cache_alias": ALIASES.get(path),
            "response_row_cap": 2000 if path == "/plays/stats" else None,
            "temporal_rule": ("never_predictor" if classification == "evaluation_sidecar" or path.startswith("/metrics/wp") else
                              "prior_completed_season_or_reconstruct_asof" if path in SEASON_AGGREGATES else
                              "verify_asof_before_predictor" if classification == "cache_only_temporal_risk" else
                              "prior_completed_game_or_pregame_snapshot"),
        })
    return {
        "version": 1,
        "source": "https://api.collegefootballdata.com/api/5.30.0/cfbd-openapi.json",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_path_count": len(paths),
        "data_availability_source": "https://api.collegefootballdata.com/data-availability",
        "classes": list(CLASSES),
        "endpoints": endpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json"))
    args = parser.parse_args()
    inventory = build_inventory(args.schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"Classified {len(inventory['endpoints'])} CFBD GET routes from {inventory['source']}")


if __name__ == "__main__":
    main()
