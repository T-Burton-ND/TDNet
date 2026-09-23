#!/usr/bin/env python3
"""Refresh only the CFBD inputs needed for a postgame poll rebuild.

The normal Monday refresh intentionally captures the full prediction-time
snapshot.  A Sunday poll rebuild has a smaller contract: completed games,
four postgame statistical feeds, and the newest available poll snapshot.  The
games response may be reused from the certified results fetch, leaving exactly
five outbound calls for this command.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from gridiron_ml.cli._paths import project_root
from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import CFBDClient, write_parquet
from gridiron_ml.publication.bundles import sha256_file


ROOT = project_root()
EASTERN = ZoneInfo("America/New_York")


def _request(client: CFBDClient, path: str, params: dict) -> pd.DataFrame:
    """Return an endpoint response without flattening nested CFBD payloads."""
    return pd.DataFrame(client.get_json(path, params))


def _merge_week_cache(existing: pd.DataFrame, current: pd.DataFrame, *, week: int) -> pd.DataFrame:
    """Replace one provider week while retaining prior weekly observations."""
    if existing.empty or "week" not in current:
        return current.reset_index(drop=True)
    if "week" not in existing:
        return current.reset_index(drop=True)
    keep = ~pd.to_numeric(existing["week"], errors="coerce").eq(int(week))
    return pd.concat([existing.loc[keep], current], ignore_index=True, sort=False)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--provider-week",
        type=int,
        required=True,
        help="CFBD's week number for the completed publication slate.",
    )
    parser.add_argument(
        "--games-json",
        type=Path,
        required=True,
        help="Previously fetched complete /games response; reused without another call.",
    )
    parser.add_argument("--raw-cache", type=Path, default=ROOT / "data/raw/cfbd/v2")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-key-env", default="CFBD_API_KEY")
    args = parser.parse_args()

    manifest_path = args.output_root / "postgame_rank_input_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing a duplicate weekly postgame API refresh: {manifest_path} exists."
        )
    if not args.games_json.exists():
        raise FileNotFoundError(args.games_json)

    games_payload = json.loads(args.games_json.read_text(encoding="utf-8"))
    games = pd.DataFrame(games_payload)
    if games.empty:
        raise RuntimeError("The reused /games response is empty.")

    client = CFBDClient(api_key_env=args.api_key_env)
    common = {
        "year": int(args.season),
        "week": int(args.provider_week),
        "seasonType": "both",
    }
    endpoint_specs = {
        "game_team_stats": ("/games/teams", common),
        "stats_advanced_game": ("/stats/game/advanced", common),
        "havoc_game": ("/stats/game/havoc", common),
        "ppa_games": ("/ppa/games", common),
        "rankings": (
            "/rankings",
            {"year": int(args.season), "seasonType": "both"},
        ),
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, (path, params) in endpoint_specs.items():
        frame = _request(client, path, params)
        # /games/teams identifies each game but omits its season/week columns.
        # The canonical loader joins on those fields, so preserve the request
        # scope explicitly without spending another API call.
        if name == "game_team_stats" and not frame.empty:
            frame["season"] = int(args.season)
            frame["week"] = int(args.provider_week)
        if frame.empty and name != "rankings":
            raise RuntimeError(f"CFBD returned no rows for required endpoint {name}.")
        frames[name] = frame

    expected_calls = len(endpoint_specs)
    if client.api_calls != expected_calls:
        raise RuntimeError(
            f"Postgame refresh made {client.api_calls} API calls; expected {expected_calls}."
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_paths: dict[str, Path] = {}
    write_parquet(
        games,
        args.raw_cache / "games" / f"{args.season}.parquet",
        snake=True,
    )
    cache_paths["games"] = args.raw_cache / "games" / f"{args.season}.parquet"
    for name, frame in frames.items():
        path = args.raw_cache / name / f"{args.season}.parquet"
        cached = pd.read_parquet(path) if path.exists() and name != "rankings" else pd.DataFrame()
        combined = _merge_week_cache(cached, frame, week=args.provider_week)
        write_parquet(combined, path, snake=True)
        cache_paths[name] = path

    created = datetime.now(EASTERN).isoformat()
    manifest = {
        "schema": "tdnet-postgame-rank-input-refresh-v1",
        "created_at_eastern": created,
        "season": int(args.season),
        "provider_week": int(args.provider_week),
        "api_call_count": int(client.api_calls),
        "games_api_call_reused": True,
        "games_source": str(args.games_json.resolve()),
        "games_source_sha256": sha256_file(args.games_json),
        "requests": {
            name: {
                "path": path,
                "params": params,
                "response_rows": int(len(frames[name])),
                "cache_rows_after_merge": int(len(pd.read_parquet(cache_paths[name]))),
            }
            for name, (path, params) in endpoint_specs.items()
        },
        "cache_files": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in cache_paths.items()
        },
        "excluded_calls": [
            "lines (frozen pregame line retained for scoring)",
            "pregame win probability (not used by postgame ranking state)",
            "stable preseason context (talent, returning production, recruiting, coaches, teams, venues)",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
