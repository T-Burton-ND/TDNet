#!/usr/bin/env python3
"""Acquire only missing Week 6–7 context for a Week 8 expanded-roster rebuild.

Requests use the shared CFBD budget. This source cache is retrospective at the
October 21 acquisition time; subsequent feature builders must still apply their
pregame availability rules and record the provider-timing limitation.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from whatif_2026_acquire import ARTIFACTS, _digest, _request, _setup, _write_json  # noqa: E402

OUT = ARTIFACTS / "week08_context_acquisition"


def requests() -> list[tuple[str, dict]]:
    games = pd.read_parquet(ROOT / "data/raw/cfbd/v2/games/2026.parquet")
    games = games.loc[
        games.week.isin([6, 7])
        & games.season_type.astype(str).str.lower().eq("regular")
        & games.completed.fillna(False).astype(bool)
        & (games.home_classification.astype(str).str.lower().eq("fbs")
           | games.away_classification.astype(str).str.lower().eq("fbs"))
    ]
    if games.empty or games.id.duplicated().any():
        raise ValueError("Missing or duplicate completed Week 6–7 FBS context games")
    items = []
    for week in (6, 7):
        base = {"year": 2026, "week": week, "seasonType": "regular", "classification": "fbs"}
        items.extend((endpoint, base.copy()) for endpoint in ("/plays", "/drives", "/games/players"))
        items.append(("/ppa/players/games", {"year": 2026, "week": week, "seasonType": "regular"}))
    items.extend(("/plays/stats", {"gameId": int(game)}) for game in sorted(games.id.astype(int)))
    return items


def main() -> None:
    items = requests()
    client, budget = _setup()
    quota = client.get_json("/info", {}, max_retries=0)
    if not isinstance(quota, dict) or not isinstance(quota.get("remainingCalls"), int):
        raise ValueError("Provider quota unavailable")
    if quota["remainingCalls"] < 1000 + len(items) or budget.status()["reserved"] + len(items) > 20_000:
        raise RuntimeError("Insufficient provider quota or shared call budget")
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = []
    for index, (endpoint, params) in enumerate(items, 1):
        token = hashlib.sha256((endpoint + json.dumps(params, sort_keys=True)).encode()).hexdigest()
        receipt_path = OUT / f"{token}.json"
        prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if prior is not None and prior.get("status") == "success_complete":
            cache = ROOT / prior["cache_path"]
            if cache.exists() and _digest(cache) == prior["sha256"]:
                receipts.append(prior)
                continue
        item = _request(client, endpoint, params)
        item["week08_context"] = True
        _write_json(receipt_path, item)
        receipts.append(item)
        if index % 10 == 0:
            print(f"Week 8 expanded context: {index}/{len(items)} requests", flush=True)
        time.sleep(0.25)
    summary = {
        "status": "complete", "requests": len(items), "provider_calls": client.api_calls,
        "source_weeks": [6, 7], "obtained_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_field_level_availability_verified": False,
        "receipts": receipts,
    }
    _write_json(OUT / "manifest.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "requests", "provider_calls")}, indent=2))


if __name__ == "__main__":
    main()
