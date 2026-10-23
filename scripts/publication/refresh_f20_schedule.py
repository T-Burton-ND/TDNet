#!/usr/bin/env python3
"""Archive one current CFBD schedule week for F20 cutoff checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.f20.capture import _write_once


def _key() -> str:
    if os.environ.get("CFBD_API_KEY"):
        return os.environ["CFBD_API_KEY"]
    path = ROOT / ".env"
    if not path.is_file():
        raise OSError("CFBD API key is unavailable")
    for line in path.read_text().splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "CFBD_API_KEY":
            return value.strip().strip("\"'")
    raise OSError("CFBD API key is unavailable")


def project_schedule(raw: bytes, week: int) -> pd.DataFrame:
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError("CFBD week schedule is empty or invalid")
    rows = []
    for game in payload:
        if int(game["season"]) != 2026 or int(game["week"]) != week:
            raise ValueError("CFBD returned a game outside the requested week")
        if game["seasonType"] != "regular":
            continue
        if game["homeClassification"] != "fbs" or game["awayClassification"] != "fbs":
            continue
        rows.append({"id": int(game["id"]), "season": 2026, "week": week,
                     "season_type": "regular", "start_date": game["startDate"],
                     "start_time_t_b_d": bool(game.get("startTimeTBD", False)),
                     "home_team": game["homeTeam"], "away_team": game["awayTeam"],
                     "home_classification": "fbs", "away_classification": "fbs"})
    if not rows:
        raise ValueError("CFBD FBS week has no games")
    frame = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    if frame.id.duplicated().any():
        raise ValueError("CFBD FBS week has no unique games")
    kickoff = pd.to_datetime(frame.start_date, utc=True, errors="coerce")
    if kickoff.isna().any():
        raise ValueError("CFBD FBS game has no parseable kickoff")
    return frame


def fetch(week: int) -> tuple[bytes, datetime, str]:
    url = f"https://api.collegefootballdata.com/games?year=2026&week={week}&seasonType=regular&classification=fbs"
    error = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"Authorization": f"Bearer {_key()}",
                                            "User-Agent": "TDNet-F20-research/1.0"})
            with urlopen(request, timeout=20) as response:
                raw = response.read(2_000_001)
                now = datetime.now(UTC)
                if response.status != 200 or len(raw) > 2_000_000 or not raw:
                    raise ValueError("CFBD schedule response failed size/status check")
                return raw, now, url
        except (OSError, ValueError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"CFBD schedule refresh failed after three attempts: {type(error).__name__}")


def refresh(archive: Path, week: int) -> dict:
    raw, captured_at, url = fetch(week)
    frame = project_schedule(raw, week)
    raw_sha = hashlib.sha256(raw).hexdigest()
    stamp = captured_at.strftime("%Y%m%dT%H%M%S%fZ")
    root = archive / "schedules"
    raw_path = root / f"{raw_sha}.json"
    schedule_path = root / f"{raw_sha}.{stamp}.parquet"
    content = frame.to_parquet(index=False)
    schedule_sha = hashlib.sha256(content).hexdigest()
    receipt = {"status": "captured", "source_id": "cfbd_games_week",
               "source_url": url, "captured_at_utc": captured_at.isoformat(),
               "season": 2026, "week": week, "raw_sha256": raw_sha,
               "raw_file": str(raw_path), "schedule_sha256": schedule_sha,
               "schedule_file": str(schedule_path), "fbs_regular_games": len(frame),
               "unconfirmed_kickoffs": int(frame.start_time_t_b_d.sum()),
               "projection_excludes_scores": True}
    _write_once(raw_path, raw)
    _write_once(schedule_path, content)
    _write_once(root / f"{raw_sha}.{stamp}.receipt.json",
                (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/private/f20/prospective")
    args = parser.parse_args()
    print(json.dumps(refresh(args.archive, args.week), indent=2))


if __name__ == "__main__":
    main()
