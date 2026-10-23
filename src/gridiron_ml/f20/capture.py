"""Archive the live Prediction Tracker table without consulting future values."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

SOURCE_URL = "https://www.thepredictiontracker.com/predncaa.php"
PARSER_VERSION = "f20-tracker-html-v4"
EXPERT_COLUMNS = {
    "Team Rankings": "teamrankings",
    "Massey Ratings": "massey",
    "ESPN FPI": "espn_fpi",
    "FEI Projections": "fei",
    "Sagarin Pred.": "sagarin_pred",
}
MARKET_COLUMNS = {"Line Open": "opening_line", "Line": "current_line",
                  "Midweek Line": "midweek_line"}
ALIASES = {
    "appalachian st": "appalachian state", "arizona st": "arizona state",
    "arkansas st": "arkansas state", "central florida": "ucf",
    "sacramento st": "sacramento state",
    "ball st": "ball state", "boise st": "boise state",
    "central mich": "central michigan", "colorado st": "colorado state",
    "eastern mich": "eastern michigan",
    "florida intl": "florida international", "florida st": "florida state",
    "fresno st": "fresno state", "iowa st": "iowa state",
    "georgia st": "georgia state", "hawaii": "hawai i",
    "jacksonville st": "jacksonville state", "kansas st": "kansas state",
    "kennesaw st": "kennesaw state", "louisiana lafayette": "louisiana",
    "massachusetts": "umass", "miami ohio": "miami ohio",
    "miami oh": "miami ohio",
    "miami fla": "miami", "kent": "kent state",
    "michigan st": "michigan state", "middle tenn": "middle tennessee",
    "mississippi": "ole miss", "mississippi st": "mississippi state",
    "missouri st": "missouri state", "new mexico st": "new mexico state",
    "north carolina st": "nc state", "north dakota st": "north dakota state",
    "northern ill": "northern illinois", "ohio st": "ohio state",
    "nc st": "nc state", "northern illinois": "northern illinois",
    "oklahoma st": "oklahoma state", "penn st": "penn state",
    "san diego st": "san diego state", "san jose st": "san jos state",
    "san jose state": "san jos state",
    "sam houston st": "sam houston", "south florida": "usf",
    "southern miss": "southern mississippi", "texas san antonio": "utsa",
    "texas st": "texas state", "utah st": "utah state",
    "washington st": "washington state", "west va": "west virginia",
    "toledo": "toledo", "troy st": "troy", "western mich": "western michigan",
    "louisiana monroe": "ul monroe", "connecticut": "uconn",
}


def _key(value: str) -> str:
    value = re.sub(r"[^a-z0-9 ]", " ", value.casefold())
    value = " ".join(value.split())
    return ALIASES.get(value, value)


class _Tables(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict] = []
        self.table: dict | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.cell_tag = ""
        self.in_caption = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.table = {"headers": [], "rows": [], "caption": ""}
        elif self.table is not None and tag == "caption":
            self.in_caption = True
        elif self.table is not None and tag == "tr":
            self.row = []
        elif self.row is not None and tag in {"th", "td"}:
            self.cell, self.cell_tag = [], tag

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)
        elif self.in_caption and self.table is not None:
            self.table["caption"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            if self.cell_tag == "th":
                self.table["headers"].append(self.row)
            else:
                self.table["rows"].append(self.row)
            self.row = None
        elif tag == "caption":
            self.in_caption = False
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def parse_tracker(raw: bytes) -> tuple[str | None, list[dict]]:
    parser = _Tables()
    parser.feed(raw.decode("utf-8", errors="replace"))
    if len(parser.tables) < 2:
        raise ValueError("Tracker page has no expert table")
    overview, experts = parser.tables[:2]
    headers = max(experts["headers"], key=len)
    if not {"home", "road", *EXPERT_COLUMNS}.issubset(set(headers)):
        raise ValueError("Tracker expert header contract changed")
    overview_headers = max(overview["headers"], key=len)
    if not {"Home", "Road", *MARKET_COLUMNS}.issubset(set(overview_headers)):
        raise ValueError("Tracker market header contract changed")
    market = {}
    for cells in overview["rows"]:
        if len(cells) == len(overview_headers):
            key = (_key(cells[0]), _key(cells[1]))
            if key in market:
                if market[key] != dict(zip(overview_headers, cells)):
                    raise ValueError(f"Conflicting tracker overview: {key}")
                continue
            market[key] = dict(zip(overview_headers, cells))
    records = []
    seen = {}
    for cells in experts["rows"]:
        if len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells))
        key = (_key(cells[0]), _key(cells[1]))
        if key in seen:
            if seen[key] != cells:
                raise ValueError(f"Conflicting tracker expert matchup: {key}")
            continue
        seen[key] = cells
        if key not in market:
            raise ValueError(f"Expert matchup missing from market table: {key}")
        records.append({"source_home": cells[0], "source_away": cells[1],
                        "key": key, "experts": {name: row[name] for name in EXPERT_COLUMNS},
                        "market": {name: market[key][name] for name in MARKET_COLUMNS}})
    if not records:
        raise ValueError("Tracker expert table is empty")
    caption = overview["caption"] or None
    return caption, records


def _number(raw: str) -> float | None:
    if not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid tracker numeric value: {raw!r}") from exc
    if not -100 <= value <= 100:
        raise ValueError(f"Out-of-range tracker margin: {value}")
    return value


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Timestamp needs timezone")
    return value.astimezone(UTC)


def _site_time(caption: str | None, year: int) -> str | None:
    if not caption:
        return None
    match = re.search(r"Last updated:\s*(\w+\s+\w+\s+\d+,?\s+\d\d?:\d\d:\d\d\s+[AP]M\s+C[DS]T)", caption)
    if not match:
        return None
    # The page omits the year; the season of the local schedule supplies it.
    value = match.group(1)
    value = re.sub(r"^\w+\s+", "", value).replace(",", "")
    tz = timezone(timedelta(hours=-5 if value.endswith("CDT") else -6))
    date = datetime.strptime(  # noqa: DTZ007 - page declares CST/CDT below
        f"{value[:-4].strip()} {year}", "%B %d %I:%M:%S %p %Y"
    )
    return date.replace(tzinfo=tz).astimezone(UTC).isoformat()


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"Immutable archive collision: {path}")
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o400)


def receipt_filename(raw_sha256: str, schedule_sha256: str | None,
                     schedule_modified_at_utc: str | None) -> str:
    if schedule_sha256:
        stamp = f"{schedule_sha256}:{schedule_modified_at_utc}"
        projection_key = f".{hashlib.sha256(stamp.encode()).hexdigest()[:16]}"
    else:
        projection_key = ""
    return f"{raw_sha256}{projection_key}.{PARSER_VERSION}.json"


def capture(raw: bytes, schedule: pd.DataFrame, archive: Path, *,
            collected_at: datetime, source_url: str = SOURCE_URL,
            schedule_sha256: str | None = None,
            schedule_modified_at_utc: str | None = None) -> dict:
    collected_at = _utc(collected_at)
    if not source_url.startswith("https://www.thepredictiontracker.com/"):
        raise ValueError("Unexpected source URL")
    required = {"id", "season", "week", "start_date", "home_team", "away_team",
                "home_classification", "away_classification", "season_type"}
    if required - set(schedule):
        raise ValueError(f"Schedule columns missing: {sorted(required - set(schedule))}")
    sha = hashlib.sha256(raw).hexdigest()
    prior = archive / "receipts" / receipt_filename(
        sha, schedule_sha256, schedule_modified_at_utc)
    if prior.exists():
        receipt = json.loads(prior.read_text())
        if receipt["raw_sha256"] != sha or hashlib.sha256(
                (archive / receipt["raw_filename"]).read_bytes()).hexdigest() != sha:
            raise ValueError("Existing capture failed its immutable hash check")
        return receipt
    caption, records = parse_tracker(raw)
    games = schedule.loc[(schedule.season == 2026)
                         & schedule.season_type.astype(str).str.casefold().eq("regular")
                         & schedule.home_classification.astype(str).str.casefold().eq("fbs")
                         & schedule.away_classification.astype(str).str.casefold().eq("fbs")].copy()
    if games.id.duplicated().any():
        raise ValueError("Duplicate schedule game ID")
    games["key"] = [(_key(h), _key(a)) for h, a in zip(games.home_team, games.away_team)]
    if games.key.duplicated().any():
        raise ValueError("Ambiguous schedule matchup")
    index = games.set_index("key")
    site_updated = _site_time(caption, 2026)
    if site_updated and datetime.fromisoformat(site_updated) > collected_at:
        raise ValueError("Page update time is after collection time")
    observations = []
    unmatched = []
    for record in records:
        key = record["key"]
        if key not in index.index:
            unmatched.append({"home": record["source_home"], "away": record["source_away"]})
            continue
        game = index.loc[[key]].iloc[0]
        kickoff = pd.Timestamp(game.start_date)
        if kickoff.tzinfo is None:
            raise ValueError("Kickoff has no timezone")
        kickoff = kickoff.tz_convert("UTC")
        if "start_time_t_b_d" in game and bool(game.start_time_t_b_d):
            status = "kickoff_unverified"
        else:
            status = "prospective_pregame" if collected_at < kickoff.to_pydatetime() else "already_started"
        values = {}
        for column, label in {**EXPERT_COLUMNS, **MARKET_COLUMNS}.items():
            original = (record["experts"] if column in EXPERT_COLUMNS else record["market"])[column]
            values[label] = {"raw_value": original, "home_margin": _number(original),
                             "status": "available" if original.strip() else "missing"}
        observations.append({"game_id": int(game.id), "season": 2026, "week": int(game.week),
                             "home_team": str(game.home_team), "away_team": str(game.away_team),
                             "kickoff_utc": kickoff.isoformat(), "status": status,
                             "source_home": record["source_home"],
                             "source_away": record["source_away"], "values": values})
    receipt = {"schema_version": 1, "parser_version": PARSER_VERSION,
               "source_id": "prediction_tracker_live", "source_url": source_url,
               "collected_at_utc": collected_at.isoformat(),
               "site_updated_at_utc": site_updated, "site_caption_raw": caption,
               "raw_sha256": sha, "raw_filename": f"snapshots/{sha}.html",
               "provenance": "prospective_capture",
               "schedule_sha256": schedule_sha256,
               "schedule_modified_at_utc": schedule_modified_at_utc,
               "table_games": len(records), "matched_games": len(observations),
               "pregame_games": sum(r["status"] == "prospective_pregame" for r in observations),
               "unmatched": unmatched, "observations": observations}
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    _write_once(archive / "snapshots" / f"{sha}.html", raw)
    _write_once(prior, payload)
    return receipt
