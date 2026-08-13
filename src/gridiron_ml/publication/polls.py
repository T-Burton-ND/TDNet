"""Normalize externally sourced poll snapshots for publication outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ap_top25(path: str | Path, *, season: int | None = None, week: int | None = None) -> pd.DataFrame:
    """Return a canonical AP Top 25 snapshot from CFBD nested or long data."""
    source = Path(path)
    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    if frame.empty:
        return _empty_poll()
    if "polls" not in frame:
        return _normalize_long_ap(frame, season=season, week=week)
    selected = frame.copy()
    if season is not None and "season" in selected:
        selected = selected[pd.to_numeric(selected["season"], errors="coerce").eq(int(season))]
    if "season_type" in selected and selected["season_type"].astype(str).str.casefold().eq("regular").any():
        selected = selected[selected["season_type"].astype(str).str.casefold().eq("regular")]
    if week is not None and "week" in selected:
        exact = selected[pd.to_numeric(selected["week"], errors="coerce").eq(int(week))]
        if not exact.empty:
            selected = exact
        else:
            prior = selected[pd.to_numeric(selected["week"], errors="coerce").le(int(week))]
            if not prior.empty:
                selected = prior
            else:
                return _empty_poll()
    if selected.empty:
        return _empty_poll()
    if "week" in selected:
        selected = selected.sort_values("week").tail(1)
    record = selected.iloc[-1]
    rows = []
    for poll in _records(record["polls"]):
        if str(poll.get("poll", "")).strip().casefold() != "ap top 25":
            continue
        for rank in _records(poll.get("ranks", [])):
            rows.append({
                "rank": rank.get("rank"), "team": rank.get("school"),
                "points": rank.get("points"), "first_place_votes": rank.get("firstPlaceVotes"),
                "conference": rank.get("conference"), "team_id": rank.get("teamId"),
                "season": record.get("season"), "week": record.get("week"), "poll": "AP Top 25",
            })
    return _clean_poll(pd.DataFrame(rows))


def load_postweek_ap_top25(path: str | Path, *, season: int, completed_week: int) -> pd.DataFrame:
    """Return the AP snapshot published after a completed football week.

    CFBD ranking week N represents the AP poll available going into week N.
    Sunday recaps for completed week N should therefore compare against the
    next AP snapshot, week N+1.  ``load_ap_top25`` already falls back to the
    latest prior snapshot when N+1 is beyond the available data.
    """
    return load_ap_top25(path, season=season, week=int(completed_week) + 1)


def add_team_records(
    poll: pd.DataFrame, games: pd.DataFrame, *, completed_week: int
) -> pd.DataFrame:
    """Attach result-derived records through ``completed_week`` to poll rows."""
    out = poll.copy()
    completed = games.copy()
    # CFBD numbers postseason games from Week 1 again.  Without this filter,
    # a retrospective season file leaks future bowl/playoff results into the
    # records displayed during the regular season.
    if "season_type" in completed:
        completed = completed.loc[
            completed["season_type"].astype(str).str.casefold().eq("regular")
        ].copy()
    if "week" in completed:
        completed = completed.loc[
            pd.to_numeric(completed["week"], errors="coerce").le(int(completed_week))
        ].copy()
    home_points = pd.to_numeric(completed.get("home_points"), errors="coerce")
    away_points = pd.to_numeric(completed.get("away_points"), errors="coerce")
    completed = completed.loc[home_points.notna() & away_points.notna()].copy()
    records: dict[str, list[int]] = {}
    for game in completed.itertuples(index=False):
        home = str(game.home_team)
        away = str(game.away_team)
        records.setdefault(home, [0, 0, 0])
        records.setdefault(away, [0, 0, 0])
        if float(game.home_points) > float(game.away_points):
            records[home][0] += 1
            records[away][1] += 1
        elif float(game.away_points) > float(game.home_points):
            records[away][0] += 1
            records[home][1] += 1
        else:
            records[home][2] += 1
            records[away][2] += 1

    def format_record(team: str) -> str:
        wins, losses, ties = records.get(str(team), [0, 0, 0])
        return f"{wins}–{losses}–{ties}" if ties else f"{wins}–{losses}"

    team_column = "keys_team" if "keys_team" in out else "team"
    out["record"] = out[team_column].astype(str).map(format_record)
    out["record_through_week"] = int(completed_week)
    return out


def _normalize_long_ap(frame, *, season, week):
    out = frame.copy()
    poll_col = next((c for c in ["poll", "poll_name", "ranking_source"] if c in out), None)
    if poll_col:
        out = out[out[poll_col].astype(str).str.casefold().eq("ap top 25")]
    if season is not None and "season" in out:
        out = out[pd.to_numeric(out["season"], errors="coerce").eq(int(season))]
    if week is not None and "week" in out:
        exact = out[pd.to_numeric(out["week"], errors="coerce").eq(int(week))]
        if not exact.empty:
            out = exact
    rename = {}
    for candidate in ["school", "team_name"]:
        if candidate in out and "team" not in out:
            rename[candidate] = "team"
    for candidate in ["ap_rank", "ranking"]:
        if candidate in out and "rank" not in out:
            rename[candidate] = "rank"
    return _clean_poll(out.rename(columns=rename))


def _records(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else [value]


def _clean_poll(frame):
    if frame.empty or not {"rank", "team"}.issubset(frame):
        return _empty_poll()
    out = frame.copy()
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out = out[out["rank"].between(1, 25) & out["team"].notna()].copy()
    out["rank"] = out["rank"].astype(int)
    out["team"] = out["team"].astype(str).str.strip()
    return out.sort_values(["rank", "team"]).drop_duplicates("team").reset_index(drop=True)


def _empty_poll():
    return pd.DataFrame(columns=["rank", "team", "points", "first_place_votes", "conference", "team_id", "season", "week", "poll"])
