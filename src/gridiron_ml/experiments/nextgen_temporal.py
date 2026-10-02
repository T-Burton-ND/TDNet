"""Regular-season next-game alignment for exploratory team-week diagnostics."""

from __future__ import annotations

import pandas as pd

from .nextgen_contract import assert_design_years


def next_game_rows(games: pd.DataFrame, *, design: bool = True) -> pd.DataFrame:
    """Align state after the previous game with the following FBS matchup.

    FCS games can update an FBS team's state. Only completed regular-season
    FBS-vs-FBS games are targets. The illustrative state is a lagged running
    average margin; feature builders must follow the same source/target cutoff.
    """
    required = {"id", "season", "week", "season_type", "completed", "start_date",
                "home_team", "away_team", "home_classification", "away_classification",
                "home_points", "away_points"}
    if not required <= set(games):
        raise ValueError(f"Missing schedule fields: {sorted(required-set(games))}")
    frame = games.loc[games.season_type.astype(str).str.lower().eq("regular") &
                      games.completed.fillna(False).astype(bool)].copy()
    if design:
        assert_design_years(frame.season.dropna().astype(int).tolist())
    frame = frame.loc[pd.to_numeric(frame.home_points, errors="coerce").notna() &
                      pd.to_numeric(frame.away_points, errors="coerce").notna()]
    rows = []
    for side, other, sign in (("home", "away", 1), ("away", "home", -1)):
        subset = frame.loc[frame[f"{side}_classification"].astype(str).str.lower().eq("fbs")]
        rows.append(pd.DataFrame({
            "season": subset.season.astype(int), "week": subset.week.astype(int),
            "game_id": subset.id.astype(int), "start_date": pd.to_datetime(subset.start_date, utc=True),
            "team": subset[f"{side}_team"].astype(str),
            "opponent": subset[f"{other}_team"].astype(str),
            "opponent_classification": subset[f"{other}_classification"].astype(str).str.lower(),
            "is_home": side == "home",
            "margin": sign * (pd.to_numeric(subset.home_points) - pd.to_numeric(subset.away_points)),
            "points_for": pd.to_numeric(subset[f"{side}_points"]),
            "points_against": pd.to_numeric(subset[f"{other}_points"]),
        }))
    team_games = pd.concat(rows, ignore_index=True).sort_values(
        ["season", "team", "start_date", "game_id"], kind="stable").reset_index(drop=True)
    groups = team_games.groupby(["season", "team"], sort=False)
    team_games["source_game_id"] = groups.game_id.shift(1)
    team_games["prior_games"] = groups.cumcount()
    team_games["prior_mean_margin"] = groups.margin.transform(
        lambda values: values.shift(1).expanding().mean())
    targets = team_games.loc[team_games.opponent_classification.eq("fbs") &
                             team_games.source_game_id.notna()].copy()
    targets = targets.rename(columns={"margin": "next_game_margin",
                                      "points_for": "next_game_points_for",
                                      "points_against": "next_game_points_against"})
    targets["next_game_win"] = targets.next_game_margin.gt(0)
    return targets.reset_index(drop=True)
