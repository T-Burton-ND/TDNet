"""Materialize the publication F5 temporal and F6 schedule-network tiers.

The canonical source frame stores a team state after each completed week and
uses that state to predict the next game.  The enrichments in this module
therefore consume the current row, but never a later row, season, or outcome.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_TEMPORAL_COLUMNS = (
    "offense_ppa",
    "offense_success_rate",
    "offense_explosiveness",
    "offense_passing_plays_ppa",
    "offense_rushing_plays_ppa",
    "defense_ppa",
    "defense_success_rate",
    "defense_explosiveness",
    "defense_havoc_rate",
    "statOff_yards_per_pass",
    "statOff_yards_per_rush_attempt",
    "statGen_turnovers",
)

TEMPORAL_PREFIX = "time_adj_"
GRAPH_PREFIX = "graph_"


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"keys_team", "keys_season", "keys_week", "games_played"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fingerprint ladder enrichment requires {missing}")
    tie_break = "keys_game_id" if "keys_game_id" in frame else "keys_week"
    return frame.sort_values(
        ["keys_season", "keys_team", "keys_week", tie_break], kind="mergesort"
    )


def build_temporal_state_features(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    half_life: float = 3.0,
    recent_window: int = 3,
    volatility_window: int = 5,
) -> pd.DataFrame:
    """Append leakage-safe recent-form features to a state-after-week frame.

    Source performance columns are season-to-date means.  When the completed
    game count advances, the latest game contribution is recovered exactly as

        x_g = (n_w * mean_w - n_(w-1) * mean_(w-1)) / (n_w - n_(w-1)).

    Bye rows do not create observations; they carry the last temporal state.
    """

    if half_life <= 0:
        raise ValueError("half_life must be positive")
    if recent_window < 1 or volatility_window < 2:
        raise ValueError("recent_window must be >=1 and volatility_window >=2")
    ordered = _ordered(pd.DataFrame(frame).copy())
    selected = [
        column
        for column in (columns or DEFAULT_TEMPORAL_COLUMNS)
        if column in ordered and pd.api.types.is_numeric_dtype(ordered[column])
    ]
    alpha = 1.0 - np.exp(np.log(0.5) / float(half_life))
    derived: dict[str, pd.Series] = {}

    for column in selected:
        last1 = pd.Series(np.nan, index=ordered.index, dtype=float)
        last_recent = pd.Series(np.nan, index=ordered.index, dtype=float)
        ewm = pd.Series(np.nan, index=ordered.index, dtype=float)
        trend = pd.Series(np.nan, index=ordered.index, dtype=float)
        volatility = pd.Series(np.nan, index=ordered.index, dtype=float)

        for _, group in ordered.groupby(["keys_season", "keys_team"], sort=False):
            history: list[float] = []
            previous_count = 0.0
            previous_mean = np.nan
            ewm_state = np.nan
            for index, row in group.iterrows():
                count = pd.to_numeric(pd.Series([row["games_played"]]), errors="coerce").iloc[0]
                mean = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                if pd.notna(count) and pd.notna(mean) and float(count) > previous_count:
                    increment = float(count) - previous_count
                    prior_total = previous_count * previous_mean if previous_count > 0 and np.isfinite(previous_mean) else 0.0
                    contribution = (float(count) * float(mean) - prior_total) / increment
                    if np.isfinite(contribution):
                        history.append(float(contribution))
                        ewm_state = (
                            float(contribution)
                            if not np.isfinite(ewm_state)
                            else alpha * float(contribution) + (1.0 - alpha) * ewm_state
                        )
                if pd.notna(count):
                    previous_count = max(previous_count, float(count))
                if pd.notna(mean):
                    previous_mean = float(mean)

                if history:
                    recent = np.asarray(history[-int(recent_window) :], dtype=float)
                    volatile = np.asarray(history[-int(volatility_window) :], dtype=float)
                    last1.at[index] = history[-1]
                    last_recent.at[index] = float(recent.mean())
                    ewm.at[index] = float(ewm_state)
                    trend.at[index] = float(recent.mean() - float(mean)) if pd.notna(mean) else np.nan
                    volatility.at[index] = float(volatile.std(ddof=0)) if len(volatile) >= 2 else 0.0

        derived[f"{TEMPORAL_PREFIX}last1__{column}"] = last1
        derived[f"{TEMPORAL_PREFIX}last{int(recent_window)}__{column}"] = last_recent
        derived[f"{TEMPORAL_PREFIX}ewm_hl{float(half_life):g}__{column}"] = ewm
        derived[f"{TEMPORAL_PREFIX}recent_minus_season__{column}"] = trend
        derived[f"{TEMPORAL_PREFIX}volatility{int(volatility_window)}__{column}"] = volatility

    return pd.concat([ordered, pd.DataFrame(derived, index=ordered.index)], axis=1).sort_index()


def _one_row_per_completed_game(season_frame: pd.DataFrame) -> pd.DataFrame:
    required = {"keys_week", "keys_game_id", "keys_team", "keys_opponent", "y_margin_this_week"}
    if not required.issubset(season_frame):
        raise ValueError(f"Schedule graph requires {sorted(required)}")
    games = season_frame.loc[
        pd.to_numeric(season_frame["y_margin_this_week"], errors="coerce").notna()
        & season_frame["keys_game_id"].notna(),
        ["keys_week", "keys_game_id", "keys_team", "keys_opponent", "y_margin_this_week"],
    ].copy()
    games["keys_week"] = pd.to_numeric(games["keys_week"], errors="coerce")
    games["y_margin_this_week"] = pd.to_numeric(games["y_margin_this_week"], errors="coerce")
    return (
        games.sort_values(["keys_week", "keys_game_id", "keys_team"], kind="mergesort")
        .drop_duplicates("keys_game_id", keep="first")
        .reset_index(drop=True)
    )


def _graph_ratings(teams: list[str], games: pd.DataFrame) -> dict[str, dict[str, float]]:
    n_teams = len(teams)
    if not n_teams:
        return {}
    index = {team: position for position, team in enumerate(teams)}
    colley = np.eye(n_teams, dtype=float) * 2.0
    rhs = np.ones(n_teams, dtype=float)
    transition = np.zeros((n_teams, n_teams), dtype=float)
    opponents: dict[str, list[str]] = defaultdict(list)
    wins: dict[str, list[str]] = defaultdict(list)
    losses: dict[str, list[str]] = defaultdict(list)
    game_counts = defaultdict(int)

    for row in games.itertuples(index=False):
        team = str(row.keys_team)
        opponent = str(row.keys_opponent)
        margin = float(row.y_margin_this_week)
        team_known = team in index
        opponent_known = opponent in index
        if team_known:
            game_counts[team] += 1
            opponents[team].append(opponent)
            colley[index[team], index[team]] += 1.0
        if opponent_known:
            game_counts[opponent] += 1
            opponents[opponent].append(team)
            colley[index[opponent], index[opponent]] += 1.0
        if team_known and opponent_known:
            colley[index[team], index[opponent]] -= 1.0
            colley[index[opponent], index[team]] -= 1.0

        if margin > 0:
            winner, loser = team, opponent
        elif margin < 0:
            winner, loser = opponent, team
        else:
            winner = loser = ""
        if winner in index:
            rhs[index[winner]] += 0.5
            wins[winner].append(loser)
        if loser in index:
            rhs[index[loser]] -= 0.5
            losses[loser].append(winner)
        if winner in index and loser in index:
            weight = 1.0 + min(abs(margin), 35.0) / 14.0
            transition[index[loser], index[winner]] += weight
            if abs(margin) <= 7.0:
                transition[index[winner], index[loser]] += 0.25

    ratings = np.linalg.solve(colley, rhs)
    row_sums = transition.sum(axis=1)
    dangling = row_sums == 0
    transition[~dangling] /= row_sums[~dangling, None]
    transition[dangling] = 1.0 / n_teams
    pagerank = np.full(n_teams, 1.0 / n_teams)
    for _ in range(50):
        pagerank = (1.0 - 0.85) / n_teams + 0.85 * transition.T.dot(pagerank)
    pagerank_sd = float(pagerank.std())
    pagerank_z = (pagerank - pagerank.mean()) / (pagerank_sd if pagerank_sd > 0 else 1.0)
    rating_map = {team: float(ratings[position]) for team, position in index.items()}

    output: dict[str, dict[str, float]] = {}
    for team, position in index.items():
        known_opponents = [opponent for opponent in opponents[team] if opponent in rating_map]
        beaten = [opponent for opponent in wins[team] if opponent in rating_map]
        lost_to = [opponent for opponent in losses[team] if opponent in rating_map]
        output[team] = {
            "colley_rating": rating_map[team],
            "pagerank_z": float(pagerank_z[position]),
            "schedule_strength": float(np.mean([rating_map[x] for x in known_opponents])) if known_opponents else 0.5,
            "win_quality": float(np.mean([rating_map[x] for x in beaten])) if beaten else 0.5,
            "loss_quality": float(np.mean([rating_map[x] for x in lost_to])) if lost_to else 0.5,
            "games_played": float(game_counts[team]),
            "unique_opponents": float(len(set(opponents[team]))),
        }
    return output


def build_schedule_graph_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append weekly Colley/PageRank schedule-network features.

    Each row at week ``w`` is computed from completed games in the same season
    with week no greater than ``w``.  ``next_opponent`` is schedule metadata;
    its rating is evaluated from that same historical graph.
    """

    ordered = _ordered(pd.DataFrame(frame).copy())
    derived_names = [
        "graph_colley_rating",
        "graph_pagerank_z",
        "graph_schedule_strength",
        "graph_win_quality",
        "graph_loss_quality",
        "graph_games_played",
        "graph_unique_opponents",
        "graph_colley_edge_next",
    ]
    derived = pd.DataFrame(np.nan, index=ordered.index, columns=derived_names, dtype=float)

    for _, season_frame in ordered.groupby("keys_season", sort=True):
        teams = sorted(season_frame["keys_team"].dropna().astype(str).unique())
        games = _one_row_per_completed_game(season_frame)
        weeks = sorted(pd.to_numeric(season_frame["keys_week"], errors="coerce").dropna().astype(int).unique())
        for week in weeks:
            history = games.loc[games["keys_week"] <= week]
            ratings = _graph_ratings(teams, history)
            rows = season_frame.index[pd.to_numeric(season_frame["keys_week"], errors="coerce").eq(week)]
            for index in rows:
                team = str(ordered.at[index, "keys_team"])
                values = ratings[team]
                opponent = str(ordered.at[index, "next_opponent"]) if "next_opponent" in ordered and pd.notna(ordered.at[index, "next_opponent"]) else ""
                opponent_rating = ratings.get(opponent, {}).get("colley_rating", 0.5)
                derived.loc[index] = [
                    values["colley_rating"],
                    values["pagerank_z"],
                    values["schedule_strength"],
                    values["win_quality"],
                    values["loss_quality"],
                    values["games_played"],
                    values["unique_opponents"],
                    values["colley_rating"] - float(opponent_rating),
                ]
    return pd.concat([ordered, derived], axis=1).sort_index()


def build_publication_ladder_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one frame containing the realized F0--F8 publication features."""

    temporal = build_temporal_state_features(frame)
    return build_schedule_graph_features(temporal)
