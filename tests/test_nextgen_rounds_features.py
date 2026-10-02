import numpy as np
import pandas as pd

from gridiron_ml.experiments.nextgen_rounds_features import (
    actor_game_statistics, context_totals, past_only_context_baselines,
    sequence_game_statistics,
)
from gridiron_ml.experiments.nextgen_rounds_market import summarize_quotes


def test_context_baseline_uses_only_earlier_seasons():
    totals = pd.DataFrame({"season": [2010, 2011], "context_id": [0, 0],
                           "n": [100, 100], "sum_yards": [300, 2000],
                           "sum_success": [40, 100]})
    baselines = past_only_context_baselines(totals, years=range(2010, 2013))
    assert baselines[2010].empty
    assert baselines[2011].set_index("context_id").loc[0, "expected_yards"] == 3
    assert baselines[2012].set_index("context_id").loc[0, "expected_yards"] == 11.5


def test_ambiguous_drive_is_excluded_from_sequence_features():
    plays = pd.DataFrame({
        "game_id": [1, 1, 1, 1], "drive_id": ["a", "a", "b", "b"],
        "period": [1, 1, 1, 1], "drive_number": [1, 1, 2, 2],
        "play_number": [1, 2, 1, 1], "id": [11, 12, 21, 22],
        "offense": ["A"] * 4, "defense": ["B"] * 4,
        "time_eligible": [True] * 4, "success": [False, True, True, True],
        "down": [1, 2, 1, 2],
    })
    result, audit = sequence_game_statistics(plays)
    assert audit["excluded_ambiguous_drives"] == 1
    row = result.loc[result.team.eq("A")].iloc[0]
    assert row.offense_failure_recovery__n == 1
    assert row.offense_failure_recovery__sum == 1


def test_skipped_event_does_not_create_fictional_transition():
    plays = pd.DataFrame({
        "game_id": [1, 1, 1], "drive_id": ["a"] * 3,
        "period": [1] * 3, "drive_number": [1] * 3,
        "play_number": [1, 2, 3], "id": [11, 12, 13],
        "offense": ["A"] * 3, "defense": ["B"] * 3,
        "time_eligible": [True, False, True], "success": [False, False, True],
        "down": [1, 2, 3],
    })
    result, audit = sequence_game_statistics(plays)
    assert audit["excluded_ambiguous_drives"] == 0
    assert result.loc[result.team.eq("A"), "offense_failure_recovery__n"].iloc[0] == 0


def test_actor_roles_exclude_wrong_team_and_yardage():
    plays = pd.DataFrame({
        "game_id": [1] * 5, "play_id": ["1", "2", "3", "4", "5"],
        "offense": ["A"] * 5, "yards_gained": [2, 1, 3, 2, 1],
        "down": [3] * 5, "distance": [2] * 5,
        "yards_to_goal": [50] * 5, "rush": [True] * 5,
        "play_type": ["Rush"] * 5,
        "dropback": [False] * 5, "time_eligible": [True] * 5,
    })
    actors = pd.DataFrame({
        "game_id": [1] * 6, "team": ["A"] * 5 + ["B"],
        "play_id": ["1", "2", "3", "4", "5", "1"],
        "athlete_id": ["x", "x", "x", "y", "z", "wrong"],
        "stat_type": ["Rush"] * 6, "stat": [2, 1, 3, 2, 99, 2],
    })
    result, audit = actor_game_statistics(actors, plays)
    assert audit["conflicting_or_unverified_rows"] == 2
    row = result.iloc[0]
    assert row.short_rush_top_share__n == 4
    assert np.isclose(row.short_rush_top_share__sum / 4, 0.75)


def test_market_movement_and_moneyline_use_one_provider():
    quotes = [
        {"provider": "consensus", "spread": -4, "overUnder": None},
        {"provider": "a", "spread": -3, "spreadOpen": -2,
         "overUnder": 50, "overUnderOpen": 49,
         "homeMoneyline": -150, "awayMoneyline": 130},
        {"provider": "b", "spread": -5, "overUnder": 52},
    ]
    row = summarize_quotes(quotes)
    assert row["market_home_spread"] == -4
    assert row["market_total"] == 51
    assert row["market_spread_move"] == -1
    assert row["market_total_move"] == 1
    assert 0.5 < row["market_home_implied_no_vig"] < 0.7
