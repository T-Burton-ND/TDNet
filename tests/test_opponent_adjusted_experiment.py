import pandas as pd
import pytest

from gridiron_ml.experiments.opponent_adjusted import (
    OpponentAdjustedVersionSpec,
    StaticFrameFingerprints,
    average_adjusted_feature_tables,
    build_method_game_contributions,
    roll_adjusted_game_contributions,
)


def test_opponent_adjusted_game_contributions_do_not_use_future_weeks():
    spec = OpponentAdjustedVersionSpec(
        "v1.2",
        "opponent_ridge",
        "test ridge residuals",
    )
    stats = ("offense_ppa", "defense_ppa", "target_team_margin")
    games = _toy_games()
    changed_future = games.copy()
    changed_future.loc[changed_future["keys_week"] == 2, "offense_ppa"] = 99.0
    changed_future.loc[changed_future["keys_week"] == 2, "defense_ppa"] = -99.0

    base = build_method_game_contributions(games=games, stat_columns=stats, spec=spec)
    changed = build_method_game_contributions(
        games=changed_future,
        stat_columns=stats,
        spec=spec,
    )

    base_week1 = base.loc[base["keys_week"] == 1, list(stats)].reset_index(drop=True)
    changed_week1 = changed.loc[changed["keys_week"] == 1, list(stats)].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(base_week1, changed_week1)


def test_opponent_adjustment_has_finite_zero_fallback_without_history():
    spec = OpponentAdjustedVersionSpec("v1.2", "opponent_ridge", "empty-history fallback")
    games = _toy_games().loc[lambda frame: frame["keys_week"].eq(1)].copy()
    result = build_method_game_contributions(
        games=games,
        stat_columns=("offense_ppa", "defense_ppa", "target_team_margin"),
        spec=spec,
    )
    assert result[["offense_ppa", "defense_ppa", "target_team_margin"]].notna().all().all()


def test_static_frame_fingerprints_feed_training_block_with_adjusted_features():
    frame = pd.DataFrame(
        {
            "keys_season": [2025, 2025],
            "keys_team": ["A", "B"],
            "keys_week": [1, 1],
            "next_game_id": [10, 10],
            "next_opponent": ["B", "A"],
            "next_game_is_home": [True, False],
            "next_week": [2, 2],
            "y_next_margin": [7.0, -7.0],
            "y_has_next_game": [True, True],
            "statOff_yards_per_pass": [8.2, 6.1],
            "opp_adj_v1_1_offense_ppa_mean_to_date": [0.2, -0.1],
            "market_spread_close": [-3.5, -3.5],
        }
    )

    X, y, meta, market = StaticFrameFingerprints(frame).training_block([2025])

    assert "opp_adj_v1_1_offense_ppa_mean_to_date" in X.columns
    assert "statOff_yards_per_pass" in X.columns
    assert "market_spread_close" not in X.columns
    assert y.tolist() == [7.0, -7.0]
    assert "next_opponent" in meta.columns
    assert "market_spread_close" in market.columns


def test_ensemble_average_aligns_versioned_adjusted_suffixes():
    keys = {
        "keys_season": [2025],
        "keys_team": ["A"],
        "keys_week": [1],
    }
    v11 = pd.DataFrame(
        {
            **keys,
            "opp_adj_v1_1_offense_ppa_mean_to_date": [1.0],
            "opp_adj_v1_1_games_played": [2.0],
        }
    )
    v12 = pd.DataFrame(
        {
            **keys,
            "opp_adj_v1_2_offense_ppa_mean_to_date": [3.0],
            "opp_adj_v1_2_games_played": [2.0],
        }
    )
    ensemble = average_adjusted_feature_tables(
        [v11, v12],
        spec=OpponentAdjustedVersionSpec("v1.7", "ensemble_average", "test"),
    )

    assert ensemble.loc[0, "opp_adj_v1_7_offense_ppa_mean_to_date"] == 2.0
    assert ensemble.loc[0, "opp_adj_v1_7_games_played"] == 2.0


def test_roll_adjusted_game_contributions_creates_stable_team_week_shape():
    spec = OpponentAdjustedVersionSpec("v1.1", "opponent_context", "test")
    contrib = pd.DataFrame(
        {
            "keys_season": [2025, 2025],
            "keys_team": ["A", "A"],
            "keys_week": [1, 2],
            "keys_game_id": [1, 2],
            "keys_opponent": ["B", "C"],
            "game_is_home": [True, False],
            "offense_ppa": [0.2, 0.4],
            "fingerprint": ["v1.1", "v1.1"],
            "method": ["opponent_context", "opponent_context"],
        }
    )

    rolled = roll_adjusted_game_contributions(contrib, spec=spec)

    assert list(rolled[["keys_season", "keys_team", "keys_week"]].iloc[1]) == [
        2025,
        "A",
        2,
    ]
    assert rolled.loc[1, "opp_adj_v1_1_offense_ppa_mean_to_date"] == pytest.approx(0.3)
    assert rolled.loc[1, "opp_adj_v1_1_unique_opponents"] == 2.0


def _toy_games():
    rows = []
    for week, margin in [(1, 10.0), (2, -4.0)]:
        rows.extend(
            [
                {
                    "keys_season": 2025,
                    "keys_week": week,
                    "keys_game_id": week,
                    "keys_team": "A",
                    "keys_opponent": "B",
                    "game_is_home": True,
                    "offense_ppa": 0.2 * week,
                    "defense_ppa": -0.1 * week,
                    "target_team_margin": margin,
                },
                {
                    "keys_season": 2025,
                    "keys_week": week,
                    "keys_game_id": week,
                    "keys_team": "B",
                    "keys_opponent": "A",
                    "game_is_home": False,
                    "offense_ppa": -0.1 * week,
                    "defense_ppa": 0.2 * week,
                    "target_team_margin": -margin,
                },
            ]
        )
    return pd.DataFrame(rows)
