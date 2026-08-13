import numpy as np
import pandas as pd

from gridiron_ml.td_sim.bootstrap import bootstrap_week0_fingerprints, schedule_team_game_table_from_cfbd_games


def test_schedule_team_game_table_from_cfbd_games_builds_canonical_home_away_rows(tmp_path):
    raw_cache = tmp_path / "raw"
    teams_dir = raw_cache / "teams_fbs"
    teams_dir.mkdir(parents=True)
    pd.DataFrame({"school": ["A", "B"]}).to_parquet(teams_dir / "2026.parquet", index=False)
    games = pd.DataFrame(
        {
            "id": [1, 2],
            "season": [2026, 2026],
            "week": [1, 1],
            "season_type": ["regular", "regular"],
            "start_date": ["2026-08-29", "2026-08-30"],
            "neutral_site": [False, True],
            "conference_game": [False, False],
            "venue": ["A Stadium", "C Stadium"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "home_conference": ["X", "Y"],
            "away_conference": ["X", "Z"],
            "home_points": [np.nan, np.nan],
            "away_points": [np.nan, np.nan],
        }
    )

    table = schedule_team_game_table_from_cfbd_games(games, season=2026, raw_cache_dir=raw_cache)

    assert len(table) == 2
    assert set(table["keys_team"]) == {"A", "B"}
    assert table.loc[table["game_is_home"], "keys_opponent"].tolist() == ["B"]
    assert {"keys_season", "keys_week", "keys_game_id", "target_team_margin"}.issubset(table.columns)


def test_bootstrap_week0_uses_team_history_and_global_fallback():
    frame = pd.DataFrame(
        {
            "keys_season": [2024, 2025, 2025],
            "keys_team": ["A", "A", "B"],
            "keys_week": [12, 12, 12],
            "keys_game_id": [1, 2, 3],
            "keys_opponent": ["B", "B", "A"],
            "game_is_home": [True, False, True],
            "games_played": [12, 12, 12],
            "offense_ppa": [0.2, 0.6, -0.2],
            "defense_ppa": [-0.2, -0.6, 0.2],
            "y_margin_this_week": [7.0, 10.0, -10.0],
            "y_next_margin": [np.nan, np.nan, np.nan],
            "y_has_next_game": [False, False, False],
        }
    )

    week0 = bootstrap_week0_fingerprints(frame, season=2026, teams=["A", "C"], seasons_back=3, recency_halflife=1.0)

    assert set(week0["keys_team"]) == {"A", "C"}
    assert week0["keys_week"].eq(0).all()
    assert week0["games_played"].eq(0).all()
    a_row = week0.loc[week0["keys_team"] == "A"].iloc[0]
    c_row = week0.loc[week0["keys_team"] == "C"].iloc[0]
    assert a_row["offense_ppa"] > c_row["offense_ppa"]
    assert pd.isna(a_row["keys_game_id"])
    assert bool(a_row["y_has_next_game"]) is False
