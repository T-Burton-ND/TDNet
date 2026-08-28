from pathlib import Path

import pandas as pd

from gridiron_ml.publication.scientific_weekly import (
    market_free_scientific_inventory,
    scientific_consensus_power_rankings,
    scientific_prediction_table,
    write_scientific_weekly_outputs,
)


def test_scientific_inventory_excludes_market_tiers_and_disabled_cells():
    inventory = pd.DataFrame(
        {
            "model_id": ["f6", "f7", "disabled"],
            "feature_config": ["F6", "F7", "F5"],
            "market_bearing": [False, True, False],
            "objective": ["margin", "margin", "margin"],
            "use_in_weekly_consensus": [True, True, False],
        }
    )
    selected = market_free_scientific_inventory(inventory)
    assert selected["model_id"].tolist() == ["f6"]


def test_scientific_predictions_make_straight_up_and_ats_picks_explicit():
    games = pd.DataFrame(
        [
            {
                "game_id": 1,
                "week": 1,
                "game_start_time_utc": "2026-08-29T19:00:00Z",
                "away_team": "Away",
                "home_team": "Home",
                "pred_winner": "Home",
                "predicted_margin": 10.0,
                "pred_home_margin": 10.0,
                "pred_home_win_probability": 0.7,
                "vegas_spread_as_of_publish": -14.0,
                "model_agreement": 0.8,
                "model_count": 42,
            },
            {
                "game_id": 2,
                "week": 1,
                "game_start_time_utc": "2026-08-29T20:00:00Z",
                "away_team": "Visitor",
                "home_team": "Host",
                "pred_winner": "Host",
                "predicted_margin": 8.0,
                "pred_home_margin": 8.0,
                "pred_home_win_probability": 0.66,
                "vegas_spread_as_of_publish": -3.0,
                "model_agreement": 0.75,
                "model_count": 42,
            },
        ]
    )
    result = scientific_prediction_table(games, reader_week=0, phase="pre_game")
    first = result.set_index("game_id").loc[1]
    second = result.set_index("game_id").loc[2]
    assert first["straight_up_pick"] == "Home"
    assert first["against_spread_pick"] == "Away +14"
    assert first["model_edge_vs_spread_points"] == 4.0
    assert second["against_spread_pick"] == "Host -3"
    assert second["model_edge_vs_spread_points"] == 5.0
    assert first["kickoff_eastern"] == "Sat 3:00 PM"


def test_scientific_writer_emits_three_csv_png_pairs(tmp_path: Path):
    games = pd.DataFrame(
        [
            {
                "game_id": 1,
                "week": 1,
                "game_start_time_utc": "2026-08-29T19:00:00Z",
                "away_team": "Away",
                "home_team": "Home",
                "pred_winner": "Home",
                "predicted_margin": 3.0,
                "pred_home_margin": 3.0,
                "pred_home_win_probability": 0.6,
                "market_spread_close": -1.0,
                "model_agreement": 1.0,
                "model_count": 1,
            }
        ]
    )
    poll = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 0,
                "poll_objective": "margin",
                "rank": rank,
                "keys_team": f"Team {rank}",
                "poll_points": 26 - rank,
                "ballots_seen": 1,
                "top25_votes": 1,
                "first_place_votes": int(rank == 1),
                "average_rank": float(rank),
                "best_rank": rank,
                "worst_rank": rank,
            }
            for rank in range(1, 26)
        ]
    )
    ballots = pd.DataFrame(
        [
            {
                "poll_objective": "margin",
                "keys_team": f"Team {rank}",
                "ballot_model": "scientific_F6_M1",
                "power_rating_vs_average": float(26 - rank),
                "ballot_rank": rank,
                "poll_points": 26 - rank,
                "top25_vote": True,
                "first_place_vote": rank == 1,
            }
            for rank in range(1, 26)
        ]
    )
    write_scientific_weekly_outputs(
        games=games,
        poll=poll,
        ballots=ballots,
        output_root=tmp_path,
        season=2026,
        week=0,
        phase="pre_game",
    )
    expected = {
        "scientific_all_game_predictions.csv",
        "scientific_all_game_predictions.png",
        "scientific_full_ballots.csv",
        "scientific_top25_ballots.png",
        "scientific_consensus_power_rankings.csv",
        "scientific_top25.png",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    assert len(pd.read_csv(tmp_path / "scientific_full_ballots.csv")) == 25
    power = pd.read_csv(tmp_path / "scientific_consensus_power_rankings.csv")
    assert power.loc[0, "predicted_margin_vs_average_team"] == 25.0
    assert power.loc[0, "poll_points"] == 25
    assert power.loc[24, "poll_points"] == 1


def test_consensus_power_ranking_averages_model_margins():
    ballots = pd.DataFrame(
        [
            {"keys_team": "A", "ballot_model": "m1", "ballot_rank": 1, "power_rating_vs_average": 7.0, "top25_vote": True, "poll_points": 25, "first_place_vote": True},
            {"keys_team": "A", "ballot_model": "m2", "ballot_rank": 2, "power_rating_vs_average": 5.0, "top25_vote": True, "poll_points": 24, "first_place_vote": False},
            {"keys_team": "B", "ballot_model": "m1", "ballot_rank": 2, "power_rating_vs_average": 2.0, "top25_vote": True, "poll_points": 24, "first_place_vote": False},
            {"keys_team": "B", "ballot_model": "m2", "ballot_rank": 1, "power_rating_vs_average": 4.0, "top25_vote": True, "poll_points": 25, "first_place_vote": True},
        ]
    )
    power = scientific_consensus_power_rankings(ballots)
    assert power["keys_team"].tolist() == ["A", "B"]
    assert power["predicted_margin_vs_average_team"].tolist() == [6.0, 3.0]
    assert power["poll_points"].tolist() == [25, 24]
    assert power["ballot_poll_points_sum"].tolist() == [49, 49]
