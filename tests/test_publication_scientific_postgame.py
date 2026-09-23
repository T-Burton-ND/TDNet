import pandas as pd

from gridiron_ml.publication.scientific_postgame import score_scientific_predictions


def test_scientific_postgame_scores_frozen_model_and_consensus_rows():
    predictions = pd.DataFrame(
        [
            {
                "game_id": 1,
                "home_team": "Home",
                "away_team": "Away",
                "model_name": model,
                "pred_home_margin": margin,
                "pred_winner": winner,
                "model_against_spread_team": ats,
                "consensus_straight_up_pick": "Home",
                "consensus_predicted_winner_margin": 3.0,
                "consensus_against_spread_team": "Away",
                "consensus_home_team_market_spread": -7.0,
            }
            for model, margin, winner, ats in [
                ("m1", 4.0, "Home", "Away"),
                ("m2", -2.0, "Away", "Home"),
            ]
        ]
    )
    results = pd.DataFrame([{"game_id": 1, "home_points": 24, "away_points": 21}])
    model_games, consensus, scorecard = score_scientific_predictions(
        predictions, results
    )
    assert len(model_games) == 2
    assert consensus.loc[0, "consensus_winner_correct"]
    assert consensus.loc[0, "consensus_ats_result"] == "Win"
    assert consensus.loc[0, "consensus_absolute_margin_error"] == 0.0
    assert len(scorecard) == 2
    assert scorecard.set_index("model_name").loc["m1", "margin_mae"] == 1.0
