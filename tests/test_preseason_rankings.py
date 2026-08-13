import pandas as pd

from gridiron_ml.publication.preseason_rankings import (
    build_preseason_performance_rankings,
    load_preseason_performance_rankings,
)


def test_preseason_rankings_are_performance_based_and_exclude_baselines(tmp_path):
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame([
        {"model_id": "winner_tree", "objective": "winner", "model_family": "tree"},
        {"model_id": "margin_linear", "objective": "margin", "model_family": "linear"},
        {"model_id": "winner_knn", "objective": "winner", "model_family": "knn"},
    ]).to_csv(inventory, index=False)
    leaderboard = tmp_path / "leaderboard.csv"
    pd.DataFrame([
        {"source": "winner_tree", "mean_total_score": .7, "mean_winner_accuracy": .8, "mean_mae": 10, "mean_rmse": 12},
        {"source": "margin_linear", "mean_total_score": .8, "mean_winner_accuracy": .7, "mean_mae": 9, "mean_rmse": 11},
        {"source": "winner_knn", "mean_total_score": 1.0, "mean_winner_accuracy": 1.0, "mean_mae": 1, "mean_rmse": 2},
    ]).to_csv(leaderboard, index=False)
    output = tmp_path / "rankings.csv"
    ranked = build_preseason_performance_rankings(inventory, leaderboard, output)
    assert ranked.iloc[0]["model_id"] == "margin_linear"
    assert ranked.loc[ranked.model_id.eq("winner_knn"), "ranking_eligible"].iloc[0] == False
    assert load_preseason_performance_rankings(output)["model_id"].tolist()[0] == "margin_linear"
