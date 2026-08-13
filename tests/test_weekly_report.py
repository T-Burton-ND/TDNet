import numpy as np
import pandas as pd

from gridiron_ml.td_run import TDEval
from gridiron_ml.td_run.poll_viz import plot_ballot_logo_grid
from gridiron_ml.td_run.season_vs_vegas import source_style
from gridiron_ml.td_run.weekly_report import summarize_matchup_predictions
from gridiron_ml.td_sim.recursive_plots import save_sim_accuracy_plot
from gridiron_ml.td_sim.recursive_simulator import RecursiveSeasonSimulator


class DummyFingerprints:
    def season_snapshot(self, season, week):
        X = pd.DataFrame({"rating": [3.0, 2.0, 1.0]})
        meta = pd.DataFrame({"keys_team": ["A", "B", "C"], "keys_season": season, "keys_week": week})
        return X, pd.Series([np.nan] * len(X)), meta, pd.DataFrame(index=X.index)

    def average_team(self, season=None, scope="season"):
        return pd.DataFrame({"rating": [2.0]})


class DummyMatchups:
    def team_vs_average(self, X_week, meta_week, average_team_df, market_df=None):
        return X_week, meta_week, pd.DataFrame(index=X_week.index)


def test_evaluator_poll_accepts_manual_ballot_without_models():
    evaluator = TDEval({}, fingerprints=DummyFingerprints(), matchup_builder=DummyMatchups(), model=object())

    poll = evaluator.poll(
        models=[],
        season=2026,
        week=1,
        top_n=3,
        manual_ballots={1: {"name": "my_poll", "teams": ["C", "B", "A"]}},
    )

    assert poll.loc[0, "keys_team"] == "C"
    assert poll.loc[0, "poll_points"] == 3
    assert evaluator.poll_ballots_["ballot_model"].tolist() == ["my_poll", "my_poll", "my_poll"]


def test_plot_ballot_logo_grid_writes_png(tmp_path):
    ballots = pd.DataFrame(
        {
            "keys_team": ["A", "B", "C"],
            "ballot_model": ["m1", "m1", "m1"],
            "ballot_rank": [1, 2, 3],
            "poll_points": [3, 2, 1],
        }
    )

    path = plot_ballot_logo_grid(ballots, tmp_path / "grid.png", top_n=3)

    assert path.exists()
    assert path.stat().st_size > 0


def test_summarize_matchup_predictions_builds_consensus_score():
    long_df = pd.DataFrame(
        {
            "keys_season": [2026, 2026],
            "next_week": [2, 2],
            "keys_team_home": ["Home", "Home"],
            "keys_team_away": ["Away", "Away"],
            "market_over_under": [50.0, 50.0],
            "model": ["a", "b"],
            "family": ["linear", "stat"],
            "pred_margin": [7.0, 3.0],
            "pred_proba_home_win": [0.70, 0.60],
        }
    )

    summary = summarize_matchup_predictions(long_df)

    assert summary.loc[0, "predicted_winner"] == "Home"
    assert summary.loc[0, "model_agreement"] == 1.0
    assert summary.loc[0, "pred_home_score"] == 27.5
    assert summary.loc[0, "pred_away_score"] == 22.5


def test_recursive_simulator_summarizes_retrospective_accuracy(tmp_path):
    games = pd.DataFrame(
        {
            "model": ["ridge", "ridge", "stat_z_index"],
            "sim_id": [0, 0, 0],
            "predicted_favorite": ["home", "away", "home"],
            "simulated_winner": ["home", "home", "away"],
            "actual_winner": ["home", "away", "away"],
        }
    )

    summary = RecursiveSeasonSimulator()._summarize_sim_accuracy(games)
    plot_path = save_sim_accuracy_plot(summary, tmp_path / "sim_accuracy.png")

    ridge = summary.loc[summary["model"] == "ridge"].iloc[0]
    assert ridge["winner_accuracy"] == 0.5
    assert ridge["chalk_accuracy"] == 1.0
    assert ridge["upset_accuracy"] == 0.0
    assert plot_path.exists()


def test_source_style_uses_family_markers():
    assert source_style("ridge", {}, 0)["marker"] == "o"
    assert source_style("stat_z_index", {}, 0)["marker"] == "s"
    assert source_style("random_forest", {}, 0)["marker"] == "^"
