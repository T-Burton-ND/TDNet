import pandas as pd

from gridiron_ml.publication.poll_recaps import (
    aggregate_receiving_votes,
    format_receiving_votes,
    model_consensus_disagreement,
)
from gridiron_ml.publication.recaps import (
    grade_postgame_predictions,
    model_chalk_upset_matrix,
    model_vegas_correctness_matrix,
    weekly_recap_metrics,
    write_model_vegas_confusion_artifacts,
)
from gridiron_ml.experiments.opponent_adjusted import DEFAULT_TRAIN_YEARS, DEFAULT_TEST_YEARS
from gridiron_ml.publication.selection import select_confirmatory_roster


def test_postgame_grading_separates_su_ats_and_pushes():
    frame = pd.DataFrame([
        {"home_team": "Home", "away_team": "Away", "home_points": 20, "away_points": 24,
         "pred_home_margin": -2, "market_spread_close": 3, "market_over_under": 44},
        {"home_team": "Push Home", "away_team": "Push Away", "home_points": 21, "away_points": 24,
         "pred_home_margin": 1, "market_spread_close": 3, "market_over_under": 45},
    ])
    graded = grade_postgame_predictions(frame)
    assert bool(graded.loc[0, "su_correct"])
    assert graded.loc[0, "ats_pick"] == "Home"
    assert graded.loc[0, "ats_result"] == "loss"
    assert graded.loc[1, "ats_result"] == "push"
    metrics = weekly_recap_metrics(graded)
    assert metrics["ats_losses"] == 1
    assert metrics["ats_pushes"] == 1
    assert metrics["ats_accuracy_excluding_pushes"] == 0.0


def test_model_vegas_confusion_matrices_and_artifacts(tmp_path):
    games = pd.DataFrame(
        [
            # Model and Vegas choose Home; Home wins: both correct, chalk result.
            {"home_team": "H1", "away_team": "A1", "pred_winner": "H1", "actual_winner": "H1", "market_spread_close": -3.0},
            # Model chooses Away against home-favorite Vegas; Away wins: model-only correct, upset call/result.
            {"home_team": "H2", "away_team": "A2", "pred_winner": "A2", "actual_winner": "A2", "market_spread_close": -4.0},
            # Model chooses home underdog; away favorite wins: Vegas-only correct, model upset call but actual chalk.
            {"home_team": "H3", "away_team": "A3", "pred_winner": "H3", "actual_winner": "A3", "market_spread_close": 2.0},
            # Model and Vegas choose away favorite; home wins: both wrong, actual upset.
            {"home_team": "H4", "away_team": "A4", "pred_winner": "A4", "actual_winner": "H4", "market_spread_close": 5.0},
            # Missing line is intentionally not graded in either matrix.
            {"home_team": "H5", "away_team": "A5", "pred_winner": "H5", "actual_winner": "H5", "market_spread_close": None},
        ]
    )
    correctness = model_vegas_correctness_matrix(games).set_index("model_outcome")
    assert correctness.loc["Model correct"].to_dict() == {"Vegas correct": 1, "Vegas wrong": 1}
    assert correctness.loc["Model wrong"].to_dict() == {"Vegas correct": 1, "Vegas wrong": 1}

    chalk = model_chalk_upset_matrix(games).set_index("model_pick")
    assert chalk.loc["Model picks chalk"].to_dict() == {"Actual chalk": 1, "Actual upset": 1}
    assert chalk.loc["Model picks upset"].to_dict() == {"Actual chalk": 1, "Actual upset": 1}

    outputs = write_model_vegas_confusion_artifacts(
        games, tmp_path, season=2025, roster_label="Test roster", dpi=60
    )
    assert set(outputs) == {
        "model_vs_vegas_correctness_confusion_matrix",
        "model_chalk_upset_vs_actual_confusion_matrix",
    }
    for csv_path in outputs.values():
        assert csv_path.exists()
        assert csv_path.with_suffix(".png").exists()
        assert not csv_path.with_suffix(".svg").exists()


def test_poll_disagreement_identifies_outlying_ballot():
    poll = pd.DataFrame({"rank": range(1, 4), "keys_team": ["A", "B", "C"]})
    ballots = pd.DataFrame([
        *[{"ballot_model": "close", "keys_team": team, "ballot_rank": rank} for rank, team in enumerate(["A", "B", "C"], 1)],
        *[{"ballot_model": "far", "keys_team": team, "ballot_rank": rank} for rank, team in enumerate(["C", "B", "A"], 1)],
    ])
    result = model_consensus_disagreement(poll, ballots, top_n=3)
    assert result.iloc[0]["ballot_model"] == "far"
    assert result.iloc[-1]["mean_absolute_rank_delta"] == 0.0


def test_receiving_votes_exclude_consensus_top25_and_sum_points():
    poll = pd.DataFrame({"rank": [1, 2], "keys_team": ["A", "B"]})
    ballots = pd.DataFrame([
        {"ballot_model": "m1", "keys_team": "A", "ballot_rank": 1, "poll_points": 25, "top25_vote": True},
        {"ballot_model": "m1", "keys_team": "C", "ballot_rank": 3, "poll_points": 23, "top25_vote": True},
        {"ballot_model": "m2", "keys_team": "C", "ballot_rank": 4, "poll_points": 22, "top25_vote": True},
        {"ballot_model": "m2", "keys_team": "D", "ballot_rank": 26, "poll_points": 0, "top25_vote": False},
    ])
    result = aggregate_receiving_votes(poll, ballots, top_n=2)
    assert result["keys_team"].tolist() == ["C"]
    assert result.iloc[0]["poll_points"] == 45
    assert format_receiving_votes(result) == "C (45)"


def test_default_holdout_years_do_not_overlap():
    assert set(DEFAULT_TRAIN_YEARS).isdisjoint(DEFAULT_TEST_YEARS)
    assert 2025 not in DEFAULT_TRAIN_YEARS


def test_roster_keeps_one_per_type_and_ranks_top_three_by_brier():
    candidates = pd.DataFrame([
        {"objective": "winner", "family": "tree", "model": "rf", "trial": 1, "brier_2024": .20, "brier_2025": .22},
        {"objective": "winner", "family": "tree", "model": "rf", "trial": 2, "brier_2024": .18, "brier_2025": .20},
        {"objective": "winner", "family": "linear", "model": "ridge", "trial": 1, "brier_2024": .19, "brier_2025": .20},
        {"objective": "margin", "family": "neural", "model": "mlp", "trial": 1, "brier_2024": .21, "brier_2025": .21},
    ])
    roster = select_confirmatory_roster(candidates)
    assert len(roster) == 3
    assert roster.iloc[0]["concrete_model_type"] == "tree/rf"
    assert roster["use_in_tdnet_poll"].all()
    assert roster["use_in_top_3_consensus"].all()
