import numpy as np
import pandas as pd
import warnings

from gridiron_ml.td_run.season_vs_vegas import (
    build_prediction_table,
    column_normalized_matrix,
    compute_all_tables,
    move_legend_outside,
    save_evaluation_plots,
    save_metric_tables,
)


class DummyPredictionModel:
    def __init__(self, offset):
        self.offset = float(offset)

    def predict(self, X):
        return pd.DataFrame(
            {"pred_margin": np.linspace(-7.0, 7.0, len(X)) + self.offset}
        )


def test_build_prediction_table_batches_columns_without_fragmentation_warning():
    matchup_X = pd.DataFrame({"feature": np.arange(6)})
    base_df = pd.DataFrame(
        {
            "actual_margin": [3.0, -4.0, 8.0, -2.0, 10.0, -14.0],
            "actual_winner": ["home", "away", "home", "away", "home", "away"],
            "vegas_implied_margin": [1.0, -1.0, 7.0, -3.0, 6.0, -10.0],
            "vegas_winner": ["home", "away", "home", "away", "home", "away"],
            "actual_is_upset": [False, False, False, False, False, False],
            "worse_record_side": ["away", "home", "away", "home", "away", "home"],
        }
    )
    model_entries = [
        {"name": f"model_{idx}", "model": DummyPredictionModel(idx / 10)}
        for idx in range(25)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions = build_prediction_table(model_entries, matchup_X, base_df)

    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, pd.errors.PerformanceWarning)
    ]
    assert "model_0__pred_margin" in predictions.columns
    assert "Vegas__picked_worse_record" in predictions.columns


def test_compute_all_tables_returns_wide_model_and_vegas_columns():
    predictions = pd.DataFrame(
        {
            "week": [1, 1, 2, 2],
            "actual_margin": [10.0, -7.0, -3.0, 14.0],
            "actual_winner": ["home", "away", "away", "home"],
            "vegas_implied_margin": [7.0, 4.0, -6.0, 10.0],
            "vegas_winner": ["home", "home", "away", "home"],
            "actual_is_upset": [False, True, False, False],
            "worse_record_side": ["away", "away", "home", "away"],
            "vegas_favorite_bucket": pd.Categorical(["3-7", "3-7", "3-7", "7-14"]),
            "Vegas__pred_margin": [7.0, 4.0, -6.0, 10.0],
            "Vegas__winner": ["home", "home", "away", "home"],
            "Vegas__error": [-3.0, 11.0, -3.0, -4.0],
            "Vegas__correct": [True, False, True, True],
            "model_a__pred_margin": [9.0, -2.0, 2.0, 11.0],
            "model_a__pred_win_prob": [0.75, 0.45, 0.55, 0.80],
            "model_a__winner": ["home", "away", "home", "home"],
            "model_a__error": [-1.0, 5.0, 5.0, -3.0],
            "model_a__correct": [True, True, False, True],
            "model_a__edge_vs_vegas": [2.0, -6.0, 8.0, 1.0],
            "model_a__called_upset": [False, True, True, False],
            "model_a__correct_upset": [False, True, False, False],
            "model_a__favorite_bucket": pd.Categorical(["7-14", "0-3", "0-3", "7-14"]),
            "model_a__confidence": [0.75, 0.55, 0.55, 0.80],
            "model_a__confidence_bucket": pd.Categorical(
                ["70-80%", "55-60%", "55-60%", "70-80%"]
            ),
            "model_b__pred_margin": [12.0, 6.0, -4.0, -1.0],
            "model_b__pred_win_prob": [0.80, 0.65, 0.40, 0.49],
            "model_b__winner": ["home", "home", "away", "away"],
            "model_b__error": [2.0, 13.0, -1.0, -15.0],
            "model_b__correct": [True, False, True, False],
            "model_b__edge_vs_vegas": [5.0, 2.0, 2.0, -11.0],
            "model_b__called_upset": [False, False, False, True],
            "model_b__correct_upset": [False, False, False, False],
            "model_b__favorite_bucket": pd.Categorical(["7-14", "3-7", "3-7", "0-3"]),
            "model_b__confidence": [0.80, 0.65, 0.60, 0.51],
            "model_b__confidence_bucket": pd.Categorical(
                ["70-80%", "60-70%", "60-70%", "50-55%"]
            ),
        }
    )

    tables = compute_all_tables(predictions)

    assert {"Vegas", "model_a", "model_b"}.issubset(
        tables["overall_margin_metrics"].columns
    )
    assert "margin_diagnostics" in tables
    assert {"source", "r2", "within_7", "abs_error_p90", "calibration_slope"}.issubset(
        tables["margin_diagnostics"].columns
    )
    assert "weekly_winner_accuracy" not in tables
    assert "ats_summary" not in tables
    assert {
        "total_score",
        "winner_accuracy",
        "upset_recall",
        "disagreement_accuracy",
        "margin_score",
    }.issubset(tables["model_score_matrix"].columns)
    assert "overall_vegas_alignment_metrics" in tables
    assert "margin_edge_3_plus_winner_accuracy" in set(
        tables["overall_winner_metrics"]["metric"]
    )
    assert "contrarian_edge_3_plus_model_minus_vegas" in set(
        tables["overall_winner_metrics"]["metric"]
    )
    alignment = tables["overall_vegas_alignment_metrics"].set_index("model")
    assert alignment.loc["model_a", "against_vegas_count"] == 2
    assert tables["model_score_matrix"].iloc[0]["source"] == "model_a"
    assert (
        tables["overall_winner_metrics"]
        .loc[
            tables["overall_winner_metrics"]["metric"] == "upset_recall",
            "model_a",
        ]
        .iloc[0]
        == 1.0
    )
    assert (
        tables["overall_winner_metrics"]
        .loc[
            tables["overall_winner_metrics"]["metric"] == "disagreement_accuracy",
            "model_a",
        ]
        .iloc[0]
        == 0.5
    )
    assert (
        tables["overall_winner_metrics"]
        .loc[
            tables["overall_winner_metrics"]["metric"] == "record_upset_recall",
            "model_a",
        ]
        .iloc[0]
        == 1.0
    )

    diagnostic_tables = compute_all_tables(
        predictions,
        eval_config={
            "artifacts": {
                "weekly_tables": True,
                "bucket_tables": True,
                "calibration_tables": True,
                "ats_tables": True,
            }
        },
    )
    assert {"Vegas", "model_a", "model_b"}.issubset(
        diagnostic_tables["weekly_winner_accuracy"].columns
    )
    assert {"model_a", "model_b"}.issubset(diagnostic_tables["ats_summary"].columns)
    assert "Vegas" not in diagnostic_tables["ats_summary"].columns


def test_prediction_sanity_reports_range_and_sign_counts():
    predictions = pd.DataFrame(
        {
            "actual_margin": [1.0, -1.0],
            "model_a__pred_margin": [3.0, -2.0],
            "model_a__error": [2.0, -1.0],
            "model_a__correct": [True, True],
        }
    )

    tables = compute_all_tables(predictions)
    sanity = tables["prediction_sanity"].set_index("metric")

    assert sanity.loc["model_a", "positive_margins"] == 1
    assert sanity.loc["model_a", "negative_margins"] == 1
    assert np.isclose(sanity.loc["model_a", "max_margin"], 3.0)


def test_save_evaluation_plots_writes_core_pngs(tmp_path):
    predictions = pd.DataFrame(
        {
            "week": [1, 1, 2, 2],
            "actual_margin": [10.0, -7.0, -3.0, 14.0],
            "actual_winner": ["home", "away", "away", "home"],
            "vegas_implied_margin": [7.0, 4.0, -6.0, 10.0],
            "vegas_winner": ["home", "home", "away", "home"],
            "actual_is_upset": [False, True, False, False],
            "worse_record_side": ["away", "away", "home", "away"],
            "vegas_favorite_bucket": pd.Categorical(["3-7", "3-7", "3-7", "7-14"]),
            "Vegas__pred_margin": [7.0, 4.0, -6.0, 10.0],
            "Vegas__winner": ["home", "home", "away", "home"],
            "Vegas__error": [-3.0, 11.0, -3.0, -4.0],
            "Vegas__correct": [True, False, True, True],
            "model_a__pred_margin": [9.0, -2.0, 2.0, 11.0],
            "model_a__pred_win_prob": [0.75, 0.45, 0.55, 0.80],
            "model_a__winner": ["home", "away", "home", "home"],
            "model_a__error": [-1.0, 5.0, 5.0, -3.0],
            "model_a__correct": [True, True, False, True],
            "model_a__edge_vs_vegas": [2.0, -6.0, 8.0, 1.0],
            "model_a__called_upset": [False, True, True, False],
            "model_a__correct_upset": [False, True, False, False],
            "model_a__favorite_bucket": pd.Categorical(["7-14", "0-3", "0-3", "7-14"]),
            "model_a__confidence": [0.75, 0.55, 0.55, 0.80],
            "model_a__confidence_bucket": pd.Categorical(
                ["70-80%", "55-60%", "55-60%", "70-80%"]
            ),
        }
    )

    tables = compute_all_tables(predictions)
    tables["game_predictions"] = predictions
    tables_dir = save_metric_tables(tables, tmp_path)
    plots_dir = save_evaluation_plots(
        tables,
        tmp_path,
        eval_config={"artifacts": {"png_plots": True}},
    )

    assert (tables_dir / "model_score_matrix.csv").exists()
    assert (tables_dir / "margin_diagnostics.csv").exists()
    assert (plots_dir / "model_score_matrix.png").exists()
    assert (plots_dir / "predicted_vs_actual_margin.png").exists()
    assert (plots_dir / "parity" / "predicted_vs_actual_margin_model_a.png").exists()
    assert (plots_dir / "margin_fit_diagnostics.png").exists()
    assert (plots_dir / "margin_error_quantiles.png").exists()
    assert (plots_dir / "table_heatmaps" / "overall_margin_metrics.png").exists()
    assert (plots_dir / "overall_winner_chalk_upset.png").exists()


def test_save_metric_tables_default_skips_optional_csv_groups(tmp_path):
    table = pd.DataFrame({"metric": ["value"], "model_a": [1.0]})
    tables = {
        "model_score_matrix": pd.DataFrame(
            {"source": ["model_a"], "total_score": [1.0]}
        ),
        "overall_winner_metrics": table,
        "overall_margin_metrics": table,
        "overall_vegas_alignment_metrics": pd.DataFrame({"model": ["model_a"]}),
        "margin_diagnostics": pd.DataFrame({"source": ["model_a"], "mae": [1.0]}),
        "winner_breakdown_counts": table,
        "game_predictions": pd.DataFrame({"keys_team": ["A"]}),
        "prediction_sanity": table,
        "weekly_rmse": table,
        "confidence_bucket_accuracy": table,
        "ats_summary": table,
    }

    tables_dir = save_metric_tables(tables, tmp_path)

    assert (tables_dir / "model_score_matrix.csv").exists()
    assert (tables_dir / "game_predictions.csv").exists()
    assert not (tables_dir / "weekly_rmse.csv").exists()
    assert not (tables_dir / "confidence_bucket_accuracy.csv").exists()
    assert not (tables_dir / "ats_summary.csv").exists()


def test_move_legend_outside_creates_external_legend():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4], label="model_a")
    move_legend_outside(ax)

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "Legend"
    plt.close(fig)


def test_column_normalized_matrix_scales_each_metric_independently():
    frame = pd.DataFrame(
        {
            "small": [1.0, 2.0, 3.0],
            "large": [1000.0, 2000.0, 3000.0],
            "flat": [7.0, 7.0, 7.0],
        }
    )

    matrix = column_normalized_matrix(frame)

    assert np.allclose(matrix[:, 0], [0.0, 0.5, 1.0])
    assert np.allclose(matrix[:, 1], [0.0, 0.5, 1.0])
    assert np.allclose(matrix[:, 2], [0.5, 0.5, 0.5])
