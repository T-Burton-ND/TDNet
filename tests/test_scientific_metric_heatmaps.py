import pandas as pd

from gridiron_ml.publication.scientific_metric_heatmaps import aggregate_metric_grid, selected_complete_trials


def test_heatmap_selection_excludes_incomplete_better_configuration_and_weights_ats():
    rows = []
    for params, folds, mae in [("incomplete", range(9), 1.0), ("complete", range(10), 2.0)]:
        for fold in folds:
            rows.append(
                {
                    "status": "success", "objective": "margin", "feature_config": "F6", "model_level": "M1",
                    "model_family": "linear", "model_config": "m.yaml", "params_json": params, "seed": 1,
                    "outer_fold": fold, "mae": mae, "winner_accuracy": .7, "brier_score": .2,
                    "favorite_correct": .8, "upset_correct": .3,
                    "ats_accuracy": .5 if fold == 0 else 1.0, "ats_n": 9 if fold == 0 else 1,
                    "n_rows": 10,
                }
            )
    selected = selected_complete_trials(pd.DataFrame(rows), expected_folds=10)
    table = aggregate_metric_grid(selected)
    assert selected["params_json"].unique().tolist() == ["complete"]
    assert table.loc[0, "mae"] == 2.0
    assert table.loc[0, "ats_accuracy"] == 0.75
