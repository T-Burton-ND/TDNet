import numpy as np
import pandas as pd
import pytest

from gridiron_ml.publication.knn_oof import run_strict_oof_knn


def _frame():
    rows = []
    for season, week in [(2020, 1), (2020, 2), (2020, 3), (2020, 4), (2020, 5), (2020, 6)]:
        rows.append({
            "game_id": f"g{week}", "season": season, "week": week,
            "home_team": f"H{week}", "away_team": f"A{week}",
            "x": float(week), "y_margin": float(week - 3),
        })
    return pd.DataFrame(rows)


def test_strict_oof_knn_has_disjoint_temporal_neighbors_and_audit():
    frame = _frame()
    predictions, audit = run_strict_oof_knn(
        frame,
        feature_columns=["x"],
        target_column="y_margin",
        folds=[
            {"train_indices": [0, 1], "test_indices": [2]},
            {"train_indices": [0, 1, 2, 3], "test_indices": [4]},
        ],
        config={"model_type": "distance", "params": {"n_neighbors": 2}},
    )
    assert len(predictions) == 2
    assert len(audit) >= 2
    assert {"neighbor_game_id", "neighbor_season", "neighbor_week", "neighbor_distance"}.issubset(audit.columns)
    assert all(
        str(target) not in set(group["neighbor_game_id"].astype(str))
        for target, group in audit.groupby("target_game_id")
    )


def test_strict_oof_knn_rejects_non_temporal_fold():
    with pytest.raises(ValueError, match="strictly temporal"):
        run_strict_oof_knn(
            _frame(),
            feature_columns=["x"],
            target_column="y_margin",
            folds=[{"train_indices": [0, 3], "test_indices": [2]}],
        )
