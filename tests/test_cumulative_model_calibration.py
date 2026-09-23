from pathlib import Path

import pandas as pd

from gridiron_ml.publication.model_calibration import (
    build_model_calibration_table,
    load_cumulative_model_predictions,
)


def test_cumulative_calibration_uses_current_roster_and_equal_count_bins(
    tmp_path: Path,
):
    root = tmp_path / "publication" / "2026"
    relative = "tables/models.csv"
    for week, models in ((0, ["keep", "retired"]), (1, ["keep"])):
        path = root / f"week_{week:02d}" / "post_game" / relative
        path.parent.mkdir(parents=True)
        rows = []
        for model in models:
            for game in range(6):
                rows.append(
                    {
                        "game_id": f"{week}-{game}",
                        "model_name": model,
                        "model_family": "linear",
                        "fingerprint": "F0",
                        "pred_home_win_probability": 0.1 + game * 0.15,
                        "actual_home_margin": 1 if game >= 3 else -1,
                    }
                )
        pd.DataFrame(rows).to_csv(path, index=False)

    predictions = load_cumulative_model_predictions(
        publication_root=root,
        completed_week=1,
        relative_path=relative,
    )
    assert set(predictions["model_name"]) == {"keep"}
    assert len(predictions) == 12

    calibration = build_model_calibration_table(predictions, bins=3)
    assert len(calibration) == 3
    assert calibration["games"].sum() == 12
    assert calibration["predicted_home_win_probability"].is_monotonic_increasing
    assert calibration["observed_home_win_rate"].between(0, 1).all()
