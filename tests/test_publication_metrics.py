import pandas as pd
import pytest

from gridiron_ml.publication.metrics import chalk_upset_table, score_predictions


def test_canonical_metrics_handle_pickem_and_directional_chalk_recall():
    frame = pd.DataFrame({
        "pred_probability_home": [0.8, 0.8, 0.2, 0.8],
        "pred_margin": [7, 3, -2, 2],
        "actual_margin": [4, -7, -1, -1],
        "vegas_spread": [-3, 4, -3, 0],
    })
    result = score_predictions(frame)
    assert result["number_of_games"] == 4
    assert result["number_of_pickem_games_excluded"] == 1
    assert result["number_of_actual_chalk_wins"] == 2
    assert result["chalk_recall"] == pytest.approx(0.5)
    assert result["upset_recall"] == pytest.approx(1.0)


def test_chalk_upset_table_is_discrete_two_by_two():
    frame = pd.DataFrame({"pred_probability_home": [0.8, 0.8], "actual_margin": [4, -7], "vegas_spread": [-3, 4]})
    table = chalk_upset_table(frame)
    assert table.set_index("prediction").loc["favorite_predicted", "favorite_actually_wins"] == 1
    assert table.set_index("prediction").loc["upset_predicted", "favorite_actually_wins"] == 1
