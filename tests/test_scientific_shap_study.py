import numpy as np
import pandas as pd

from gridiron_ml.publication.scientific_shap_study import aggregate_source_importance, source_effects


def test_source_importance_sums_home_and_away_absolute_contributions():
    values = np.array([[1.0, -2.0, 3.0, -4.0], [-1.0, 4.0, -5.0, 2.0]])

    result = aggregate_source_importance(values, ["a", "b"])

    assert result.set_index("source_feature").loc["a", "mean_abs_shap"] == 5.0
    assert result.set_index("source_feature").loc["b", "mean_abs_shap"] == 6.0
    assert np.isclose(result["normalized_importance"].sum(), 1.0)


def test_source_effects_retains_home_and_away_coordinates():
    values = np.array([[1.0, 2.0, 3.0, 4.0]])
    pairs = pd.DataFrame([[10.0, 20.0, 30.0, 40.0]])

    result = source_effects(values, pairs, ["a", "b"])

    assert result.to_dict("records") == [
        {"explained_row": 0, "team_side": "home", "source_feature": "a", "feature_value": 10.0, "shap_value": 1.0},
        {"explained_row": 0, "team_side": "home", "source_feature": "b", "feature_value": 20.0, "shap_value": 2.0},
        {"explained_row": 0, "team_side": "away", "source_feature": "a", "feature_value": 30.0, "shap_value": 3.0},
        {"explained_row": 0, "team_side": "away", "source_feature": "b", "feature_value": 40.0, "shap_value": 4.0},
    ]
