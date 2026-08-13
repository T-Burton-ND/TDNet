import numpy as np

from gridiron_ml.publication.scientific_shap_study import aggregate_source_importance


def test_source_importance_sums_home_and_away_absolute_contributions():
    values = np.array([[1.0, -2.0, 3.0, -4.0], [-1.0, 4.0, -5.0, 2.0]])

    result = aggregate_source_importance(values, ["a", "b"])

    assert result.set_index("source_feature").loc["a", "mean_abs_shap"] == 5.0
    assert result.set_index("source_feature").loc["b", "mean_abs_shap"] == 6.0
    assert np.isclose(result["normalized_importance"].sum(), 1.0)
