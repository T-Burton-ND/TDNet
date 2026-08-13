import numpy as np
import pandas as pd

from gridiron_ml.td_run.shap_analysis import save_shap_analysis_for_models


class DummyLinearShapModel:
    model_name = "dummy_linear"
    model_family = "linear"
    feature_names_ = ["a", "b"]
    weights_ = np.array([2.0, -1.0])
    intercept_ = 0.5
    standardize = False
    medians_ = pd.Series({"a": 0.0, "b": 0.0})
    means_ = pd.Series({"a": 0.0, "b": 0.0})
    stds_ = pd.Series({"a": 1.0, "b": 1.0})

    def _transform_features(self, X):
        return X.loc[:, self.feature_names_].to_numpy(dtype=float)


def test_save_shap_analysis_for_linear_model_writes_pngs_and_table(tmp_path):
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0, 0.0]})

    artifacts = save_shap_analysis_for_models(
        [{"name": "dummy_linear", "model": DummyLinearShapModel()}],
        X,
        tmp_path,
        eval_config={
            "artifacts": {
                "shap": True,
                "shap_summary_plots": True,
                "shap_bar_plots": True,
            },
            "shap": {"max_background": 2, "max_explain": 3, "max_display": 2},
        },
    )

    assert artifacts.loc[0, "status"] == "ok"
    assert (tmp_path / "plots" / "shap" / "dummy_linear_shap_summary.png").exists()
    assert (tmp_path / "plots" / "shap" / "dummy_linear_shap_bar.png").exists()
    assert (tmp_path / "tables" / "shap" / "dummy_linear_shap_importance.csv").exists()
