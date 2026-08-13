import pandas as pd

from gridiron_ml.publication.baselines import fit_predict_baseline


def test_point_differential_baseline_uses_only_precomputed_to_date_columns():
    train = pd.DataFrame({"actual_margin": [10.0, -4.0], "actual_home_win": [1, 0]})
    test = pd.DataFrame(
        {
            "home_point_diff_to_date": [7.0],
            "away_point_diff_to_date": [2.0],
            "actual_margin": [1000.0],  # never read for prediction
        }
    )
    out = fit_predict_baseline(
        train,
        test,
        name="season_to_date_point_differential",
    )
    assert out.loc[0, "pred_margin"] == 5.0


def test_vegas_baseline_is_explicit_and_separate():
    train = pd.DataFrame({"actual_margin": [10.0], "actual_home_win": [1]})
    test = pd.DataFrame({"market_spread_close": [-3.5]})
    out = fit_predict_baseline(train, test, name="vegas_declared_line")
    assert out.loc[0, "baseline"] == "vegas_declared_line"
    assert out.loc[0, "pred_margin"] == -3.5
