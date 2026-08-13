import numpy as np
import pandas as pd

from gridiron_ml.publication.inference import (
    calibration_summary,
    empirical_power_precision,
    equivalence_result,
    fit_margin_calibrator,
    holm_adjust,
    mcnemar_test,
    margin_to_probability,
    season_clustered_bootstrap,
    season_clustered_mean_bootstrap,
    temporal_cross_fitted_margin_calibration,
)


def test_margin_link_and_calibrator_are_bounded_and_monotone():
    margins = np.linspace(-10, 10, 40)
    y = (margins > 0).astype(int)
    raw = margin_to_probability(margins)
    calibrator = fit_margin_calibrator(margins, y)
    calibrated = calibrator.predict(margins)
    assert np.all((raw >= 0) & (raw <= 1))
    assert np.all(np.diff(calibrated) >= 0)
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_calibration_summary_reports_bins_and_metrics():
    summary = calibration_summary([0, 1, 1, 0], [0.2, 0.8, 0.7, 0.3], bins=4)
    assert summary["n"] == 4
    assert 0 <= summary["brier_score"] <= 1
    assert summary["probability_min"] >= 0
    assert summary["probability_max"] <= 1
    assert sum(row["count"] for row in summary["reliability_bins"]) == 4


def test_season_clustered_bootstrap_is_reproducible():
    frame = pd.DataFrame({"season": [2020] * 3 + [2021] * 3, "difference": [-1, 0, 1, 2, 3, 4]})
    first = season_clustered_bootstrap(frame, n_resamples=100, seed=7)
    second = season_clustered_bootstrap(frame, n_resamples=100, seed=7)
    assert first == second
    assert first["n_seasons"] == 2


def test_vectorized_clustered_mean_bootstrap_is_reproducible():
    frame = pd.DataFrame({"season": [2020] * 3 + [2021] * 3, "difference": [-1, 0, 1, 2, 3, 4]})
    first = season_clustered_mean_bootstrap(frame, n_resamples=100, seed=7)
    second = season_clustered_mean_bootstrap(frame, n_resamples=100, seed=7)
    assert first == second
    assert first["estimate"] == frame["difference"].mean()
    assert first["n_seasons"] == 2


def test_temporal_calibration_never_uses_current_or_future_season():
    frame = pd.DataFrame({
        "keys_game_id": range(12),
        "season": np.repeat([2021, 2022, 2023], 4),
        "pred_margin": [-7, -2, 2, 7] * 3,
        "actual_margin": [-3, -1, 1, 4] * 3,
    })
    result = temporal_cross_fitted_margin_calibration(frame)
    assert sorted(result["season"].unique().tolist()) == [2022, 2023]
    assert result.loc[result["season"].eq(2022), "calibration_train_through"].eq(2021).all()
    assert result.loc[result["season"].eq(2023), "calibration_train_through"].eq(2022).all()
    assert result["keys_game_id"].tolist() == list(range(4, 12))
    assert result["calibrated_probability_home"].between(0, 1).all()


def test_mcnemar_holm_and_equivalence():
    result = mcnemar_test([1, 1, 0, 0], [1, 0, 0, 1])
    assert result["discordant"] == 2
    assert np.allclose(holm_adjust([0.01, 0.02, 0.5]), [0.03, 0.04, 0.5])
    assert equivalence_result(0.01, -0.02, 0.03, 0.05)["decision"] == "practical_equivalence"


def test_power_precision_has_declared_sample_sizes():
    frame = pd.DataFrame({"season": np.repeat([2020, 2021, 2022], 20), "difference": np.tile(np.linspace(-1, 1, 20), 3)})
    output = empirical_power_precision(frame, sample_sizes=[20, 40], effect_sizes=[0.5], n_resamples=20, seed=4)
    assert output["sample_size"].tolist() == [20, 40]
    assert output["superiority_power"].between(0, 1).all()
