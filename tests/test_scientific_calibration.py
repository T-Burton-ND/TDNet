import pandas as pd

from gridiron_ml.cli.publication.fit_scientific_frozen_calibrators import save_calibration_plot
from gridiron_ml.cli.publication.run_scientific_cross_fitted_oof import as_bool


def test_manifest_boolean_parser_and_current_market_contract():
    assert as_bool(True) is True
    assert as_bool("true") is True
    assert as_bool("False") is False


def test_dedicated_calibration_plot_writes_png_only(tmp_path):
    frame = pd.DataFrame({
        "actual_home_win": [0, 0, 0, 1, 0, 1, 1, 1] * 3,
        "calibrated_probability_home": [0.08, 0.16, 0.28, 0.39, 0.47, 0.63, 0.79, 0.92] * 3,
    })
    target = tmp_path / "F0" / "M1" / "calibration_curve"
    save_calibration_plot(frame, target, title="F0/M1 temporal calibration")
    assert target.with_suffix(".png").stat().st_size > 0
    assert not target.with_suffix(".svg").exists()
