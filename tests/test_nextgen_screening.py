import json
from pathlib import Path

import numpy as np
import pytest

from gridiron_ml.experiments.nextgen_screening import (
    source_to_matchup, future_metrics, residual_probability, generation_barrier,
    build_estimator, permutation_source_shap,
    verify_reusable_result,
)
from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file


def test_resume_rejects_changed_computation_and_corrupt_outputs(tmp_path):
    binding = {"setpoint": {"alpha": 1}, "market_sha256": "original"}
    names = ("predictions.parquet", "source_shap.parquet")
    for name in names:
        (tmp_path / name).write_bytes(b"verified output")
    prior = {"execution_binding": binding,
             "output_sha256": {name: sha256_file(tmp_path / name) for name in names}}
    verify_reusable_result(prior, binding, tmp_path)
    with pytest.raises(ValueError, match="computation"):
        verify_reusable_result(prior, {**binding, "market_sha256": "changed"}, tmp_path)
    (tmp_path / names[0]).write_bytes(b"truncated")
    with pytest.raises(ValueError, match="changed"):
        verify_reusable_result(prior, binding, tmp_path)
    (tmp_path / names[0]).unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_reusable_result(prior, binding, tmp_path)


def records():
    return [
        {"name": "offense", "matchup_counterpart": "defense", "matchup_formula": "home.[offense]+away.[defense]"},
        {"name": "defense", "matchup_counterpart": "offense", "matchup_formula": "home.[defense]+away.[offense]"},
        {"name": "talent", "matchup_counterpart": "talent", "matchup_formula": "home.[talent]-away.[talent]"},
    ]


def test_matchup_uses_declared_opposing_unit_and_global_contrasts():
    x = np.array([[10, 3, 8, 7, 2, 4]])
    np.testing.assert_equal(source_to_matchup(x, records()), [[12, 10, 4]])
    bad = records()
    bad[0]["matchup_formula"] = "undeclared formula"
    with pytest.raises(ValueError, match="declared"):
        source_to_matchup(x, bad)


def test_market_metrics_are_evaluation_only_with_push_exclusion():
    metrics = future_metrics([10, -3, 7], [8, 1, 6], [.8, .6, .7], [-7, -2, -7])
    assert metrics["mae"] == pytest.approx(7/3)
    assert metrics["ats_accuracy"] == 1
    assert metrics["chalk_accuracy"] == 1
    assert metrics["upset_accuracy"] == 0
    assert future_metrics([10], [8], [.8])["ats_accuracy"] is None


def test_calibration_is_monotone_and_never_exact_zero_or_one():
    p = residual_probability([-1000, 0, 1000], np.arange(-100., 100.))
    assert 0 < p[0] < p[1] < p[2] < 1
    with pytest.raises(ValueError, match="Insufficient"):
        residual_probability([1], [0])


def test_generation_barrier_requires_all_finalization_parts(tmp_path):
    generation_barrier(tmp_path, "F06")
    with pytest.raises(ValueError, match="has not finalized"):
        generation_barrier(tmp_path, "F09")
    path = tmp_path / "results/F06/finalization.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"training_finalized": True}))
    with pytest.raises(ValueError, match="incomplete"):
        generation_barrier(tmp_path, "F09")


def test_common_permutation_shap_closes_prediction_for_both_architectures():
    # Real estimator/explainer integration, tiny fixture only (not screening).
    points = json.loads((Path(__file__).resolve().parents[1] / "configs/experiments/nextgen_screening_setpoints_v1.json").read_text())
    rng = np.random.default_rng(1701)
    raw = rng.normal(size=(160, 6))
    x = source_to_matchup(raw, records())
    y = 3*x[:, 0]-2*x[:, 1]+x[:, 2]
    for architecture in ("M2", "M4"):
        m = build_estimator(architecture, points[architecture][0])
        m.fit(x, y)
        table, report = permutation_source_shap(m, records(), raw[:8], raw[-3:])
        assert report["max_absolute_additivity_error"] < 1e-6
        assert table.normalized_importance.sum() == pytest.approx(1)
