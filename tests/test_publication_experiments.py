from pathlib import Path

import pandas as pd
import pytest

from gridiron_ml.experiments.publication import (
    assert_disk_guardrail,
    build_experiment_manifest,
    expand_feature_registry,
    filter_frame_for_feature_config,
    materialize_split_rows,
    selected_feature_families,
    select_finalists,
)
from gridiron_ml.fingerprints.features import split_frame


ROOT = Path(__file__).resolve().parents[1]


def test_feature_registry_ignores_contract_columns_and_expands_features():
    columns = ["keys_season", "next_game_id", "fp_version", "target_points_for_avg", "statOff_success_rate"]
    registry = expand_feature_registry(columns, registry_path=ROOT / "configs/features/feature_registry.yaml")
    assert set(registry) == {"target_points_for_avg", "statOff_success_rate"}


def test_feature_registry_fails_closed_on_unknown_predictor():
    with pytest.raises(ValueError, match="no entry"):
        expand_feature_registry(["totally_new_predictor"], registry_path=ROOT / "configs/features/feature_registry.yaml")


def test_split_and_ladder_configs_materialize_without_2026():
    rows = materialize_split_rows(ROOT / "configs/splits/rolling_origin.yaml")
    assert rows
    assert all(2026 not in row["train_seasons"] + row["val_seasons"] + row["test_seasons"] for row in rows)
    import yaml
    ladders = yaml.safe_load((ROOT / "configs/features/feature_ladders.yaml").read_text())
    assert "market" in selected_feature_families("F8", ladders)


def test_disk_guardrail_accepts_new_descendant(tmp_path):
    assert_disk_guardrail(tmp_path / "not" / "created" / "yet", 0)


def test_market_line_is_retained_as_eval_sidecar_for_market_free_tier():
    frame = pd.DataFrame(
        {
            "keys_season": [2025],
            "offense_ppa": [0.2],
            "market_spread_close": [-3.5],
            "y_next_margin": [7.0],
        }
    )
    filtered, metadata = filter_frame_for_feature_config(
        frame,
        feature_config="F6",
        registry_path=ROOT / "configs/features/feature_registry.yaml",
        ladders_path=ROOT / "configs/features/feature_ladders.yaml",
    )
    features, _, _, market = split_frame(filtered)

    assert "offense_ppa" in metadata["selected_features"]
    assert "market_spread_close" not in metadata["selected_features"]
    assert "market_spread_close" not in features
    assert "market_spread_close" in market


def test_one_run_per_task_hyperparameter_manifest(tmp_path):
    manifest, chunks = build_experiment_manifest(
        project_root=ROOT,
        config_path=ROOT / "configs/publication/hps_spline.yaml",
        data_path=ROOT / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet",
        output_root=tmp_path,
        max_trials=9,
    )
    assert len(manifest) == len(chunks) == 9
    assert chunks["trial_count"].eq(1).all()
    assert manifest["params_json"].nunique() > 1
    assert manifest["chunk_id"].equals(manifest["task_id"])


def test_corrected_f6_wide_manifest_disables_legacy_temporal_expansion(tmp_path):
    manifest, _ = build_experiment_manifest(
        project_root=ROOT,
        config_path=ROOT / "configs/publication/hps_corrected_f6_wide_margin_gap.yaml",
        data_path=ROOT / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet",
        output_root=tmp_path,
        max_trials=1,
    )

    assert manifest["temporal_feature_expansion"].eq(False).all()


def test_finalist_selection_aggregates_the_same_hyperparameters_across_folds():
    rows = []
    for params, scores in [("{\"alpha\": 1}", [.20, .22]), ("{\"alpha\": 10}", [.30, .32])]:
        for fold, score in enumerate(scores):
            rows.append({"status": "success", "objective": "winner", "feature_config": "F1",
                         "model_level": "K1", "model_family": "kernel", "model_config": "k.yaml",
                         "params_json": params, "seed": 1, "outer_fold": fold,
                         "experiment_id": "same-cell", "brier_score": score})
    selected = select_finalists(pd.DataFrame(rows), max_per_cell=1)
    assert len(selected) == 1
    assert selected.iloc[0]["params_json"] == "{\"alpha\": 1}"
    assert selected.iloc[0]["cv_fold_count"] == 2


def test_finalist_selection_can_exclude_incomplete_parameter_sets():
    rows = []
    for params, scores in [
        ('{"alpha": 1}', [.10]),
        ('{"alpha": 10}', [.20, .22]),
    ]:
        for fold, score in enumerate(scores):
            rows.append(
                {
                    "status": "success",
                    "objective": "winner",
                    "feature_config": "F8",
                    "model_level": "M3",
                    "model_family": "tree",
                    "model_config": "tree.yaml",
                    "params_json": params,
                    "seed": 1,
                    "outer_fold": fold,
                    "brier_score": score,
                }
            )

    selected = select_finalists(
        pd.DataFrame(rows), max_per_cell=1, minimum_fold_count=2
    )

    assert len(selected) == 1
    assert selected.iloc[0]["params_json"] == '{"alpha": 10}'
    assert selected.iloc[0]["cv_fold_count"] == 2
