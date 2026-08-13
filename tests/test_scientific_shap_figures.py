from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gridiron_ml.publication.scientific_shap_figures import (
    FigureSettings,
    MODEL_ORDER,
    OBJECTIVE_ORDER,
    build_scientific_shap_figures,
    prepare_importance,
)


def _fixture_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = [f"feature_{index:02d}" for index in range(8)]
    families = [
        "structural", "box_score", "efficiency", "opponent_adjusted",
        "temporal", "temporal", "schedule_graph", "schedule_graph",
    ]
    contract = pd.DataFrame({"source_feature": features, "feature_family": families})
    importance_rows = []
    effect_rows = []
    for objective_index, objective in enumerate(OBJECTIVE_ORDER):
        for model_index, model in enumerate(MODEL_ORDER):
            for fold in range(2):
                for feature_index, feature in enumerate(features):
                    importance_rows.append(
                        {
                            "objective": objective,
                            "model_level": model,
                            "outer_fold": fold,
                            "source_feature": feature,
                            "mean_abs_shap": 1.0 + feature_index + model_index / 10 + fold / 20,
                            "explainer_method": "fixture",
                            "n_explained": 16,
                            "runtime_seconds": 2 + model_index,
                            "additivity_error": 1e-7,
                            "common_permutation_rank_rho": 0.85,
                        }
                    )
                    for sample in range(8):
                        value = -1.75 + 0.5 * sample
                        effect_rows.append(
                            {
                                "objective": objective,
                                "model_level": model,
                                "outer_fold": fold,
                                "source_feature": feature,
                                "feature_value_z": value,
                                "shap_value": value * (feature_index + 1) / 20 + objective_index / 100,
                            }
                        )
    return contract, pd.DataFrame(importance_rows), pd.DataFrame(effect_rows)


def test_renderer_covers_every_feature_in_every_all_feature_suite(tmp_path: Path) -> None:
    contract, importance, effects = _fixture_tables()
    report = build_scientific_shap_figures(
        importance=importance,
        effects=effects,
        feature_contract=contract,
        output_root=tmp_path,
        settings=FigureSettings(
            features_per_atlas_page=4,
            features_per_dependence_page=4,
            raster_dpi=30,
            minimum_valid_folds=2,
            bootstrap_resamples=30,
        ),
    )

    assert report["complete_coverage"] is True
    assert report["source_feature_count"] == 8
    manifest = pd.read_csv(tmp_path / "figure_manifest.csv")
    for figure_type in ["feature_atlas", "rank_stability", "direction_atlas", "dependence"]:
        for objective in OBJECTIVE_ORDER:
            rows = manifest.loc[
                manifest["figure_type"].eq(figure_type)
                & manifest["objective"].eq(objective)
            ]
            represented = set()
            for encoded in rows["features_json"]:
                represented.update(json.loads(encoded))
            assert represented == set(contract["source_feature"])

    summary = pd.read_csv(tmp_path / "all_feature_importance_summary.csv")
    assert {"importance_ci_low", "importance_ci_high", "cell_top_25_frequency"} <= set(summary)
    assert (tmp_path / "supplement_complete_feature_atlas.pdf").is_file()
    assert (tmp_path / "supplement_all_feature_dependence.pdf").is_file()
    assert (tmp_path / "figure_01_family_allocation.svg").is_file()


def test_incomplete_fold_feature_coverage_fails() -> None:
    contract, importance, _ = _fixture_tables()
    broken = importance.loc[
        ~(
            importance["objective"].eq("margin")
            & importance["model_level"].eq("M1")
            & importance["outer_fold"].eq(1)
            & importance["source_feature"].eq("feature_00")
        )
    ]
    with pytest.raises(ValueError, match="Incomplete all-feature SHAP coverage"):
        prepare_importance(broken, contract, minimum_valid_folds=2)


def test_duplicate_source_aggregation_rows_fail() -> None:
    contract, importance, _ = _fixture_tables()
    duplicate = pd.concat([importance, importance.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one source-aggregated row"):
        prepare_importance(duplicate, contract, minimum_valid_folds=2)
