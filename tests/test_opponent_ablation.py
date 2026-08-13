from pathlib import Path

import pandas as pd

from gridiron_ml.experiments.opponent_ablation import (
    DEFAULT_ABLATION_SPECS,
    ablation_spec_from_name,
    apply_ablation_view,
    build_ablation_job_manifest,
    build_shap_master_fragment,
    summarize_ablation_features,
)
from gridiron_ml.experiments.opponent_adjusted import OpponentAdjustedVersionSpec
from gridiron_ml.td_run.training import ModelRunSpec


def test_ablation_views_keep_expected_feature_families():
    frame = pd.DataFrame(
        {
            "keys_season": [2025, 2025],
            "keys_team": ["A", "B"],
            "keys_week": [1, 1],
            "next_game_id": [10, 10],
            "next_opponent": ["B", "A"],
            "next_game_is_home": [True, False],
            "next_week": [2, 2],
            "y_next_margin": [7.0, -7.0],
            "y_has_next_game": [True, True],
            "statOff_yards_per_pass": [8.2, 6.1],
            "opp_adj_v1_4_offense_ppa_mean_to_date": [0.2, -0.1],
            "opp_adj_v1_4_games_played": [1.0, 1.0],
            "opp_adj_v1_4_elo_rating_edge_ewm": [12.0, -12.0],
            "market_spread_close": [-3.5, -3.5],
            "fp_subversion": [14, 14],
        }
    )

    raw = apply_ablation_view(frame, ablation_spec_from_name("raw_baseline"))
    residuals = apply_ablation_view(
        frame,
        ablation_spec_from_name("adjusted_residuals_only"),
    )
    context = apply_ablation_view(frame, ablation_spec_from_name("context_only"))
    raw_context = apply_ablation_view(frame, ablation_spec_from_name("raw_plus_context"))

    assert "statOff_yards_per_pass" in raw.columns
    assert "opp_adj_v1_4_offense_ppa_mean_to_date" not in raw.columns
    assert "fp_subversion" not in raw.columns
    assert "next_week" in raw.columns
    assert "market_spread_close" in raw.columns

    assert "opp_adj_v1_4_offense_ppa_mean_to_date" in residuals.columns
    assert "opp_adj_v1_4_games_played" not in residuals.columns
    assert "statOff_yards_per_pass" not in residuals.columns

    assert "opp_adj_v1_4_games_played" in context.columns
    assert "opp_adj_v1_4_elo_rating_edge_ewm" in context.columns
    assert "opp_adj_v1_4_offense_ppa_mean_to_date" not in context.columns

    assert "statOff_yards_per_pass" in raw_context.columns
    assert "opp_adj_v1_4_games_played" in raw_context.columns
    assert "opp_adj_v1_4_offense_ppa_mean_to_date" not in raw_context.columns


def test_ablation_feature_summary_counts_roles():
    frame = pd.DataFrame(
        {
            "keys_season": [2025],
            "keys_team": ["A"],
            "keys_week": [1],
            "y_next_margin": [7.0],
            "y_has_next_game": [True],
            "statOff_yards_per_pass": [8.2],
            "opp_adj_v1_4_offense_ppa_mean_to_date": [0.2],
            "opp_adj_v1_4_games_played": [1.0],
        }
    )
    ablated = apply_ablation_view(frame, ablation_spec_from_name("raw_plus_adjusted_all"))

    summary = summarize_ablation_features(ablated)

    assert summary["feature_count"] == 3
    assert summary["raw_feature_count"] == 1
    assert summary["adjusted_residual_feature_count"] == 1
    assert summary["adjusted_context_feature_count"] == 1


def test_build_ablation_manifest_without_fingerprint_build(tmp_path: Path):
    version = OpponentAdjustedVersionSpec("v1.1", "opponent_context", "test")
    model = ModelRunSpec("ridge", "linear", "configs/models/linear/config_ridge.yaml")

    manifest = build_ablation_job_manifest(
        project_root=tmp_path,
        output_root=tmp_path / "out",
        source_fingerprint_root=tmp_path / "source",
        version_specs=(version,),
        ablation_specs=(DEFAULT_ABLATION_SPECS[0],),
        model_specs=(model,),
        ensure_fingerprints=False,
    )

    assert len(manifest) == 1
    assert manifest.loc[0, "job_index"] == 0
    assert manifest.loc[0, "sge_task_id"] == 1
    assert (tmp_path / "out" / "job_manifest.csv").exists()
    assert (tmp_path / "out" / "ablation_manifest.json").exists()
    assert (tmp_path / "out" / "README.md").exists()


def test_shap_master_fragment_adds_feature_metadata(tmp_path: Path):
    shap_dir = tmp_path / "season_eval" / "tables" / "shap"
    shap_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "feature": [
                "home_opp_adj_v1_4_offense_ppa_mean_to_date",
                "net_statOff_yards_per_pass",
            ],
            "mean_abs_shap": [0.5, 0.2],
            "mean_shap": [0.1, -0.05],
        }
    ).to_csv(shap_dir / "ridge_shap_importance.csv", index=False)

    fragment = build_shap_master_fragment(
        output_dir=tmp_path,
        version_spec=OpponentAdjustedVersionSpec("v1.4", "elo_context", "test"),
        ablation_spec=ablation_spec_from_name("raw_plus_adjusted_all"),
        model_spec=ModelRunSpec("ridge", "linear", "configs/models/linear/config_ridge.yaml"),
    )

    first = fragment.iloc[0]
    assert first["feature_role"] == "adjusted_residual"
    assert first["feature_side"] == "home"
    assert first["feature_normalized"] == "opp_adj_offense_ppa_mean_to_date"
    assert (tmp_path / "shap_importance_master_fragment.csv").exists()
