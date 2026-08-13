import pandas as pd
import pytest

from gridiron_ml.publication.consensus import build_equal_weight_consensus, select_compact_components


def _predictions():
    return pd.DataFrame([
        {"game_id": "g1", "model_name": "a", "pred_margin": 2.0, "season": 2022, "actual_margin": 1.0},
        {"game_id": "g1", "model_name": "b", "pred_margin": 4.0, "season": 2022, "actual_margin": 1.0},
        {"game_id": "g2", "model_name": "a", "pred_margin": -2.0, "season": 2023, "actual_margin": -1.0},
        {"game_id": "g2", "model_name": "b", "pred_margin": -4.0, "season": 2023, "actual_margin": -1.0},
    ])


def test_all_model_consensus_records_effective_membership_without_imputation():
    consensus, membership = build_equal_weight_consensus(_predictions())
    assert consensus.loc[consensus.game_id.eq("g1"), "consensus_margin"].iloc[0] == 3.0
    assert consensus["effective_model_count"].tolist() == [2, 2]
    assert len(membership) == 4


def test_compact_selection_rejects_holdout_or_prospective_seasons():
    with pytest.raises(ValueError, match="2025"):
        select_compact_components(_predictions().assign(season=2025), minimum_seasons=1)
