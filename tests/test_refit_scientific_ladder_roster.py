from types import SimpleNamespace

import pandas as pd
import pytest

from gridiron_ml.cli.publication.refit_scientific_ladder_roster import finalize, roster_cell


def test_roster_cell_enumerates_nine_tiers_six_margin_models():
    cells = [roster_cell(i) for i in range(54)]
    assert len(set(cells)) == 54
    assert cells[0] == ("F0", "M1")
    assert cells[-1] == ("F8", "M10")
    with pytest.raises(ValueError):
        roster_cell(54)


def test_finalize_adds_stable_publication_model_labels(tmp_path):
    for task_id in range(54):
        tier, level = roster_cell(task_id)
        directory = tmp_path / "runs" / f"task_{task_id:03d}"
        directory.mkdir(parents=True)
        pd.DataFrame([{
            "model_id": f"scientific_{tier}_{level}",
            "feature_config": tier,
            "model_level": level,
        }]).to_parquet(directory / "inventory_fragment.parquet", index=False)
    finalize(SimpleNamespace(output_root=tmp_path, training_end_season=2024, holdout_season=2025))
    inventory = pd.read_csv(tmp_path / "final_model_inventory.csv")
    assert inventory["final_model_name"].equals(inventory["model_id"])
