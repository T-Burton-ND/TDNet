import json
from pathlib import Path

def test_frozen_scientific_inventory_matches_explicit_owner_registry():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/publication_2026/ROSTER_REGISTRY.json").read_text())
    assert len(registry["confirmatory_scientific_models"]) == 42
    assert len(registry["market_aware_comparison_models"]) == 12
    model_ids = registry["confirmatory_scientific_models"] + registry["market_aware_comparison_models"]
    assert len(model_ids) == 54
    assert {model_id.split("_")[1] for model_id in model_ids} == {f"F{i}" for i in range(9)}
