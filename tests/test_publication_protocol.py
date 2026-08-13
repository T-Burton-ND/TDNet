from pathlib import Path

import pandas as pd
import pytest
import yaml

from gridiron_ml.publication.protocol import (
    CANONICAL_TIERS,
    materialize_feature_manifest,
    resolve_tier_families,
    validate_feature_frame,
    validate_ladder_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_ladder_is_nested_and_has_explicit_market_boundary():
    ladder = yaml.safe_load((ROOT / "configs/features/feature_ladders.yaml").read_text())
    report = validate_ladder_config(ladder)
    assert tuple(report["canonical_order"]) == CANONICAL_TIERS
    market_free_ladder = tuple(f"F{i}" for i in range(7))
    for previous, current in zip(market_free_ladder, market_free_ladder[1:]):
        assert set(report["resolved_families"][previous]).issubset(report["resolved_families"][current])
    assert "market" not in report["resolved_families"]["F6"]
    assert report["resolved_families"]["F7"] == ["market"]
    assert set(report["resolved_families"]["F8"]) == (
        set(report["resolved_families"]["F6"]) | {"market"}
    )


def test_f7_is_market_only_and_f8_is_f6_plus_market():
    ladder = yaml.safe_load((ROOT / "configs/features/feature_ladders.yaml").read_text())
    assert "F8" in ladder["canonical_order"]
    assert resolve_tier_families("F7", ladder) == {"market"}
    assert resolve_tier_families("F8", ladder) == (
        resolve_tier_families("F6", ladder) | {"market"}
    )


def test_f0_is_exact_minimal_strength_baseline():
    manifest = materialize_feature_manifest(
        [
            "games_played",
            "roster_talent",
            "target_points_for_avg",
            "statOff_first_downs",
        ],
        tier="F0",
        registry_path=ROOT / "configs/features/feature_registry.yaml",
        ladder_path=ROOT / "configs/features/feature_ladders.yaml",
    )
    assert manifest["label"] == "Minimal strength baseline"
    assert manifest["feature_names"] == ["games_played", "roster_talent"]


def test_feature_manifest_has_stable_order_and_hash():
    columns = [
        "target_points_for_avg",
        "statOff_success_rate",
        "market_spread_close",
        "keys_game_id",
    ]
    first = materialize_feature_manifest(
        columns,
        tier="F8",
        registry_path=ROOT / "configs/features/feature_registry.yaml",
        ladder_path=ROOT / "configs/features/feature_ladders.yaml",
    )
    second = materialize_feature_manifest(
        list(reversed(columns)),
        tier="F8",
        registry_path=ROOT / "configs/features/feature_registry.yaml",
        ladder_path=ROOT / "configs/features/feature_ladders.yaml",
    )
    assert first["feature_names"] == sorted(first["feature_names"])
    assert first["schema_hash"] == second["schema_hash"]
    assert first["feature_count"] == len(first["feature_names"])


def test_market_boundary_rejects_market_column_in_f6():
    with pytest.raises(ValueError, match="Market"):
        materialize_feature_manifest(
            ["target_points_for_avg", "market_spread_close"],
            tier="F6",
            registry_path=ROOT / "configs/features/feature_registry.yaml",
            ladder_path=ROOT / "configs/features/feature_ladders.yaml",
        )


def test_market_sidecar_can_coexist_with_market_free_frame():
    manifests = validate_feature_frame(
        pd.DataFrame({
            "games_played": [0],
            "target_points_for_avg": [1.0],
            "market_spread_close": [-3.5],
        }),
        registry_path=ROOT / "configs/features/feature_registry.yaml",
        ladder_path=ROOT / "configs/features/feature_ladders.yaml",
    )
    assert manifests["F6"]["market_derived"] is False


def test_frame_validator_materializes_all_canonical_tiers():
    frame = pd.DataFrame(
        {
            "games_played": [0],
            "target_points_for_avg": [1.0],
            "statOff_success_rate": [0.5],
        }
    )
    manifests = validate_feature_frame(
        frame,
        registry_path=ROOT / "configs/features/feature_registry.yaml",
        ladder_path=ROOT / "configs/features/feature_ladders.yaml",
    )
    assert list(manifests) == list(CANONICAL_TIERS)
    assert all("schema_hash" in value for value in manifests.values())
    market = materialize_feature_manifest(
        [*frame.columns, "market_spread_close"],
        tier="F8",
        registry_path=ROOT / "configs/features/feature_registry.yaml",
        ladder_path=ROOT / "configs/features/feature_ladders.yaml",
    )
    assert "market_spread_close" in market["feature_names"]


def test_market_only_manifest_excludes_football_features():
    market = materialize_feature_manifest(
        ["games_played", "target_points_for_avg", "market_spread_close"],
        tier="F7",
        registry_path=ROOT / "configs/features/feature_registry.yaml",
        ladder_path=ROOT / "configs/features/feature_ladders.yaml",
    )
    assert market["feature_names"] == ["market_spread_close"]
