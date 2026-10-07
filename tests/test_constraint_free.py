"""Safety properties for the paired F18/F19 representation experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron_ml.experiments.constraint_free import (
    FeatureRecord, SNAPSHOT_MARKET_COLUMNS, assert_feature_contract,
    metadata_for_matrix, representation,
)


def test_f18_rejects_market_by_metadata_and_name():
    tagged = FeatureRecord("market_home_spread", "market", True, "CFBD", "pregame")
    with pytest.raises(ValueError, match="F18 market contamination"):
        assert_feature_contract([tagged.name], [tagged], "F18")
    disguised = FeatureRecord("matchup__betting_provider_signal", "historical")
    with pytest.raises(ValueError, match="Untagged market-like"):
        assert_feature_contract([disguised.name], [disguised], "F18")


def test_f19_requires_explicit_and_prospectively_available_provenance():
    with pytest.raises(ValueError, match="explicitly include"):
        assert_feature_contract(["matchup__tempo"],
                                [FeatureRecord("matchup__tempo", "historical")], "F19")
    with pytest.raises(ValueError, match="Incomplete market provenance"):
        assert_feature_contract(["market_home_spread"],
                                [FeatureRecord("market_home_spread", "market", True)], "F19")
    with pytest.raises(ValueError, match="Unavailable prospective"):
        assert_feature_contract(["market_quote_count"],
                                [FeatureRecord("market_quote_count", "market", True,
                                               "CFBD", "pregame", True)], "F19")
    names = ["matchup__tempo", *SNAPSHOT_MARKET_COLUMNS]
    assert_feature_contract(names, metadata_for_matrix(names, "F19"), "F19")


@pytest.mark.parametrize("kind", ["pca90", "family_pca90", "select128", "robust_pca90"])
def test_held_out_outcome_and_sentinel_do_not_change_fitted_transform(kind):
    rng = np.random.default_rng(1701)
    x_train = rng.normal(size=(60, 8))
    y_train = rng.normal(size=60)
    held_out = rng.normal(size=(10, 8))
    names = [f"matchup__feature_{i}" for i in range(8)]
    records = metadata_for_matrix(names, "F18")
    first = representation(kind, records)
    second = representation(kind, records)
    fit_first = first.fit_transform(x_train, y_train)
    fit_second = second.fit_transform(x_train, y_train)
    np.testing.assert_allclose(fit_first, fit_second)
    changed = held_out.copy()
    changed[0, 0] = 1e12
    # A target-season sentinel and outcome change can affect transformed
    # target rows, but cannot change the training-fitted preprocessing state.
    np.testing.assert_allclose(first.transform(x_train), second.transform(x_train))
    np.testing.assert_allclose(first.transform(held_out)[1:], second.transform(changed)[1:])


def test_search_input_fails_if_2026_outcome_enters(monkeypatch):
    import constraint_free_search as search

    def contaminated(_root, _stage):
        meta = pd.DataFrame({"target_game_id": [1, 2], "season": [2025, 2026],
                             "next_game_margin": [0.0, 1e12]})
        return np.zeros((2, 1)), meta, np.zeros(2), {
            "source_features": ["tempo"], "market_features": []}

    monkeypatch.setattr(search, "load_stage_matrix", contaminated)
    with pytest.raises(ValueError, match="outside 2013–2025"):
        search.load_historical("F18")


def test_manifest_task_dispatch_uses_representation_name(tmp_path, monkeypatch):
    import json
    import constraint_free_search as search

    manifest = {
        "config_sha256": search.digest(search.CONFIG),
        "source_sha256": search.digest(search.Path(search.__file__)),
        "safety_source_sha256": search.digest(
            search.ROOT / "src/gridiron_ml/experiments/constraint_free.py"),
        "tasks": [{"tier": "F18", "architecture": "M1", "representation": "raw"}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    seen = []

    def fake_evaluate(tier, architecture, kind, config):
        seen.append((tier, architecture, kind))
        return {"mean_mae": 1.0}

    monkeypatch.setattr(search, "evaluate_task", fake_evaluate)
    result = search.run_task(path, 1)
    assert result["status"] == "success"
    assert seen == [("F18", "M1", "raw")]
