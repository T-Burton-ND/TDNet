"""Matched reducer search and fold-only transformation checks."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from gridiron_ml.experiments.constraint_free_extra_reducers import (
    CANDIDATES, extra_representation,
)


@pytest.mark.parametrize("kind", ["pca975", "pls32", "autoencoder32", "mi128",
                                    "raw_missing_scaled"])
def test_extra_reducer_is_fitted_only_on_training_rows(kind):
    rng = np.random.default_rng(1729)
    train = rng.normal(size=(150, 40))
    train[rng.random(train.shape) < 0.04] = np.nan
    future = rng.normal(size=(10, 40))
    labels = rng.normal(size=150)
    first = extra_representation(kind, 40)
    second = extra_representation(kind, 40)
    fit1 = first.fit_transform(train, labels)
    fit2 = second.fit_transform(train, labels)
    np.testing.assert_allclose(fit1, fit2)
    changed = future.copy()
    changed[0, 0] = 1e12
    np.testing.assert_allclose(first.transform(future)[1:], second.transform(changed)[1:])


def test_reducer_screen_and_promotion_are_symmetric(tmp_path, monkeypatch):
    import constraint_free_extra_search as search

    def fake_load(_tier):
        return np.zeros((4, 3)), pd.DataFrame(), [], {}

    monkeypatch.setattr(search, "load_broad_historical", fake_load)
    cfg = json.loads(search.CONFIG.read_text())
    assert tuple(cfg["candidates"]) == CANDIDATES
    screen = search.plan(tmp_path, 1)
    manifest = json.loads((tmp_path / "stage1/manifest.json").read_text())
    assert screen["per_tier"] == 6 * len(CANDIDATES)
    assert manifest["years_to_score"] == [2025]
    rows = []
    for task in manifest["tasks"]:
        rows.append({**task, "status": "success", "mean_mae": CANDIDATES.index(task["representation"]),
                     "mean_brier": 0.2})
    pd.DataFrame(rows).to_csv(tmp_path / "stage1/leaderboard.csv", index=False)
    confirm = search.plan(tmp_path, 2)
    promoted = json.loads((tmp_path / "stage2/manifest.json").read_text())
    assert confirm["per_tier"] == 12
    assert promoted["years_to_score"] == [2022, 2023, 2024, 2025]
    assert {task["representation"] for task in promoted["tasks"]} == set(CANDIDATES[:2])
