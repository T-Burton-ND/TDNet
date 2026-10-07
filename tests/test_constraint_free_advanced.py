"""Focused checks for fold-local advanced representations and matched searches."""
from __future__ import annotations

import numpy as np

from gridiron_ml.experiments.constraint_free import metadata_for_matrix
from gridiron_ml.experiments.constraint_free_advanced import (
    GenomeRepresentation, genome_hash, random_genome,
)


def test_genome_representation_ignores_held_out_sentinel_during_fit():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(80, 10))
    held_out = rng.normal(size=(10, 10))
    names = [f"matchup__feature_{i}" for i in range(10)]
    records = metadata_for_matrix(names, "F18")
    genome = {
        "mode": "hybrid", "variance": 0.9, "select_count": 64,
        "correlation": 0.0, "nonlinear": True,
        "family_mask": 2 ** 2, "market_on": False, "robust": False,
        "hp_index": 1, "target_mode": "direct",
    }
    first = GenomeRepresentation(genome, records, "F18").fit(train, rng.normal(size=80))
    second = GenomeRepresentation(genome, records, "F18").fit(train, rng.normal(size=80))
    changed = held_out.copy()
    changed[0, 0] = 1e12
    np.testing.assert_allclose(first.transform(held_out)[1:], first.transform(changed)[1:])
    assert first.columns_.tolist() == second.columns_.tolist()


def test_ga_uses_exact_candidate_budget_and_records_parents(monkeypatch, tmp_path):
    import constraint_free_advanced_search as search

    calls = []

    def fake_candidate(tier, architecture, genome, **kwargs):
        calls.append((genome_hash(genome), kwargs["generation"], kwargs["parents"]))
        return {
            "status": "success", "genome": genome, "genome_hash": genome_hash(genome),
            "mean_mae": float(len(calls) % 11), "mean_brier": 0.2,
            "worst_year_mae": 20.0,
        }

    monkeypatch.setattr(search, "candidate", fake_candidate)
    cfg = {"seed": 1729, "architectures": ["M1"], "ga_population": 8,
           "ga_generations": 3, "ga_elites": 2, "development_years": [2025]}
    results = search.run_ga("F18", "M1", tmp_path, None, None, None, cfg)
    assert len(results) == 24
    assert len({item[0] for item in calls}) == 24
    assert [sum(item[1] == generation for item in calls) for generation in range(3)] == [8, 8, 8]
    assert all(item[2] for item in calls[8:])


def test_random_genome_is_valid_and_hash_stable():
    rng = np.random.default_rng(1701)
    genome = random_genome(rng)
    assert genome_hash(genome) == genome_hash(dict(reversed(list(genome.items()))))
