"""src.gridiron_ml.td_sim.probability.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Run recursive season simulations with evolving synthetic fingerprints.
"""

import numpy as np
import pandas as pd


def sigmoid_margin_to_prob(margin, scale=13.5, random_scaling_factor=1.0):
    """Convert predicted home margin into home win probability.

    Positive margins favor the home team. Larger random_scaling_factor values
    flatten the sigmoid and make simulated outcomes less deterministic.
    """
    adjusted_scale = max(float(scale) * max(float(random_scaling_factor), 1e-8), 1e-8)
    margin = pd.to_numeric(pd.Series(margin), errors="coerce").astype(float)
    logits = np.clip(margin / adjusted_scale, -60.0, 60.0)
    return pd.Series(1.0 / (1.0 + np.exp(-logits)), index=margin.index)


def clip_probabilities(probabilities, min_prob=0.01, max_prob=0.99):
    """Run the clip_probabilities step and return its normalized result."""
    prob = pd.to_numeric(pd.Series(probabilities), errors="coerce").astype(float)
    return prob.clip(float(min_prob), float(max_prob))


def sample_margin_with_noise(margin, rng, noise_std, random_scaling_factor=1.0):
    """Run the sample_margin_with_noise step and return its normalized result."""
    margin = np.asarray(pd.to_numeric(pd.Series(margin), errors="coerce"), dtype=float)
    std = float(noise_std) * float(random_scaling_factor)
    return margin + rng.normal(0.0, std, size=len(margin))


def score_from_margin(margin, total_points=None, rng=None, score_noise_std=5.0):
    """Generate plausible integer scores from a predicted home margin.

    This is intentionally lightweight. It gives TD Sim game-level scorelines
    for evolving simulated fingerprints without pretending to be a scoring model.
    """
    rng = rng or np.random.default_rng()
    margin = np.asarray(margin, dtype=float)
    if total_points is None:
        total = rng.normal(52.0, float(score_noise_std), size=len(margin))
    else:
        total = np.asarray(total_points, dtype=float)
        total = total + rng.normal(0.0, float(score_noise_std), size=len(margin))
    total = np.maximum(total, np.abs(margin) + 14.0)
    home = np.maximum(0.0, (total + margin) / 2.0)
    away = np.maximum(0.0, total - home)
    return np.rint(home).astype(int), np.rint(away).astype(int)
