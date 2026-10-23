#!/usr/bin/env python3
"""Fit and freeze F20 M1–M5/M10 on 2017–2024 matched historical games."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from fit_f20_baselines import ROOT, cohort, sha
from sklearn.metrics import brier_score_loss

from gridiron_ml.f20.model import FEATURE_COLUMNS, historical_features
from gridiron_ml.f20.scientific import MODEL_IDS, estimator, predict_panel


def fit(frame):
    if set(frame.season) != set(range(2017, 2025)):
        raise ValueError("F20 training seasons must be exactly 2017–2024")
    features = historical_features(frame)[list(FEATURE_COLUMNS)]
    labels = frame.home_win.astype(int).to_numpy()
    scores = {model_id: [] for model_id in (*MODEL_IDS, "F20")}
    for season in range(2021, 2025):
        train = frame.season.lt(season).to_numpy()
        test = frame.season.eq(season).to_numpy()
        if train.sum() < 1000 or test.sum() < 100:
            raise ValueError("Chronological fold is too small")
        fitted = {model_id: estimator(model_id).fit(features.loc[train], labels[train])
                  for model_id in MODEL_IDS}
        fold = predict_panel(fitted, features.loc[test])
        for model_id, probability in fold.items():
            scores[model_id].append({"test_season": season, "games": int(test.sum()),
                                     "brier": float(brier_score_loss(labels[test], probability))})
    final = {model_id: estimator(model_id).fit(features, labels) for model_id in MODEL_IDS}
    weighted = {model_id: float(np.average([item["brier"] for item in folds],
                                           weights=[item["games"] for item in folds]))
                for model_id, folds in scores.items()}
    return final, scores, weighted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=ROOT / "data/private/f20/historical")
    parser.add_argument("--schedule-dir", type=Path, default=ROOT / "data/raw/cfbd/v2/games")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/private/f20/models/f20_scientific_2017_2024_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame, source_audit = cohort(args.historical, args.schedule_dir, range(2017, 2025))
    models, folds, weighted = fit(frame)
    args.output.mkdir(parents=True)
    hashes = {}
    for model_id, model in models.items():
        path = args.output / f"{model_id}.joblib"
        joblib.dump(model, path)
        hashes[model_id] = sha(path)
    manifest = {"canonical_name": "F20", "status": "frozen_scientific_probability_panel",
                "fit_at_utc": datetime.now(UTC).isoformat(),
                "training_seasons": list(range(2017, 2025)), "training_games": len(frame),
                "excluded_seasons": [2025, 2026], "historical_source_timing": "archived_pregame_unverified",
                "feature_columns": list(FEATURE_COLUMNS), "model_ids": list(MODEL_IDS),
                "consensus": "equal_mean_of_six_home_win_probabilities",
                "chronological_folds": folds, "pooled_fold_brier": weighted,
                "source_audit": source_audit, "model_sha256": hashes}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "training_games": len(frame),
                      "pooled_fold_brier": weighted}, indent=2))


if __name__ == "__main__":
    main()
