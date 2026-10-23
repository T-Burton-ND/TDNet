#!/usr/bin/env python3
"""Freeze six F20 margin regressors beside the unchanged probability models."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

from fit_f20_baselines import ROOT, cohort, sha
from gridiron_ml.f20.model import FEATURE_COLUMNS, historical_features
from gridiron_ml.f20.scientific import MODEL_IDS, margin_estimator, predict_margin_panel


def fit(frame):
    if set(frame.season) != set(range(2017, 2025)):
        raise ValueError("F20 margin training seasons must be exactly 2017–2024")
    features = historical_features(frame)[list(FEATURE_COLUMNS)]
    target = (frame.hscore.astype(float) - frame.vscore.astype(float)).to_numpy()
    if not np.isfinite(target).all():
        raise ValueError("Historical margin target is incomplete")
    folds = {model_id: [] for model_id in (*MODEL_IDS, "F20")}
    for season in range(2021, 2025):
        train = frame.season.lt(season).to_numpy()
        test = frame.season.eq(season).to_numpy()
        if train.sum() < 1000 or test.sum() < 100:
            raise ValueError("Chronological margin fold is too small")
        fitted = {model_id: margin_estimator(model_id).fit(features.loc[train], target[train])
                  for model_id in MODEL_IDS}
        predicted = predict_margin_panel(fitted, features.loc[test])
        for model_id, margin in predicted.items():
            folds[model_id].append({"test_season": season, "games": int(test.sum()),
                                    "mae": float(mean_absolute_error(target[test], margin))})
    final = {model_id: margin_estimator(model_id).fit(features, target) for model_id in MODEL_IDS}
    weighted = {model_id: float(np.average([item["mae"] for item in items],
                                           weights=[item["games"] for item in items]))
                for model_id, items in folds.items()}
    return final, folds, weighted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=ROOT / "data/private/f20/historical")
    parser.add_argument("--schedule-dir", type=Path, default=ROOT / "data/raw/cfbd/v2/games")
    parser.add_argument("--probability-models", type=Path,
                        default=ROOT / "data/private/f20/models/f20_scientific_2017_2024_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/private/f20/models/f20_scientific_2017_2024_v2_margin")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Frozen F20 output already exists: {args.output}")
    old = json.loads((args.probability_models / "manifest.json").read_text())
    if (old.get("canonical_name") != "F20" or old.get("training_seasons") != list(range(2017, 2025))
            or old.get("model_ids") != list(MODEL_IDS)
            or old.get("feature_columns") != list(FEATURE_COLUMNS)):
        raise ValueError("Existing F20 probability panel differs from frozen contract")
    for model_id in MODEL_IDS:
        if sha(args.probability_models / f"{model_id}.joblib") != old["model_sha256"][model_id]:
            raise ValueError(f"Existing F20/{model_id} probability model hash differs")
    frame, audit = cohort(args.historical, args.schedule_dir, range(2017, 2025))
    if len(frame) != old["training_games"]:
        raise ValueError("Margin training games differ from probability panel")
    if audit != old["source_audit"]:
        raise ValueError("Historical source audit differs from probability panel")
    models, folds, weighted = fit(frame)
    args.output.mkdir(parents=True)
    for model_id in MODEL_IDS:
        shutil.copy2(args.probability_models / f"{model_id}.joblib", args.output / f"{model_id}.joblib")
    hashes = {}
    for model_id, model in models.items():
        path = args.output / f"{model_id}_margin.joblib"
        joblib.dump(model, path)
        hashes[model_id] = sha(path)
    manifest = {**old, "status": "frozen_scientific_probability_and_margin_panel",
                "fit_at_utc": datetime.now(UTC).isoformat(),
                "probability_fit_at_utc": old["fit_at_utc"],
                "margin_model_sha256": hashes,
                "margin_consensus": "equal_mean_of_six_predicted_home_margins",
                "margin_chronological_folds": folds, "pooled_fold_margin_mae": weighted,
                "margin_training_target": "home_points_minus_away_points",
                "margin_model_was_frozen_before_2026_games": False,
                "probability_parent_manifest_sha256": sha(args.probability_models / "manifest.json")}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "training_games": len(frame),
                      "pooled_fold_margin_mae": weighted}, indent=2))


if __name__ == "__main__":
    main()
