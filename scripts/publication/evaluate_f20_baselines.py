#!/usr/bin/env python3
"""Exploratory paired 2025 check of frozen market and F20-A baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
from fit_f20_baselines import ROOT, cohort
from sklearn.metrics import brier_score_loss

from gridiron_ml.f20.model import BASE_COLUMNS, FEATURE_COLUMNS, historical_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=ROOT / "data/private/f20/historical")
    parser.add_argument("--schedule-dir", type=Path, default=ROOT / "data/raw/cfbd/v2/games")
    parser.add_argument("--models", type=Path, default=ROOT / "data/private/f20/models/baseline_2017_2024_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/publication_2026/f20_2025_exploratory.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Exploratory evaluation already exists")
    manifest = json.loads((args.models / "manifest.json").read_text())
    if manifest["training_seasons"][-1] != 2024:
        raise ValueError("2025 is not held out from fitting")
    data, audit = cohort(args.historical, args.schedule_dir, range(2025, 2026))
    X = historical_features(data)
    y = data.home_win.to_numpy(int)
    scores = {}
    game_losses = {}
    for arm, columns in {"market_only": BASE_COLUMNS, "F20-A": FEATURE_COLUMNS}.items():
        path = args.models / f"{arm}.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["model_sha256"][arm]:
            raise ValueError(f"Model hash mismatch: {arm}")
        p = joblib.load(path).predict_proba(X.loc[:, list(columns)])[:, 1]
        scores[arm] = float(brier_score_loss(y, p))
        game_losses[arm] = (p - y) ** 2
    report = {"status": "exploratory_consumed_2025", "matched_games": len(data),
              "paired_cohort": True, "brier": scores,
              "market_minus_f20a_brier": scores["market_only"] - scores["F20-A"],
              "source_audit": audit["2025"],
              "interpretation": "Historical archive publication times unverified; 2025 was previously consumed by TDNet research. No model choice uses this result."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
