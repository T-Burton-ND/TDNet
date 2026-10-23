#!/usr/bin/env python3
"""Fit frozen F20-A and market controls using only 2017–2024 archive rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.f20.capture import _key
from gridiron_ml.f20.model import (
    BASE_COLUMNS,
    FEATURE_COLUMNS,
    historical_features,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort(historical: Path, schedule_dir: Path, seasons: range) -> tuple[pd.DataFrame, dict]:
    frames = []
    audit = {}
    for season in seasons:
        source = historical / f"ncaa{season}.csv"
        schedule_path = schedule_dir / f"{season}.parquet"
        tracker = pd.read_csv(source)
        games = pd.read_parquet(schedule_path)
        games = games.loc[(games.season == season)
                          & games.season_type.astype(str).str.casefold().eq("regular")
                          & games.home_classification.astype(str).str.casefold().eq("fbs")
                          & games.away_classification.astype(str).str.casefold().eq("fbs")].copy()
        games["key"] = [(_key(h), _key(a)) for h, a in zip(games.home_team, games.away_team)]
        if games.id.duplicated().any():
            raise ValueError(f"Duplicate {season} schedule game ID")
        games = games.loc[~games.key.duplicated(keep=False)].copy()
        tracker["key"] = [(_key(h), _key(a)) for h, a in zip(tracker.Home, tracker.Road)]
        duplicates = tracker.key.duplicated(keep=False)
        tracker = tracker.loc[~duplicates].copy()
        joined = tracker.merge(games[["key", "id", "week", "home_points", "away_points"]],
                               on="key", how="inner", validate="one_to_one")
        scores = joined[["hscore", "vscore", "home_points", "away_points"]].apply(
            pd.to_numeric, errors="coerce")
        good = scores.notna().all(axis=1) & scores.hscore.eq(scores.home_points) & scores.vscore.eq(scores.away_points)
        joined = joined.loc[good].copy()
        joined["season"] = season
        joined["home_win"] = (joined.hscore > joined.vscore).astype(int)
        # Tie labels are excluded. Features below use only the explicit pregame whitelist.
        joined = joined.loc[joined.hscore.ne(joined.vscore)].copy()
        if joined.id.duplicated().any():
            raise ValueError(f"Duplicate {season} game ID")
        frames.append(joined)
        audit[str(season)] = {"archive_rows": len(pd.read_csv(source)),
                              "matched_scored_regular_fbs_games": len(joined),
                              "matched_games_by_cfbd_week": {
                                  str(k): int(v) for k, v in joined.week_y.value_counts().sort_index().items()},
                              "archive_sha256": sha(source),
                              "schedule_sha256": sha(schedule_path),
                              "provenance": "archived_pregame_unverified"}
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["season", "id"]).any():
        raise ValueError("Duplicate historical season/game ID")
    return result, audit


def fit_models(frame: pd.DataFrame) -> tuple[dict, dict]:
    if frame.season.max() > 2024 or frame.season.min() < 2017:
        raise ValueError("Training years must be 2017–2024 only")
    X = historical_features(frame)
    y = frame.home_win.astype(int).to_numpy()
    models, cv = {}, {}
    for arm, columns in {"market_only": BASE_COLUMNS, "F20-A": FEATURE_COLUMNS}.items():
        # Fixed C and solver; no search or selection on 2025/2026 outcomes.
        def estimator():
            return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                 StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        folds = []
        for test_season in range(2021, 2025):
            train = frame.season < test_season
            test = frame.season == test_season
            if not train.any() or not test.any():
                raise ValueError("Chronological fold is empty")
            candidate = estimator().fit(X.loc[train, list(columns)], y[train])
            probabilities = candidate.predict_proba(X.loc[test, list(columns)])[:, 1]
            folds.append({"test_season": test_season, "train_through": test_season - 1,
                          "games": int(test.sum()), "brier": float(brier_score_loss(y[test], probabilities))})
        models[arm] = estimator().fit(X.loc[:, list(columns)], y)
        cv[arm] = folds
    return models, cv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=ROOT / "data/private/f20/historical")
    parser.add_argument("--schedule-dir", type=Path, default=ROOT / "data/raw/cfbd/v2/games")
    parser.add_argument("--output", type=Path, default=ROOT / "data/private/f20/models/baseline_2017_2024_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Frozen baseline output exists")
    data, audit = cohort(args.historical, args.schedule_dir, range(2017, 2025))
    if len(data) < 3000:
        raise ValueError("Too few audited historical games")
    models, cv = fit_models(data)
    args.output.mkdir(parents=True)
    model_hashes = {}
    for arm, model in models.items():
        path = args.output / f"{arm}.joblib"
        joblib.dump(model, path)
        model_hashes[arm] = sha(path)
    manifest = {"status": "frozen_developmental", "fit_at_utc": datetime.now(UTC).isoformat(),
                "training_seasons": [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
                "training_games": len(data), "excluded_seasons": [2025, 2026],
                "historical_source_timing": "archived_pregame_unverified",
                "feature_columns": {"market_only": BASE_COLUMNS, "F20-A": FEATURE_COLUMNS},
                "estimator": "median imputation, standard scaling, L2 logistic regression C=1",
                "chronological_folds": cv, "source_audit": audit,
                "model_sha256": model_hashes}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"training_games": len(data), "chronological_folds": cv,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
