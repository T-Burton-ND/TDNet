#!/usr/bin/env python3
"""Compare F20 arms on one historical game cohort with rolling-origin fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from constraint_free_broad_universe import load_broad_historical
from fit_f20_baselines import cohort

from gridiron_ml.f20.model import BASE_COLUMNS, FEATURE_COLUMNS, historical_features
from gridiron_ml.f20.variants import EXPERT_ONLY, select_compact_tdnet

ARMS = ("market_only", "F20-A", "F20-B", "F20-C")
PAIRS = (("market_only", "F20-A"), ("F20-A", "F20-B"),
         ("F20-A", "F20-C"), ("F20-B", "F20-C"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_inputs(historical: Path, schedule_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                     pd.DataFrame, list[dict], dict]:
    tracker, tracker_audit = cohort(historical, schedule_dir, range(2017, 2026))
    matrix, meta, records, _ = load_broad_historical("F19")
    if len(meta) != len(matrix) or meta.target_game_id.duplicated().any():
        raise ValueError("F19 historical game identity is invalid")
    positions = pd.Series(np.arange(len(meta)), index=meta.target_game_id.astype(int))
    matched = tracker.loc[tracker.id.astype(int).isin(positions.index)].copy()
    matched = matched.sort_values(["season", "id"]).reset_index(drop=True)
    if matched.empty or matched.id.duplicated().any():
        raise ValueError("Tracker/F19 matched cohort is empty or duplicated")
    indices = positions.loc[matched.id.astype(int)].to_numpy(int)
    target = meta.iloc[indices].reset_index(drop=True)
    if not matched.season.reset_index(drop=True).eq(target.season).all():
        raise ValueError("Tracker and F19 season mismatch")
    if not matched.home_win.reset_index(drop=True).eq(target.next_game_margin.gt(0).astype(int)).all():
        raise ValueError("Tracker and F19 outcome mismatch")
    if np.isinf(matrix[indices]).any():
        raise ValueError("F19 input contains infinity")
    expert = historical_features(matched)
    names = [record.name for record in records]
    if len(names) != len(set(names)) or set(names) & set(FEATURE_COLUMNS):
        raise ValueError("F19 and expert features have duplicate names")
    f19 = pd.DataFrame(matrix[indices], columns=names)
    eligible = [{"name": record.name, "market_derived": record.market_derived,
                 "target_derived": False, "pregame_win_probability_derived": False,
                 "temporal_cutoff": "before_target_game",
                 "prospective_allowed": record.prospective_allowed,
                 "source_timestamp_semantics": record.timestamp_semantics}
                for record in records if record.prospective_allowed and not record.market_derived]
    if len(eligible) < 20:
        raise ValueError("F19 has too few audited TDNet candidates")
    identity = pd.DataFrame({"game_id": matched.id.astype(int),
                             "season": matched.season.astype(int),
                             "week": target.week.astype(int),
                             "home_win": matched.home_win.astype(int)})
    provenance = {"tracker_audit": tracker_audit,
                  "f19_historical_matrix_sha256": hashlib.sha256(
                      np.ascontiguousarray(matrix).tobytes()).hexdigest(),
                  "f19_historical_game_ids_sha256": hashlib.sha256(
                      meta.target_game_id.astype("int64").to_numpy().tobytes()).hexdigest(),
                  "f19_feature_count": len(names), "b_candidate_count": len(eligible),
                  "historical_expert_publication_timing": "unverified",
                  "historical_f19_market_quote_timing": "unverified"}
    return identity, expert, f19, eligible, provenance


def estimator(regularization: float):
    # Same probability model and candidate grid in every arm; preprocessing fits within each fold.
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=regularization, max_iter=2000))


def arm_columns(arm: str, selected: tuple[str, ...], f19: pd.DataFrame) -> list[str]:
    if arm == "market_only":
        return list(BASE_COLUMNS)
    if arm == "F20-A":
        return list(FEATURE_COLUMNS)
    if arm == "F20-B":
        return [*FEATURE_COLUMNS, *selected]
    if arm == "F20-C":
        return [*f19.columns, *EXPERT_ONLY]
    raise ValueError(f"Unknown F20 arm: {arm}")


def paired_interval(loss_a: np.ndarray, loss_b: np.ndarray, seed: int) -> dict:
    difference = loss_b - loss_a
    rng = np.random.default_rng(seed)
    means = np.empty(2000)
    for index in range(len(means)):
        means[index] = difference[rng.integers(len(difference), size=len(difference))].mean()
    return {"candidate_minus_reference": float(difference.mean()),
            "game_bootstrap_95pct": [float(v) for v in np.quantile(means, [.025, .975])]}


def evaluate(identity: pd.DataFrame, expert: pd.DataFrame, f19: pd.DataFrame,
             eligible: list[dict], c_grid: tuple[float, ...] = (1.0,)) -> tuple[pd.DataFrame, dict, dict,
                                                                                pd.DataFrame]:
    if not c_grid or any(not np.isfinite(c) or c <= 0 for c in c_grid):
        raise ValueError("Regularization grid must contain positive finite C values")
    features = pd.concat([expert, f19], axis=1)
    y = identity.home_win.to_numpy(int)
    seasons = identity.season.to_numpy(int)
    candidates = []
    folds = []
    final_models = {}
    for test_season in (2021, 2022, 2023, 2024):
        train = (seasons >= 2017) & (seasons < test_season)
        test = seasons == test_season
        if train.sum() < 1000 or test.sum() < 100:
            raise ValueError(f"Insufficient matched games for {test_season}")
        selected = select_compact_tdnet(f19.loc[train], identity.home_win.loc[train],
                                        identity.season.loc[train], eligible, max_features=20)
        folds.append({"test_season": test_season, "training_games": int(train.sum()),
                      "test_games": int(test.sum()), "b_selected_tdnet": list(selected)})
        for arm in ARMS:
            columns = arm_columns(arm, selected, f19)
            for c in c_grid:
                model = estimator(c).fit(features.loc[train, columns], y[train])
                p = model.predict_proba(features.loc[test, columns])[:, 1]
                if not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
                    raise ValueError(f"Invalid {arm} probability")
                part = identity.loc[test].copy()
                part["arm"] = arm
                part["logistic_C"] = c
                part["probability"] = p
                part["brier_loss"] = (p - y[test]) ** 2
                part["evaluation"] = "2021_2024_rolling_origin"
                candidates.append(part)
    grid_predictions = pd.concat(candidates, ignore_index=True)
    grid_scores = grid_predictions.groupby(["arm", "logistic_C"]).brier_loss.mean()
    selected_c = {arm: float(min(c_grid, key=lambda c: (grid_scores.loc[(arm, c)], c)))
                  for arm in ARMS}
    selected_cv = grid_predictions.loc[[row.logistic_C == selected_c[row.arm]
                                        for row in grid_predictions.itertuples()]].copy()
    final_train = seasons <= 2024
    final_test = seasons == 2025
    if not final_test.any() or final_train.sum() < 3000:
        raise ValueError("2025 comparison cohort is missing")
    final_selected = select_compact_tdnet(f19.loc[final_train], identity.home_win.loc[final_train],
                                          identity.season.loc[final_train], eligible, max_features=20)
    folds.append({"test_season": 2025, "training_games": int(final_train.sum()),
                  "test_games": int(final_test.sum()), "b_selected_tdnet": list(final_selected)})
    final_predictions = []
    for arm in ARMS:
        columns = arm_columns(arm, final_selected, f19)
        model = estimator(selected_c[arm]).fit(features.loc[final_train, columns], y[final_train])
        p = model.predict_proba(features.loc[final_test, columns])[:, 1]
        part = identity.loc[final_test].copy()
        part["arm"] = arm
        part["logistic_C"] = selected_c[arm]
        part["probability"] = p
        part["brier_loss"] = (p - y[final_test]) ** 2
        part["evaluation"] = "2025_consumed_exploratory"
        final_predictions.append(part)
        final_models[arm] = {"model": model, "feature_columns": columns}
    result = pd.concat([selected_cv, *final_predictions], ignore_index=True)
    if result.duplicated(["game_id", "arm"]).any():
        raise ValueError("Comparison has duplicate game/arm predictions")
    expected = set(ARMS)
    if any(set(group.arm) != expected for _, group in result.groupby("game_id")):
        raise ValueError("Comparison arms have different game cohorts")
    return result, {"folds": folds, "regularization_grid": list(c_grid),
                    "selected_C_by_2021_2024_cv": selected_c,
                    "development_grid_brier": {arm: {str(c): float(grid_scores.loc[(arm, c)])
                                                     for c in c_grid} for arm in ARMS},
                    "selection_bias_note": "The reported 2021–2024 CV score is also the C-selection score when the grid has multiple values.",
                    "b_max_tdnet_features": 20,
                    "b_selection": "training_only_univariate_F_statistic"}, final_models, grid_predictions


def summarize(predictions: pd.DataFrame) -> dict:
    report = {}
    for scope, cohort_rows in predictions.groupby("evaluation"):
        per_arm = {}
        pivot = cohort_rows.pivot(index="game_id", columns="arm", values="brier_loss")
        for arm, group in cohort_rows.groupby("arm"):
            p = np.clip(group.probability.to_numpy(float), 1e-12, 1 - 1e-12)
            y = group.home_win.to_numpy(int)
            per_arm[arm] = {"games": len(group), "brier": float(group.brier_loss.mean()),
                            "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                            "winner_accuracy": float(((p >= .5) == y).mean())}
        pairs = {f"{a}_vs_{b}": paired_interval(pivot[a].to_numpy(), pivot[b].to_numpy(),
                                               seed=2026 + index)
                 for index, (a, b) in enumerate(PAIRS)}
        report[scope] = {"games_per_arm": len(pivot), "arms": per_arm,
                         "paired_brier": pairs,
                         "per_season_brier": {
                             str(year): {arm: float(group.brier_loss.mean())
                                         for arm, group in rows.groupby("arm")}
                             for year, rows in cohort_rows.groupby("season")}}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=ROOT / "data/private/f20/historical")
    parser.add_argument("--schedule-dir", type=Path, default=ROOT / "data/raw/cfbd/v2/games")
    parser.add_argument("--output", type=Path, default=ROOT / "data/private/f20/abc_comparison_v1")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "docs/publication_2026/f20_abc_comparison_v1.json")
    parser.add_argument("--c-grid", default="1", help="Comma-separated logistic C values selected by 2021–2024 rolling CV.")
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("F20 A/B/C comparison output already exists")
    identity, expert, f19, eligible, provenance = paired_inputs(args.historical, args.schedule_dir)
    c_grid = tuple(float(item) for item in args.c_grid.split(","))
    predictions, design, final_models, grid_predictions = evaluate(
        identity, expert, f19, eligible, c_grid)
    metrics = summarize(predictions)
    args.output.mkdir(parents=True)
    path = args.output / "game_predictions.parquet"
    predictions.to_parquet(path, index=False)
    grid_path = args.output / "development_grid_predictions.parquet"
    grid_predictions.to_parquet(grid_path, index=False)
    models_dir = args.output / "models"
    models_dir.mkdir()
    model_receipts = {}
    for arm, fitted in final_models.items():
        model_path = models_dir / f"{arm}.joblib"
        joblib.dump(fitted["model"], model_path)
        model_receipts[arm] = {"sha256": digest(model_path),
                               "feature_columns": fitted["feature_columns"]}
    summary = {"status": "developmental_paired_comparison_not_canonical",
               "created_at_utc": datetime.now(UTC).isoformat(),
               "scope": "regular_FBS_vs_FBS_tracker_F19_matched_game_ids",
               "training_seasons": list(range(2017, 2025)),
               "2025_status": "consumed_exploratory_not_untouched_holdout",
               "provenance": provenance, "design": design, "metrics": metrics,
               "private_predictions_sha256": digest(path),
               "private_development_grid_predictions_sha256": digest(grid_path),
               "private_model_artifacts": model_receipts,
               "interpretation": "Historical expert publication and F19 market quote times are unverified. Use 2021–2024 rolling-origin results for development; 2025 is an already-consumed exploratory check. No arm is selected canonical here."}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"matched_games": len(identity),
                      "scope": {key: value["games_per_arm"] for key, value in metrics.items()},
                      "brier": {key: {arm: row["brier"] for arm, row in value["arms"].items()}
                                for key, value in metrics.items()},
                      "summary": str(args.summary), "private_output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
