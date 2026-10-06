#!/usr/bin/env python3
"""Build the separate 2026 F17-market what-if from archived pregame snapshots."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/publication")]

from build_2026_f09_whatif import DATA, ROUND_DATA
from build_2026_f13_f16_whatif import extended_target_matrix, target_state
from nextgen_rounds_scientific_train import make_model
from nextgen_rounds_train import load_stage_matrix
from pregame_market_snapshot_2026 import build_pregame_market_state

from gridiron_ml.experiments.nextgen_rounds_market import MARKET_COLUMNS
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import (
    build_estimator,
    residual_probability,
)
from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file

MODEL_IDS = ("M1", "M2", "M3", "M4", "M5", "M10")
FAMILIES = {"M1": "linear", "M2": "spline", "M3": "tree", "M4": "boosted", "M5": "neural", "M10": "knn"}
PRED_DIR = DATA / "f17_market_predictions"
FEATURE_DIR = DATA / "f17_market_features"


def current_market_state(target_meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    result, report = build_pregame_market_state(target_meta)
    if set(MARKET_COLUMNS) - set(result):
        raise ValueError("Current market feature schema differs from the historical F17 contract")
    if len(result) != len(target_meta) or result.target_game_id.duplicated().any():
        raise ValueError("F17 archived pregame market matrix is incomplete")
    return result, report


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = DATA / "f13_f16_features/f13_f16_target_state_2026.parquet"
    if not state_path.exists():
        raise FileNotFoundError("Build F13–F16 current-season states first")
    combined_state = target_state(pd.read_parquet(state_path))
    x16, target_meta, _ = extended_target_matrix(
        combined_state, load_stage_matrix(ROUND_DATA, "F16")[3]
    )
    xhist, meta, _, evidence = load_stage_matrix(ROUND_DATA, "F17_market")
    evidence16 = load_stage_matrix(ROUND_DATA, "F16")[3]
    if evidence["source_features"] != evidence16["source_features"]:
        raise ValueError("F17 market changed the inherited source-feature ordering")
    if evidence["market_features"] != list(MARKET_COLUMNS):
        raise ValueError("F17 market columns differ from the frozen market contract")
    market, market_report = current_market_state(target_meta)
    market = target_meta[["target_game_id"]].merge(market, on="target_game_id", validate="one_to_one")
    spreads = market.market_home_spread.to_numpy(float)
    market.to_parquet(FEATURE_DIR / "f17_market_target_state_2026.parquet", index=False, compression="zstd")
    x_target = np.column_stack([x16, market[list(MARKET_COLUMNS)].to_numpy(float)])
    if x_target.shape[1] != xhist.shape[1]:
        raise ValueError(f"Historical/target F17 matrix mismatch: {xhist.shape[1]} vs {x_target.shape[1]}")
    columns = [f"matchup__{name}" for name in evidence["source_features"]] + list(evidence["market_features"])
    X = pd.DataFrame(xhist, columns=columns)
    T = pd.DataFrame(x_target, columns=columns)
    y = meta.next_game_margin.to_numpy(float)
    cfg = json.loads((ROOT / "configs/experiments/nextgen_rounds_scientific_models_v1.json").read_text())
    points = json.loads((ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json").read_text())
    path = PRED_DIR / "predictions.parquet"
    old_receipt_path = PRED_DIR / "receipt.json"
    prior_market = None
    if old_receipt_path.exists():
        prior_market = json.loads(old_receipt_path.read_text()).get("market_source")
    # A changed target-market snapshot invalidates every saved F17 prediction.
    records = (pd.read_parquet(path).to_dict("records")
               if path.exists() and prior_market == market_report else [])
    completed = {str(row["model_name"]).rsplit("_", 1)[-1] for row in records}
    for architecture in MODEL_IDS:
        if architecture in completed:
            continue
        print(f"F17-market what-if refit {architecture}", flush=True)
        specs = ([('setpoint', point) for point in points[architecture]]
                 if architecture in ("M2", "M4")
                 else [('seed', seed) for seed in ([1701] if architecture == "M3" else cfg["seeds"])])
        candidates = []
        for kind, spec in specs:
            if kind == "setpoint":
                saved = pd.read_parquet(ROUND_DATA / "experiments/F17_market" / architecture / spec["id"] / "predictions.parquet")
                residuals = (saved.actual_margin - saved.predicted_margin).to_numpy(float)
                model = build_estimator(architecture, spec)
                model.fit(X, y)
                prediction = np.asarray(model.predict(T), dtype=float)
            else:
                seed = int(spec)
                saved = pd.read_parquet(ROUND_DATA / "scientific_model_runs/experiments/F17_market" / architecture / f"seed_{seed}" / "predictions.parquet")
                residuals = (saved.actual_margin - saved.predicted_margin).to_numpy(float)
                model, _ = make_model(architecture, seed, "F17_market")
                model.train(X, y)
                prediction = np.asarray(model.predict_margin(T), dtype=float).reshape(-1)
            candidates.append((prediction, residual_probability(prediction, residuals)))
        prediction = np.mean([candidate[0] for candidate in candidates], axis=0)
        probability = np.mean([candidate[1] for candidate in candidates], axis=0)
        for index, game in target_meta.reset_index(drop=True).iterrows():
            records.append({
                "game_id": int(game.target_game_id), "season": 2026, "week": int(game.week),
                "home_team": game.home_team, "away_team": game.away_team,
                "model_name": f"scientific_F17-market_{architecture}",
                "model_family": FAMILIES[architecture], "fingerprint": "F17-market",
                "pred_home_margin": float(prediction[index]),
                "pred_home_win_probability": float(probability[index]),
                "market_spread_close": float(spreads[index]) if np.isfinite(spreads[index]) else np.nan,
                "market_win_probability": float(market.market_home_implied_no_vig.iloc[index])
                if np.isfinite(market.market_home_implied_no_vig.iloc[index]) else np.nan,
                "quote_time_status": "pregame_snapshot_quote_time_unavailable",
                "market_snapshot_captured_pre_kickoff": bool(market.market_snapshot_captured_pre_kickoff.iloc[index]),
                "market_snapshot_timestamp_utc": str(market.snapshot_timestamp_utc.iloc[index])
                if pd.notna(market.snapshot_timestamp_utc.iloc[index]) else None,
            })
        pd.DataFrame(records).to_parquet(path, index=False, compression="zstd")
        completed.add(architecture)
    receipt = {
        "status": "success_pregame_snapshot_quote_time_unavailable" if len(completed) == len(MODEL_IDS) else "partial",
        "generation": "F17-market", "models": sorted(completed),
        "games": int(target_meta.target_game_id.nunique()), "features": int(x_target.shape[1]),
        "training_through_season": 2025, "calibration_years": [2024, 2025],
        "market_features_used": True, "market_research_only": True,
        "quote_timestamp_available": False,
        "all_available_target_market_snapshots_pre_kickoff": True,
        "target_games_without_market_snapshot": market_report["games_without_pregame_market_snapshot"],
        "M3_replicates": 1, "other_seed_replicates": 3, "M2_M4_setpoints": 10,
        "market_source": market_report,
        "prediction_sha256": sha256_file(path),
        "market_features_sha256": sha256_file(FEATURE_DIR / "f17_market_target_state_2026.parquet"),
    }
    (PRED_DIR / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    (FEATURE_DIR / "receipt.json").write_text(json.dumps(market_report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
