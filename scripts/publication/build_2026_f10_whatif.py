#!/usr/bin/env python3
"""Train the scientific roster through 2025 and forecast 2026 with F10-A."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/publication")]
from build_2026_f09_whatif import DATA, ROUND_DATA, target_matrix  # noqa: E402
from nextgen_rounds_train import load_stage_matrix  # noqa: E402
from nextgen_rounds_scientific_train import make_model  # noqa: E402
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import (  # noqa: E402
    build_estimator, residual_probability,
)
from gridiron_ml.models import TDKNN, TDLinear, TDMLP, TDTree  # noqa: E402,F401

OUT = DATA / "f10_predictions"


def _records(architecture, prediction, probability, meta, spreads):
    family = {"M1": "linear", "M2": "spline", "M3": "tree",
              "M4": "boosted", "M5": "neural", "M10": "knn"}[architecture]
    rows = []
    for i, row in meta.reset_index(drop=True).iterrows():
        rows.append({"game_id": int(row.target_game_id), "season": 2026, "week": int(row.week),
                     "home_team": row.home_team, "away_team": row.away_team,
                     "model_name": f"scientific_F10_{architecture}", "model_family": family,
                     "fingerprint": "F10", "pred_home_margin": float(prediction[i]),
                     "pred_home_win_probability": float(probability[i]),
                     "market_spread_close": float(spreads[i]) if np.isfinite(spreads[i]) else np.nan,
                     "market_win_probability": np.nan})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state9 = pd.read_parquet(DATA / "f09_predictions/f09_target_state.parquet")
    state10 = pd.read_parquet(DATA / "f10_features/f10_target_state.parquet")
    manifest = json.loads(Path("/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/fingerprints/F12_F_a/feature_manifest.json").read_text())
    f10_names = [r["name"] for r in manifest if r["generation"] == "F10"]
    if len(f10_names) != 56 or set(f10_names) - set(state10):
        raise ValueError("F10 target state does not cover the complete frozen A feature family")
    state = state9.merge(state10[["target_game_id", "team", *f10_names]],
                         on=["target_game_id", "team"], how="inner", validate="one_to_one")
    x_hist, meta, _, evidence = load_stage_matrix(ROUND_DATA, "F10")
    x_target, target_meta, spreads = target_matrix(state, evidence)
    names = [f"matchup__{name}" for name in evidence["source_features"]]
    hist, target = pd.DataFrame(x_hist, columns=names), pd.DataFrame(x_target, columns=names)
    y, years = meta.next_game_margin.to_numpy(float), meta.season.to_numpy(int)
    heldout_root = ROUND_DATA / "experiments/F10"
    seeds_config = json.loads((ROOT / "configs/experiments/nextgen_rounds_scientific_models_v1.json").read_text())
    points = json.loads((ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json").read_text())
    prediction_path = OUT / "predictions.parquet"
    rows = pd.read_parquet(prediction_path).to_dict("records") if prediction_path.exists() else []
    complete = {str(row["model_name"]).rsplit("_", 1)[-1] for row in rows}
    family = {"M1": "linear", "M2": "spline", "M3": "tree",
              "M4": "boosted", "M5": "neural", "M10": "knn"}
    for row in rows:
        row["model_family"] = family.get(str(row["model_name"]).rsplit("_", 1)[-1], row["model_family"])
    for architecture in ("M1", "M2", "M3", "M4", "M5", "M10"):
        if architecture in complete:
            print(f"F10 {architecture} already saved", flush=True)
            continue
        print(f"F10 refit {architecture}", flush=True)
        if architecture in ("M2", "M4"):
            replicates = [("setpoint", p) for p in points[architecture]]
        else:
            seeds = [1701] if architecture == "M3" else seeds_config["seeds"]
            replicates = [("seed", seed) for seed in seeds]
        individual = []
        for kind, spec in replicates:
            if kind == "setpoint":
                saved = pd.read_parquet(heldout_root / architecture / spec["id"] / "predictions.parquet")
                residuals = (saved.actual_margin - saved.predicted_margin).to_numpy(float)
                model = build_estimator(architecture, spec)
                model.fit(hist, y)
                pred = np.asarray(model.predict(target), dtype=float)
            else:
                seed = int(spec)
                saved = pd.read_parquet(ROUND_DATA / f"scientific_model_runs/experiments/F10/{architecture}/seed_{seed}/predictions.parquet")
                residuals = (saved.actual_margin - saved.predicted_margin).to_numpy(float)
                model, _ = make_model(architecture, seed, "F10")
                model.train(hist, y)
                pred = np.asarray(model.predict_margin(target), dtype=float).reshape(-1)
            individual.append((pred, residual_probability(pred, residuals)))
        pred = np.mean([item[0] for item in individual], axis=0)
        probability = np.mean([item[1] for item in individual], axis=0)
        rows.extend(_records(architecture, pred, probability, target_meta, spreads))
        pd.DataFrame(rows).to_parquet(prediction_path, index=False, compression="zstd")
        complete.add(architecture)
        print(f"F10 {architecture} saved", flush=True)
    missing = sorted(set(("M1", "M2", "M3", "M4", "M5", "M10")) - complete)
    receipt = {"status": "success" if not missing else "partial", "generation": "F10 A",
               "models": sorted(complete), "unscored_models": missing,
               "target_games": int(target_meta.target_game_id.nunique()),
               "target_rows": int(len(target_meta)), "f10_source_features": len(f10_names),
               "training_through_season": 2025, "calibration_source_years": [2024, 2025],
               "calibration_models_trained_through_season": 2023,
               "market_features_used": False, "feature_state_join": "F0-F8 weekly ladder + corrected F9 state + full F10 A state",
               "frozen_recruiting_classes": [2022, 2023, 2024, 2025],
               "M3_replicates": 1, "other_seed_replicates": 3, "M2_M4_setpoints": 10}
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
