#!/usr/bin/env python3
"""Apply frozen F18/F19 bundles to reconstructed Week 6–7 inputs.

F19 Week 7 uses market data retrieved after games; results are retrospective
and are never passed through the prospective prediction API.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import constraint_free_2026_matrix as builder  # noqa: E402
from constraint_free_fit_finalists import _predict  # noqa: E402
from gridiron_ml.experiments.constraint_free import SNAPSHOT_MARKET_COLUMNS  # noqa: E402

STATES = ROOT / "data/what_if_2026_fingerprints/week06_07_retrospective_states"
OUT = ROOT / "publication/2026/week_07/post_game/retrospective_replay/scientific/full_f0_f19"
FROZEN = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/broad_search/final_fit/freeze/freeze_manifest.json")
MARKET = OUT / "f17_f19_market_snapshot.parquet"
ROSTER = ("M1", "M2", "M3", "M4", "M5", "M10")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_schedule() -> pd.DataFrame:
    frame = pd.read_parquet(STATES / "target_schedule_without_scores.parquet")
    frame = frame.rename(columns={"id": "target_game_id", "start_date": "target_start_utc"})
    frame["target_start_utc"] = pd.to_datetime(frame.target_start_utc, utc=True)
    frame = frame.sort_values("target_game_id").reset_index(drop=True)
    if len(frame) != 117 or set(frame.week) != {6, 7}:
        raise ValueError("Week 6–7 target schedule changed")
    return frame[["target_game_id", "week", "target_start_utc", "home_team", "away_team"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    builder.LADDER = ROOT / "data/publication/2026/weekly_operations/week_06/fingerprint_ladder_v3/canonical_fingerprint.parquet"
    builder.STATE_PATHS = {
        "F09": STATES / "f09_target_state.parquet",
        "F10": STATES / "f10_target_state.parquet",
        "F11": STATES / "f11_target_state.parquet",
        "F12": STATES / "f12_target_state.parquet",
        "research": STATES / "research_target_state.parquet",
    }
    builder.target_schedule = target_schedule
    f18, meta, report = builder.build()
    if len(f18) != 117 or not f18.target_game_id.equals(meta.target_game_id):
        raise ValueError("F18 retrospective matrix does not match target schedule")
    market = pd.read_parquet(MARKET)
    if len(market) != 117 or set(market.target_game_id) != set(meta.target_game_id):
        raise ValueError("Retrospective market cohort mismatch")
    f19 = f18.merge(market[["target_game_id", *SNAPSHOT_MARKET_COLUMNS]],
                    on="target_game_id", validate="one_to_one")
    _, _, records, _ = builder.load_broad_historical("F19")
    if list(f19.columns[1:]) != [r.name for r in records]:
        raise ValueError("F19 historical column contract changed")
    files = {"F18": OUT / "f18_inputs.parquet", "F19": OUT / "f19_inputs.parquet"}
    f18.to_parquet(files["F18"], index=False, compression="zstd")
    f19.to_parquet(files["F19"], index=False, compression="zstd")
    freeze = json.loads(FROZEN.read_text())
    if freeze["training_seasons"][-1] != 2025:
        raise ValueError("Frozen bundles have post-2025 training labels")
    rows, hashes = [], {}
    market_provenance = market.set_index("target_game_id").market_provenance
    for tier, matrix in (("F18", f18), ("F19", f19)):
        for model_id in ROSTER:
            cell = next(c for c in freeze["cells"] if c["tier"] == tier and c["architecture"] == model_id)
            path = Path(cell["model_bundle"])
            if digest(path) != cell["model_bundle_sha256"]:
                raise ValueError(f"Frozen model hash changed: {tier}/{model_id}")
            with path.open("rb") as handle:
                bundle = pickle.load(handle)
            names = bundle["source_feature_names"]
            if list(matrix.columns[1:]) != names:
                raise ValueError(f"Frozen input order changed: {tier}/{model_id}")
            raw = matrix[names].to_numpy(float)
            transformed = bundle["transformer"].transform(raw)
            margin = _predict(bundle["model"], bundle["backend"], transformed)
            if bundle["genome"] and bundle["genome"]["target_mode"] == "residual":
                margin = (margin - raw[:, names.index("market_home_spread")]
                          if tier == "F19" else margin + bundle["residual_baseline_mean"])
            cal = freeze["margin_to_win_probability_calibration"][tier][model_id]
            prob = expit(cal["intercept"] + cal["margin_coefficient"] * margin)
            if len(margin) != 117 or not np.isfinite(margin).all() or not np.isfinite(prob).all():
                raise ValueError(f"Invalid {tier}/{model_id} retrospective prediction")
            row = meta[["target_game_id", "week", "home_team", "away_team"]].rename(columns={"target_game_id": "game_id"}).copy()
            row["fingerprint"] = tier
            row["model_id"] = model_id
            row["model_name"] = f"scientific_{tier}_{model_id}"
            row["pred_home_margin"] = margin
            row["pred_home_win_probability"] = prob
            row["provenance_class"] = ("retrospective_score_free_features" if tier == "F18" else
                row.game_id.map(market_provenance).map({
                    "frozen_pregame_summary": "retrospective_features_frozen_market",
                    "week06_ex_post_market_fallback": "retrospective_ex_post_market_after_games",
                    "ex_post_oct21_cache_unverified_quote_time": "retrospective_ex_post_market_after_games",
                }))
            rows.append(row)
            hashes[str(path)] = digest(path)
    result = pd.concat(rows, ignore_index=True)
    path = OUT / "f18_f19_retrospective_predictions.parquet"
    result.to_parquet(path, index=False, compression="zstd")
    report.update({"market_provenance": market.groupby("market_provenance").size().to_dict(),
                   "issued_before_kickoff": False, "frozen_model_sha256": hashes,
                   "target_input_sha256": {k: digest(v) for k, v in files.items()},
                   "market_source_sha256": digest(MARKET), "predictions_sha256": digest(path),
                   "historical_quote_timing_verified": False})
    (OUT / "f18_f19_retrospective_receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print({"games": 117, "model_cells": 12, "rows": len(result)}, flush=True)


if __name__ == "__main__":
    main()
