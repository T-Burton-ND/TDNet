#!/usr/bin/env python3
"""Generate frozen F18/F19 forecasts from score-free 2026 inputs only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.special import expit

from constraint_free_fit_finalists import _predict
from constraint_free_freeze import apply_rule
from constraint_free_search import digest, write_json
from gridiron_ml.experiments.constraint_free import ROSTER

EXPERIMENT = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1")
FREEZE = EXPERIMENT / "broad_search/final_fit/freeze/freeze_manifest.json"
DEFAULT_OUTPUT = EXPERIMENT / "broad_search/final_fit/prospective"


def _stable_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _calibrate(margin: np.ndarray, spec: dict) -> np.ndarray:
    return expit(spec["intercept"] + spec["margin_coefficient"] * margin)


def predict(freeze_path: Path = FREEZE, output: Path = DEFAULT_OUTPUT) -> dict:
    frozen = json.loads(freeze_path.read_text())
    if not frozen.get("no_2026_outcomes_read") or frozen["primary_consensus"] != "equal":
        raise ValueError("Unverified prospective freeze")
    if (output / "predictions.parquet").exists() or (output / "receipt.json").exists():
        raise FileExistsError("Prospective predictions are immutable")
    for ref in (frozen["target_schedule_without_scores"],
                frozen["feature_source_availability"], frozen["target_input_receipt"]):
        if digest(Path(ref["path"])) != ref["sha256"]:
            raise ValueError(f"Frozen input changed: {ref['path']}")
    schedule = pd.read_parquet(frozen["target_schedule_without_scores"]["path"])
    availability = pd.read_parquet(frozen["feature_source_availability"]["path"])
    schedule = schedule.merge(availability, on="target_game_id", validate="one_to_one")
    market_ref = frozen["market_policy"]
    market_path = Path(market_ref["market_source"])
    if digest(market_path) != market_ref["market_source_sha256"]:
        raise ValueError("Archived market source changed after freeze")
    market = pd.read_parquet(market_path, columns=[
        "target_game_id", "snapshot_timestamp_utc", "market_snapshot_captured_pre_kickoff"])
    schedule = schedule.merge(market, on="target_game_id", validate="one_to_one")
    snapshot = pd.to_datetime(schedule.snapshot_timestamp_utc, utc=True, errors="coerce")
    start = pd.to_datetime(schedule.target_start_utc, utc=True)
    eligible_market = schedule.market_snapshot_captured_pre_kickoff.fillna(False).astype(bool)
    eligible_market &= snapshot < start
    if int(eligible_market.sum()) != market_ref["eligible_games"] or schedule.loc[
            ~eligible_market, "target_game_id"].astype(int).tolist() != market_ref["unavailable_game_ids"]:
        raise ValueError("F19 market eligibility changed after freeze")
    rows = []
    for tier in ("F18", "F19"):
        matrix_ref = frozen["target_input_matrices"][tier]
        if digest(Path(matrix_ref["path"])) != matrix_ref["sha256"]:
            raise ValueError(f"Frozen {tier} input matrix changed")
        matrix = pd.read_parquet(matrix_ref["path"])
        if len(matrix) != len(schedule) or not matrix.target_game_id.equals(schedule.target_game_id):
            raise ValueError(f"{tier} game identities changed")
        eligible = np.ones(len(schedule), dtype=bool) if tier == "F18" else eligible_market.to_numpy(bool)
        predictions = np.full((len(schedule), len(ROSTER)), np.nan)
        cell_refs = []
        for column, architecture in enumerate(ROSTER):
            cell = next((entry for entry in frozen["cells"]
                         if entry["tier"] == tier and entry["architecture"] == architecture), None)
            if cell is None:
                raise ValueError(f"Missing frozen model: {tier} {architecture}")
            path = Path(cell["model_bundle"])
            if digest(path) != cell["model_bundle_sha256"]:
                raise ValueError(f"Frozen model changed: {path}")
            if digest(Path(cell["preprocessing_state"])) != cell["preprocessing_state_sha256"]:
                raise ValueError("Frozen preprocessing state changed")
            with path.open("rb") as handle:
                bundle = pickle.load(handle)
            names = bundle["source_feature_names"]
            if list(matrix.columns[1:]) != names or bundle["tier"] != tier or bundle["architecture"] != architecture:
                raise ValueError("Prospective feature order differs from fitted model")
            data = matrix.loc[eligible, names].to_numpy(float)
            transformed = bundle["transformer"].transform(data)
            margin = _predict(bundle["model"], bundle["backend"], transformed)
            if bundle["genome"] and bundle["genome"]["target_mode"] == "residual":
                if tier == "F19":
                    spread = data[:, names.index("market_home_spread")]
                    if not np.isfinite(spread).all():
                        raise ValueError("Eligible F19 residual forecast lacks market spread")
                    margin = margin - spread
                else:
                    margin = margin + bundle["residual_baseline_mean"]
            if margin.shape != (int(eligible.sum()),) or not np.isfinite(margin).all():
                raise ValueError(f"Frozen {tier} {architecture} model produced invalid margins")
            predictions[eligible, column] = margin
            cell_refs.append(cell)
            calibration = frozen["margin_to_win_probability_calibration"][tier][architecture]
            probability = np.full(len(schedule), np.nan)
            probability[eligible] = _calibrate(margin, calibration)
            rows.append(_output_rows(schedule, tier, architecture, eligible, margin,
                                     probability, cell["preprocessing_state_sha256"],
                                     cell["model_bundle_sha256"], _stable_digest(calibration)))
            del bundle
        for method, rule in frozen["ensemble_rules"][tier].items():
            margins = apply_rule(predictions[eligible], rule)
            if not np.isfinite(margins).all():
                raise ValueError(f"Frozen {tier} {method} consensus produced nonfinite margins")
            calibration = frozen["margin_to_win_probability_calibration"][tier][method]
            probability = np.full(len(schedule), np.nan)
            probability[eligible] = _calibrate(margins, calibration)
            rows.append(_output_rows(schedule, tier, method, eligible, margins,
                                     probability, _stable_digest({
                                         "members": [cell["preprocessing_state_sha256"] for cell in cell_refs]}),
                                     _stable_digest(rule), _stable_digest(calibration)))
    table = pd.concat(rows, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "predictions.parquet"
    table.to_parquet(destination, index=False, compression="zstd")
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest": str(freeze_path), "freeze_manifest_sha256": digest(freeze_path),
        "predictions": str(destination), "predictions_sha256": digest(destination),
        "rows": len(table), "games": len(schedule),
        "tier_coverage": {tier: int(table.loc[(table.tier == tier) &
                                              (table.model_id == "equal"), "status"].eq("available").sum())
                          for tier in ("F18", "F19")},
        "no_2026_outcomes_read": True,
    }
    write_json(output / "receipt.json", receipt)
    return receipt


def _output_rows(schedule: pd.DataFrame, tier: str, model_id: str,
                 eligible: np.ndarray, margin: np.ndarray, probability: np.ndarray,
                 representation_sha: str, model_sha: str, calibration_sha: str) -> pd.DataFrame:
    prediction = np.full(len(schedule), np.nan)
    prediction[eligible] = margin
    input_as_of = pd.to_datetime(schedule.latest_documented_source_utc, utc=True)
    if tier == "F19":
        input_as_of = pd.concat([input_as_of, pd.to_datetime(
            schedule.snapshot_timestamp_utc, utc=True, errors="coerce")], axis=1).max(axis=1)
    result = schedule[["target_game_id", "week", "target_start_utc", "home_team", "away_team"]].copy()
    result.insert(0, "tier", tier)
    result.insert(1, "model_id", model_id)
    result["status"] = np.where(eligible, "available", "unavailable_market_snapshot")
    result["prediction_cutoff_utc"] = schedule.target_start_utc
    result["latest_documented_input_utc"] = input_as_of
    result["market_snapshot_timestamp_utc"] = (
        schedule.snapshot_timestamp_utc if tier == "F19" else pd.NaT)
    result["representation_sha256"] = representation_sha
    result["model_sha256"] = model_sha
    result["calibration_sha256"] = calibration_sha
    result["pred_margin"] = prediction
    result["home_win_probability"] = probability
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(predict(args.freeze, args.output), indent=2))


if __name__ == "__main__":
    main()
