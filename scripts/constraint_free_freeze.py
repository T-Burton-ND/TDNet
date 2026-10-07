#!/usr/bin/env python3
"""Freeze F18/F19 models, OOF calibration, ensembles, and market cutoff policy.

This program may inspect 2026 input columns and snapshot timestamps, but it
does not import, read, or derive a 2026 score or other outcome.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge

from constraint_free_search import digest, write_json
from constraint_free_2026_matrix import ARCHIVE, source_frames
from gridiron_ml.experiments.constraint_free import ROSTER, SNAPSHOT_MARKET_COLUMNS

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1")
FIT = EXPERIMENT / "broad_search/final_fit"
TARGET = EXPERIMENT / "broad_target_inputs"
MARKET = REPO / "data/what_if_2026_fingerprints/f17_market_features/f17_market_target_state_2026.parquet"
YEARS = (2022, 2023, 2024, 2025)


def _probability_calibration(margin: np.ndarray, actual: np.ndarray) -> dict:
    model = LogisticRegression(C=100, solver="lbfgs", max_iter=1000)
    model.fit(margin.reshape(-1, 1), actual > 0)
    return {"intercept": float(model.intercept_[0]),
            "margin_coefficient": float(model.coef_[0, 0]),
            "fit_games": len(actual), "fit_seasons": list(YEARS)}


def _ensemble_rules(predictions: np.ndarray, actual: np.ndarray) -> dict:
    size = predictions.shape[1]
    equal = np.full(size, 1 / size)
    model_mae = np.mean(np.abs(predictions - actual[:, None]), axis=0)
    inverse = (1 / np.maximum(model_mae, 1e-6))
    inverse /= inverse.sum()
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * size
    smooth_mae = minimize(
        lambda w: np.mean(np.sqrt((actual - predictions @ w) ** 2 + 0.1 ** 2))
        + 0.02 * np.square(w - equal).sum(),
        equal, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-10},
    )
    mse = minimize(
        lambda w: np.mean(np.square(actual - predictions @ w))
        + 0.02 * np.square(w - equal).sum(),
        equal, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-10},
    )
    if not smooth_mae.success or not mse.success:
        raise ValueError(f"Historical ensemble optimization failed: {smooth_mae.message}; {mse.message}")
    stack = Ridge(alpha=100).fit(predictions, actual)
    return {
        "equal": {"type": "weighted", "weights": equal.tolist(), "intercept": 0.0},
        "median": {"type": "median"},
        "trimmed": {"type": "trimmed", "drop_each_end": 1},
        "inverse_mae": {"type": "weighted", "weights": inverse.tolist(), "intercept": 0.0},
        "optimized_mae": {"type": "weighted", "weights": smooth_mae.x.tolist(), "intercept": 0.0,
                          "objective": "smooth MAE with 0.02 equal-weight shrinkage"},
        "nonnegative_mse": {"type": "weighted", "weights": mse.x.tolist(), "intercept": 0.0,
                            "objective": "MSE with 0.02 equal-weight shrinkage"},
        "ridge_stack": {"type": "weighted", "weights": stack.coef_.tolist(),
                        "intercept": float(stack.intercept_), "alpha": 100},
    }


def apply_rule(predictions: np.ndarray, rule: dict) -> np.ndarray:
    if rule["type"] == "weighted":
        return predictions @ np.asarray(rule["weights"]) + rule["intercept"]
    if rule["type"] == "median":
        return np.median(predictions, axis=1)
    if rule["type"] == "trimmed":
        count = int(rule["drop_each_end"])
        return np.sort(predictions, axis=1)[:, count:-count].mean(axis=1)
    raise ValueError(f"Unknown ensemble rule: {rule}")


def _market_snapshot_policy(schedule: pd.DataFrame) -> dict:
    columns = ["target_game_id", "snapshot_timestamp_utc",
               "market_snapshot_captured_pre_kickoff",
               "market_quote_timestamp_available", *SNAPSHOT_MARKET_COLUMNS]
    market = pd.read_parquet(MARKET, columns=columns)
    if len(market) != len(schedule) or market.target_game_id.duplicated().any():
        raise ValueError("Market snapshot cohort differs from frozen schedule")
    joined = schedule.merge(market, on="target_game_id", validate="one_to_one")
    captured = joined.market_snapshot_captured_pre_kickoff.fillna(False).astype(bool)
    timestamp = pd.to_datetime(joined.snapshot_timestamp_utc, utc=True, errors="coerce")
    start = pd.to_datetime(joined.target_start_utc, utc=True)
    if (captured & ~(timestamp < start)).any():
        raise ValueError("Archived market snapshot was captured after kickoff")
    if joined.market_quote_timestamp_available.fillna(False).astype(bool).any():
        raise ValueError("Quote-level timing semantics changed")
    if joined.loc[~captured, list(SNAPSHOT_MARKET_COLUMNS)].notna().any().any():
        raise ValueError("Unavailable game has nonmissing market inputs")
    gap = (start[captured] - timestamp[captured]).dt.total_seconds() / 3600
    if len(gap) != 263 or (gap <= 0).any():
        raise ValueError("Pregame market snapshot coverage changed")
    return {
        "forecast_cutoff": "target_start_utc exclusive",
        "actual_market_as_of": "archived snapshot_timestamp_utc for each eligible game",
        "eligible_games": int(captured.sum()),
        "unavailable_games": int((~captured).sum()),
        "unavailable_game_ids": joined.loc[~captured, "target_game_id"].astype(int).tolist(),
        "snapshot_hours_before_kickoff_min": float(gap.min()),
        "snapshot_hours_before_kickoff_median": float(gap.median()),
        "historical_quote_level_timestamp_verified": False,
        "prospective_quote_level_timestamp_available": False,
        "market_source": str(MARKET), "market_source_sha256": digest(MARKET),
    }


def _documented_source_availability(schedule: pd.DataFrame, output_path: Path) -> dict:
    manifests = {}
    for design in "abc":
        path = ARCHIVE / "fingerprints" / f"F12_F_{design}" / "feature_manifest.json"
        manifests[design] = json.loads(path.read_text())
    frames, teams = source_frames(schedule, manifests)
    names = {
        "F06": ("keys_game_date",),
        "F09": ("latest_source_available_utc",),
        "F10": ("latest_source_available_utc", "recruitment_available_utc"),
        "F11": ("staff_available_utc",),
        "F12": ("latest_source_available_utc_st", "latest_source_available_utc_ppa"),
        "research": ("feature_available_utc",),
    }
    stamps = []
    for generation, columns in names.items():
        for column in columns:
            stamps.append(pd.to_datetime(frames[generation][column], utc=True,
                                         errors="coerce").reset_index(drop=True))
    latest = pd.concat(stamps, axis=1).max(axis=1)
    documented = teams[["target_game_id", "side", "team"]].copy()
    documented["latest_documented_source_utc"] = latest.to_numpy()
    grouped = documented.groupby("target_game_id", sort=False)["latest_documented_source_utc"].max()
    out = schedule[["target_game_id", "target_start_utc"]].merge(
        grouped.rename("latest_documented_source_utc"), on="target_game_id", validate="one_to_one")
    if out.latest_documented_source_utc.isna().any() or not (
            out.latest_documented_source_utc < out.target_start_utc).all():
        raise ValueError("Documented football source availability is not pregame")
    out[["target_game_id", "latest_documented_source_utc"]].to_parquet(output_path, index=False)
    return {"path": str(output_path), "sha256": digest(output_path),
            "semantics": "latest timestamp among saved prior-game, roster, staff, F12 and F13–F16 source state fields"}


def freeze(output: Path, fit_manifest: Path = FIT / "manifest.json") -> dict:
    manifest = json.loads(fit_manifest.read_text())
    if len(manifest["tasks"]) != 12 or tuple(manifest["years"]) != YEARS:
        raise ValueError("Final fit manifest is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "freeze_manifest.json").exists():
        raise FileExistsError("Freeze manifest is immutable")
    schedule_path = TARGET / "target_schedule_without_scores.parquet"
    schedule = pd.read_parquet(schedule_path)
    if len(schedule) != 271 or schedule.target_game_id.duplicated().any():
        raise ValueError("Score-free 2026 cohort changed")
    matrix_sha = {}
    for tier in ("F18", "F19"):
        path = TARGET / f"{tier.lower()}_inputs.parquet"
        matrix = pd.read_parquet(path)
        if len(matrix) != 271 or not matrix.target_game_id.equals(schedule.target_game_id):
            raise ValueError(f"{tier} target input order or coverage changed")
        matrix_sha[tier] = digest(path)
    source_receipt = json.loads((TARGET / "receipt.json").read_text())
    if matrix_sha["F18"] != source_receipt["f18_inputs_sha256"] or matrix_sha["F19"] != source_receipt["f19_inputs_sha256"]:
        raise ValueError("Pregame input matrix differs from construction receipt")
    availability = _documented_source_availability(
        schedule, output / "feature_source_availability.parquet")
    cells = []
    oof_all = []
    ensemble_specs = {}
    calibrations = {}
    for tier in ("F18", "F19"):
        frames = []
        reference = None
        tier_cells = []
        for architecture in ROSTER:
            selected = next((row for row in manifest["tasks"]
                             if row["tier"] == tier and row["architecture"] == architecture), None)
            if selected is None:
                raise ValueError(f"Missing fitted finalist: {tier} {architecture}")
            folder = fit_manifest.parent / f"{tier}_{architecture}"
            result_path = folder / "result.json"
            result = json.loads(result_path.read_text())
            if result.get("status") != "success" or result.get("manifest_sha256") != digest(fit_manifest):
                raise ValueError(f"Incomplete final fit: {folder}")
            for name, expected in result["artifacts"].items():
                if digest(folder / name) != expected:
                    raise ValueError(f"Fitted artifact changed: {folder / name}")
            frame = pd.read_parquet(folder / "oof.parquet").sort_values("target_game_id")
            if len(frame) != 2226 or frame.target_game_id.duplicated().any() or tuple(sorted(frame.season.unique())) != YEARS:
                raise ValueError(f"OOF cohort changed: {folder}")
            identity = frame[["target_game_id", "season", "actual_margin"]].reset_index(drop=True)
            if reference is None:
                reference = identity
            elif not identity.equals(reference):
                raise ValueError(f"OOF games or outcomes differ: {folder}")
            frames.append(frame.pred_margin.to_numpy(float))
            tier_cells.append({
                "tier": tier, "architecture": architecture,
                "selected_candidate": selected,
                "fit_result": str(result_path), "fit_result_sha256": digest(result_path),
                "model_bundle": str(folder / "fitted_bundle.pkl"),
                "model_bundle_sha256": result["artifacts"]["fitted_bundle.pkl"],
                "preprocessing_state": str(folder / "preprocessing.pkl"),
                "preprocessing_state_sha256": result["artifacts"]["preprocessing.pkl"],
                "pca_loadings": (str(folder / "pca_loadings.npz")
                                 if "pca_loadings.npz" in result["artifacts"] else None),
                "pca_loadings_sha256": result["artifacts"].get("pca_loadings.npz"),
                "pca_loading_blocks": result["pca_loading_blocks"],
                "selected_source_features": result["selected_source_features"],
                "representation_features": result["representation_features"],
                "oof_path": str(folder / "oof.parquet"),
                "oof_sha256": result["artifacts"]["oof.parquet"],
            })
        actual = reference.actual_margin.to_numpy(float)
        matrix = np.column_stack(frames)
        rules = _ensemble_rules(matrix, actual)
        ensemble_specs[tier] = rules
        tier_calibration = {}
        output_frame = reference.copy()
        for column, architecture in enumerate(ROSTER):
            prediction = matrix[:, column]
            output_frame[f"pred_margin_{architecture}"] = prediction
            tier_calibration[architecture] = _probability_calibration(prediction, actual)
        for method, rule in rules.items():
            prediction = apply_rule(matrix, rule)
            output_frame[f"pred_margin_{method}"] = prediction
            tier_calibration[method] = _probability_calibration(prediction, actual)
            rule["development_oof_mae"] = float(np.abs(actual - prediction).mean())
            rule["development_oof_games"] = len(actual)
        calibrations[tier] = tier_calibration
        output_frame.insert(0, "tier", tier)
        oof_all.append(output_frame)
        cells.extend(tier_cells)
    oof_path = output / "historical_oof_and_ensembles.parquet"
    pd.concat(oof_all, ignore_index=True).to_parquet(oof_path, index=False)
    freeze_record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                           text=True).strip(),
        "fit_manifest": str(fit_manifest), "fit_manifest_sha256": digest(fit_manifest),
        "finalist_selection": manifest["selection"],
        "finalist_selection_sha256": manifest["selection_sha256"],
        "training_seasons": [2013, 2025], "oof_seasons": list(YEARS),
        "scientific_architectures": list(ROSTER),
        "cells": cells, "ensemble_rules": ensemble_specs,
        "margin_to_win_probability_calibration": calibrations,
        "primary_consensus": "equal",
        "historical_oof": str(oof_path), "historical_oof_sha256": digest(oof_path),
        "target_input_matrices": {tier: {"path": str(TARGET / f"{tier.lower()}_inputs.parquet"),
                                         "sha256": sha} for tier, sha in matrix_sha.items()},
        "target_schedule_without_scores": {"path": str(schedule_path),
                                           "sha256": digest(schedule_path)},
        "target_input_receipt": {"path": str(TARGET / "receipt.json"),
                                 "sha256": digest(TARGET / "receipt.json")},
        "feature_source_availability": availability,
        "market_policy": _market_snapshot_policy(schedule),
        "development_score_warning": "Selected and calibrated using 2022–2025; OOF development metrics are optimistic for the selected system",
        "historical_market_warning": "F19 historical quote-level timing could not be proven",
        "no_2026_outcomes_read": True,
    }
    path = output / "freeze_manifest.json"
    write_json(path, freeze_record)
    return {"freeze_manifest": str(path), "sha256": digest(path),
            "fitted_cells": len(cells), "prospective_games": len(schedule),
            "f19_market_games": freeze_record["market_policy"]["eligible_games"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIT / "freeze")
    parser.add_argument("--fit-manifest", type=Path, default=FIT / "manifest.json")
    args = parser.parse_args()
    print(json.dumps(freeze(args.output, args.fit_manifest), indent=2))


if __name__ == "__main__":
    main()
