#!/usr/bin/env python3
"""Paired M2/M4 A-screening for corrected F12 and F13–F17 research rounds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.experiments.nextgen_rounds_features import F13_METRICS, F14_METRICS, F15_METRICS
from gridiron_ml.experiments.nextgen_rounds_market import MARKET_COLUMNS
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import (
    build_estimator, future_metrics, residual_probability, source_to_matchup,
)
from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file


STAGES = ("F12_original", "F12_corrected", "F13", "F14", "F15", "F16", "F17_market")
EARLIER_STAGES = ("F09", "F10", "F11")
ALL_STAGES = (*EARLIER_STAGES, *STAGES)
SOURCE_F13 = tuple(f"{side}_{metric}" for side in ("offense", "defense") for metric in F13_METRICS)
SOURCE_F14 = tuple(f"{side}_{metric}" for side in ("offense", "defense") for metric in F14_METRICS)
SOURCE_F15 = F15_METRICS
SOURCE_F16 = tuple(f"{side}_{metric}_recent_change" for side in ("offense", "defense")
                   for metric in ("resid_yards", "resid_success")) + tuple(
                       f"{side}_recovery_game_sd" for side in ("offense", "defense"))
ADDED = {"F13": SOURCE_F13, "F14": SOURCE_F14, "F15": SOURCE_F15, "F16": SOURCE_F16}


def stage_sources(stage: str) -> tuple[str, ...]:
    if stage not in ALL_STAGES:
        raise ValueError(f"Unknown stage {stage}")
    stop = ALL_STAGES.index(stage)
    return tuple(name for generation in ("F13", "F14", "F15", "F16")
                 if ALL_STAGES.index(generation) <= stop for name in ADDED[generation])


def _record(name: str) -> dict:
    if name.startswith("offense_"):
        counterpart = name.replace("offense_", "defense_", 1)
    elif name.startswith("defense_"):
        counterpart = name.replace("defense_", "offense_", 1)
    else:
        counterpart = name
    return {"name": name, "matchup_counterpart": counterpart,
            "matchup_formula": f"home.[{name}]{'-' if counterpart == name else '+'}away.[{counterpart}]"}


def _checked_prepare(output: Path) -> dict:
    receipt_path = output / "prepare_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "prepared_untrained":
        raise ValueError("Prepared research inputs are not finalized")
    for filename, digest in receipt["outputs"].items():
        if sha256_file(output / filename) != digest:
            raise ValueError(f"Prepared artifact changed: {filename}")
    return receipt


def load_stage_matrix(output: Path, stage: str):
    """Every round uses one game-ID cohort, source labels and training years."""
    receipt = _checked_prepare(output)
    archive = Path(receipt["source_archive"])
    parent_path = archive / "fingerprints/F12_F_a/values.parquet"
    if sha256_file(parent_path) != receipt["f12_parent_sha256"]:
        raise ValueError("Saved F12 parent changed")
    parent_manifest = archive / "fingerprints/F12_F_a/feature_manifest.json"
    provenance = json.loads((archive / "fingerprints/F12_F_a/provenance.json").read_text())
    if (sha256_file(parent_manifest) != provenance["manifest_sha256"]
            or sha256_file(parent_path) != provenance["data_sha256"]):
        raise ValueError("F12 parent provenance mismatch")
    original_records = json.loads(parent_manifest.read_text())
    if stage in {"F09", "F10", "F11"}:
        generation_limit = int(stage[1:])
        records = [r for r in original_records
                   if r["generation"] == "F06" or
                   (r["generation"].startswith("F") and
                    r["generation"][1:].isdigit() and
                    9 <= int(r["generation"][1:]) <= generation_limit)]
    else:
        records = original_records
    original_names = [r["name"] for r in records]
    parent = pd.read_parquet(parent_path)
    parent = parent.loc[parent.season.between(2013, 2025)].copy()
    key = ["target_game_id", "team"]
    if parent.duplicated(key).any():
        raise ValueError("Duplicate parent target team")
    corrected = pd.read_parquet(output / "corrected_f09_a_target_features.parquet")
    state = pd.read_parquet(output / "f13_f16_target_state.parquet")
    market = pd.read_parquet(output / "f17_market_game_features.parquet")
    if corrected.duplicated(key).any() or state.duplicated(key).any() or market.target_game_id.duplicated().any():
        raise ValueError("Duplicate prepared feature identity")
    f09_names = [c for c in corrected if c not in key]
    original_f09 = {r["name"] for r in original_records if r["generation"] == "F09"}
    if set(f09_names) != original_f09:
        raise ValueError("Correction must cover exactly the inherited F09 A features")
    state_columns = list(stage_sources(stage))
    state_meta = ["latest_source_game_id", "latest_source_game_utc", "feature_available_utc"]
    frame = parent.merge(corrected, on=key, how="inner", validate="one_to_one", suffixes=("", "__corrected"))
    frame = frame.merge(state[key + state_meta + state_columns], on=key, how="inner",
                        validate="one_to_one", suffixes=("", "__research"))
    frame = frame.merge(market[["target_game_id", *MARKET_COLUMNS]], on="target_game_id",
                        how="inner", validate="many_to_one")
    if len(frame) != len(parent):
        raise ValueError("Prepared features change the F12 cohort; resolve before fitting")
    if not (pd.to_datetime(frame["latest_source_game_utc__research"], utc=True)
            < pd.to_datetime(frame["feature_available_utc__research"], utc=True)).all():
        raise ValueError("Research source game is not before its availability")
    if not (pd.to_datetime(frame["feature_available_utc__research"], utc=True)
            < pd.to_datetime(frame.target_start_utc, utc=True)).all():
        raise ValueError("Research feature is not pre-target")
    if frame["latest_source_game_id__research"].eq(frame.target_game_id).any():
        raise ValueError("Target game contributed to its own features")
    if stage != "F12_original":
        for name in f09_names:
            frame[name] = frame[name + "__corrected"]
    added = list(stage_sources(stage))
    records = records + [_record(name) for name in added]
    names = original_names + added
    home = frame.loc[frame.is_home].set_index("target_game_id").sort_index()
    away = frame.loc[~frame.is_home].set_index("target_game_id").sort_index()
    if not home.index.equals(away.index):
        raise ValueError("Unpaired home and away rows")
    if not np.allclose(home.next_game_margin.to_numpy(float), -away.next_game_margin.to_numpy(float)):
        raise ValueError("Reciprocal outcomes differ")
    raw = np.column_stack([home[names].to_numpy(float), away[names].to_numpy(float)])
    x = source_to_matchup(raw, records)
    if stage == "F17_market":
        x = np.column_stack([x, home[list(MARKET_COLUMNS)].to_numpy(float)])
    meta = home[["season", "week", "target_start_utc", "next_game_margin"]].reset_index()
    spread = home.market_home_spread.to_numpy(float)
    counts = meta.groupby("season").size().to_dict()
    return x, meta, spread, {"source_features": names,
                              "market_features": list(MARKET_COLUMNS) if stage == "F17_market" else [],
                              "games_by_season": {str(k): int(v) for k, v in counts.items()},
                              "f12_parent_sha256": receipt["f12_parent_sha256"],
                              "prepare_receipt_sha256": sha256_file(output / "prepare_receipt.json")}


def run(output: Path, stage: str, model: str, setpoint_id: str):
    if stage not in ALL_STAGES or model not in {"M2", "M4"}:
        raise ValueError("Unknown research stage or architecture")
    config = ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json"
    points = json.loads(config.read_text())
    point = next((p for p in points[model] if p["id"] == setpoint_id), None)
    if point is None:
        raise ValueError("Unknown frozen setpoint")
    x, meta, spreads, evidence = load_stage_matrix(output, stage)
    out = output / "experiments" / stage / model / setpoint_id
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    binding = {"stage": stage, "model": model, "setpoint": point, "seed": 1701,
               "input": evidence, "code_sha256": sha256_file(Path(__file__)),
               "setpoints_sha256": sha256_file(config)}
    if result_path.exists():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "success":
            if (prior.get("binding") != binding or
                    prior.get("predictions_sha256") != sha256_file(out / "predictions.parquet")):
                raise ValueError("Existing successful research result has changed")
            return prior
    years = meta.season.to_numpy(int)
    y = meta.next_game_margin.to_numpy(float)
    train = years <= 2023
    if not (years.min() == 2013 and years.max() == 2025 and train.sum() > 3000):
        raise ValueError("Research training cohort is incomplete")
    started = time.monotonic()
    result = {"status": "incomplete", "stage": stage, "model": model,
              "setpoint": setpoint_id, "binding": binding,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "training_games": int(train.sum()),
              "development_games": {str(y): int((years == y).sum()) for y in (2024, 2025)},
              "max_design_year": 2025,
              "market_quote_timestamp_available": False if stage == "F17_market" else None,
              "market_research_only": stage == "F17_market"}
    try:
        residuals = []
        for cutoff in (2019, 2021):
            fit = years <= cutoff
            calibrate = (years > cutoff) & (years <= cutoff + 2)
            estimator = build_estimator(model, point)
            estimator.fit(x[fit], y[fit])
            residuals.extend(y[calibrate] - estimator.predict(x[calibrate]))
        estimator = build_estimator(model, point)
        estimator.fit(x[train], y[train])
        eval_mask = years >= 2024
        predicted = estimator.predict(x[eval_mask])
        probabilities = residual_probability(predicted, residuals)
        eval_meta = meta.loc[eval_mask].reset_index(drop=True)
        prediction = pd.DataFrame({
            "target_game_id": eval_meta.target_game_id.astype(int),
            "season": eval_meta.season.astype(int),
            "actual_margin": y[eval_mask], "predicted_margin": predicted,
            "home_win_probability": probabilities,
        })
        for year in (2024, 2025):
            mask = eval_meta.season.eq(year).to_numpy()
            result[f"metrics_{year}"] = future_metrics(
                y[eval_mask][mask], predicted[mask], probabilities[mask],
                spreads[eval_mask][mask])
        target = out / "predictions.parquet"
        temp = out / "predictions.tmp.parquet"
        prediction.to_parquet(temp, index=False, compression="zstd")
        temp.replace(target)
        result.update(status="success", predictions_sha256=sha256_file(target),
                      calibration_residuals=len(residuals))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=8))
    result["runtime_seconds"] = time.monotonic() - started
    temp = out / "result.tmp.json"
    temp.write_text(json.dumps(result, sort_keys=True, indent=2, default=str) + "\n")
    temp.replace(result_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/nextgen_rounds_2026")
    parser.add_argument("--stage", choices=ALL_STAGES, required=True)
    parser.add_argument("--model", choices=("M2", "M4"), required=True)
    parser.add_argument("--setpoint", required=True)
    args = parser.parse_args()
    result = run(args.output, args.stage, args.model, args.setpoint)
    print(json.dumps({k: v for k, v in result.items() if k not in {"binding", "traceback"}}, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
