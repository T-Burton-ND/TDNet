#!/usr/bin/env python3
"""Build the two fixed equal-weight canonical 2025 consensus products."""
from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.publication.inference import MarginCalibrator, calibration_summary  # noqa: E402
from gridiron_ml.publication.metrics import chalk_upset_table, score_predictions  # noqa: E402


REQUIRED = {"model_role", "fingerprint_id", "keys_game_id", "season", "pred_margin", "pred_probability_home", "actual_margin", "vegas_spread"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def consensus(frame: pd.DataFrame, members: set[str], label: str, calibrators: dict[str, MarginCalibrator]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.assign(member_id=frame["model_role"].astype(str) + "|" + frame["fingerprint_id"].astype(str))
    work = work.loc[work["member_id"].isin(members)].copy()
    if set(work["member_id"].unique()) != members:
        raise ValueError(f"{label}: member coverage mismatch")
    key = "keys_game_id"
    numeric = ["pred_margin", "pred_probability_home", "actual_margin", "vegas_spread"]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["worker_probability_home"] = work["pred_probability_home"]
    work["calibrated_probability_home"] = float("nan")
    for member_id, calibrator in calibrators.items():
        mask = work["member_id"].eq(member_id)
        work.loc[mask, "calibrated_probability_home"] = calibrator.predict(work.loc[mask, "pred_margin"])
    if work.duplicated(["member_id", key]).any():
        raise ValueError(f"{label}: duplicate member/game rows")
    grouped = work.groupby(key, sort=True)
    aggregations = {
        "season": ("season", "first"),
        "keys_week": ("keys_week", "first"),
        "pred_margin": ("pred_margin", "mean"),
        "pred_probability_home": ("pred_probability_home", "mean"),
        "actual_margin": ("actual_margin", "first"),
        "vegas_spread": ("vegas_spread", "first"),
        "effective_model_count": ("member_id", "nunique"),
    }
    if "vegas_probability_home" in work.columns:
        aggregations["vegas_probability_home"] = ("vegas_probability_home", "first")
    calibrated_complete = members.issubset(set(calibrators))
    if calibrated_complete:
        aggregations["pred_probability_home"] = ("calibrated_probability_home", "mean")
        aggregations["worker_probability_home"] = ("worker_probability_home", "mean")
    out = grouped.agg(**aggregations).reset_index()
    out.insert(0, "consensus", label)
    out["probability_source"] = "cross_fitted_logistic_margin" if calibrated_complete else "worker_probability_unrecalibrated"
    out["consensus_members"] = "|".join(sorted(members))
    if not out["effective_model_count"].eq(len(members)).all():
        raise ValueError(f"{label}: incomplete game-level membership")
    membership = work[[key, "member_id"]].drop_duplicates().sort_values([key, "member_id"])
    membership.insert(2, "consensus", label)
    return out, membership


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/publication/canonical_2025")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.predictions)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical consensus input missing columns: {missing}")
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    calibrators: dict[str, MarginCalibrator] = {}
    if args.calibration_root is not None:
        for path in sorted((args.calibration_root / "F5").glob("*/selection_through_2024.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            member_id = f"{record['model_role']}|F5"
            calibrators[member_id] = MarginCalibrator(intercept=float(record["intercept"]), slope=float(record["slope"]), fit_rows=int(record["fit_rows"]), fit_hash=record.get("fit_hash"))
    selected = str(decisions["selected_fingerprint_market_free"])
    all_members = set(frame.loc[frame["fingerprint_id"].eq(selected), "model_role"].astype(str) + "|" + selected)
    top_members = set(decisions.get("top10_members", []))
    if len(all_members) != 6 or len(top_members) != 10:
        raise ValueError(f"Unexpected fixed roster sizes: all={len(all_members)}, top10={len(top_members)}")
    outputs = []
    memberships = []
    for label, members in (("all_model_consensus", all_members), ("top10_brier_consensus", top_members)):
        product, membership = consensus(frame, members, label, calibrators)
        outputs.append(product)
        memberships.append(membership)
        product.to_parquet(args.output_root / f"{label}.parquet", index=False)
        product.to_csv(args.output_root / f"{label}.csv", index=False)
        metrics = score_predictions(product)
        metrics["consensus"] = label
        metrics["selected_market_free_fingerprint"] = selected
        metrics["member_count"] = len(members)
        metrics["calibration_probability_source"] = str(product["probability_source"].iloc[0])
        pd.DataFrame([metrics]).to_csv(args.output_root / f"{label}_metrics.csv", index=False)
        chalk_upset_table(product).to_csv(args.output_root / f"{label}_chalk_upset.csv", index=False)
        actual_home = product["actual_margin"].gt(0).astype(int)
        probability = product["pred_probability_home"].to_numpy(dtype=float)
        valid = np.isfinite(probability)
        calibration = calibration_summary(actual_home.to_numpy()[valid], probability[valid])
        (args.output_root / f"{label}_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    pd.concat(outputs, ignore_index=True).to_csv(args.output_root / "consensus_predictions.csv", index=False)
    pd.concat(memberships, ignore_index=True).to_csv(args.output_root / "consensus_membership.csv", index=False)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "canonical_2025_consensus_complete_primary_calibrated_secondary_exploratory",
        "primary_consensus": "all_model_consensus",
        "secondary_consensus": "top10_brier_consensus",
        "equal_weights": True,
        "membership_fixed_before_2026": True,
        "selected_market_free_fingerprint": selected,
        "all_model_members": sorted(all_members),
        "top10_members": sorted(top_members),
        "calibrators_loaded": sorted(calibrators),
        "primary_calibration_status": "cross_fitted_logistic_margin" if all_members.issubset(set(calibrators)) else "missing_primary_calibrators",
        "source_prediction_sha256": sha256(args.predictions),
        "source_results": str(args.results),
        "calibration_note": "Primary F5 consensus uses the through-2024 cross-fitted logistic calibrators when supplied; secondary top10 cells outside F5 remain exploratory and use worker probabilities unless their own OOF calibrators are supplied.",
    }
    (args.output_root / "consensus_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
