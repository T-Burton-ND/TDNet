#!/usr/bin/env python3
"""Fit and bind 54 frozen logistic margin calibrators from OOF archives."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridiron_ml.cli._paths import project_root
from gridiron_ml.publication.inference import (
    calibration_summary,
    fit_margin_calibrator,
    season_clustered_mean_bootstrap,
    temporal_cross_fitted_margin_calibration,
)

ROOT = project_root()
LEVELS = ("M1", "M2", "M3", "M4", "M5", "M10")
EXPECTED_SEASONS = list(range(2011, 2026))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_summary(frame: pd.DataFrame) -> dict[str, object]:
    y = frame["actual_home_win"].to_numpy(dtype=int)
    p = frame["calibrated_probability_home"].to_numpy(dtype=float)
    summary = calibration_summary(y, p, bins=10)
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    brier = frame[["season"]].copy()
    brier["difference"] = (p - y) ** 2
    logloss = frame[["season"]].copy()
    logloss["difference"] = -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
    summary["brier_score_season_bootstrap"] = season_clustered_mean_bootstrap(brier)
    summary["log_loss_season_bootstrap"] = season_clustered_mean_bootstrap(logloss)
    summary["evaluation_seasons"] = sorted(frame["season"].astype(int).unique().tolist())
    summary["protocol"] = "expanding-window calibration; each season calibrated using earlier OOF seasons only"
    return summary


def save_calibration_plot(frame: pd.DataFrame, target_base: Path, *, title: str) -> None:
    """Write a dedicated equal-count reliability plot for one scientific cell."""
    work = frame[["actual_home_win", "calibrated_probability_home"]].copy()
    work["bin"] = pd.qcut(
        work["calibrated_probability_home"], q=min(10, len(work)), duplicates="drop"
    )
    grouped = work.groupby("bin", observed=True)
    points = grouped.agg(
        predicted=("calibrated_probability_home", "mean"),
        observed=("actual_home_win", "mean"),
        count=("actual_home_win", "size"),
    ).reset_index(drop=True)
    # Wilson score intervals avoid invalid error bars near zero and one.
    z = 1.959963984540054
    n = points["count"].to_numpy(dtype=float)
    observed = points["observed"].to_numpy(dtype=float)
    center = (observed + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(observed * (1 - observed) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lower, upper = np.maximum(0, center - half), np.minimum(1, center + half)

    fig, (ax_curve, ax_hist) = plt.subplots(
        2, 1, figsize=(6.4, 6.8), height_ratios=(4, 1), constrained_layout=True
    )
    ax_curve.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax_curve.errorbar(
        points["predicted"].to_numpy(dtype=float), observed,
        yerr=np.maximum(0.0, np.vstack([observed - lower, upper - observed])),
        fmt="o-", color="#145DA0", capsize=3, linewidth=1.5, label="Temporal validation",
    )
    ax_curve.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted home-win probability", ylabel="Observed home-win rate", title=title)
    ax_curve.grid(alpha=0.2)
    ax_curve.legend(loc="upper left", frameon=False)
    ax_hist.hist(work["calibrated_probability_home"], bins=np.linspace(0, 1, 21), color="#145DA0", alpha=0.8)
    ax_hist.set(xlim=(0, 1), xlabel="Predicted home-win probability", ylabel="Games")
    ax_hist.grid(axis="y", alpha=0.2)
    target_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_base.with_suffix(".png"), dpi=180, metadata={"Software": "TDNet"})
    fig.savefig(target_base.with_suffix(".svg"), metadata={"Date": None, "Creator": "TDNet"})
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "models/season_2026_scientific_f0_f8_bundle")
    args = parser.parse_args()
    inventory = pd.read_csv(args.bundle / "final_model_inventory.csv")
    inventory_updates: dict[tuple[str, str], dict[str, object]] = {}
    records, errors = [], []
    for tier in [f"F{i}" for i in range(9)]:
        for level in LEVELS:
            frozen = inventory.loc[
                inventory["fingerprint"].astype(str).eq(tier)
                & inventory["model_id"].astype(str).str.endswith(f"_{level}")
            ]
            source = args.oof_root / tier / level
            oof_path, status_path = source / "oof_predictions.parquet", source / "status.json"
            if len(frozen) != 1 or not oof_path.exists() or not status_path.exists():
                errors.append(f"missing complete OOF evidence for {tier}/{level}")
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "success" or status.get("frozen_checkpoint_sha256") != str(frozen.iloc[0]["checkpoint_sha256"]):
                errors.append(f"invalid OOF/checkpoint binding for {tier}/{level}")
                continue
            if status.get("successful_seasons") != EXPECTED_SEASONS or status.get("failed_seasons"):
                errors.append(f"incomplete OOF season evidence for {tier}/{level}")
                continue
            if status.get("oof_predictions_sha256") != sha256(oof_path):
                errors.append(f"OOF file hash mismatch for {tier}/{level}")
                continue
            frame = pd.read_parquet(oof_path)
            required = {"keys_game_id", "season", "pred_margin", "actual_margin", "fingerprint_id", "model_role"}
            if required - set(frame):
                errors.append(f"OOF schema mismatch for {tier}/{level}")
                continue
            fit = frame.dropna(subset=["pred_margin", "actual_margin"]).copy()
            seasons = sorted(pd.to_numeric(fit["season"], errors="coerce").dropna().astype(int).unique().tolist())
            if seasons != EXPECTED_SEASONS:
                errors.append(f"invalid OOF cutoff for {tier}/{level}")
                continue
            if fit.duplicated(["keys_game_id", "season"]).any():
                errors.append(f"duplicate OOF games for {tier}/{level}")
                continue
            if not fit["fingerprint_id"].astype(str).eq(tier).all() or not fit["model_role"].astype(str).eq(str(frozen.iloc[0]["model_id"])).all():
                errors.append(f"OOF row binding mismatch for {tier}/{level}")
                continue
            fit_hash = hashlib.sha256(pd.util.hash_pandas_object(fit[["season", "pred_margin", "actual_margin"]], index=True).values.tobytes()).hexdigest()
            calibrator = fit_margin_calibrator(fit["pred_margin"], fit["actual_margin"].gt(0).astype(int), fit_hash=fit_hash)
            if not np.isfinite([calibrator.intercept, calibrator.slope]).all() or calibrator.slope <= 0:
                errors.append(f"non-monotone or non-finite calibrator for {tier}/{level}")
                continue
            validation = temporal_cross_fitted_margin_calibration(fit)
            metrics = validation_summary(validation)
            report_root = args.bundle / "calibration_reports" / tier / level
            report_root.mkdir(parents=True, exist_ok=True)
            validation_path = report_root / "temporal_validation_predictions.parquet"
            validation.to_parquet(validation_path, index=False)
            metrics_path = report_root / "calibration_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            plot_base = report_root / "calibration_curve"
            save_calibration_plot(validation, plot_base, title=f"{tier}/{level} temporal calibration")
            relative_metrics = metrics_path.relative_to(args.bundle)
            relative_plot_png = plot_base.with_suffix(".png").relative_to(args.bundle)
            relative_plot_svg = plot_base.with_suffix(".svg").relative_to(args.bundle)
            relative_validation = validation_path.relative_to(args.bundle)
            record = {
                "artifact_type": "frozen_cross_fitted_margin_calibrator",
                "method": "cross_fitted_logistic_margin",
                "fingerprint_id": tier,
                "model_level": level,
                "model_role": str(frozen.iloc[0]["model_id"]),
                "frozen_checkpoint_sha256": str(frozen.iloc[0]["checkpoint_sha256"]),
                "oof_predictions_sha256": sha256(oof_path),
                "oof_status_sha256": sha256(status_path),
                "production_calibration_cutoff": 2025,
                "oof_seasons": status.get("successful_seasons", []),
                "failed_oof_seasons": status.get("failed_seasons", []),
                "temporal_validation_seasons": metrics["evaluation_seasons"],
                "temporal_validation_rows": int(len(validation)),
                "calibration_metrics_path": str(relative_metrics),
                "calibration_metrics_sha256": sha256(metrics_path),
                "calibration_plot_png_path": str(relative_plot_png),
                "calibration_plot_png_sha256": sha256(plot_base.with_suffix(".png")),
                "calibration_plot_svg_path": str(relative_plot_svg),
                "calibration_plot_svg_sha256": sha256(plot_base.with_suffix(".svg")),
                "temporal_validation_predictions_path": str(relative_validation),
                "temporal_validation_predictions_sha256": sha256(validation_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                **calibrator.to_dict(),
            }
            target = args.bundle / "calibrators" / tier / level / "calibrator.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            record["calibrator_sha256"] = sha256(target)
            relative_calibrator = target.relative_to(args.bundle)
            record["calibrator_path"] = str(relative_calibrator)
            inventory_updates[(tier, level)] = {
                "calibration_status": "complete_cross_fitted_oof_through_2025",
                "calibrator_path": str(relative_calibrator),
                "calibrator_sha256": record["calibrator_sha256"],
                "calibration_metrics_path": str(relative_metrics),
                "calibration_metrics_sha256": record["calibration_metrics_sha256"],
                "calibration_plot_png_path": str(relative_plot_png),
                "calibration_plot_svg_path": str(relative_plot_svg),
            }
            records.append(record)
    manifest = {
        "status": "pass" if not errors and len(records) == 54 else "blocked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "cross_fitted_logistic_margin",
        "production_calibration_cutoff": 2025,
        "records": records,
        "errors": errors,
    }
    (args.bundle / "CALIBRATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if manifest["status"] == "pass":
        report_index = pd.DataFrame([{
            "fingerprint": record["fingerprint_id"],
            "model_level": record["model_level"],
            "model_role": record["model_role"],
            "png": record["calibration_plot_png_path"],
            "svg": record["calibration_plot_svg_path"],
            "metrics": record["calibration_metrics_path"],
            "validation_predictions": record["temporal_validation_predictions_path"],
        } for record in records]).sort_values(["fingerprint", "model_level"])
        reports_root = args.bundle / "calibration_reports"
        report_index.to_csv(reports_root / "plot_index.csv", index=False)
        readme_lines = [
            "# Individual temporal calibration reports",
            "",
            "Each plot evaluates one model cell with an expanding-window protocol: every season is calibrated only on earlier out-of-fold seasons.",
            "",
            "| Fingerprint | Model | PNG | SVG | Metrics |",
            "|---|---:|---|---|---|",
        ]
        for row in report_index.to_dict("records"):
            png = Path(str(row["png"])).relative_to("calibration_reports")
            svg = Path(str(row["svg"])).relative_to("calibration_reports")
            metrics = Path(str(row["metrics"])).relative_to("calibration_reports")
            readme_lines.append(
                f"| {row['fingerprint']} | {row['model_level']} | [PNG]({png}) | [SVG]({svg}) | [JSON]({metrics}) |"
            )
        (reports_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
        for (tier, level), updates in inventory_updates.items():
            mask = inventory["fingerprint"].astype(str).eq(tier) & inventory["model_level"].astype(str).eq(level)
            for column, value in updates.items():
                inventory.loc[mask, column] = value
        inventory_path = args.bundle / "final_model_inventory.csv"
        inventory.to_csv(inventory_path, index=False)
        freeze_path = args.bundle / "freeze_manifest.json"
        if freeze_path.exists():
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            for artifact in freeze.get("artifacts", []):
                key = (str(artifact.get("fingerprint")), str(artifact.get("model_level")))
                if key in inventory_updates:
                    artifact.update(inventory_updates[key])
            freeze["calibration"] = {
                "required": True,
                "status": "complete_cross_fitted_oof_through_2025",
                "manifest": "CALIBRATION_MANIFEST.json",
                "manifest_sha256": sha256(args.bundle / "CALIBRATION_MANIFEST.json"),
            }
            freeze["inventory_sha256"] = sha256(inventory_path)
            freeze["status"] = "candidate_calibrated_independent_verification_pending"
            freeze["remaining_gates"] = [
                "independent verification", "durable external archive", "owner publication approval"
            ]
            freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "records": len(records), "errors": errors}, indent=2))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
