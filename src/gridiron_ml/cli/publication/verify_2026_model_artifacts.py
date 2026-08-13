#!/usr/bin/env python3
"""Verify the current external 2026 model artifacts and roster contracts."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from gridiron_ml.cli._paths import project_root


ROOT = project_root()
LEVELS = ("M1", "M2", "M3", "M4", "M5", "M10")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _resolve(value: object, inventory: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = (ROOT / path, inventory.parent / path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _check_hashes(frame: pd.DataFrame, inventory: Path) -> list[str]:
    errors = []
    for row in frame.to_dict("records"):
        path = _resolve(row.get("checkpoint_path", ""), inventory)
        label = str(row.get("model_id", path.name))
        if not path.is_file():
            errors.append(f"{label}: checkpoint missing")
        elif _digest(path) != str(row.get("checkpoint_sha256", "")):
            errors.append(f"{label}: checkpoint SHA-256 mismatch")
    return errors


def _scientific(path: Path) -> dict:
    frame = pd.read_csv(path)
    tiers = frame["feature_config"].astype(str)
    observed = set(zip(tiers, frame["model_level"].astype(str)))
    expected = {(f"F{i}", level) for i in range(9) for level in LEVELS}
    errors = []
    if len(frame) != 54 or observed != expected:
        errors.append("scientific roster is not the exact 54-cell F0-F8 x M matrix")
    if not frame["objective"].astype(str).eq("margin").all():
        errors.append("scientific roster contains a non-margin objective")
    market = frame["market_bearing"].astype(str).str.casefold().isin({"1", "true", "yes", "y"})
    if set(tiers[market]) != {"F7", "F8"} or tiers[~market].isin({"F7", "F8"}).any():
        errors.append("scientific market boundary is not exactly F7/F8")
    if not pd.to_numeric(frame["training_end_season"], errors="coerce").eq(2025).all():
        errors.append("scientific training boundary is not through 2025")
    errors.extend(_check_hashes(frame, path))
    expected_calibration_status = "complete_cross_fitted_oof_through_2025"
    if not frame.get("calibration_status", pd.Series(dtype=str)).astype(str).eq(expected_calibration_status).all():
        errors.append("scientific calibration status is incomplete")
    for field in ("calibrator_path", "calibrator_sha256", "calibration_metrics_path", "calibration_metrics_sha256"):
        if field not in frame:
            errors.append(f"scientific inventory lacks {field}")
    if not errors:
        for row in frame.to_dict("records"):
            for path_field, hash_field in (
                ("calibrator_path", "calibrator_sha256"),
                ("calibration_metrics_path", "calibration_metrics_sha256"),
            ):
                artifact = _resolve(row[path_field], path)
                if not artifact.is_file():
                    errors.append(f"{row['model_id']}: missing {path_field}")
                elif _digest(artifact) != str(row[hash_field]):
                    errors.append(f"{row['model_id']}: {path_field} SHA-256 mismatch")
    return {
        "status": "pass" if not errors else "fail",
        "cells": int(len(frame)),
        "market_free_prediction_cells": int((~market).sum()),
        "market_comparison_cells": int(market.sum()),
        "calibration_statuses": sorted(frame.get("calibration_status", pd.Series("undeclared")).astype(str).unique()),
        "inventory_sha256": _digest(path),
        "errors": errors,
    }


def _wide(path: Path) -> dict:
    frame = pd.read_csv(path)
    ensembles = frame["model_family"].astype(str).eq("ensemble")
    learned = frame.loc[~ensembles]
    errors = []
    if len(frame) != 36 or len(learned) != 34 or int(ensembles.sum()) != 2:
        errors.append("wide roster must contain 34 learned checkpoints and two ensembles")
    if not learned["feature_config"].astype(str).eq("F6").all():
        errors.append("every learned wide-margin checkpoint must use corrected F6")
    if not frame["objective"].astype(str).eq("margin").all():
        errors.append("wide roster contains a non-margin objective")
    if not pd.to_numeric(frame["training_end_season"], errors="coerce").eq(2025).all():
        errors.append("wide training boundary is not through 2025")
    errors.extend(_check_hashes(frame, path))
    return {
        "status": "pass" if not errors else "fail",
        "cells": int(len(frame)),
        "learned_f6_cells": int(len(learned)),
        "ensembles": int(ensembles.sum()),
        "inventory_sha256": _digest(path),
        "errors": errors,
    }


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--scientific-inventory", type=Path,
        default=Path(os.environ.get(
            "TDNET_SCIENTIFIC_ROSTER_INVENTORY",
            "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
            "scientific_roster_refits/f0_f8_margin_through_2025_v1/final_model_inventory.csv",
        )),
    )
    parser.add_argument(
        "--wide-inventory", type=Path,
        default=Path(
            "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
            "corrected_f6_wide_margin_roster/through_2025_v1/final_model_inventory.csv"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific": _scientific(args.scientific_inventory.resolve()),
        "wide_margin": _wide(args.wide_inventory.resolve()),
    }
    result["status"] = (
        "pass" if result["scientific"]["status"] == result["wide_margin"]["status"] == "pass" else "fail"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
