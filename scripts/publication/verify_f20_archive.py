#!/usr/bin/env python3
"""Verify F20 forecasts from archived bytes and frozen model artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.f20.model import (
    BASE_COLUMNS,
    FEATURE_COLUMNS,
    captured_features,
)
from gridiron_ml.f20.scientific import MODEL_IDS, predict_panel


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_matching_file(root: Path, digest: str, pattern: str) -> Path:
    matches = [path for path in root.glob(pattern) if sha(path) == digest]
    if len(matches) != 1:
        raise ValueError(f"Expected one archived file with hash {digest}; got {len(matches)}")
    return matches[0]


def verify(archive: Path, models_root: Path) -> dict:
    records = sorted((archive / "forecasts").glob("*/*.json"))
    records = [path for path in records if path.name != "official.json"]
    if not records:
        raise ValueError("No F20 forecasts in archive")
    model_cache = {}
    panel_counts = {"legacy_A_control": 0, "F20_six_model": 0}
    for path in records:
        row = json.loads(path.read_text())
        raw_path = archive / "snapshots" / f"{row['raw_snapshot_sha256']}.html"
        if sha(raw_path) != row["raw_snapshot_sha256"]:
            raise ValueError(f"Raw snapshot changed: {path}")
        receipt_path = one_matching_file(archive / "receipts", row["receipt_sha256"],
                                         f"{row['raw_snapshot_sha256']}*.json")
        receipt = json.loads(receipt_path.read_text())
        found = [obs for obs in receipt["observations"] if obs["game_id"] == row["game_id"]]
        if len(found) != 1 or found[0]["values"] != row["input_values"]:
            raise ValueError(f"Forecast values differ from receipt: {path}")
        if not (datetime.fromisoformat(row["source_collected_at_utc"])
                <= datetime.fromisoformat(row["issued_at_utc"])
                < datetime.fromisoformat(row["kickoff_utc"])):
            raise ValueError(f"Forecast timing invalid: {path}")
        model_hash = row["model_manifest_sha256"]
        if model_hash not in model_cache:
            manifest_path = one_matching_file(models_root, model_hash, "*/manifest.json")
            manifest = json.loads(manifest_path.read_text())
            if manifest["training_seasons"][-1] > 2024 or 2026 not in manifest["excluded_seasons"]:
                raise ValueError("Archived model trained on an evaluation season")
            scientific = tuple(manifest.get("model_ids", [])) == MODEL_IDS
            if scientific and (manifest.get("canonical_name") != "F20"
                               or manifest.get("feature_columns") != list(FEATURE_COLUMNS)):
                raise ValueError("Frozen F20 scientific panel contract changed")
            arms = MODEL_IDS if scientific else ("market_only", "F20-A")
            models = {}
            for arm in arms:
                model_path = manifest_path.parent / f"{arm}.joblib"
                if sha(model_path) != manifest["model_sha256"][arm]:
                    raise ValueError(f"Frozen model changed: {arm}")
                models[arm] = joblib.load(model_path)
            model_cache[model_hash] = (models, scientific, manifest["model_sha256"])
        X = captured_features(found[0])
        models, scientific, frozen_hashes = model_cache[model_hash]
        if row["model_sha256"] != frozen_hashes:
            raise ValueError(f"Forecast model hashes differ from frozen panel: {path}")
        if scientific:
            predicted = {arm: float(value[0]) for arm, value in
                         predict_panel(models, X.loc[:, list(FEATURE_COLUMNS)]).items()}
            panel_counts["F20_six_model"] += 1
        else:
            predicted = {arm: float(models[arm].predict_proba(X.loc[:, list(columns)])[0, 1])
                         for arm, columns in {"market_only": BASE_COLUMNS, "F20-A": FEATURE_COLUMNS}.items()}
            panel_counts["legacy_A_control"] += 1
        if set(row["home_win_probability"]) != set(predicted):
            raise ValueError(f"Forecast probability keys changed: {path}")
        for arm, value in predicted.items():
            if not np.isclose(value, row["home_win_probability"][arm], atol=1e-12):
                raise ValueError(f"Forecast probability cannot be reproduced: {path}")
    official_count = 0
    for pointer_path in (archive / "forecasts").glob("*/official.json"):
        pointer = json.loads(pointer_path.read_text())
        record_path = pointer_path.parent / f"{pointer['forecast_id']}.json"
        if sha(record_path) != pointer["record_sha256"]:
            raise ValueError(f"Official pointer hash changed: {pointer_path}")
        record = json.loads(record_path.read_text())
        if record["issuance_class"] != "eligible_weekly":
            raise ValueError(f"Official pointer designates ineligible record: {pointer_path}")
        official_count += 1
    return {"status": "verified", "forecast_records": len(records),
            "official_game_pointers": official_count, "model_manifests": len(model_cache),
            "panel_records": panel_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.models_root), indent=2))


if __name__ == "__main__":
    main()
