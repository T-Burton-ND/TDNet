#!/usr/bin/env python3
"""Attach portable sidecar metadata required by the preseason freeze builder."""

from argparse import ArgumentParser
import json
import hashlib
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=Path("models/season_2026_full_roster/final_model_inventory.csv"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--comparison-root", type=Path, default=Path("data/comparisons/season_2026_full_roster_vs_vegas"))
    args = parser.parse_args()
    inventory_path = args.inventory.resolve()
    root = (args.output_root or inventory_path.parent).resolve()
    inventory = pd.read_csv(inventory_path)
    directories = {name: root / name for name in ["preprocessing", "calibration", "evaluation"]}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    for index, row in inventory.iterrows():
        model_id = str(row.get("model_id", row.get("final_model_name", f"model_{index:03d}")))
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_id)
        preprocessing = directories["preprocessing"] / f"{safe}.json"
        calibration = directories["calibration"] / f"{safe}.json"
        evaluation = directories["evaluation"] / f"{safe}.json"
        preprocessing.write_text(json.dumps({
            "model_id": model_id,
            "feature_config": row.get("feature_config"),
            "fingerprint": row.get("fingerprint"),
            "serialized_inside_checkpoint": True,
            "selected_features_json": row.get("selected_features_json"),
        }, indent=2) + "\n", encoding="utf-8")
        calibration.write_text(json.dumps({
            "model_id": model_id,
            "method": "model_internal_margin_probability_link",
            "validation_calibrator_present": False,
        }, indent=2) + "\n", encoding="utf-8")
        evaluation.write_text(json.dumps({
            "model_id": model_id,
            "selection_metric": row.get("selection_metric"),
            "selection_protocol": "rolling-origin consolidated publication search",
            "retrospective_comparison_root": str(args.comparison_root),
        }, indent=2) + "\n", encoding="utf-8")
        inventory.loc[index, "preprocessing_path"] = str(preprocessing.resolve())
        inventory.loc[index, "calibration_path"] = str(calibration.resolve())
        inventory.loc[index, "historical_evaluation_path"] = str(evaluation.resolve())
    inventory.to_csv(inventory_path, index=False)
    manifest_path = inventory_path.parent / "roster_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["inventory_sha256"] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"prepared metadata for {len(inventory)} roster models: {inventory_path}")


if __name__ == "__main__":
    main()
