#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Build a replacement preseason freeze with the two recovered tree roles."""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = project_root()
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gridiron_ml.publication.freeze import build_preseason_freeze


TARGETS = {"margin_boosted_hist_gradient_boosted", "margin_tree_random_forest"}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--base-bundle",
        type=Path,
        default=ROOT / "models/season_2026_wide_margin_frozen_bundle",
    )
    parser.add_argument(
        "--supplement-root",
        type=Path,
        default=ROOT / "models/season_2026_missing_margin_tree_roles",
    )
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--freeze-version", default="2026-full-roster-v3")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Build for audit/rehearsal while preserving the worktree dirty flag in provenance.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    base = args.base_bundle.resolve()
    supplement = args.supplement_root.resolve()
    output = args.output_bundle.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output bundle: {output}")

    base_inventory = pd.read_csv(base / "final_model_inventory.csv")
    supplement_inventory = pd.read_csv(supplement / "final_model_inventory.csv")
    actual_targets = set(supplement_inventory["model_id"].astype(str))
    if actual_targets != TARGETS:
        raise ValueError(f"Supplement contains {sorted(actual_targets)}, expected {sorted(TARGETS)}")
    if set(base_inventory["model_id"].astype(str)) & TARGETS:
        raise ValueError("Base bundle already contains one of the supplement model IDs")

    # Make every source path explicit and local. Existing frozen bytes remain
    # the source of truth; only the two new rows come from the supplement.
    base_inventory = base_inventory.copy()
    base_inventory["checkpoint_path"] = [
        str((base / str(relative)).resolve())
        for relative in base_inventory["bundle_checkpoint_path"]
    ]
    supplement_inventory = supplement_inventory.copy()
    supplement_inventory["checkpoint_path"] = supplement_inventory["checkpoint_path"].map(
        lambda path: str(Path(path).resolve())
    )
    combined = pd.concat([base_inventory, supplement_inventory], ignore_index=True, sort=False)
    combined = combined.sort_values("model_id", kind="mergesort").reset_index(drop=True)
    if len(combined) != 39 or combined["model_id"].duplicated().any():
        raise ValueError(f"Expected 39 unique combined rows, got {len(combined)}")

    source_root = output.parent / f"{output.name}_source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "preprocessing").mkdir(exist_ok=True)
    (source_root / "calibration").mkdir(exist_ok=True)
    (source_root / "evaluation").mkdir(exist_ok=True)
    for index, row in combined.iterrows():
        model_id = str(row["model_id"])
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_id)
        preprocessing = source_root / "preprocessing" / f"{safe}.json"
        calibration = source_root / "calibration" / f"{safe}.json"
        evaluation = source_root / "evaluation" / f"{safe}.json"
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
        }, indent=2) + "\n", encoding="utf-8")
        combined.loc[index, "preprocessing_path"] = str(preprocessing)
        combined.loc[index, "calibration_path"] = str(calibration)
        combined.loc[index, "historical_evaluation_path"] = str(evaluation)

    inventory_path = source_root / "final_model_inventory.csv"
    combined.to_csv(inventory_path, index=False)
    manifest = build_preseason_freeze(
        project_root=project_root,
        bundle_root=output,
        inventory_path=inventory_path,
        selection_report_path=base / "model_selection_report.md",
        feature_registry_path=base / "feature_registry.yaml",
        feature_ladders_path=base / "feature_ladders.yaml",
        split_paths=[base / "split_definitions" / name for name in (
            "rolling_origin.yaml", "leave_one_season_out.yaml", "final_historical_holdout.yaml"
        )],
        environment_lock_path=base / "environment" / "environment.yaml",
        data_snapshot_manifest_path=base / "data_snapshot_manifest.json",
        schedule_snapshot_path=project_root / "data/raw/cfbd/v2/games/2026.parquet",
        freeze_version=args.freeze_version,
        include_checkpoints=True,
        allow_dirty=args.allow_dirty,
    )
    print(json.dumps({
        "freeze_version": manifest["freeze_version"],
        "model_count": len(manifest["models"]),
        "output_bundle": str(output),
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
