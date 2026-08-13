import hashlib
import json

import pandas as pd

from gridiron_ml.publication.locked_bundle import validate_canonical_2026_inventory


def test_canonical_inventory_requires_exact_locked_artifacts(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    records = []
    rows = []
    for level, family in zip(("M1", "M2", "M3", "M4", "M5", "M10"), ("linear", "spline", "tree", "boosted", "neural", "knn")):
        checkpoint = bundle / "checkpoints" / level / "checkpoint.pkl"
        calibrator = bundle / "calibrators" / level / "production_through_2025.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        calibrator.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(level.encode())
        calibrator.write_text("{}\n", encoding="utf-8")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({
            "model_level": level,
            "model_family": family,
            "checkpoint": f"checkpoints/{level}/checkpoint.pkl",
            "checkpoint_sha256": digest(checkpoint),
            "calibrator": f"calibrators/{level}/production_through_2025.json",
            "calibrator_sha256": digest(calibrator),
        })
        rows.append({
            "model_level": level,
            "model_family": family,
            "objective": "margin",
            "feature_config": "F4",
            "market_bearing": False,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "calibrator_path": str(calibrator),
            "calibrator_sha256": digest(calibrator),
            "fingerprint_path": "data/fingerprint.parquet",
            "training_end_season": 2025,
        })
    manifest = bundle / "LOCK_MANIFEST.json"
    manifest.write_text(json.dumps({
        "status": "locked_candidate_not_final_publication_freeze",
        "feature_config": "F4",
        "market_bearing": False,
        "fit_train_seasons": list(range(2010, 2026)),
        "artifacts": records,
    }), encoding="utf-8")
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)
    result = validate_canonical_2026_inventory(inventory, manifest, project_root=tmp_path)
    assert result["status"] == "pass"


def test_canonical_inventory_rejects_legacy_roster(tmp_path):
    inventory = tmp_path / "legacy.csv"
    pd.DataFrame([{"checkpoint_path": "old.pkl", "fingerprint_path": "old.parquet"}]).to_csv(inventory, index=False)
    manifest = tmp_path / "LOCK_MANIFEST.json"
    manifest.write_text(json.dumps({"status": "locked_candidate_not_final_publication_freeze"}), encoding="utf-8")
    result = validate_canonical_2026_inventory(inventory, manifest, project_root=tmp_path)
    assert result["status"] == "invalid"
    assert any("missing required locked-bundle columns" in error for error in result["errors"])
