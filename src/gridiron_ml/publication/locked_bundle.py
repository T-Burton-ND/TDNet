"""Contracts for using the frozen 2026 candidate in weekly production."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_LEVELS = {"M1", "M2", "M3", "M4", "M5", "M10"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_canonical_2026_inventory(
    inventory_path: str | Path,
    bundle_manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, object]:
    """Fail closed unless a weekly inventory exactly names the locked bundle."""
    inventory_path = Path(inventory_path)
    manifest_path = Path(bundle_manifest_path)
    root = Path(project_root).resolve()
    inventory = pd.read_csv(inventory_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("status") != "locked_candidate_not_final_publication_freeze":
        errors.append("locked bundle is not an explicit candidate manifest")
    if manifest.get("feature_config") != "F4" or manifest.get("market_bearing") is not False:
        errors.append("locked bundle is not market-free F4")
    if manifest.get("fit_train_seasons") != list(range(2010, 2026)):
        errors.append("locked bundle fit boundary is not 2010-2025")
    required = {
        "model_level", "model_family", "objective", "feature_config",
        "market_bearing", "checkpoint_path", "checkpoint_sha256",
        "calibrator_path", "calibrator_sha256", "fingerprint_path",
        "training_end_season",
    }
    missing = sorted(required - set(inventory.columns))
    if missing:
        errors.append("inventory missing required locked-bundle columns: " + ", ".join(missing))
    records = {str(item.get("model_level")): item for item in manifest.get("artifacts", [])}
    if set(records) != EXPECTED_LEVELS:
        errors.append("locked bundle artifact levels are incomplete")
    if not missing:
        levels = set(inventory["model_level"].astype(str))
        if levels != EXPECTED_LEVELS or len(inventory) != len(EXPECTED_LEVELS):
            errors.append("inventory must contain exactly one row for each canonical level")
        if not inventory["objective"].astype(str).str.lower().eq("margin").all():
            errors.append("canonical weekly inventory must be margin-only")
        if not inventory["feature_config"].astype(str).eq("F4").all():
            errors.append("canonical weekly inventory must use F4")
        if inventory["market_bearing"].astype(str).str.lower().isin({"true", "1", "yes"}).any():
            errors.append("canonical weekly inventory contains market-bearing rows")
        if pd.to_numeric(inventory["training_end_season"], errors="coerce").gt(2025).any():
            errors.append("canonical weekly inventory includes post-2025 training")
        for row in inventory.to_dict("records"):
            level = str(row["model_level"])
            record = records.get(level)
            if record is None:
                errors.append(f"{level}: not present in locked manifest")
                continue
            for field in ("model_family", "checkpoint_sha256", "calibrator_sha256"):
                if str(row[field]) != str(record[field]):
                    errors.append(f"{level}: {field} disagrees with locked manifest")
            for field in ("checkpoint_path", "calibrator_path"):
                path = Path(str(row[field]))
                if not path.is_absolute():
                    path = root / path
                manifest_field = "checkpoint" if field == "checkpoint_path" else "calibrator"
                expected = manifest_path.parent / str(record[manifest_field])
                if path.resolve() != expected.resolve():
                    errors.append(f"{level}: {field} is not the locked path")
                if not path.exists():
                    errors.append(f"{level}: missing {field} {path}")
                else:
                    manifest_hash_field = "checkpoint_sha256" if field == "checkpoint_path" else "calibrator_sha256"
                    if sha256_file(path) != str(record[manifest_hash_field]):
                        errors.append(f"{level}: {field} hash mismatch")
    return {"status": "pass" if not errors else "invalid", "errors": errors, "inventory": str(inventory_path), "bundle_manifest": str(manifest_path)}
