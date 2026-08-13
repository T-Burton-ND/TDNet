"""Deterministic preseason freeze construction and verification."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import tempfile
from typing import Any

import pandas as pd
import yaml

from .bundles import canonical_json_bytes, git_state, sha256_file


FREEZE_DIRECTORIES = (
    "model_cards",
    "checkpoints",
    "preprocessing",
    "calibration",
    "split_definitions",
    "environment",
)


def build_model_cards(inventory: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Render one auditable Markdown card per frozen inventory row."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cards = []
    for index, row in inventory.reset_index(drop=True).iterrows():
        model_id = str(row.get("model_id", row.get("final_model_name", row.get("model_name", f"model_{index:02d}"))))
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_id)
        fields = {
            "Stable model ID": model_id,
            "Family": row.get("model_family", row.get("family", "unknown")),
            "Objective": row.get("objective", "unknown"),
            "Feature configuration": row.get("feature_config", "unknown"),
            "Checkpoint SHA-256": row.get("checkpoint_sha256", "not recorded"),
            "Historical evaluation": row.get("historical_evaluation_path", "not recorded"),
            "Calibration": row.get("calibration_path", "not recorded"),
            "Intended use": "Prospective NCAA football game prediction under the declared 2026 protocol.",
            "Limitations": "Predictions are uncertain, distribution shift is possible, and outputs are not betting advice.",
        }
        text = f"# TDNet model card: {model_id}\n\n" + "\n".join(
            f"- **{name}:** {value}" for name, value in fields.items()
        ) + "\n"
        path = output / f"{safe_id}.md"
        path.write_text(text, encoding="utf-8")
        cards.append(path)
    return cards


def build_preseason_freeze(
    *,
    project_root: str | Path,
    bundle_root: str | Path,
    inventory_path: str | Path,
    selection_report_path: str | Path,
    feature_registry_path: str | Path,
    feature_ladders_path: str | Path,
    split_paths: list[str | Path],
    environment_lock_path: str | Path,
    data_snapshot_manifest_path: str | Path,
    schedule_snapshot_path: str | Path,
    preseason_ranking_path: str | Path | None = None,
    freeze_version: str = "2026-preseason-v1",
    include_checkpoints: bool = True,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Build a new fail-closed freeze directory without overwriting an old one."""
    project_root = Path(project_root).resolve()
    bundle = Path(bundle_root).resolve()
    required_paths = {
        "inventory": Path(inventory_path),
        "selection report": Path(selection_report_path),
        "feature registry": Path(feature_registry_path),
        "feature ladders": Path(feature_ladders_path),
        "environment lock": Path(environment_lock_path),
        "data snapshot manifest": Path(data_snapshot_manifest_path),
        "schedule snapshot": Path(schedule_snapshot_path),
    }
    if preseason_ranking_path is not None:
        required_paths["preseason rankings"] = Path(preseason_ranking_path)
    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {name} is missing: {path}")
    if bundle.exists() and any(bundle.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty freeze bundle: {bundle}")
    state = git_state(project_root)
    if state["git_dirty"] and not allow_dirty:
        raise RuntimeError("Refusing to freeze a dirty worktree.")

    inventory = pd.read_csv(required_paths["inventory"])
    _validate_freeze_inventory(inventory, project_root)
    _validate_feature_registry(required_paths["feature registry"])
    for directory in FREEZE_DIRECTORIES:
        (bundle / directory).mkdir(parents=True, exist_ok=True)

    copied_inventory = inventory.copy()
    checkpoint_records = []
    for index, row in inventory.iterrows():
        source = _resolve(project_root, row["checkpoint_path"])
        actual_hash = sha256_file(source)
        expected_hash = str(row.get("checkpoint_sha256", "")).strip()
        if expected_hash and expected_hash.lower() not in {"nan", actual_hash}:
            raise ValueError(f"Checkpoint hash mismatch for {source}")
        copied_inventory.loc[index, "checkpoint_sha256"] = actual_hash
        copied_inventory.loc[index, "checkpoint_size_bytes"] = source.stat().st_size
        copied_inventory.loc[index, "checkpoint_storage_class"] = "public_bundle" if include_checkpoints else "private_hash_only"
        if include_checkpoints:
            destination = bundle / "checkpoints" / source.name
            shutil.copy2(source, destination)
            copied_inventory.loc[index, "bundle_checkpoint_path"] = str(destination.relative_to(bundle))
        checkpoint_records.append({
            "model_id": str(row.get("model_id", row.get("final_model_name", row.get("model_name", index)))),
            "sha256": actual_hash,
            "size_bytes": source.stat().st_size,
            "included": bool(include_checkpoints),
        })
        for column, destination_dir in [("preprocessing_path", "preprocessing"), ("calibration_path", "calibration")]:
            source_artifact = _resolve(project_root, row[column])
            shutil.copy2(source_artifact, bundle / destination_dir / f"{index:03d}_{source_artifact.name}")

    copied_inventory.to_csv(bundle / "final_model_inventory.csv", index=False)
    shutil.copy2(required_paths["selection report"], bundle / "model_selection_report.md")
    shutil.copy2(required_paths["feature registry"], bundle / "feature_registry.yaml")
    shutil.copy2(required_paths["feature ladders"], bundle / "feature_ladders.yaml")
    shutil.copy2(required_paths["environment lock"], bundle / "environment" / required_paths["environment lock"].name)
    shutil.copy2(required_paths["data snapshot manifest"], bundle / "data_snapshot_manifest.json")
    if preseason_ranking_path is not None:
        shutil.copy2(required_paths["preseason rankings"], bundle / "preseason_model_rankings.csv")
    for path in split_paths:
        split = Path(path)
        if not split.exists():
            raise FileNotFoundError(f"Split definition is missing: {split}")
        shutil.copy2(split, bundle / "split_definitions" / split.name)
    build_model_cards(copied_inventory, bundle / "model_cards")

    provenance = {
        **state,
        "python": os.sys.version,
        "project_root": str(project_root),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(bundle / "code_provenance.json", provenance)
    files = _file_inventory(bundle, excluded={"freeze_manifest.json", "SHA256SUMS", "README.md"})
    data_manifest_hash = sha256_file(bundle / "data_snapshot_manifest.json")
    manifest = {
        "freeze_version": freeze_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **state,
        "models": checkpoint_records,
        "feature_manifest_sha256": sha256_file(bundle / "feature_registry.yaml"),
        "data_snapshot_sha256": data_manifest_hash,
        "schedule_snapshot_sha256": sha256_file(required_paths["schedule snapshot"]),
        "preseason_ranking_sha256": sha256_file(bundle / "preseason_model_rankings.csv") if preseason_ranking_path is not None else None,
        "environment_lock_sha256": sha256_file(bundle / "environment" / required_paths["environment lock"].name),
        "container_sha256": None,
        "files": files,
    }
    manifest["manifest_sha256"] = sha256(canonical_json_bytes(manifest)).hexdigest()
    _atomic_json(bundle / "freeze_manifest.json", manifest)
    write_sha256sums(bundle)
    render_freeze_readme(bundle)
    return manifest


def write_sha256sums(bundle_root: str | Path) -> Path:
    bundle = Path(bundle_root)
    paths = sorted(p for p in bundle.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    output = bundle / "SHA256SUMS"
    output.write_text("".join(f"{sha256_file(path)}  {path.relative_to(bundle)}\n" for path in paths), encoding="utf-8")
    return output


def render_freeze_readme(bundle_root: str | Path) -> Path:
    bundle = Path(bundle_root)
    manifest = json.loads((bundle / "freeze_manifest.json").read_text(encoding="utf-8"))
    output = bundle / "README.md"
    output.write_text(
        "# TDNet 2026 preseason freeze\n\n"
        f"Freeze version: `{manifest['freeze_version']}`  \n"
        f"Created: `{manifest['created_at_utc']}`  \n"
        f"Git commit: `{manifest['git_commit']}`  \n"
        f"Frozen models: {len(manifest['models'])}  \n"
        f"Manifest SHA-256: `{manifest['manifest_sha256']}`\n\n"
        "Run `python src/gridiron_ml/cli/publication/verify_preseason_freeze.py --bundle <this-directory>` before use.\n",
        encoding="utf-8",
    )
    return output


def verify_preseason_freeze(bundle_root: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_root)
    failures = []
    manifest_path = bundle / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256", None)
    actual = sha256(canonical_json_bytes(manifest)).hexdigest()
    if claimed != actual:
        failures.append("manifest self-hash mismatch")
    for relative, expected in manifest.get("files", {}).items():
        path = bundle / relative
        if not path.exists():
            failures.append(f"missing file: {relative}")
        elif sha256_file(path) != expected["sha256"] or path.stat().st_size != int(expected["size_bytes"]):
            failures.append(f"file integrity mismatch: {relative}")
    inventory = pd.read_csv(bundle / "final_model_inventory.csv")
    represented = set(inventory["checkpoint_sha256"].astype(str))
    for model in manifest.get("models", []):
        if str(model["sha256"]) not in represented:
            failures.append(f"model absent from inventory: {model['model_id']}")
    return {"valid": not failures, "manifest_sha256": claimed, "failures": failures}


def _validate_freeze_inventory(inventory: pd.DataFrame, project_root: Path) -> None:
    required = {"checkpoint_path", "preprocessing_path", "calibration_path", "historical_evaluation_path"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Final inventory missing required columns: {sorted(missing)}")
    if inventory.empty:
        raise ValueError("Final model inventory is empty.")
    for column in required:
        for raw in inventory[column]:
            path = _resolve(project_root, raw)
            if not path.exists():
                raise FileNotFoundError(f"Inventory artifact is missing: {path}")


def _validate_feature_registry(path: Path) -> None:
    registry = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = registry.get("defaults", {})
    if "availability_rule" not in defaults:
        for name, spec in registry.get("features", {}).items():
            if "availability_rule" not in spec:
                raise ValueError(f"Feature '{name}' has no availability rule.")


def _resolve(project_root: Path, raw: Any) -> Path:
    if raw is None or pd.isna(raw):
        raise ValueError("Required inventory artifact path is null.")
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def _file_inventory(bundle: Path, *, excluded: set[str]) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(bundle)): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)
