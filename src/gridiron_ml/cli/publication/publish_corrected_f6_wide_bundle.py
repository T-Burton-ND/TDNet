#!/usr/bin/env python3
"""Validate and atomically install the corrected-F6 operational wide roster."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from gridiron_ml.cli._paths import project_root


ROOT = project_root()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=ROOT / "models/season_2026_wide_margin_frozen_bundle")
    parser.add_argument("--backup", type=Path, default=ROOT / "models/season_2026_wide_margin_frozen_bundle_pre_corrected_f6_20260805")
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    target = args.target.resolve()
    backup = args.backup.resolve()
    inventory_path = candidate / "final_model_inventory.csv"
    report_path = candidate / "roster_manifest.json"
    inventory = pd.read_csv(inventory_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    failures = []
    if report.get("status") != "complete" or report.get("training_end_season") != 2025:
        failures.append("candidate is not a complete through-2025 roster")
    if len(inventory) != 36:
        failures.append(f"expected 36 rows, found {len(inventory)}")
    family = inventory["model_family"].astype(str)
    if int(family.eq("ensemble").sum()) != 2 or int(family.ne("ensemble").sum()) != 34:
        failures.append("candidate must contain 34 learned estimators and two ensembles")
    learned = inventory.loc[family.ne("ensemble")]
    if not learned["feature_config"].astype(str).eq("F6").all():
        failures.append("every learned estimator must use F6")
    if learned["selected_features_json"].astype(str).str.contains("market|vegas", case=False, regex=True).any():
        failures.append("market feature found in corrected-F6 learned roster")
    for row in inventory.to_dict("records"):
        checkpoint = Path(str(row["checkpoint_path"]))
        if not checkpoint.exists() or file_sha256(checkpoint) != str(row["checkpoint_sha256"]):
            failures.append(f"checkpoint verification failed: {row['model_id']}")
    if failures:
        raise RuntimeError("Refusing roster replacement: " + "; ".join(failures))
    if backup.exists():
        raise FileExistsError(f"Recovery backup already exists: {backup}")

    temporary = Path(tempfile.mkdtemp(prefix=".season_2026_wide_margin_f6_", dir=target.parent))
    try:
        checkpoint_root = temporary / "checkpoints"
        checkpoint_root.mkdir(parents=True)
        installed = inventory.copy()
        bundle_paths = []
        installed_paths = []
        for row in installed.to_dict("records"):
            source = Path(str(row["checkpoint_path"]))
            relative = Path("checkpoints") / f"{row['model_id']}.pkl"
            destination = temporary / relative
            shutil.copy2(source, destination)
            if file_sha256(destination) != str(row["checkpoint_sha256"]):
                raise RuntimeError(f"Copied checkpoint hash mismatch: {row['model_id']}")
            bundle_paths.append(str(relative))
            installed_paths.append(str((target / relative).relative_to(ROOT)))
        installed["bundle_checkpoint_path"] = bundle_paths
        installed["checkpoint_path"] = installed_paths
        installed.to_csv(temporary / "final_model_inventory.csv", index=False)
        shutil.copy2(candidate / "preseason_model_rankings.csv", temporary / "preseason_model_rankings.csv")
        # Selection provenance is shared by the through-2024 and through-2025
        # refits and therefore lives in the sibling selection directory, not
        # necessarily inside each fitted candidate.
        provenance_candidates = {
            "refit_manifest.csv": [candidate / "refit_manifest.csv", candidate.parent / "selection_v1" / "refit_manifest.csv"],
            "selection_report.json": [candidate / "selection_report.json", candidate.parent / "selection_v1" / "selection_report.json"],
        }
        for name, sources in provenance_candidates.items():
            source = next((path for path in sources if path.exists()), None)
            if source is not None:
                shutil.copy2(source, temporary / name)
        install_report = {
            **report,
            "installed_at_utc": datetime.now(timezone.utc).isoformat(),
            "installed_target": str(target),
            "recovery_backup": str(backup),
            "inventory_sha256": file_sha256(temporary / "final_model_inventory.csv"),
            "source_candidate": str(candidate),
        }
        (temporary / "roster_manifest.json").write_text(json.dumps(install_report, indent=2, sort_keys=True) + "\n")
        (temporary / "README.md").write_text(
            "# Corrected-F6 wide margin roster\n\n"
            "This operational bundle contains 34 learned margin estimators and two equal-weight ensembles. "
            "Every learned estimator was selected on the corrected, market-free, fixed 681-coordinate F6 contract "
            "and refit through the 2025 season. The displaced bundle is retained at the recovery path recorded in "
            "`roster_manifest.json`.\n",
            encoding="utf-8",
        )
        if target.exists():
            target.replace(backup)
        try:
            temporary.replace(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({"status": "installed", "target": str(target), "backup": str(backup), "models": 36}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
