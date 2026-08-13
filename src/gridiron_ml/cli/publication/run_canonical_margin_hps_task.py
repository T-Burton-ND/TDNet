#!/usr/bin/env python3
"""Run one canonical margin-only pre-2025 HPS gap-fill task.

The worker audits the task boundary before fitting. A task is rejected if it
contains a non-margin objective, a non-canonical tier, a 2025/2026 fit or
validation season, missing provenance, or a source frame without a season
column. Results are written only to the task's manifest-defined group path.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.experiments.hyperparameter_search import run_search_combo  # noqa: E402
from gridiron_ml.publication.protocol import CANONICAL_TIERS  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_row(row: dict) -> dict:
    tier = str(row.get("canonical_feature_config", ""))
    if str(row.get("objective", "")) != "margin":
        raise ValueError("canonical HPS gap-fill accepts margin objective only")
    if tier not in CANONICAL_TIERS:
        raise ValueError(f"canonical_feature_config must be F0-F8, got {tier!r}")
    fit_years = json.loads(row.get("train_years_json", "[]") or "[]")
    val_years = json.loads(row.get("val_years_json", "[]") or "[]")
    test_years = json.loads(row.get("test_years_json", "[]") or "[]")
    illegal = sorted({int(x) for x in fit_years + val_years + test_years if int(x) >= 2025})
    if illegal:
        raise ValueError(f"holdout/prospective seasons in task boundary: {illegal}")
    required = ["fingerprint_path", "model_config_path", "feature_registry", "feature_ladders", "output_dir"]
    missing = [name for name in required if not str(row.get(name, "")).strip()]
    if missing:
        raise ValueError(f"task missing provenance fields: {missing}")
    source = Path(str(row["fingerprint_path"]))
    if not source.exists():
        raise FileNotFoundError(source)
    columns = pd.read_parquet(source, engine="pyarrow").columns
    season_column = "season" if "season" in columns else "keys_season" if "keys_season" in columns else None
    if season_column is None:
        raise ValueError("source fingerprint has no season key (expected season or keys_season)")
    frame = pd.read_parquet(source, columns=[season_column])
    seasons = [int(x) for x in sorted(pd.to_numeric(frame[season_column], errors="coerce").dropna().astype(int).unique())]
    excluded_2026_rows = int(pd.to_numeric(frame[season_column], errors="coerce").ge(2026).sum())
    return {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "margin",
        "canonical_feature_config": tier,
        "train_years": [int(x) for x in fit_years],
        "validation_years": [int(x) for x in val_years],
        "test_years": [int(x) for x in test_years],
        "source_fingerprint_path": str(source),
        "source_fingerprint_sha256": sha256_file(source),
        "source_seasons": seasons,
        "source_2026_rows_excluded_before_fit": excluded_2026_rows,
        "holdout_2025_excluded": True,
        "prospective_2026_excluded": True,
        "status": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_parquet(args.manifest) if args.manifest.suffix == ".parquet" else pd.read_csv(args.manifest)
    selected = manifest.loc[manifest["sge_task_id"].astype(int).eq(args.task_id)]
    if len(selected) != 1:
        raise ValueError(f"expected one task for SGE_TASK_ID={args.task_id}, found {len(selected)}")
    row = selected.iloc[0].to_dict()
    output = Path(str(row["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "leakage_audit.json"
    try:
        audit = audit_row(row)
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        result = run_search_combo(
            project_root=ROOT,
            config_path=args.config,
            row=row,
            train_years=tuple(json.loads(row.get("train_years_json", "[]") or "[]")),
            val_years=tuple(json.loads(row.get("val_years_json", "[]") or "[]")),
            test_years=tuple(json.loads(row.get("test_years_json", "[]") or "[]")),
            force=False,
        )
        result["leakage_audit_path"] = str(audit_path)
        (output / "task_result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps({"status": result.get("status"), "output_dir": str(output), "audit": str(audit_path)}))
        return 0 if result.get("status") in {"success", "skipped_existing"} else 1
    except Exception as exc:
        failure = {"status": "failed", "error": str(exc), "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        (output / "task_result.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
