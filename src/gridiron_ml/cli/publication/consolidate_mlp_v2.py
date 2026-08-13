#!/usr/bin/env python3
"""Incrementally consolidate the bounded publication MLP array.

The array writes one status and (on success) one result parquet per trial.
This reducer is safe to run while the array is active: missing/running trials
remain visible in the report and never enter the selection table.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = project_root()
import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.experiments.publication import select_finalists  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(os.environ.get("TDNET_MLP_ARTIFACT_ROOT", ROOT / "data/experiments/publication_hps_mlp_v2")),
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/experiments/publication_mlp_v2_consolidation")
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_root / "job_manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    rows = []
    counts = {"success": 0, "failed": 0, "running": 0, "missing": 0, "malformed": 0}
    for row in manifest.itertuples(index=False):
        run_root = Path(row.output_path)
        status_path = run_root / "status.json"
        if not status_path.exists():
            counts["missing"] += 1
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            state = str(status.get("status", "missing"))
        except Exception:
            counts["malformed"] += 1
            continue
        if state == "success":
            result_path = run_root / "result.parquet"
            if not result_path.exists():
                counts["malformed"] += 1
                continue
            try:
                result = pd.read_parquet(result_path)
                if len(result) != 1:
                    counts["malformed"] += 1
                    continue
                merged = {**row._asdict(), **result.iloc[0].to_dict()}
                rows.append(merged)
                counts["success"] += 1
            except Exception:
                counts["malformed"] += 1
        elif state in counts:
            counts[state] += 1
        else:
            counts["failed"] += 1

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["task_id"], kind="mergesort").reset_index(drop=True)
    result_path = output_root / "successful_results.parquet"
    csv_path = output_root / "successful_results.csv.gz"
    results.to_parquet(result_path, index=False)
    results.to_csv(csv_path, index=False, compression="gzip")
    finalists = select_finalists(results) if not results.empty else pd.DataFrame()
    finalists_path = output_root / "finalists.parquet"
    finalists.to_parquet(finalists_path, index=False)
    finalists.to_csv(output_root / "finalists.csv", index=False)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "planned_trials": int(len(manifest)),
        "counts": counts,
        "complete": counts["success"] == len(manifest) and not any(
            counts[key] for key in ("failed", "running", "missing", "malformed")
        ),
        "selection_status": "eligible_for_selection" if counts["success"] == len(manifest) else "incomplete_not_eligible",
        "output_sha256": sha256_file(result_path),
        "output_path": str(result_path),
        "finalist_rows": int(len(finalists)),
        "finalists_path": str(finalists_path),
        "selection_metric": {"margin": "mae", "winner": "brier_score"},
    }
    (output_root / "consolidation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
