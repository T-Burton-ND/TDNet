#!/usr/bin/env python3
"""Package retained pregame prediction rows into a verifiable frozen bundle.

This is a recovery tool for a week whose predictions were generated and
retained before kickoff but whose normal Tuesday bundle was not written. It
never recalculates predictions and records that packaging occurred later.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import UTC, datetime
import json
from pathlib import Path

import pandas as pd

from gridiron_ml.cli._paths import project_root
from gridiron_ml.publication.bundles import (
    build_prediction_bundle,
    prepare_public_prediction_table,
    sha256_file,
)


def main() -> int:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report-manifest", type=Path, required=True)
    parser.add_argument("--schedule-snapshot", type=Path, required=True)
    parser.add_argument("--model-inventory", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--prediction-deadline-utc", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--prediction-code-commit",
        default="not_recorded_recovered_from_retained_pregame_artifact",
    )
    args = parser.parse_args()

    source = pd.read_csv(args.predictions)
    report = json.loads(args.report_manifest.read_text(encoding="utf-8"))
    prepared = prepare_public_prediction_table(
        source,
        prediction_deadline_utc=args.prediction_deadline_utc,
        feature_manifest_sha256=sha256_file(args.model_inventory),
        data_snapshot_sha256=sha256_file(args.schedule_snapshot),
        schedule_snapshot_sha256=sha256_file(args.schedule_snapshot),
        git_commit=args.prediction_code_commit,
        pipeline_version="retained-pregame-recovery-v1",
        environment_lock_sha256=sha256_file(args.environment_lock),
        kickoff_time_confirmed=True,
    )
    result = build_prediction_bundle(
        prepared,
        output_root=args.output_root,
        project_root=root,
        supporting_files={
            "source_predictions.csv": args.predictions,
            "report_manifest.json": args.report_manifest,
            "schedule_snapshot.parquet": args.schedule_snapshot,
            "model_inventory.csv": args.model_inventory,
        },
        metadata={
            "recovery_packaging_only": True,
            "predictions_recalculated": False,
            "recovery_packaged_at_utc": datetime.now(UTC).isoformat(),
            "retained_prediction_source": str(args.predictions),
            "retained_prediction_source_sha256": sha256_file(args.predictions),
            "retained_prediction_created_at_utc": report.get("created_at_utc"),
            "source_report_manifest_sha256": sha256_file(args.report_manifest),
        },
        allow_dirty_code=True,
    )
    print(
        json.dumps(
            {
                "status": "recovered_from_retained_pregame_predictions",
                "bundle": str(result["root"]),
                "prediction_rows": len(prepared),
                "game_count": int(prepared["game_id"].nunique()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
