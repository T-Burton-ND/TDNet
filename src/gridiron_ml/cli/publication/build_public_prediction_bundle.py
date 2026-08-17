#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Prepare and hash a public-safe immutable prediction bundle."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess

import pandas as pd

from gridiron_ml.publication import (
    build_prediction_bundle,
    prepare_public_prediction_table,
)
from gridiron_ml.publication.bundles import sha256_file


def main():
    parser = ArgumentParser()
    parser.add_argument("--project-root", default=project_root())
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prediction-deadline", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--data-snapshot-manifest", required=True)
    parser.add_argument("--schedule-snapshot", required=True)
    parser.add_argument("--environment-lock", required=True)
    parser.add_argument("--pipeline-version", default="gridiron_ml-0.1.2")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--kickoff-times-confirmed", action="store_true")
    parser.add_argument("--allow-dirty-code", action="store_true")
    args = parser.parse_args()
    source = Path(args.predictions)
    predictions = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.project_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    prepared = prepare_public_prediction_table(
        predictions,
        prediction_deadline_utc=args.prediction_deadline,
        feature_manifest_sha256=sha256_file(args.feature_manifest),
        data_snapshot_sha256=sha256_file(args.data_snapshot_manifest),
        schedule_snapshot_sha256=sha256_file(args.schedule_snapshot),
        git_commit=commit,
        pipeline_version=args.pipeline_version,
        environment_lock_sha256=sha256_file(args.environment_lock),
        kickoff_time_confirmed=args.kickoff_times_confirmed,
    )
    result = build_prediction_bundle(
        prepared,
        output_root=args.output_root,
        project_root=args.project_root,
        supporting_files={
            "feature_manifest": args.feature_manifest,
            "data_snapshot_manifest": args.data_snapshot_manifest,
            "schedule_snapshot": args.schedule_snapshot,
            "environment_lock": args.environment_lock,
        },
        allow_dirty_code=args.allow_dirty_code,
    )
    print(result["manifest"]["manifest_sha256"])


if __name__ == "__main__":
    main()
