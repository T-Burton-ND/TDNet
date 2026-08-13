#!/usr/bin/env python
"""Run manifest-driven fingerprint hyperparameter search jobs."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = project_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gridiron_ml.experiments.hyperparameter_search import (
    build_search_manifest,
    default_output_root,
    default_source_fingerprint_root,
    merge_search_outputs,
    run_manifest_job,
)
from gridiron_ml.experiments.opponent_adjusted import (
    DEFAULT_TEST_YEARS,
    DEFAULT_TRAIN_YEARS,
    DEFAULT_VAL_YEARS,
)
from gridiron_ml.td_run.training import DEFAULT_MODEL_SPECS, ModelRunSpec


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest")
    add_common_args(build)
    add_selection_args(build)

    run = subparsers.add_parser("run-job")
    add_common_args(run)
    run.add_argument("--job-index", type=int, default=None)
    run.add_argument("--sge-task-id", type=int, default=None)
    run.add_argument("--job-manifest", type=Path, default=None)
    run.add_argument("--force", action="store_true")

    local = subparsers.add_parser("run-local")
    add_common_args(local)
    local.add_argument("--job-manifest", type=Path, default=None)
    local.add_argument("--limit", type=int, default=1)
    local.add_argument("--start-index", type=int, default=0)
    local.add_argument("--force", action="store_true")

    merge = subparsers.add_parser("merge")
    add_common_args(merge)

    return parser.parse_args()


def add_common_args(parser):
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--config", type=Path, default=Path("configs/models/tuning/fingerprint_hyperparameter_search.yaml"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--source-fingerprint-root", type=Path, default=None)
    parser.add_argument("--train-years", nargs="*", type=int, default=list(DEFAULT_TRAIN_YEARS))
    parser.add_argument("--val-years", nargs="*", type=int, default=list(DEFAULT_VAL_YEARS))
    parser.add_argument("--test-years", nargs="*", type=int, default=list(DEFAULT_TEST_YEARS))


def add_selection_args(parser):
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None, choices=["stat", "linear", "tree"])


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(root)
    source_root = (
        Path(args.source_fingerprint_root).resolve()
        if args.source_fingerprint_root
        else default_source_fingerprint_root(root)
    )
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    if args.command == "build-manifest":
        manifest = build_search_manifest(
            project_root=root,
            config_path=config_path,
            output_root=output_root,
            source_fingerprint_root=source_root,
            model_specs=select_models(args.models, args.families),
        )
        print(f"Manifest: {output_root / 'job_manifest.csv'}")
        print(f"Jobs: {len(manifest)}")
        return

    if args.command == "run-job":
        result = run_manifest_job(
            project_root=root,
            output_root=output_root,
            config_path=config_path,
            job_index=args.job_index,
            sge_task_id=args.sge_task_id,
            job_manifest=args.job_manifest,
            train_years=tuple(args.train_years),
            val_years=tuple(args.val_years),
            test_years=tuple(args.test_years),
            force=args.force,
        )
        print(f"Status: {result.get('status')}")
        print(f"Output: {result.get('output_dir')}")
        return

    if args.command == "run-local":
        manifest_path = args.job_manifest or output_root / "job_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        completed = 0
        for row in manifest.sort_values("job_index").itertuples(index=False):
            job_index = int(row.job_index)
            if job_index < int(args.start_index):
                continue
            if Path(row.metrics_path).exists() and not args.force:
                continue
            result = run_manifest_job(
                project_root=root,
                output_root=output_root,
                config_path=config_path,
                job_index=job_index,
                job_manifest=manifest_path,
                train_years=tuple(args.train_years),
                val_years=tuple(args.val_years),
                test_years=tuple(args.test_years),
                force=args.force,
            )
            completed += 1
            print(f"{completed}/{args.limit}: job_index={job_index} status={result.get('status')}")
            if completed >= int(args.limit):
                break
        print(f"Local jobs run: {completed}")
        return

    if args.command == "merge":
        merged = merge_search_outputs(output_root=output_root)
        print(f"Metrics rows: {len(merged['metrics'])}")
        print(f"Best rows: {len(merged['best'])}")
        return


def select_models(names, families):
    selected_names = {str(name) for name in names or []}
    selected_families = {str(family) for family in families or []}
    out = []
    for spec in DEFAULT_MODEL_SPECS:
        spec = ModelRunSpec.from_mapping(spec)
        if selected_names and spec.name not in selected_names:
            continue
        if selected_families and spec.family not in selected_families:
            continue
        out.append(spec)
    return tuple(out)


if __name__ == "__main__":
    main()
