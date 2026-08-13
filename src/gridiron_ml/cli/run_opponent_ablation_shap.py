#!/usr/bin/env python
"""Run opponent-adjusted ablation + SHAP experiments."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from pathlib import Path

import pandas as pd

from gridiron_ml.experiments.opponent_ablation import (
    DEFAULT_ABLATION_SPECS,
    DEFAULT_SHAP_MAX_BACKGROUND,
    DEFAULT_SHAP_MAX_EXPLAIN,
    DEFAULT_SOURCE_EXPERIMENT_NAME,
    DEFAULT_ABLATION_EXPERIMENT_NAME,
    ablation_spec_from_name,
    build_ablation_job_manifest,
    default_output_root,
    default_source_fingerprint_root,
    merge_ablation_outputs,
    run_manifest_job,
    version_spec_from_label,
)
from gridiron_ml.experiments.opponent_adjusted import (
    DEFAULT_TEST_YEARS,
    DEFAULT_TRAIN_YEARS,
    DEFAULT_VAL_YEARS,
    DEFAULT_VERSION_SPECS,
)
from gridiron_ml.td_run.training import DEFAULT_MODEL_SPECS, ModelRunSpec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build/run/merge the TDNet opponent-adjusted ablation SHAP sweep."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest", help="Create the job manifest.")
    add_common_args(build)
    add_selection_args(build)
    build.add_argument(
        "--no-ensure-fingerprints",
        action="store_true",
        help="Do not build missing source adjusted fingerprint parquet files.",
    )
    build.add_argument(
        "--overwrite-fingerprints",
        action="store_true",
        help="Rebuild source adjusted fingerprint parquet files.",
    )

    run = subparsers.add_parser("run-job", help="Run one manifest row.")
    add_common_args(run)
    add_shap_args(run)
    run.add_argument("--job-index", type=int, default=None, help="Zero-based manifest row.")
    run.add_argument("--sge-task-id", type=int, default=None, help="One-based SGE task id.")
    run.add_argument("--job-manifest", type=Path, default=None, help="Manifest CSV path.")
    run.add_argument("--force", action="store_true", help="Overwrite an existing job result.")
    run.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Persist trained model checkpoints for this job.",
    )

    local = subparsers.add_parser(
        "run-local",
        help="Run a small local batch from the manifest, useful for smoke testing.",
    )
    add_common_args(local)
    add_shap_args(local)
    local.add_argument("--job-manifest", type=Path, default=None, help="Manifest CSV path.")
    local.add_argument("--limit", type=int, default=1, help="Number of pending jobs to run.")
    local.add_argument("--start-index", type=int, default=0, help="First zero-based job index.")
    local.add_argument("--force", action="store_true", help="Overwrite existing job results.")
    local.add_argument("--keep-checkpoints", action="store_true", help="Persist checkpoints.")

    merge = subparsers.add_parser("merge", help="Merge completed job outputs.")
    add_common_args(merge)

    return parser.parse_args()


def add_common_args(parser):
    parser.add_argument(
        "--project-root",
        default=project_root(),
        type=Path,
        help="TDNet repository root.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        type=Path,
        help=f"Experiment output root. Defaults to data/experiments/{DEFAULT_ABLATION_EXPERIMENT_NAME}.",
    )
    parser.add_argument(
        "--source-fingerprint-root",
        default=None,
        type=Path,
        help=f"Source adjusted fingerprint root. Defaults to data/experiments/{DEFAULT_SOURCE_EXPERIMENT_NAME}.",
    )
    parser.add_argument(
        "--train-years",
        nargs="*",
        type=int,
        default=list(DEFAULT_TRAIN_YEARS),
        help="Training seasons.",
    )
    parser.add_argument(
        "--val-years",
        nargs="*",
        type=int,
        default=list(DEFAULT_VAL_YEARS),
        help="Validation seasons.",
    )
    parser.add_argument(
        "--test-years",
        nargs="*",
        type=int,
        default=list(DEFAULT_TEST_YEARS),
        help="Test/report seasons.",
    )


def add_selection_args(parser):
    parser.add_argument(
        "--fingerprints",
        nargs="*",
        default=None,
        help="Optional fingerprint labels, e.g. v1.4 v1.7.",
    )
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=None,
        help="Optional ablation names.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model names.",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        choices=["stat", "linear", "tree"],
        help="Optional model families.",
    )


def add_shap_args(parser):
    parser.add_argument(
        "--max-background",
        type=int,
        default=DEFAULT_SHAP_MAX_BACKGROUND,
        help="Maximum SHAP background rows.",
    )
    parser.add_argument(
        "--max-explain",
        type=int,
        default=DEFAULT_SHAP_MAX_EXPLAIN,
        help="Maximum SHAP explanation rows.",
    )
    parser.add_argument(
        "--shap-plots",
        action="store_true",
        help="Also save SHAP summary/bar PNGs for each job.",
    )


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(project_root)
    source_root = (
        Path(args.source_fingerprint_root).resolve()
        if args.source_fingerprint_root
        else default_source_fingerprint_root(project_root)
    )

    if args.command == "build-manifest":
        manifest = build_ablation_job_manifest(
            project_root=project_root,
            output_root=output_root,
            source_fingerprint_root=source_root,
            version_specs=select_versions(args.fingerprints),
            ablation_specs=select_ablations(args.ablations),
            model_specs=select_models(args.models, args.families),
            train_years=tuple(args.train_years),
            val_years=tuple(args.val_years),
            test_years=tuple(args.test_years),
            ensure_fingerprints=not args.no_ensure_fingerprints,
            overwrite=args.overwrite_fingerprints,
        )
        print(f"Manifest: {output_root / 'job_manifest.csv'}")
        print(f"Jobs: {len(manifest)}")
        return

    if args.command == "run-job":
        result = run_manifest_job(
            project_root=project_root,
            output_root=output_root,
            job_index=args.job_index,
            sge_task_id=args.sge_task_id,
            job_manifest=args.job_manifest,
            train_years=tuple(args.train_years),
            val_years=tuple(args.val_years),
            test_years=tuple(args.test_years),
            max_background=args.max_background,
            max_explain=args.max_explain,
            shap_plots=args.shap_plots,
            keep_checkpoints=args.keep_checkpoints,
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
            metrics_path = Path(row.metrics_path)
            if metrics_path.exists() and not args.force:
                continue
            result = run_manifest_job(
                project_root=project_root,
                output_root=output_root,
                job_index=job_index,
                job_manifest=manifest_path,
                train_years=tuple(args.train_years),
                val_years=tuple(args.val_years),
                test_years=tuple(args.test_years),
                max_background=args.max_background,
                max_explain=args.max_explain,
                shap_plots=args.shap_plots,
                keep_checkpoints=args.keep_checkpoints,
                force=args.force,
            )
            completed += 1
            print(f"{completed}/{args.limit}: job_index={job_index} status={result.get('status')}")
            if completed >= int(args.limit):
                break
        print(f"Local jobs run: {completed}")
        return

    if args.command == "merge":
        merged = merge_ablation_outputs(output_root=output_root)
        print(f"Metrics rows: {len(merged['metrics'])}")
        print(f"SHAP rows: {len(merged['shap'])}")
        print(f"Summary: {output_root / 'summary'}")
        return


def select_versions(labels):
    if not labels:
        return DEFAULT_VERSION_SPECS
    return tuple(version_spec_from_label(label) for label in labels)


def select_ablations(names):
    if not names:
        return DEFAULT_ABLATION_SPECS
    return tuple(ablation_spec_from_name(name) for name in names)


def select_models(names, families):
    requested_names = {str(name) for name in (names or [])}
    requested_families = {str(family) for family in (families or [])}
    selected = []
    for raw in DEFAULT_MODEL_SPECS:
        spec = ModelRunSpec.from_mapping(raw)
        if requested_names and spec.name not in requested_names:
            continue
        if requested_families and spec.family not in requested_families:
            continue
        selected.append(spec)
    if not selected:
        raise ValueError("No model specs matched the requested filters.")
    return tuple(selected)


if __name__ == "__main__":
    main()
