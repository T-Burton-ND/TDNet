#!/usr/bin/env python
"""Build and smoke-test time-adjusted fingerprint frames."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from pathlib import Path
import sys

REPO_ROOT = project_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gridiron_ml.experiments.time_adjusted import (
    DEFAULT_VERSION_SPECS,
    TimeAdjustedVersionSpec,
    build_time_adjusted_experiment_frames,
    default_output_root,
    default_source_fingerprint_root,
    safe_label,
    smoke_train_time_adjusted_models,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    add_common_args(build)
    build.add_argument("--labels", nargs="*", default=None)
    build.add_argument("--source-labels", nargs="*", default=None)
    build.add_argument("--max-features", type=int, default=120)
    build.add_argument("--overwrite", action="store_true")

    smoke = subparsers.add_parser("smoke-train")
    add_common_args(smoke)
    smoke.add_argument("--frame-label", default=None)
    smoke.add_argument("--frame-path", type=Path, default=None)
    smoke.add_argument("--smoke-output-root", type=Path, default=Path("/tmp/tdnet_time_adjusted_smoke"))
    smoke.add_argument("--models", nargs="*", default=["stat_weighted", "ridge", "random_forest"])
    smoke.add_argument("--train-years", nargs="*", type=int, default=[2010, 2011, 2012])
    smoke.add_argument("--val-years", nargs="*", type=int, default=[2013])
    smoke.add_argument("--test-years", nargs="*", type=int, default=[2014])

    return parser.parse_args()


def add_common_args(parser):
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--source-fingerprint-root", type=Path, default=None)


def main():
    args = parse_args()
    root = args.project_root.resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(root)
    source_root = (
        Path(args.source_fingerprint_root).resolve()
        if args.source_fingerprint_root
        else default_source_fingerprint_root(root)
    )

    if args.command == "build":
        specs = select_specs(args.labels, args.source_labels)
        artifacts = build_time_adjusted_experiment_frames(
            project_root=root,
            output_root=output_root,
            source_fingerprint_root=source_root,
            version_specs=specs,
            max_features=args.max_features,
            overwrite=args.overwrite,
        )
        print(f"Output root: {output_root}")
        for label, artifact in artifacts.items():
            print(f"{label}: {artifact.path} rows={len(artifact.frame)} cols={len(artifact.frame.columns)}")
        return

    if args.command == "smoke-train":
        frame_path = args.frame_path
        if frame_path is None:
            frame_label = args.frame_label or DEFAULT_VERSION_SPECS[0].label
            frame_path = output_root / "fingerprints" / safe_label(frame_label) / "canonical_fingerprint.parquet"
        summary = smoke_train_time_adjusted_models(
            project_root=root,
            frame_path=frame_path,
            output_root=args.smoke_output_root,
            model_names=tuple(args.models),
            train_years=tuple(args.train_years),
            val_years=tuple(args.val_years),
            test_years=tuple(args.test_years),
        )
        print(f"Smoke output: {args.smoke_output_root}")
        print(summary[["family", "model", "winner_accuracy", "rmse"]].to_string(index=False))
        return


def select_specs(labels, source_labels):
    selected_labels = {str(label) for label in labels or []}
    selected_sources = {str(label) for label in source_labels or []}
    specs = []
    for spec in DEFAULT_VERSION_SPECS:
        if selected_labels and spec.label not in selected_labels:
            continue
        if selected_sources and spec.source_label not in selected_sources:
            continue
        specs.append(spec)
    if specs:
        return tuple(specs)
    if selected_labels:
        known_by_label = {spec.label: spec for spec in DEFAULT_VERSION_SPECS}
        return tuple(known_by_label[label] for label in selected_labels if label in known_by_label)
    return tuple(DEFAULT_VERSION_SPECS)


if __name__ == "__main__":
    main()
