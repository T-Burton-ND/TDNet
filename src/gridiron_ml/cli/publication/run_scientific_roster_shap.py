#!/usr/bin/env python3
"""Build, run, and finalize the scientific-roster F6 SHAP study."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path

from gridiron_ml.cli._paths import project_root
from gridiron_ml.publication.scientific_shap_study import (
    build_shap_manifest,
    finalize_shap,
    run_shap_task,
)


def main() -> None:
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["build-manifest", "run-task", "finalize"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, required=True)
        cmd.add_argument("--output-root", type=Path, required=True)
    run = sub.choices["run-task"]
    run.add_argument("--manifest", type=Path)
    run.add_argument("--task-id", type=int)
    run.add_argument("--sge-task-id", type=int)
    run.add_argument("--force", action="store_true")
    final = sub.choices["finalize"]
    final.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = project_root()
    manifest = args.manifest if hasattr(args, "manifest") and args.manifest else args.output_root / "job_manifest.parquet"
    if args.command == "build-manifest":
        frame = build_shap_manifest(project_root=root, config_path=args.config, output_root=args.output_root)
        print(json.dumps({"tasks": len(frame), "manifest": str(manifest)}, indent=2))
    elif args.command == "run-task":
        task_id = args.task_id
        if task_id is None:
            sge_id = args.sge_task_id or int(os.environ["SGE_TASK_ID"])
            task_id = int(sge_id) - 1
        result = run_shap_task(manifest_path=manifest, config_path=args.config, task_id=task_id, force=args.force)
        print(json.dumps(result, indent=2))
        if result["status"] != "success":
            raise SystemExit(1)
    else:
        print(json.dumps(finalize_shap(manifest_path=manifest, output_root=args.output_root), indent=2))


if __name__ == "__main__":
    main()
