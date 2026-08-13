#!/usr/bin/env python3
"""Run one local or SGE publication experiment chunk."""

from argparse import ArgumentParser
import json
import os

from gridiron_ml.experiments.publication import run_experiment_chunk


def main():
    parser = ArgumentParser()
    parser.add_argument("--job-manifest", required=True)
    parser.add_argument("--chunk-id", type=int)
    parser.add_argument("--sge-task-id", type=int)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("NSLOTS", 4)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    args = parser.parse_args()
    sge_task_id = args.sge_task_id or (
        int(os.environ["SGE_TASK_ID"]) if os.environ.get("SGE_TASK_ID") else None
    )
    results = run_experiment_chunk(
        job_manifest=args.job_manifest,
        chunk_id=args.chunk_id,
        sge_task_id=sge_task_id,
        workers=args.workers,
        force=args.force,
        retry_incomplete=args.retry_incomplete,
    )
    print(json.dumps(results, indent=2, default=str))
    if any(row.get("status") == "failed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
