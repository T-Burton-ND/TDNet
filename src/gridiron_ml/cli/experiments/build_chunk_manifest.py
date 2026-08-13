#!/usr/bin/env python3
"""Rebuild a chunk manifest from an existing trial manifest."""

from argparse import ArgumentParser
from pathlib import Path

from gridiron_ml.experiments.publication import atomic_write_frame, read_frame


def main():
    parser = ArgumentParser()
    parser.add_argument("--job-manifest", required=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--output")
    args = parser.parse_args()
    jobs = read_frame(args.job_manifest)
    jobs["chunk_id"] = jobs["task_id"].astype(int) // int(args.chunk_size)
    chunks = jobs.groupby("chunk_id", as_index=False).agg(
        task_start=("task_id", "min"),
        task_end=("task_id", "max"),
        trial_count=("task_id", "size"),
    )
    chunks["sge_task_id"] = chunks["chunk_id"] + 1
    output = Path(args.output or Path(args.job_manifest).with_name("chunk_manifest.parquet"))
    atomic_write_frame(chunks, output)
    print(f"chunks={len(chunks)}")
    print(output)


if __name__ == "__main__":
    main()

