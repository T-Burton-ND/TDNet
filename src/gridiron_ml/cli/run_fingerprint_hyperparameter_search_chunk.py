#!/usr/bin/env python
"""Run a small chunk of fingerprint hyperparameter-search manifest rows."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = project_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gridiron_ml.experiments.hyperparameter_search import run_manifest_job
from gridiron_ml.experiments.opponent_adjusted import (
    DEFAULT_TEST_YEARS,
    DEFAULT_TRAIN_YEARS,
    DEFAULT_VAL_YEARS,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job-manifest", type=Path, default=None)
    parser.add_argument("--chunk-index", type=int, default=None, help="Zero-based chunk index.")
    parser.add_argument("--sge-task-id", type=int, default=None, help="One-based SGE task id.")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-years", nargs="*", type=int, default=list(DEFAULT_TRAIN_YEARS))
    parser.add_argument("--val-years", nargs="*", type=int, default=list(DEFAULT_VAL_YEARS))
    parser.add_argument("--test-years", nargs="*", type=int, default=list(DEFAULT_TEST_YEARS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    manifest_path = args.job_manifest or output_root / "job_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    chunk_index = resolve_chunk_index(args.chunk_index, args.sge_task_id)
    start = int(chunk_index) * int(args.chunk_size)
    stop = min(start + int(args.chunk_size), len(manifest))
    rows = manifest.iloc[start:stop].copy()
    if rows.empty:
        raise IndexError(f"Chunk {chunk_index} has no rows for manifest size {len(manifest)}")

    print(
        f"Running chunk={chunk_index} rows={start}-{stop - 1} "
        f"workers={args.workers} chunk_size={args.chunk_size}",
        flush=True,
    )
    results = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = [
            pool.submit(
                run_one,
                project_root=str(args.project_root.resolve()),
                output_root=str(output_root),
                config_path=str(resolve(args.project_root.resolve(), args.config)),
                manifest_path=str(manifest_path),
                job_index=int(row.job_index),
                train_years=tuple(args.train_years),
                val_years=tuple(args.val_years),
                test_years=tuple(args.test_years),
                force=bool(args.force),
            )
            for row in rows.itertuples(index=False)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"job_index={result.get('job_index')} status={result.get('status')}", flush=True)

    chunk_dir = output_root / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(chunk_dir / f"chunk_{chunk_index:06d}.csv", index=False)
    print(f"Chunk complete: {chunk_dir / f'chunk_{chunk_index:06d}.csv'}", flush=True)


def run_one(
    *,
    project_root: str,
    output_root: str,
    config_path: str,
    manifest_path: str,
    job_index: int,
    train_years: tuple[int, ...],
    val_years: tuple[int, ...],
    test_years: tuple[int, ...],
    force: bool,
) -> dict:
    try:
        result = run_manifest_job(
            project_root=project_root,
            output_root=output_root,
            config_path=config_path,
            job_index=job_index,
            job_manifest=manifest_path,
            train_years=train_years,
            val_years=val_years,
            test_years=test_years,
            force=force,
        )
        return {"job_index": job_index, **dict(result)}
    except Exception as exc:
        return {"job_index": job_index, "status": "failed", "error": repr(exc)}


def resolve_chunk_index(chunk_index: int | None, sge_task_id: int | None) -> int:
    if chunk_index is not None:
        return int(chunk_index)
    if sge_task_id is None:
        raise ValueError("Provide --chunk-index or --sge-task-id.")
    return int(sge_task_id) - 1


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
