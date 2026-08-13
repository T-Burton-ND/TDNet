#!/usr/bin/env python3
"""Build one preseason state artifact selected by manifest task ID."""

from argparse import ArgumentParser
import os
import pandas as pd

from gridiron_ml.publication import materialize_preseason_state


def main():
    parser = ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sge-task-id", type=int)
    args = parser.parse_args()
    task = args.sge_task_id or int(os.environ.get("SGE_TASK_ID", "1"))
    manifest = pd.read_csv(args.manifest)
    selected = manifest.loc[manifest["sge_task_id"].astype(int) == int(task)]
    if len(selected) != 1:
        raise IndexError(f"Expected one preseason-state row for SGE task {task}.")
    row = selected.iloc[0]
    result = materialize_preseason_state(
        source_path=row.source_path,
        output_dir=row.output_dir,
        season=int(row.season),
        fingerprint=str(row.fingerprint),
    )
    print(result)


if __name__ == "__main__":
    main()
