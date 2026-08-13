#!/usr/bin/env python3
"""Merge compact publication experiment fragments."""

from argparse import ArgumentParser

from gridiron_ml.experiments.publication import merge_experiment_chunks


def main():
    parser = ArgumentParser()
    parser.add_argument("--job-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    tables = merge_experiment_chunks(
        job_manifest=args.job_manifest, output_root=args.output_root
    )
    for name, table in tables.items():
        print(f"{name}={len(table)}")


if __name__ == "__main__":
    main()

