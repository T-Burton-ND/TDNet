#!/usr/bin/env python3
"""Validate publication manifest and merged status tables."""

from argparse import ArgumentParser
import json

from gridiron_ml.experiments.publication import validate_experiment_output


def main():
    parser = ArgumentParser()
    parser.add_argument("--job-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = validate_experiment_output(
        job_manifest=args.job_manifest, output_root=args.output_root
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

