from gridiron_ml.cli._paths import project_root
#!/usr/bin/env python3
"""Build publication trial/chunk manifests without submitting SGE work."""

from argparse import ArgumentParser
import os
from pathlib import Path

from gridiron_ml.experiments.publication import (
    assert_disk_guardrail,
    build_experiment_manifest,
    estimate_disk_bytes,
)


def main():
    parser = ArgumentParser()
    parser.add_argument("--project-root", default=project_root())
    parser.add_argument("--config", default="configs/publication/feature_model_matrix.yaml")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--minimum-free-gb", type=float, default=30.0)
    args = parser.parse_args()
    output_root = args.output_root or os.environ.get("TDNET_ARTIFACT_ROOT", "publication_artifacts")
    assert_disk_guardrail(output_root, args.minimum_free_gb)
    manifest, chunks = build_experiment_manifest(
        project_root=args.project_root,
        config_path=args.config,
        data_path=args.data_path,
        output_root=output_root,
        max_trials=args.max_trials,
    )
    print(f"trials={len(manifest)}")
    print(f"chunks={len(chunks)}")
    print(f"estimated_compact_fragment_bytes={estimate_disk_bytes(manifest)}")


if __name__ == "__main__":
    main()
