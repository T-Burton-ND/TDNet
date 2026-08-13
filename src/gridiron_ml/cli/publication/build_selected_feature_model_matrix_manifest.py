#!/usr/bin/env python3
"""Build the publication architecture matrix manifest on the selected frame."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.experiments.publication import build_experiment_manifest  # noqa: E402


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "data/experiments/publication_feature_model_matrix_v2",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_4/canonical_fingerprint.parquet",
    )
    args = parser.parse_args()
    manifest, chunks = build_experiment_manifest(
        project_root=args.project_root,
        config_path=args.project_root / "configs/publication/feature_model_matrix.yaml",
        data_path=args.data_path,
        output_root=args.artifact_root,
    )
    report = {
        "status": "manifest_built_selected_v1_4",
        "data_path": str(args.data_path),
        "artifact_root": str(args.artifact_root),
        "task_count": int(len(manifest)),
        "chunk_count": int(len(chunks)),
        "objectives": sorted(manifest["objective"].unique().tolist()),
        "feature_tiers": sorted(manifest["feature_config"].unique().tolist()),
        "model_levels": sorted(manifest["model_level"].unique().tolist()),
        "seeds": sorted(manifest["seed"].unique().tolist()),
        "split_configs": sorted(manifest["split_config"].unique().tolist()),
        "excludes_2026_from_fit": all(
            2026 not in json.loads(value) for value in manifest["train_seasons_json"]
        ) and all(2026 not in json.loads(value) for value in manifest["val_seasons_json"]),
    }
    (args.artifact_root / "manifest_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
