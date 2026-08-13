from gridiron_ml.cli._paths import project_root
#!/usr/bin/env python3
"""Build the deterministic 2026 preseason freeze bundle."""

from argparse import ArgumentParser
from pathlib import Path

from gridiron_ml.publication import build_preseason_freeze


def main():
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--feature-registry", type=Path, default=root / "configs/features/feature_registry.yaml")
    parser.add_argument("--feature-ladders", type=Path, default=root / "configs/features/feature_ladders.yaml")
    parser.add_argument("--split", type=Path, action="append", default=[])
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--data-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--schedule-snapshot", type=Path, required=True)
    parser.add_argument("--preseason-rankings", type=Path, help="Frozen historical-performance ranking sidecar used for Top-1/Top-3 publication views.")
    parser.add_argument("--freeze-version", default="2026-preseason-v1")
    parser.add_argument("--private-checkpoints", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="Rehearsal only; recorded in the manifest.")
    args = parser.parse_args()
    splits = args.split or [
        root / "configs/splits/rolling_origin.yaml",
        root / "configs/splits/leave_one_season_out.yaml",
        root / "configs/splits/final_historical_holdout.yaml",
    ]
    manifest = build_preseason_freeze(
        project_root=args.project_root,
        bundle_root=args.bundle,
        inventory_path=args.inventory,
        selection_report_path=args.selection_report,
        feature_registry_path=args.feature_registry,
        feature_ladders_path=args.feature_ladders,
        split_paths=splits,
        environment_lock_path=args.environment_lock,
        data_snapshot_manifest_path=args.data_snapshot_manifest,
        schedule_snapshot_path=args.schedule_snapshot,
        preseason_ranking_path=args.preseason_rankings,
        freeze_version=args.freeze_version,
        include_checkpoints=not args.private_checkpoints,
        allow_dirty=args.allow_dirty,
    )
    print(manifest["manifest_sha256"])


if __name__ == "__main__":
    main()
