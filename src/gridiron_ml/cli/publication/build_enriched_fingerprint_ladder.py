#!/usr/bin/env python3
"""Build the realized F5 temporal and F6 graph publication source frame."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from gridiron_ml.cli._paths import project_root
from gridiron_ml.fingerprints.ladder import build_publication_ladder_frame
from gridiron_ml.publication.protocol import validate_feature_frame


ROOT = project_root()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_4/canonical_fingerprint.parquet",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--max-seasons", type=int)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    frame = pd.read_parquet(source)
    if args.max_seasons is not None:
        seasons = sorted(pd.to_numeric(frame["keys_season"], errors="coerce").dropna().astype(int).unique())
        keep = seasons[-int(args.max_seasons) :]
        frame = frame.loc[pd.to_numeric(frame["keys_season"], errors="coerce").isin(keep)].copy()
    enriched = build_publication_ladder_frame(frame)
    manifests = validate_feature_frame(
        enriched,
        registry_path=ROOT / "configs/features/feature_registry.yaml",
        ladder_path=ROOT / "configs/features/feature_ladders.yaml",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output, index=False)
    metadata_path = (args.metadata or output.with_suffix(".metadata.json")).resolve()
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": file_sha256(source),
        "output": str(output),
        "output_sha256": file_sha256(output),
        "rows": int(len(enriched)),
        "columns": int(enriched.shape[1]),
        "season_min": int(pd.to_numeric(enriched["keys_season"], errors="coerce").min()),
        "season_max": int(pd.to_numeric(enriched["keys_season"], errors="coerce").max()),
        "feature_counts": {tier: int(value["feature_count"]) for tier, value in manifests.items()},
        "schema_hashes": {tier: value["schema_hash"] for tier, value in manifests.items()},
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
