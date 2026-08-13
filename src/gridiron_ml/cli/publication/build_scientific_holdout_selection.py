#!/usr/bin/env python3
"""Condense the audited heatmap manifest into one margin selection per cell."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json

import pandas as pd

from gridiron_ml.cli._paths import project_root


ROOT = project_root()
TIERS = tuple(f"F{i}" for i in range(9))
LEVELS = ("M1", "M2", "M3", "M4", "M5", "M10")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--heatmap-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_parquet(args.heatmap_manifest)
    selected = source.loc[source["objective"].astype(str).eq("margin")].copy()
    identity = [
        "feature_config", "model_level", "model_family", "model_config",
        "params_json", "seed", "data_path", "selection_source",
    ]
    conflicts = selected.groupby(["feature_config", "model_level"])[identity].nunique(dropna=False)
    if conflicts.gt(1).any().any():
        raise ValueError("Heatmap folds disagree on the selected configuration for a scientific cell.")
    selected = selected.sort_values("outer_fold").drop_duplicates(["feature_config", "model_level"])
    expected = {(tier, level) for tier in TIERS for level in LEVELS}
    observed = set(zip(selected["feature_config"].astype(str), selected["model_level"].astype(str)))
    if len(selected) != len(expected) or observed != expected:
        raise ValueError(f"Expected 54 F0-F8 margin cells; found {len(selected)}.")

    selected["status"] = "success"
    selected["selection_manifest"] = str(args.heatmap_manifest.resolve())
    keep = [
        "objective", "feature_config", "model_level", "model_family",
        "model_config", "params_json", "seed", "data_path", "status",
        "selection_source", "selection_manifest",
    ]
    selected = selected[keep].sort_values(["feature_config", "model_level"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    report = {
        "status": "ready",
        "cell_count": len(selected),
        "objective": "margin",
        "tiers": list(TIERS),
        "levels": list(LEVELS),
        "source_manifest": str(args.heatmap_manifest.resolve()),
        "source_manifest_sha256": sha256(args.heatmap_manifest.resolve()),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output.resolve()),
        "holdout_selection_uses_2025": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
