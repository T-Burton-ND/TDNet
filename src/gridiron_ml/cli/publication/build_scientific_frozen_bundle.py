#!/usr/bin/env python3
"""Materialize one portable, hashable scientific F0--F8 margin bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


LEVELS = ("M1", "M2", "M3", "M4", "M5", "M10")
TIERS = tuple(f"F{i}" for i in range(9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--external-margin-root", type=Path, required=True)
    parser.add_argument("--local-refit-root", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    expected = {(tier, level) for tier in TIERS for level in LEVELS}
    observed = set(zip(manifest["feature_config"], manifest["model_level"]))
    if observed != expected or len(manifest) != 54:
        raise ValueError("scientific manifest must contain exactly 54 F0-F8 x M1/M2/M3/M4/M5/M10 cells")

    bundle = args.output_bundle.resolve()
    checkpoints = bundle / "checkpoints"
    if bundle.exists() and not args.reuse_existing:
        raise FileExistsError(f"refusing to overwrite existing bundle: {bundle}")
    checkpoints.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in manifest.sort_values(["feature_config", "model_level"]).to_dict("records"):
        tier, level = str(row["feature_config"]), str(row["model_level"])
        local = args.local_refit_root / "cells" / tier / level / "checkpoint.pkl"
        external = args.external_margin_root / tier / level / "checkpoint.pkl"
        source = local if local.exists() else external
        if not source.exists():
            raise FileNotFoundError(f"missing checkpoint for {tier}/{level}: {local} or {external}")
        destination = checkpoints / tier / level / "checkpoint.pkl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        try:
            inventory_checkpoint = str(destination.relative_to(Path.cwd().resolve()))
        except ValueError:
            inventory_checkpoint = str(destination)
        rows.append(
            {
                "model_id": f"scientific_{tier}_{level}",
                "final_model_name": f"scientific_{tier}_{level}",
                "model_family": row["model_family"],
                "objective": "margin",
                "feature_config": tier,
                "fingerprint": tier,
                "fingerprint_path": row["source_fingerprint_path"],
                "checkpoint_path": inventory_checkpoint,
                "checkpoint_sha256": sha256(destination),
                "checkpoint_size_bytes": destination.stat().st_size,
                "training_seasons": row["fit_train_seasons_json"],
                "use_in_tdnet_poll": tier not in {"F7", "F8"},
                "use_in_weekly_consensus": tier not in {"F7", "F8"},
                "comparative_only": tier in {"F7", "F8"},
                "market_bearing": tier in {"F7", "F8"},
                "selection_source": row["selection_source"],
                "source_fingerprint_sha256": row["source_sha256"],
                "selected_feature_count": None,
                "checkpoint_source": "local_refit" if source == local else "durable_existing_refit",
            }
        )

    inventory = pd.DataFrame(rows)
    inventory.to_csv(bundle / "final_model_inventory.csv", index=False)
    sums = "".join(f"{row['checkpoint_sha256']}  {row['checkpoint_path']}\n" for row in rows)
    (bundle / "CHECKPOINT_SHA256SUMS").write_text(sums, encoding="utf-8")
    metadata = {
        "bundle_version": "scientific-f0-f8-margin-2026-v2",
        "cell_count": 54,
        "fingerprints": list(TIERS),
        "architectures": list(LEVELS),
        "objective": "margin",
        "fit_train_seasons": list(range(2010, 2026)),
        "prospective_2026_used_for_fit": False,
        "confirmatory_fingerprints": list(TIERS[:7]),
        "comparative_only_fingerprints": ["F7", "F8"],
        "manifest_sha256": sha256(args.manifest.resolve()),
        "checkpoint_count": len(rows),
        "market_bearing_tiers_retrained_with_market_features": ["F7", "F8"],
    }
    (bundle / "BUNDLE_MANIFEST.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (bundle / "README.md").write_text(
        "# TDNet scientific F0-F8 frozen bundle\n\n"
        "This bundle contains 54 margin models: six architectures across F0-F8. "
        "All models fit seasons 2010-2025 and exclude 2026 data. F0-F6 are the "
        "confirmatory market-free fingerprints; F7 is market-only and F8 is F6 plus market. "
        "F7/F8 are research comparators and are excluded from official predictions and polls.\n\n"
        "Use `final_model_inventory.csv` with the shared weekly publication loader. "
        "Verify checkpoint hashes against `CHECKPOINT_SHA256SUMS` before use.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
