#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Create the normalized ranking sidecar for a margin-only roster."""

from argparse import ArgumentParser
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranking = pd.read_csv(args.source)
    inventory = pd.read_csv(args.inventory)
    if "model_id" not in ranking or "model_id" not in inventory:
        raise ValueError("Ranking and inventory must contain model_id")
    ranking = ranking.copy()
    ranking["model_id"] = ranking["model_id"].astype(str).str.replace(r"^margin_", "", regex=True)
    if "final_model_name" in ranking:
        ranking["final_model_name"] = ranking["final_model_name"].astype(str).str.replace(r"^margin_", "", regex=True)
    expected = set(inventory["model_id"].astype(str))
    ranking = ranking.loc[ranking["model_id"].isin(expected)].copy()
    missing = expected - set(ranking["model_id"])
    if missing:
        raise ValueError(f"Normalized ranking is missing inventory models: {sorted(missing)}")
    if ranking["model_id"].duplicated().any():
        raise ValueError("Normalized ranking contains duplicate model IDs")
    # Active artifact references must come from the normalized inventory.  The
    # source ranking is retained only as provenance in the sidecar manifest;
    # otherwise old margin_* checkpoint paths would leak back into the new
    # publication package.
    inventory_by_id = inventory.set_index("model_id")
    for column in inventory.columns:
        if column in ranking.columns and column != "model_id":
            ranking[column] = ranking["model_id"].map(inventory_by_id[column])
    if "ranking_source" in ranking.columns:
        ranking["ranking_source"] = "normalized_margin_replay_inventory"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output, index=False)
    manifest = {
        "source": str(args.source),
        "inventory": str(args.inventory),
        "output": str(args.output),
        "rows": int(len(ranking)),
        "removed_margin_prefix": True,
        "status": "normalized_margin_ranking_ready",
    }
    args.output.with_name("normalized_margin_ranking_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
