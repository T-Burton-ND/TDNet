#!/usr/bin/env python3
"""Build a reproducible objective-specific public model-set manifest.

The default output is metadata-only: checkpoints remain local, while every
checkpoint path and SHA-256 is recorded.  Pass ``--copy-checkpoints`` only for
an explicitly requested model release that is allowed to carry binaries.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd

from gridiron_ml.publication.bundles import sha256_file


BASELINE_FAMILIES = {"knn", "naive"}
POLL_EXCLUDED_IDS_BY_OBJECTIVE = {
    "winner": {
        "winner_linear_ols",
        "winner_linear_huber",
        "winner_linear_ridge",
        "winner_linear_sgd",
    },
    "margin": set(),
}
MARKET_MARKERS = ("market", "vegas", "spread", "moneyline", "over_under")
INVALID_FEATURE_CONFIGS = {"F7", "F8"}


def _contains_market(value: object) -> bool:
    text = str(value).lower()
    return any(marker in text for marker in MARKET_MARKERS)


def _model_card(row: pd.Series, checkpoint_storage: str) -> str:
    return (
        f"# TDNet model card: {row['model_id']}\n\n"
        f"- Objective: `{row['objective']}`\n"
        f"- Family/type: `{row.get('model_family', '')}` / `{row.get('model_type', '')}`\n"
        f"- Fingerprint configuration: `{row.get('feature_config', '')}`\n"
        f"- Training seasons: `{row.get('training_seasons', '')}`\n"
        f"- Checkpoint SHA-256: `{row.get('checkpoint_sha256', '')}`\n"
        f"- Checkpoint storage: `{checkpoint_storage}`\n"
        "- Market inputs: excluded from training and prediction.\n"
        "- Intended use: prospective NCAA football prediction under the declared TDNet protocol.\n"
        "- Limitations: predictions are uncertain and are not betting advice.\n"
    )


def build_model_set(
    *, inventory_path: Path, ranking_path: Path | None, output_root: Path,
    objective: str, copy_checkpoints: bool = False, overwrite_existing: bool = False,
) -> pd.DataFrame:
    inventory = pd.read_csv(inventory_path).copy()
    required = {"model_id", "objective", "checkpoint_path", "checkpoint_sha256", "model_family"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Inventory is missing {sorted(missing)}.")
    selected = inventory.loc[inventory["objective"].astype(str).eq(objective)].copy()
    selected = selected.loc[~selected["model_family"].astype(str).str.lower().isin(BASELINE_FAMILIES)].copy()
    if selected.empty:
        raise ValueError(f"No learned {objective} models remain in {inventory_path}.")
    if selected["feature_config"].astype(str).isin(INVALID_FEATURE_CONFIGS).any():
        raise ValueError("Refusing to publish F7/F8 model rows.")
    if selected.apply(lambda row: _contains_market(row.get("feature_config", "")) or _contains_market(row.get("selected_features_json", "")), axis=1).any():
        raise ValueError("Refusing to publish a market-bearing model row.")

    poll_exclusions = POLL_EXCLUDED_IDS_BY_OBJECTIVE.get(objective, set())
    selected["use_in_weekly_consensus"] = True
    selected["use_in_tdnet_poll"] = ~selected["model_id"].astype(str).isin(poll_exclusions)
    selected["use_in_comparisons"] = True
    selected["comparative_only"] = False
    selected["public_model_set_objective"] = objective
    selected = selected.sort_values(["use_in_tdnet_poll", "selection_metric", "model_id"], ascending=[False, True, True], kind="mergesort")

    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not overwrite_existing:
        raise FileExistsError(f"Refusing to overwrite non-empty model set: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "model_cards").mkdir(exist_ok=True)
    (output_root / "checkpoints").mkdir(exist_ok=True)

    storage = "public_bundle" if copy_checkpoints else "private_hash_only"
    copied = selected.copy()
    for index, row in copied.iterrows():
        source = Path(str(row["checkpoint_path"])).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Missing checkpoint for {row['model_id']}: {source}")
        actual = sha256_file(source)
        if str(row["checkpoint_sha256"]).strip().lower() not in {"", "nan", actual.lower()}:
            raise ValueError(f"Checkpoint hash mismatch for {row['model_id']}.")
        copied.loc[index, "checkpoint_sha256"] = actual
        copied.loc[index, "checkpoint_storage_class"] = storage
        if copy_checkpoints:
            destination = output_root / "checkpoints" / source.name
            shutil.copy2(source, destination)
            copied.loc[index, "bundle_checkpoint_path"] = str(destination.relative_to(output_root))
        safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(row["model_id"]))
        (output_root / "model_cards" / f"{safe_id}.md").write_text(_model_card(copied.loc[index], storage), encoding="utf-8")

    copied.to_csv(output_root / "final_model_inventory.csv", index=False)
    if ranking_path is not None and ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        ranking = ranking.loc[ranking["model_id"].astype(str).isin(copied["model_id"].astype(str))].copy()
        ranking.to_csv(output_root / "preseason_model_rankings.csv", index=False)

    files = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"freeze_manifest.json", "SHA256SUMS", "README.md"}:
            files[str(path.relative_to(output_root))] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "freeze_version": f"2026-{objective}-model-set-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": objective,
        "training_cutoff": 2025,
        "market_inputs_used_for_training": False,
        "comparative_baselines_included": False,
        "poll_exclusions": sorted(poll_exclusions),
        "checkpoint_storage_class": storage,
        "source_inventory": str(inventory_path.resolve()),
        "source_inventory_sha256": sha256_file(inventory_path),
        "source_ranking": str(ranking_path.resolve()) if ranking_path else None,
        "model_count": int(len(copied)),
        "poll_model_count": int(copied["use_in_tdnet_poll"].sum()),
        "files": files,
    }
    manifest["manifest_sha256"] = __import__("hashlib").sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (output_root / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_root / "README.md").write_text(
        f"# TDNet 2026 {objective} model set\n\n"
        f"This frozen set contains {len(copied)} learned models trained through 2025. "
        "KNN and naive models are comparative baselines only and are intentionally absent. "
        "Market inputs are excluded.\n\n"
        f"Poll-eligible models: {int(copied['use_in_tdnet_poll'].sum())}.\n"
        f"Manifest SHA-256: `{manifest['manifest_sha256']}`\n",
        encoding="utf-8",
    )
    sums = "".join(f"{sha256_file(path)}  {path.relative_to(output_root)}\n" for path in sorted(output_root.rglob("*")) if path.is_file() and path.name != "SHA256SUMS")
    (output_root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return copied.reset_index(drop=True)


def main() -> None:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=root / "models/season_2026_no_market_roster/final_model_inventory.csv")
    parser.add_argument("--ranking", type=Path, default=None)
    parser.add_argument("--objective", choices=("winner", "margin"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-checkpoints", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true", help="Regenerate an existing model set in place.")
    args = parser.parse_args()
    frame = build_model_set(
        inventory_path=args.inventory.resolve(), ranking_path=args.ranking.resolve() if args.ranking else None,
        output_root=args.output_root, objective=args.objective, copy_checkpoints=args.copy_checkpoints,
        overwrite_existing=args.overwrite_existing,
    )
    print(f"wrote {len(frame)} {args.objective} model rows to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
