#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Materialize the learned 2026 weekly/paper roster from the freeze bundle.

The full freeze bundle may contain explicit naive baselines for comparison.
Weekly ballots exclude those rows, retain every learned role, and retain KNN
because KNN is an actual model ballot in the TDNet workflow.
"""

from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd


ROOT = project_root()
DEFAULT_WIDE_BUNDLE = Path(
    "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
    "corrected_f6_wide_margin_roster/through_2025_v1"
)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_WIDE_BUNDLE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/publication_2026/weekly_learned_model_inventory.csv",
    )
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    inventory = pd.read_csv(bundle / "final_model_inventory.csv")
    if "model_family" not in inventory or "use_in_tdnet_poll" not in inventory:
        raise ValueError("Freeze inventory lacks model-family or poll-enable columns.")
    selected = inventory.loc[
        inventory["use_in_tdnet_poll"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
        & inventory["model_family"].astype(str).str.lower().ne("naive")
    ].copy()
    if "market_bearing" in selected:
        selected = selected.loc[
            ~selected["market_bearing"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
        ].copy()
    if "feature_config" in selected:
        selected = selected.loc[~selected["feature_config"].astype(str).isin({"F7", "F8"})].copy()
    if selected.empty:
        raise ValueError("No learned poll models remain after excluding naive baselines.")
    if "bundle_checkpoint_path" in selected:
        selected["checkpoint_path"] = selected["bundle_checkpoint_path"].map(
            lambda value: str((bundle / str(value)).resolve())
        )
    def portable_or_absolute(value: object) -> str:
        resolved = (Path(str(value)) if Path(str(value)).is_absolute() else ROOT / str(value)).resolve()
        try:
            return str(resolved.relative_to(ROOT))
        except ValueError:
            return str(resolved)

    selected["fingerprint_path"] = selected["fingerprint_path"].map(portable_or_absolute)
    missing = [path for path in selected["checkpoint_path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing weekly checkpoint bytes: {missing[:5]}")
    if selected["model_family"].astype(str).str.lower().eq("knn").sum() == 0:
        raise ValueError("The learned weekly roster must retain at least one KNN ballot.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    summary = {
        "bundle": str(bundle),
        "output": str(args.output),
        "model_count": int(len(selected)),
        "excluded_from_poll_count": int(len(inventory) - len(selected)),
        "excluded_naive_count": int(
            inventory["model_family"].astype(str).str.lower().eq("naive").sum()
        ),
        "knn_model_count": int(selected["model_family"].astype(str).str.lower().eq("knn").sum()),
        "model_families": sorted(selected["model_family"].astype(str).unique()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
