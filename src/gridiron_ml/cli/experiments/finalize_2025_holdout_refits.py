#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Build honest holdout polls after all 36 refit tasks complete."""

from argparse import ArgumentParser
from pathlib import Path
import os

import pandas as pd

from gridiron_ml.experiments.hyperparameter_search import default_source_fingerprint_root
from scripts.finalize_fingerprint_hyperparameter_search import build_poll_set, load_completed_entries


def main():
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("TDNET_HOLDOUT_ROOT", "holdout_2025_refits")))
    args = parser.parse_args()
    for objective in ["winner", "margin"]:
        selected = pd.read_csv(root / f"data/experiments/fingerprint_hyperparameter_search/{objective}/final_artifacts/selected_best_by_model.csv")
        selected.insert(0, "objective", objective)
        final_root = args.output_root / objective / "final_artifacts"
        entries = load_completed_entries(
            selected=selected, source_root=default_source_fingerprint_root(root),
            final_root=final_root, objective=objective,
        )
        if len(entries) != len(selected):
            raise RuntimeError(f"Only {len(entries)}/{len(selected)} {objective} refits are complete.")
        pd.DataFrame([entry["inventory"] for entry in entries]).to_csv(final_root / "final_model_inventory.csv", index=False)
        build_poll_set(
            trained=entries, season=2025, weeks=tuple(range(0, 17)), top_n=25,
            output_dir=final_root / "polls/2025_full_season",
            logo_dir=root / "data/meta/logos/by_team",
        )


if __name__ == "__main__":
    main()
