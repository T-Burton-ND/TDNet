#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Create one-task-per-model honest 2025 holdout refit manifest."""

from argparse import ArgumentParser
from pathlib import Path
import json
import os

import pandas as pd


def main():
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("TDNET_HOLDOUT_ROOT", "holdout_2025_refits")))
    args = parser.parse_args()
    rows = []
    for objective in ["winner", "margin"]:
        selected_path = root / f"data/experiments/fingerprint_hyperparameter_search/{objective}/final_artifacts/selected_best_by_model.csv"
        for _, row in pd.read_csv(selected_path).iterrows():
            rows.append({
                "task_id": len(rows) + 1,
                "objective": objective,
                "family": row["family"],
                "model": row["model"],
                "selected_row_json": json.dumps(row.to_dict()),
                "output_root": str(args.output_root.resolve()),
                "train_years_json": json.dumps(list(range(2010, 2024))),
                "val_years_json": json.dumps([2024]),
                "eval_years_json": json.dumps([2025]),
            })
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_root / "job_manifest.csv", index=False)
    manifest.to_parquet(args.output_root / "job_manifest.parquet", index=False)
    provenance = {
        "train_years": list(range(2010, 2024)), "validation_years": [2024], "evaluation_years": [2025],
        "fit_eval_overlap": False, "task_count": len(manifest), "one_model_per_task": True,
        "selection_leakage_audit_passed": False,
        "ready_to_run": False,
        "block_reason": "The selected rows came from a hyperparameter search scored and selected on 2025.",
    }
    (args.output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(manifest[["task_id", "objective", "family", "model"]].to_string(index=False))


if __name__ == "__main__":
    main()
