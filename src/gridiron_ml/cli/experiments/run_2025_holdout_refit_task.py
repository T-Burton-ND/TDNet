from gridiron_ml.cli._paths import project_root
#!/usr/bin/env python3
"""Train and evaluate one honest 2025 holdout finalist."""

from argparse import ArgumentParser
from pathlib import Path
import json

import pandas as pd

from gridiron_ml.experiments.hyperparameter_search import default_source_fingerprint_root
from scripts.finalize_fingerprint_hyperparameter_search import train_selected_row


def main():
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sge-task-id", type=int, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    provenance_path = args.manifest.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    if not provenance.get("selection_leakage_audit_passed", False):
        raise RuntimeError(
            "Refusing refit: selected hyperparameters have not passed the train/evaluation overlap audit."
        )
    row = manifest[manifest["task_id"].eq(args.sge_task_id)]
    if len(row) != 1:
        raise IndexError(f"Expected exactly one row for task {args.sge_task_id}.")
    task = row.iloc[0]
    output = Path(task["output_root"]) / task["objective"] / "final_artifacts"
    result = train_selected_row(
        row=json.loads(task["selected_row_json"]), root=root,
        source_root=default_source_fingerprint_root(root), final_root=output,
        objective=task["objective"],
        train_years=tuple(json.loads(task["train_years_json"])),
        val_years=tuple(json.loads(task["val_years_json"])),
        eval_years=tuple(json.loads(task["eval_years_json"])),
    )
    print(result["inventory"])


if __name__ == "__main__":
    main()
