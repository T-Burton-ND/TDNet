#!/usr/bin/env python3
"""Select reproducible top settings after the complete metrics-only HPS array finishes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "publication/2026/research/scientific_120_hps_v1/candidate_tasks.jsonl"
DATA = Path("/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1/data")
RESULTS = Path("/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1/results")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finalize(result_root: Path, data_root: Path, *, require_all: bool = True) -> dict:
    tasks = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    if len(tasks) != 12000:
        raise ValueError("Expected 12,000 planned candidate configurations")
    seen, statuses, metrics = set(), [], []
    for task in tasks:
        task_id = task["task_id"]
        path = result_root / f"task_{task_id:05d}.json"
        if not path.exists():
            statuses.append({"task_id": task_id, "fingerprint": task["fingerprint"],
                             "architecture": task["architecture"], "trial_id": task["trial_id"],
                             "status": "missing"})
            continue
        result = json.loads(path.read_text())
        if result["task_id"] != task_id or result["binding"]["task"] != task:
            raise ValueError(f"Task {task_id} result has a changed identity")
        if result["binding"]["manifest_sha256"] != sha256(MANIFEST):
            raise ValueError("A result was computed from a different candidate manifest")
        receipt = json.loads((data_root / task["fingerprint"] / "receipt.json").read_text())
        if result["binding"]["matrix_sha256"] != receipt["matrix_sha256"]:
            raise ValueError("A result was computed from a different input matrix")
        if task["architecture"] == "M5" and result["status"] == "success":
            history = result_root / f"task_{task_id:05d}_history.csv"
            if not history.is_file() or sha256(history) != result["history_sha256"]:
                raise ValueError(f"M5 task {task_id} has missing or changed loss history")
        statuses.append({"task_id": task_id, "fingerprint": task["fingerprint"],
                         "architecture": task["architecture"], "trial_id": task["trial_id"],
                         "status": result["status"], "error": result.get("error", "")})
        if result["status"] == "success":
            folds = result["fold_metrics"]
            if sorted(row["test_season"] for row in folds) != list(range(2021, 2026)):
                raise ValueError(f"Task {task_id} lacks five historical folds")
            metrics.extend(folds)
            seen.add(task_id)
    status_frame = pd.DataFrame(statuses)
    if require_all and status_frame.status.ne("success").any():
        counts = status_frame.status.value_counts().to_dict()
        raise ValueError(f"HPS is incomplete; refusing final selection: {counts}")
    ledger = pd.DataFrame(metrics)
    if ledger.empty:
        raise ValueError("No successful HPS results")
    summaries = (ledger.groupby(["fingerprint", "architecture", "trial_id"])
                 .margin_mae.agg(["mean", "std", "count"]).reset_index())
    if summaries["count"].ne(5).any():
        raise ValueError("A successful candidate has fewer than five fold metrics")
    winners = []
    task_index = {(t["fingerprint"], t["architecture"], t["trial_id"]): t for t in tasks}
    for (fingerprint, architecture), cell in summaries.groupby(["fingerprint", "architecture"]):
        chosen = cell.sort_values(["mean", "std", "trial_id"]).iloc[0]
        task = task_index[(fingerprint, architecture, int(chosen.trial_id))]
        receipt = json.loads((data_root / fingerprint / "receipt.json").read_text())
        winners.append({"fingerprint": fingerprint, "architecture": architecture,
                        "trial_id": int(chosen.trial_id), "params": task["params"],
                        "fit_seed": task["fit_seed"], "five_fold_mean_mae": float(chosen["mean"]),
                        "five_fold_sd_mae": float(chosen["std"]),
                        "complete_candidates": int(len(cell)),
                        "feature_matrix": str(data_root / fingerprint / "features.parquet"),
                        "feature_matrix_sha256": receipt["matrix_sha256"],
                        "feature_names": receipt["feature_names"],
                        "representation_receipt_sha256": (
                            json.loads((result_root / f"task_{task['task_id']:05d}.json").read_text())
                            ["binding"]["winner_receipt_sha256"]),
                        "model_config_base": {
                            "M1": "configs/models/linear/config_ridge.yaml",
                            "M2": "configs/models/spline/config_spline_ridge.yaml",
                            "M3": "configs/models/tree/config_random_forest.yaml",
                            "M4": "configs/models/boosted/config_hist_gradient_boosted.yaml",
                            "M5": "configs/models/neural/config_mlp.yaml",
                            "M10": "configs/models/knn/config_knn_distance.yaml",
                        }[architecture],
                        "source_task_id": task["task_id"]})
    if require_all and len(winners) != 120:
        raise ValueError(f"Expected 120 selected configurations, got {len(winners)}")
    result_root.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(result_root / "fold_metrics.csv", index=False)
    status_frame.to_csv(result_root / "candidate_status.csv", index=False)
    result = {"status": "complete" if len(seen) == 12000 else "partial",
              "candidate_tasks": len(tasks), "successful_tasks": len(seen),
              "selected_cells": len(winners), "manifest_sha256": sha256(MANIFEST),
              "winners": winners, "checkpoint_retained": False,
              "selection_rule": "mean_five_fold_margin_mae_then_fold_sd_then_trial_id",
              "interpretation": "retrospective_development_not_untouched_holdout"}
    temp = result_root / "selected_configs.tmp.json"
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temp.replace(result_root / "selected_configs.json")
    return {key: result[key] for key in ("status", "candidate_tasks", "successful_tasks", "selected_cells")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULTS)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(finalize(args.result_root, args.data_root, require_all=not args.allow_partial), indent=2))


if __name__ == "__main__":
    main()
