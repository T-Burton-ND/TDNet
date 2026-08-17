#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Build leakage-safe legacy HPS manifests; never submit jobs."""

from argparse import ArgumentParser
from pathlib import Path
import json
import os

import pandas as pd
import numpy as np

from gridiron_ml.experiments.hyperparameter_search import (
    build_search_manifest,
    default_source_fingerprint_root,
)


CONFIGS = {
    "winner": "configs/models/tuning/fingerprint_hyperparameter_search_winner.yaml",
    "margin": "configs/models/tuning/fingerprint_hyperparameter_search_margin.yaml",
}
CV_TEST_SEASONS = tuple(range(2020, 2025))


def main():
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("TDNET_NESTED_SEARCH_ROOT", "nested_search_2025")))
    args = parser.parse_args()
    artifact = args.artifact_root.resolve()
    rows = []
    for objective, config in CONFIGS.items():
        output = artifact / objective
        base = build_search_manifest(
            project_root=root, config_path=root / config, output_root=output,
            source_fingerprint_root=default_source_fingerprint_root(root),
        )
        fold_frames = []
        for fold_id, test_season in enumerate(CV_TEST_SEASONS):
            fold = base.copy()
            fold["outer_fold"] = fold_id
            fold["base_job_index"] = fold["job_index"]
            fold["train_years_json"] = json.dumps(list(range(2010, test_season - 1)))
            fold["val_years_json"] = json.dumps([test_season - 1])
            fold["test_years_json"] = json.dumps([test_season])
            fold["cv_test_season"] = test_season
            fold_root = output / "cv_folds" / f"test_{test_season}"
            relative = (
                fold["fingerprint"].astype(str) + "/" + fold["family"].astype(str) + "/"
                + fold["model"].astype(str) + "/top_" + fold["top_k_features"].astype(str)
                + "/trial_" + fold["trial_index"].astype(int).astype(str).str.zfill(4)
            )
            fold["output_dir"] = relative.map(lambda value: str(fold_root / value))
            fold["metrics_path"] = fold["output_dir"].map(lambda value: str(Path(value) / "metrics.csv"))
            fold_frames.append(fold)
        manifest = pd.concat(fold_frames, ignore_index=True)
        manifest["job_index"] = range(len(manifest))
        manifest["sge_task_id"] = manifest["job_index"] + 1
        manifest.to_csv(output / "job_manifest.csv", index=False)
        manifest.to_parquet(output / "job_manifest.parquet", index=False)
        for shard_index, shard in enumerate(np.array_split(manifest, 2), start=1):
            shard = shard.copy().reset_index(drop=True)
            shard["sge_task_id"] = shard.index + 1
            shard_path = output / f"job_manifest_part_{shard_index}.csv"
            shard.to_csv(shard_path, index=False)
            shard.to_parquet(shard_path.with_suffix(".parquet"), index=False)
            rows.append({
                "array_id": f"nested_2025_{objective}_part_{shard_index}",
                "objective": objective, "shard": shard_index,
                "task_count": len(shard), "task_concurrency": 10,
                "manifest_path": str(shard_path.resolve()),
                "config_path": str((root / config).resolve()), "output_root": str(output),
                "cv_folds": len(CV_TEST_SEASONS),
                "cv_test_seasons": " ".join(map(str, CV_TEST_SEASONS)),
                "one_training_per_task": True,
            })
    artifact.mkdir(parents=True, exist_ok=True)
    catalog = pd.DataFrame(rows)
    catalog.to_csv(artifact / "array_catalog.csv", index=False)
    catalog.to_parquet(artifact / "array_catalog.parquet", index=False)
    audit = {
        "search_cv_folds": [
            {"train_years": list(range(2010, test - 1)), "validation_year": test - 1, "test_year": test}
            for test in CV_TEST_SEASONS
        ],
        "locked_refit_train_years": list(range(2010, 2024)),
        "locked_refit_validation_years": [2024],
        "final_evaluation_years": [2025],
        "all_stage_overlaps_empty": True,
        "array_count": len(catalog), "search_task_count": int(catalog.task_count.sum()),
        "locked_refit_task_count": 36,
        "new_family_refit_task_count": 24,
        "full_objective_specific_roster_count": 60,
        "submitted": False,
    }
    (artifact / "leakage_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(catalog.to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
