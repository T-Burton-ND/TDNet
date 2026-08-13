#!/usr/bin/env python3
"""Build every publication/recovery/data manifest without submitting SGE jobs."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
from pathlib import Path
import json
import os

import pandas as pd

from gridiron_ml.experiments.publication import build_experiment_manifest


ROOT = project_root()
DEFAULT_ARTIFACT_ROOT = Path(os.environ.get("TDNET_ARTIFACT_ROOT", "publication_artifacts"))
DATA_PATH = ROOT / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet"


def main():
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--local-limit-gb", type=float, default=10.0)
    parser.add_argument("--future-reserve-gb", type=float, default=8.0)
    args = parser.parse_args()
    root = args.project_root.resolve()
    artifact = args.artifact_root.resolve()
    suite = artifact / "suite_manifests"
    suite.mkdir(parents=True, exist_ok=True)

    arrays = []
    generic = [
        ("publication_matrix", "configs/publication/feature_model_matrix.yaml", 10),
        ("hps_spline", "configs/publication/hps_spline.yaml", 10),
        ("hps_hist_gradient_boosted", "configs/publication/hps_hist_gradient_boosted.yaml", 10),
        ("hps_mlp", "configs/publication/hps_mlp.yaml", 10),
        ("hps_structured_mlp", "configs/publication/hps_structured_mlp.yaml", 10),
        ("hps_kernel", "configs/publication/hps_kernel.yaml", 10),
        ("hps_temporal", "configs/publication/hps_temporal.yaml", 10),
    ]
    for array_id, config, tc in generic:
        manifest, _ = build_experiment_manifest(
            project_root=root,
            config_path=config,
            data_path=DATA_PATH,
            output_root=artifact,
        )
        experiment = json.loads((Path(manifest.iloc[0].output_path).parents[1] / "resolved_config.json").read_text())["experiment_name"]
        manifest_path = artifact / "experiments" / experiment / "job_manifest.parquet"
        arrays.append(array_row(array_id, manifest_path, len(manifest), tc, "scripts/sge/experiment_chunk_task.sge", config, 32_000))

    balanced = legacy_missing_manifest(root, artifact, "balanced")
    arrays.append(array_row("legacy_balanced_recovery", balanced, row_count(balanced), 10, "scripts/sge/legacy_hps_recovery_task.sge", "configs/models/tuning/fingerprint_hyperparameter_search_balanced.yaml", 48_000))
    margin = legacy_failed_manifest(root, artifact, "margin")
    arrays.append(array_row("legacy_margin_recovery", margin, row_count(margin), 10, "scripts/sge/legacy_hps_recovery_task.sge", "configs/models/tuning/fingerprint_hyperparameter_search_margin.yaml", 48_000))
    ablation = ablation_recovery_manifest(root, artifact)
    arrays.append(array_row("legacy_ablation_recovery", ablation, row_count(ablation), 10, "scripts/sge/opponent_ablation_shap_task.sge", "", 2_000_000))
    diagnostics = diagnostics_manifest(root, artifact)
    arrays.append(array_row("final_model_diagnostics", diagnostics, row_count(diagnostics), 10, "scripts/sge/final_model_diagnostics_task.sge", "", 100_000_000))
    states = preseason_state_manifest(root, artifact, season=2026)
    arrays.append(array_row("preseason_saved_states", states, row_count(states), 10, "scripts/sge/preseason_state_task.sge", "", 3_000_000))

    catalog = pd.DataFrame(arrays)
    catalog["estimated_total_bytes"] = catalog["task_count"] * catalog["estimated_bytes_per_task"]
    catalog["storage_root"] = str(artifact)
    catalog["one_training_per_task"] = True
    catalog.to_csv(suite / "array_catalog.csv", index=False)
    catalog.to_parquet(suite / "array_catalog.parquet", index=False)
    total = int(catalog["estimated_total_bytes"].sum())
    projected_total = total + int(float(args.future_reserve_gb) * 1024**3)
    report = {
        "array_count": len(catalog),
        "task_count": int(catalog["task_count"].sum()),
        "estimated_total_bytes": total,
        "estimated_total_gb": total / 1024**3,
        "future_finalist_reserve_gb": float(args.future_reserve_gb),
        "projected_program_total_gb": projected_total / 1024**3,
        "local_limit_gb": float(args.local_limit_gb),
        "projected_total_exceeds_local_limit": projected_total > float(args.local_limit_gb) * 1024**3,
        "artifact_root": str(artifact),
        "submitted": False,
    }
    (suite / "suite_estimate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(catalog[["array_id", "task_count", "task_concurrency", "estimated_total_bytes"]].to_string(index=False))
    print(json.dumps(report, indent=2))


def array_row(array_id, manifest, tasks, tc, worker, config, bytes_per_task):
    return {
        "array_id": array_id,
        "manifest_path": str(Path(manifest).resolve()),
        "task_count": int(tasks),
        "task_concurrency": int(tc),
        "worker_script": worker,
        "config_path": str((ROOT / config).resolve()) if config else "",
        "estimated_bytes_per_task": int(bytes_per_task),
    }


def row_count(path):
    path = Path(path)
    return len(pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path))


def rewrite_outputs(frame, output_root):
    out = frame.copy().reset_index(drop=True)
    out["source_job_index"] = out["job_index"]
    out["job_index"] = range(len(out))
    out["sge_task_id"] = out["job_index"] + 1
    for index in out.index:
        directory = Path(output_root) / "runs" / f"task_{index:05d}"
        out.at[index, "output_dir"] = str(directory)
        out.at[index, "metrics_path"] = str(directory / "metrics.csv")
        out.at[index, "status_path"] = str(directory / "status.json")
    return out


def legacy_missing_manifest(root, artifact, objective):
    source = root / f"data/experiments/fingerprint_hyperparameter_search/{objective}"
    manifest = pd.read_csv(source / "job_manifest.csv")
    results = pd.read_parquet(source / "summary/tables/master_hyperparameter_results.parquet")
    done = set(pd.to_numeric(results["job_index"], errors="coerce").dropna().astype(int))
    missing = manifest.loc[~manifest["job_index"].astype(int).isin(done)]
    output = artifact / f"recovery/legacy_{objective}"
    output.mkdir(parents=True, exist_ok=True)
    recovered = rewrite_outputs(missing, output)
    path = output / "job_manifest.csv"
    recovered.to_csv(path, index=False)
    return path


def legacy_failed_manifest(root, artifact, objective):
    source = root / f"data/experiments/fingerprint_hyperparameter_search/{objective}"
    manifest = pd.read_csv(source / "job_manifest.csv")
    results = pd.read_parquet(source / "summary/tables/master_hyperparameter_results.parquet")
    failed_ids = set(pd.to_numeric(results.loc[results["status"].eq("failed"), "job_index"], errors="coerce").dropna().astype(int))
    failed = manifest.loc[manifest["job_index"].astype(int).isin(failed_ids)].copy()
    for index, row in failed.iterrows():
        params = json.loads(row["params_json"])
        if row["model"] == "orthogonal_matching_pursuit":
            top_k = pd.to_numeric(pd.Series([row["top_k_features"]]), errors="coerce").iloc[0]
            params["params.n_nonzero_coefs"] = int(min(params.get("params.n_nonzero_coefs", 10), top_k if pd.notna(top_k) else 50))
        elif row["model"] == "ransac":
            params["params.min_samples"] = 0.9
            params["params.max_trials"] = max(300, int(params.get("params.max_trials", 300)))
            params["params.base_estimator"] = "ridge"
            params["params.estimator_params.alpha"] = 0.1
            params["training.standardize"] = True
        failed.at[index, "params_json"] = json.dumps(params, sort_keys=True)
    output = artifact / f"recovery/legacy_{objective}"
    output.mkdir(parents=True, exist_ok=True)
    recovered = rewrite_outputs(failed, output)
    path = output / "job_manifest.csv"
    recovered.to_csv(path, index=False)
    return path


def ablation_recovery_manifest(root, artifact):
    source = root / "data/experiments/opponent_adjusted_ablation_shap"
    manifest = pd.read_csv(source / "job_manifest.csv")
    metrics = pd.read_csv(source / "summary/tables/master_model_ablation_metrics.csv")
    failed = set(pd.to_numeric(metrics.loc[metrics["status"].eq("failed"), "job_index"], errors="coerce").dropna().astype(int))
    selected = manifest.loc[manifest["job_index"].astype(int).isin(failed)].copy().reset_index(drop=True)
    selected["model_config_path"] = "configs/models/linear/config_ransac_recovery.yaml"
    output = artifact / "recovery/legacy_ablation"
    for index in selected.index:
        directory = output / "runs" / f"task_{index:05d}"
        selected.at[index, "source_job_index"] = selected.at[index, "job_index"]
        selected.at[index, "job_index"] = index
        selected.at[index, "sge_task_id"] = index + 1
        selected.at[index, "output_dir"] = str(directory)
        selected.at[index, "metrics_path"] = str(directory / "metrics.csv")
        selected.at[index, "shap_fragment_path"] = str(directory / "shap.csv")
        selected.at[index, "status_path"] = str(directory / "status.json")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "job_manifest.csv"; selected.to_csv(path, index=False); return path


def diagnostics_manifest(root, artifact):
    source = pd.read_csv(root / "data/experiments/fingerprint_hyperparameter_search/final_model_diagnostics/job_manifest.csv")
    output = artifact / "diagnostics/final_models"
    source["diagnostic_dir"] = [str(output / "runs" / f"task_{i:05d}") for i in range(len(source))]
    source["job_index"] = range(len(source)); source["sge_task_id"] = source["job_index"] + 1
    output.mkdir(parents=True, exist_ok=True)
    path = output / "job_manifest.csv"; source.to_csv(path, index=False); return path


def preseason_state_manifest(root, artifact, season):
    rows = []
    for number in range(1, 8):
        label = f"v1.{number}"; safe = label.replace(".", "_")
        rows.append({
            "job_index": len(rows), "sge_task_id": len(rows) + 1, "season": season,
            "fingerprint": label,
            "source_path": str(root / f"data/experiments/opponent_adjusted_fingerprints/fingerprints/{safe}/canonical_fingerprint.parquet"),
            "output_dir": str(artifact / f"preseason_states/{season}/{safe}"),
        })
    output = artifact / f"preseason_states/{season}"; output.mkdir(parents=True, exist_ok=True)
    path = output / "job_manifest.csv"; pd.DataFrame(rows).to_csv(path, index=False); return path


if __name__ == "__main__":
    main()
