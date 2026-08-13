#!/usr/bin/env python3
"""Build the margin-only canonical F2/F3/F5 HPS gap-fill manifest."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.experiments.hyperparameter_search import sample_setpoints, stable_seed  # noqa: E402


MODEL_LEVELS = {
    "spline_ridge": "M2", "mlp": "M5", "structured_mlp": "M6",
    "gaussian_process": "K3", "nystroem_ridge": "K4", "rbf_kernel_ridge": "K1", "rbf_svr": "K2",
    "decay_ridge": "T1", "trend_elastic_net": "T2", "temporal_random_forest": "T3", "temporal_hist_gradient_boosted": "T4",
}
CONFIGS = {
    "boosted": ("configs/models/boosted/config_hist_gradient_boosted.yaml", "configs/publication/hps_hist_gradient_boosted.yaml"),
    "kernel": (None, "configs/publication/hps_kernel.yaml"),
    "neural": ("configs/models/neural/config_mlp.yaml", "configs/publication/hps_mlp.yaml"),
    "structured_neural": ("configs/models/neural/config_structured_mlp.yaml", "configs/publication/hps_structured_mlp.yaml"),
    "spline": ("configs/models/spline/config_spline_ridge.yaml", "configs/publication/hps_spline.yaml"),
    "temporal": (None, "configs/publication/hps_temporal.yaml"),
}
EXCLUDED_FAMILIES = {"ensemble", "naive", "knn"}
GAP_TIERS = ("F2", "F3", "F5")


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def legacy_space(root: Path, family: str, model: str) -> tuple[dict, int, int]:
    cfg = load(root / "configs/models/tuning/fingerprint_hyperparameter_search_margin.yaml")
    space = dict(cfg.get("families", {}).get(family, {}).get("params", {}))
    space.update(dict(cfg.get("models", {}).get(model, {}).get("params", {})))
    trials = int(cfg.get("families", {}).get(family, {}).get("trials", cfg.get("global", {}).get("trials_per_model", 24)))
    return space, trials, int(cfg.get("global", {}).get("seed", 23003))


def modern_space(root: Path, family: str, model: str) -> tuple[dict, int, int, Path]:
    base, hps_path = CONFIGS[family]
    cfg = load(root / hps_path)
    level = MODEL_LEVELS[model]
    space = dict(cfg.get("parameter_search", {}).get("spaces", {}).get(level, {}))
    trials = int(cfg.get("parameter_search", {}).get("trials_per_level", {}).get(level, 24))
    seed = int(cfg.get("parameter_search", {}).get("seed", 26031))
    config_path = root / (base or cfg["model_configs"][level])
    return space, trials, seed, config_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=ROOT / "models/season_2026_wide_margin_frozen_bundle/final_model_inventory.csv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/publication/canonical_margin_hps_gap_v1")
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get(
        "TDNET_CANONICAL_HPS_ARTIFACT_ROOT",
        str(ROOT / "outputs/publication/canonical_margin_hps_gap_v1/artifacts"),
    )))
    args = parser.parse_args()
    inventory = pd.read_csv(args.inventory)
    inventory = inventory.loc[(inventory.objective.astype(str) == "margin") & ~inventory.model_family.astype(str).isin(EXCLUDED_FAMILIES)].copy()
    if len(inventory) != 28:
        raise ValueError(f"Expected 28 trainable HPS roles after exclusions, found {len(inventory)}")
    rows = []
    for record in inventory.sort_values("model_id").to_dict("records"):
        family, model = str(record["model_family"]), str(record["model_type"])
        if family in {"stat", "linear", "tree"}:
            space, trials, seed = legacy_space(ROOT, family, model)
            config_path = ROOT / f"configs/models/{family}/config_{model}.yaml"
        else:
            space, trials, seed, config_path = modern_space(ROOT, family, model)
        setpoints = sample_setpoints(space, n=trials, seed=stable_seed(seed, model))
        for tier in GAP_TIERS:
            for trial, params in enumerate(setpoints):
                task_id = len(rows) + 1
                out = args.artifact_root / tier / family / model / f"trial_{trial:04d}"
                rows.append({
                    "job_index": task_id - 1, "sge_task_id": task_id,
                    "objective": "margin", "model": model, "family": family,
                    "canonical_feature_config": tier, "fingerprint": tier,
                    "model_config_path": str(config_path.resolve()), "top_k_features": "all",
                    "trial_index": trial, "params_json": json.dumps(params, sort_keys=True),
                    "fingerprint_path": str((ROOT / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet").resolve()),
                    "feature_registry": str((ROOT / "configs/features/feature_registry.yaml").resolve()),
                    "feature_ladders": str((ROOT / "configs/features/feature_ladders.yaml").resolve()),
                    "output_dir": str(out), "metrics_path": str(out / "metrics.csv"),
                    "train_years_json": json.dumps(list(range(2010, 2023))),
                    "val_years_json": json.dumps([2023]), "test_years_json": json.dumps([2024]),
                    "expected_source_model_id": str(record["model_id"]),
                })
    manifest = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_root / "job_manifest.csv", index=False)
    manifest.to_parquet(args.output_root / "job_manifest.parquet", index=False)
    report = {
        "status": "ready_for_smoke_and_review", "task_count": len(manifest),
        "trainable_roles": len(inventory), "gap_tiers": list(GAP_TIERS),
        "fit_seasons": list(range(2010, 2023)), "validation_seasons": [2023], "scoring_seasons": [2024],
        "holdout_2025_excluded": True, "prospective_2026_excluded": True,
        "artifact_root": str(args.artifact_root), "objective": "margin",
    }
    (args.output_root / "manifest_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
