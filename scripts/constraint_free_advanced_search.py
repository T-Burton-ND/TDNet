#!/usr/bin/env python3
"""Matched hierarchical genetic and Gaussian-process representation searches.

Each task is one tier/architecture. Candidate evaluations use 2013–2025 only;
the search does not import, read, or score a 2026 target outcome.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from constraint_free_search import (
    DEFAULT_OUTPUT, SETPOINTS, digest, fit_predict, load_historical,
    make_architecture, write_json,
)
from gridiron_ml.experiments.constraint_free_advanced import (
    CORRELATIONS, FAMILIES, MODES, SELECT_COUNTS, VARIANCES,
    GenomeRepresentation, encode_genome, genome_hash, normalize_genome,
    random_genome,
)
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import build_estimator

CONFIG = ROOT / "configs/experiments/constraint_free_advanced_v1.json"
CORE = ROOT / "src/gridiron_ml/experiments/constraint_free_advanced.py"


def adjusted_model(architecture: str, hp_index: int):
    # Preserve architecture identity while varying one compatible setting.
    base_cfg = json.loads((ROOT / "configs/experiments/constraint_free_v1.json").read_text())
    if architecture in ("M2", "M4"):
        choices = {"M2": ("m2_03", "m2_05", "m2_08"),
                   "M4": ("m4_03", "m4_04", "m4_09")}
        points = json.loads(SETPOINTS.read_text())
        point = next(p for p in points[architecture] if p["id"] == choices[architecture][hp_index])
        return build_estimator(architecture, point), "sklearn", point["id"]
    model, backend = make_architecture(architecture, base_cfg)
    if architecture == "M1":
        model.params["alpha"] *= (0.25, 1.0, 4.0)[hp_index]
    elif architecture == "M3":
        model.params["max_features"] = (0.2, 0.35, 0.5)[hp_index]
    elif architecture == "M5":
        model.learning_rate *= (0.5, 1.0, 2.0)[hp_index]
    elif architecture == "M10":
        model.params["n_neighbors"] = (10, 20, 40)[hp_index]
    else:
        raise ValueError("Architecture outside the scientific roster")
    return model, backend, hp_index


def evaluate_genome(tier: str, architecture: str, genome: dict,
                    X: np.ndarray, meta, records, years_to_score: list[int]) -> dict:
    genome = normalize_genome(genome)
    years = meta.season.to_numpy(int)
    y = meta.next_game_margin.to_numpy(float)
    spread_index = next((i for i, record in enumerate(records)
                         if record.name == "market_home_spread"), None)
    folds = []
    for year in years_to_score:
        train, test = years < year, years == year
        if train.sum() < 1000 or test.sum() < 100:
            raise ValueError(f"Invalid rolling-origin fold {year}")
        transform = GenomeRepresentation(genome, records, tier)
        fit = transform.fit_transform(X[train], y[train])
        future = transform.transform(X[test])
        if not np.isfinite(fit).all() or not np.isfinite(future).all():
            raise ValueError("Genome produced nonfinite representation")
        model, backend, hp = adjusted_model(architecture, genome["hp_index"])
        if genome["target_mode"] == "residual":
            if tier == "F19":
                if spread_index is None:
                    raise ValueError("F19 residual target requires a market spread")
                baseline_train = -X[train, spread_index]
                baseline_test = -X[test, spread_index]
                if not np.isfinite(baseline_train).all() or not np.isfinite(baseline_test).all():
                    raise ValueError("F19 residual target lacks a historical market spread")
            else:
                baseline_train = np.full(train.sum(), y[train].mean())
                baseline_test = np.full(test.sum(), y[train].mean())
            prediction = baseline_test + fit_predict(model, backend, fit,
                                                     y[train] - baseline_train, future)
        else:
            prediction = fit_predict(model, backend, fit, y[train], future)
        actual = y[test]
        probability = expit(prediction / max(float(np.std(y[train])), 3.0))
        folds.append({
            "year": year, "games": int(test.sum()),
            "mae": float(np.abs(actual - prediction).mean()),
            "rmse": float(np.sqrt(np.square(actual - prediction).mean())),
            "brier": float(np.square(probability - (actual > 0)).mean()),
            "winner_accuracy": float(((prediction > 0) == (actual > 0)).mean()),
            "features": int(fit.shape[1]), "hp_choice": hp,
            "target_mode": genome["target_mode"],
        })
    return {
        "folds": folds, "mean_mae": float(np.mean([f["mae"] for f in folds])),
        "mean_brier": float(np.mean([f["brier"] for f in folds])),
        "worst_year_mae": float(max(f["mae"] for f in folds)),
        "model_fits": len(folds), "preprocessing_fits": len(folds),
    }


def score_key(result: dict):
    if result["status"] != "success":
        return (float("inf"), float("inf"), float("inf"))
    return (result["mean_mae"], result["mean_brier"], result["worst_year_mae"])


def candidate(tier: str, architecture: str, genome: dict, *, trial: int,
              stage: str, out: Path, X, meta, records, years, parents=None,
              generation=None, acquisition=None, mutations=None) -> dict:
    genome = normalize_genome(genome)
    identity = genome_hash(genome)
    path = out / "candidates" / f"{identity}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("genome") != genome or saved.get("tier") != tier or saved.get("architecture") != architecture:
            raise ValueError("Candidate cache identity collision")
        return saved
    result = {
        "status": "running", "tier": tier, "architecture": architecture,
        "track": stage, "trial": trial, "generation": generation,
        "parents": parents or [], "mutations": mutations or [], "acquisition": acquisition,
        "genome": genome, "genome_hash": identity,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    started = time.monotonic()
    try:
        result.update(evaluate_genome(tier, architecture, genome, X, meta, records, years))
        result["status"] = "success"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=12))
    result["runtime_seconds"] = time.monotonic() - started
    write_json(path, result)
    return result


def crossover(a: dict, b: dict, rng: np.random.Generator) -> tuple[dict, list[str]]:
    child = {key: (a[key] if rng.integers(2) else b[key]) for key in a}
    domains = {
        "mode": MODES, "variance": VARIANCES, "select_count": SELECT_COUNTS,
        "correlation": CORRELATIONS, "nonlinear": (False, True),
        "family_mask": range(1, 2 ** len(FAMILIES)),
        "market_on": (False, True), "robust": (False, True), "hp_index": (0, 1, 2),
        "target_mode": ("direct", "residual"),
    }
    mutations = []
    for key, domain in domains.items():
        if rng.random() < 0.20:
            value = rng.choice(domain)
            child[key] = value.item() if isinstance(value, np.generic) else value
            mutations.append(key)
    return normalize_genome(child), mutations


def unique_random(rng, used: set[str]) -> dict:
    for _ in range(1000):
        genome = random_genome(rng)
        if genome_hash(genome) not in used:
            return genome
    raise ValueError("Unable to generate a novel genome")


def run_ga(tier, architecture, out, X, meta, records, cfg):
    rng = np.random.default_rng(int(cfg["seed"]) + list(cfg["architectures"]).index(architecture))
    population = []
    initial = set()
    for _ in range(int(cfg["ga_population"])):
        genome = unique_random(rng, initial)
        population.append((genome, [], []))
        initial.add(genome_hash(genome))
    all_results, used = [], set()
    for generation in range(int(cfg["ga_generations"])):
        for genome, parent_ids, mutations in population:
            identity = genome_hash(genome)
            if identity in used:
                raise ValueError("GA repeated a candidate within its fixed budget")
            result = candidate(tier, architecture, genome, trial=len(all_results) + 1,
                               stage="ga", out=out, X=X, meta=meta, records=records,
                               years=cfg["development_years"], generation=generation,
                               parents=parent_ids, mutations=mutations)
            all_results.append(result)
            used.add(identity)
        if generation == int(cfg["ga_generations"]) - 1:
            break
        ranked = sorted(all_results, key=score_key)
        parents = ranked[:max(4, int(cfg["ga_elites"]))]
        population = []
        next_ids = set(used)
        while len(population) < int(cfg["ga_population"]):
            left, right = rng.choice(len(parents), size=2, replace=False)
            genome, mutations = crossover(parents[left]["genome"], parents[right]["genome"], rng)
            parent_ids = [parents[left]["genome_hash"], parents[right]["genome_hash"]]
            if genome_hash(genome) in next_ids:
                genome = unique_random(rng, next_ids)
                mutations.append("novelty_replacement")
            population.append((genome, parent_ids, mutations))
            next_ids.add(genome_hash(genome))
    return all_results


def run_bo(tier, architecture, out, X, meta, records, cfg):
    rng = np.random.default_rng(int(cfg["seed"]) + list(cfg["architectures"]).index(architecture))
    all_results, used = [], set()
    for trial in range(int(cfg["bo_trials"])):
        if trial < int(cfg["bo_initial_random"]):
            genome = unique_random(rng, used)
            acquisition = "random_initialization"
        else:
            successful = [r for r in all_results if r["status"] == "success"]
            if len(successful) < 3:
                genome = unique_random(rng, used)
                acquisition = "random_after_failures"
            else:
                observations = np.vstack([encode_genome(r["genome"]) for r in successful])
                losses = np.array([r["mean_mae"] for r in successful])
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-4)
                surrogate = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                     random_state=int(cfg["seed"]), n_restarts_optimizer=1)
                surrogate.fit(observations, losses)
                pool = []
                pool_ids = set(used)
                for _ in range(int(cfg["bo_acquisition_pool"])):
                    item = unique_random(rng, pool_ids)
                    pool.append(item)
                    pool_ids.add(genome_hash(item))
                mean, std = surrogate.predict(np.vstack([encode_genome(g) for g in pool]), return_std=True)
                safe_std = np.maximum(std, 1e-9)
                z = (losses.min() - mean) / safe_std
                expected_improvement = (losses.min() - mean) * norm.cdf(z) + safe_std * norm.pdf(z)
                chosen = int(np.argmax(expected_improvement))
                genome = pool[chosen]
                acquisition = {"policy": "expected_improvement", "value": float(expected_improvement[chosen])}
        result = candidate(tier, architecture, genome, trial=trial + 1,
                           stage="bo", out=out, X=X, meta=meta, records=records,
                           years=cfg["development_years"], acquisition=acquisition)
        all_results.append(result)
        used.add(genome_hash(genome))
    return all_results


def plan(output: Path, track: str) -> dict:
    cfg = json.loads(CONFIG.read_text())
    if cfg["sge_task_concurrency_max"] > 10:
        raise ValueError("TC may never exceed 10")
    tasks = [{"tier": tier, "architecture": architecture}
             for tier in cfg["tiers"] for architecture in cfg["architectures"]]
    for tier in cfg["tiers"]:
        load_historical(tier)
    manifest = {
        "track": track, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(CONFIG), "source_sha256": digest(Path(__file__)),
        "advanced_source_sha256": digest(CORE),
        "base_search_sha256": digest(ROOT / "scripts/constraint_free_search.py"),
        "tasks": tasks, "development_years": cfg["development_years"],
        "candidate_budget_per_task": (cfg["ga_population"] * cfg["ga_generations"]
                                      if track == "ga" else cfg["bo_trials"]),
        "task_concurrency_max": cfg["sge_task_concurrency_max"],
        "historical_market_quote_timing": "unverified; development only",
    }
    path = output / track / "manifest.json"
    if path.exists():
        raise FileExistsError(f"Immutable track manifest exists: {path}")
    write_json(path, manifest)
    return {"manifest": str(path), "sha256": digest(path), "tasks": len(tasks),
            "candidate_budget_per_tier": len(cfg["architectures"]) * manifest["candidate_budget_per_task"]}


def run(manifest_path: Path, task_id: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    for path, expected in ((CONFIG, "config_sha256"), (Path(__file__), "source_sha256"),
                           (CORE, "advanced_source_sha256"),
                           (ROOT / "scripts/constraint_free_search.py", "base_search_sha256")):
        if digest(path) != manifest[expected]:
            raise ValueError(f"Search binding changed after planning: {path}")
    task = manifest["tasks"][task_id - 1]
    out = manifest_path.parent / f"{task['tier']}_{task['architecture']}"
    summary = out / "summary.json"
    if summary.exists():
        prior = json.loads(summary.read_text())
        if prior.get("status") == "success" and prior.get("manifest_sha256") == digest(manifest_path):
            return prior
        raise FileExistsError(f"Nonreusable task summary exists: {summary}")
    X, meta, records, _ = load_historical(task["tier"])
    cfg = json.loads(CONFIG.read_text())
    started = time.monotonic()
    search = run_ga if manifest["track"] == "ga" else run_bo
    candidates = search(task["tier"], task["architecture"], out, X, meta, records, cfg)
    best = min(candidates, key=score_key)
    status = "success" if all(c["status"] == "success" for c in candidates) else "partial"
    result = {
        **task, "track": manifest["track"], "status": status,
        "manifest_sha256": digest(manifest_path),
        "candidates": len(candidates),
        "successful_candidates": sum(c["status"] == "success" for c in candidates),
        "model_fits": sum(c.get("model_fits", 0) for c in candidates),
        "best_genome": best["genome"], "best_genome_hash": best["genome_hash"],
        "best_mean_mae": best.get("mean_mae"),
        "best_mean_brier": best.get("mean_brier"),
        "runtime_seconds": time.monotonic() - started,
        "historical_market_quote_timing": "unverified" if task["tier"] == "F19" else "none",
    }
    write_json(summary, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "run"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--track", choices=("ga", "bo"), default="ga")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--task-id", type=int)
    args = p.parse_args()
    if args.action == "plan":
        result = plan(args.output, args.track)
    else:
        if not args.manifest or not args.task_id:
            p.error("run requires --manifest and --task-id")
        result = run(args.manifest, args.task_id)
    print(json.dumps(result, indent=2, default=str), flush=True)
    if result.get("status") not in (None, "success") and args.action == "run":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
