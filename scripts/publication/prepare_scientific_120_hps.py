#!/usr/bin/env python3
"""Freeze the 120-cell HPS search plan; no model fits are launched here."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

CONFIG = ROOT / "configs/experiments/scientific_120_hps_v1.yaml"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(seed: int, architecture: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{architecture}".encode()).digest()[:8], "big")


def sample_value(spec: dict, rng: np.random.Generator):
    kind = spec.get("type", "categorical")
    if kind == "categorical":
        choices = spec["choices"]
        return choices[int(rng.integers(0, len(choices)))]
    low, high = float(spec["low"]), float(spec["high"])
    value = math.exp(rng.uniform(math.log(low), math.log(high))) if spec.get("log") else rng.uniform(low, high)
    return int(round(value)) if kind == "int" else float(value)


def make_plan() -> tuple[dict, list[dict]]:
    cfg = yaml.safe_load(CONFIG.read_text())
    if (cfg["test_seasons"] != list(range(2021, 2026)) or
            cfg["fingerprints"] != [f"F{i}" for i in range(20)] or
            cfg["architectures"] != ["M1", "M2", "M3", "M4", "M5", "M10"] or
            cfg["candidate_sets_per_cell"] != 100 or
            cfg["max_concurrent_tasks"] > min(10, cfg["hard_max_running_tasks"]) or
            cfg["threads_per_fit"] > cfg["hard_max_cores_per_task"] or
            cfg["hard_max_running_tasks"] > 25 or cfg["hard_max_cores_per_task"] > 2 or
            not cfg["cpu_only"] or
            cfg["save_checkpoints"]):
        raise ValueError("Frozen roster, budget, CPU, or artifact contract changed")
    source = ROOT / cfg["search_space_source"]
    spaces = yaml.safe_load(source.read_text())["parameter_search"]["spaces"]
    excluded = set(cfg["search_space_exclude"])
    candidates = {}
    for architecture in cfg["architectures"]:
        rng = np.random.default_rng(stable_seed(cfg["sampling_seed"], architecture))
        space = {k: v for k, v in spaces[architecture].items() if k not in excluded}
        space.update(cfg.get("search_space_overrides", {}).get(architecture, {}))
        unique = set()
        rows = []
        attempts = 0
        while len(rows) < cfg["candidate_sets_per_cell"] and attempts < 100000:
            attempts += 1
            params = {key: sample_value(spec, rng) for key, spec in space.items()}
            signature = json.dumps(params, sort_keys=True, separators=(",", ":"))
            if signature not in unique:
                unique.add(signature)
                rows.append(params)
        if len(rows) != cfg["candidate_sets_per_cell"]:
            raise ValueError(f"Search space has only {len(rows)} distinct sampled configurations for {architecture}")
        candidates[architecture] = rows
    tasks = []
    for fingerprint in cfg["fingerprints"]:
        for architecture in cfg["architectures"]:
            for trial_id, params in enumerate(candidates[architecture]):
                tasks.append({
                    "task_id": len(tasks) + 1,
                    "fingerprint": fingerprint,
                    "architecture": architecture,
                    "trial_id": trial_id,
                    "params": params,
                    "fit_seed": cfg["fit_seed"],
                    "folds": [{"test_season": y,
                               "fit_seasons": [cfg["earliest_training_season"], y - 2],
                               "inner_validation_season": y - 1}
                              for y in cfg["test_seasons"]],
                })
    summary = {
        "study": cfg["study"], "status": "planned_unrun",
        "configuration_sha256": digest(CONFIG), "search_space_sha256": digest(source),
        "planner_sha256": digest(Path(__file__)),
        "cells": len(cfg["fingerprints"]) * len(cfg["architectures"]),
        "candidate_sets_per_cell": cfg["candidate_sets_per_cell"],
        "candidate_tasks": len(tasks),
        "candidate_fits": len(tasks) * len(cfg["test_seasons"]),
        "incumbent_control_fits_if_added": len(cfg["fingerprints"]) * len(cfg["architectures"]) * len(cfg["test_seasons"]),
        "cpu_only": True, "threads_per_fit": cfg["threads_per_fit"],
        "max_concurrent_tasks": cfg["max_concurrent_tasks"],
        "hard_max_running_tasks": cfg["hard_max_running_tasks"],
        "hard_max_cores_per_task": cfg["hard_max_cores_per_task"],
        "save_checkpoints": False,
        "search_space_pairing": "same 100 architecture setpoints in each of 20 fingerprints",
        "historical_input_audit": "required_before_launch",
        "market_quote_audit": "required_for_F7_F8_F17_F19_interpretation",
    }
    return summary, tasks


def write_plan(output: Path) -> dict:
    summary, tasks = make_plan()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "candidate_tasks.jsonl"
    manifest.write_text("".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks))
    summary["manifest_sha256"] = digest(manifest)
    (output / "plan.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "publication/2026/research/scientific_120_hps_v1")
    args = parser.parse_args()
    print(json.dumps(write_plan(args.output), indent=2))


if __name__ == "__main__":
    main()
