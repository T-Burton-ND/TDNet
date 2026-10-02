#!/usr/bin/env python3
"""Plan and submit the CPU scientific-model extension for F12-corrected–F17."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nextgen_rounds_scientific_train import (
    CONFIG_PATH,
    DEFAULT_DATA_ROOT,
    load_config,
    run_task,
)
from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file
from nextgen_rounds_train import load_stage_matrix, run as run_screening


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def make_plan(data_root: Path) -> tuple[dict, dict]:
    cfg = load_config()
    cohort_hashes = {}
    feature_counts = {}
    reference_ids = None
    for stage in cfg["stages"]:
        x, meta, _, evidence = load_stage_matrix(data_root, stage)
        eval_ids = tuple(sorted(meta.loc[meta.season.isin([2024, 2025]),
                                        "target_game_id"].astype(int).tolist()))
        if reference_ids is None:
            reference_ids = eval_ids
        elif eval_ids != reference_ids:
            raise ValueError(f"Evaluation game IDs changed for {stage}")
        if len(eval_ids) != 1179 or len(set(eval_ids)) != 1179:
            raise ValueError(f"Unexpected common 2024/2025 cohort for {stage}")
        cohort_hashes[stage] = sha256_bytes(
            ",".join(str(game_id) for game_id in eval_ids).encode("ascii"))
        feature_counts[stage] = {
            "model_features": int(x.shape[1]),
            "source_features": len(evidence["source_features"]),
            "market_features": len(evidence["market_features"]),
            "games_by_season": evidence["games_by_season"],
        }
    completed_screen = pd.read_csv(data_root / "run_results.csv")
    reused = []
    for stage in cfg["stages"]:
        if stage in {"F09", "F10", "F11"}:
            continue
        for model in ("M2", "M4"):
            cell = completed_screen.loc[
                completed_screen.stage.eq(stage) & completed_screen.model.eq(model)]
            if len(cell) != 10 or set(cell.status) != {"success"}:
                raise ValueError(f"Existing M2/M4 cell is not complete: {stage}/{model}")
            reused.append({"stage": stage, "model": model, "successful_setpoints": len(cell)})
    jobs = []
    task_id = 1
    frozen_points = json.loads((ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json").read_text())
    for stage in cfg["stages"][:3]:
        for model in ("M2", "M4"):
            for point in frozen_points[model]:
                jobs.append({"task_id": task_id, "stage": stage, "model": model,
                             "setpoint": point["id"], "run_type": "frozen_setpoint"})
                task_id += 1
    for stage in cfg["stages"]:
        for model_name in cfg["models"]:
            for seed in cfg["seeds"]:
                jobs.append({"task_id": task_id, "stage": stage,
                             "model": model_name, "seed": int(seed),
                             "run_type": "scientific_model"})
                task_id += 1
    manifest = {
        "version": 1,
        "experiment": cfg["experiment"],
        "data_root": str(data_root.resolve()),
        "config_path": str(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "m2_m4_setpoints_sha256": sha256_file(
            ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json"),
        "selection_source": cfg["selection_source"],
        "stages": cfg["stages"],
        "models": list(cfg["models"]),
        "seeds": cfg["seeds"],
        "split": cfg["split"],
        "market_research_only_stage": cfg["market_research_only_stage"],
        "cohort_id_sha256": cohort_hashes,
        "feature_counts": feature_counts,
        "jobs": jobs,
        "reused_completed_m2_m4": reused,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt = {
        "status": "planned",
        "manifest_sha256": sha256_bytes(json.dumps(
            manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "tasks": len(jobs),
        "evaluation_games_per_stage": len(reference_ids or ()),
        "common_evaluation_ids": True,
        "feature_counts": feature_counts,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return manifest, receipt


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temp.replace(path)


def plan(data_root: Path, manifest_path: Path | None = None) -> dict:
    root = data_root / "scientific_model_runs"
    manifest_path = manifest_path or root / "job_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing job manifest: {manifest_path}")
    manifest, receipt = make_plan(data_root)
    atomic_json(manifest_path, manifest)
    receipt["manifest_file_sha256"] = sha256_file(manifest_path)
    atomic_json(manifest_path.with_name("plan_receipt.json"), receipt)
    return {"manifest": str(manifest_path), **receipt}


def run(manifest_path: Path, task_id: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != sha256_file(CONFIG_PATH):
        raise ValueError("Scientific model configuration changed after planning")
    if manifest["m2_m4_setpoints_sha256"] != sha256_file(
            ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json"):
        raise ValueError("M2/M4 setpoints changed after planning")
    jobs = [job for job in manifest["jobs"] if int(job["task_id"]) == task_id]
    if len(jobs) != 1:
        raise ValueError(f"Unknown or duplicate array task: {task_id}")
    job = jobs[0]
    if job["run_type"] == "frozen_setpoint":
        return run_screening(Path(manifest["data_root"]), job["stage"],
                             job["model"], job["setpoint"])
    return run_task(Path(manifest["data_root"]), job["stage"], job["model"],
                    int(job["seed"]), sha256_file(manifest_path))


def submit(manifest_path: Path, *, concurrency: int, repo_root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256_file(manifest_path)
    if manifest["config_sha256"] != sha256_file(CONFIG_PATH):
        raise ValueError("Scientific model configuration changed after planning")
    if manifest["m2_m4_setpoints_sha256"] != sha256_file(
            ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json"):
        raise ValueError("M2/M4 setpoints changed after planning")
    smoke_path = Path(manifest["data_root"]) / "scientific_model_runs" / "smoke_test.json"
    smoke = json.loads(smoke_path.read_text())
    if smoke.get("status") != "passed" or smoke.get("config_sha256") != manifest["config_sha256"]:
        raise ValueError("CPU smoke-test receipt is absent or stale")
    plan_receipt_path = manifest_path.with_name("plan_receipt.json")
    plan_receipt = json.loads(plan_receipt_path.read_text())
    if plan_receipt.get("manifest_file_sha256") != manifest_sha:
        raise ValueError("Plan receipt does not match the job manifest")
    qstat_xml = subprocess.check_output(["qstat", "-xml", "-u", os.environ["USER"]],
                                        text=True)
    active = [entry for entry in ET.fromstring(qstat_xml).iter("job_list")
              if (entry.findtext("JB_name") or "").startswith("tdrounds_sc")]
    if active:
        raise ValueError("A nextgen scientific-model CPU array is already active")
    logs = Path(manifest["data_root"]) / "scientific_model_runs" / "scheduler_logs"
    logs.mkdir(parents=True, exist_ok=True)
    task_count = len(manifest["jobs"])
    command = [
        "qsub", "-terse", "-clear", "-cwd", "-V", "-j", "y", "-q", "long",
        "-N", "tdrounds_sc", "-t", f"1-{task_count}", "-tc", str(concurrency),
        "-pe", "smp", "1", "-l", "h_rt=48:00:00", "-l", "h_vmem=16G",
        "-o", str(logs), "-v",
        f"REPO_ROOT={repo_root.resolve()},ROUND_JOB_MANIFEST={manifest_path.resolve()}",
        str(repo_root / "scripts/sge/nextgen_rounds_scientific_task.sge"),
    ]
    receipt = subprocess.check_output(command, text=True).strip()
    submission = {
        "status": "submitted", "receipt": receipt, "command": command,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_sha,
        "tasks": task_count, "concurrency": concurrency,
        "gpu_request": False, "cpu_slots_per_task": 1,
        "memory_per_task": "16G", "walltime": "48:00:00",
    }
    atomic_json(manifest_path.with_name("submission_receipt.json"), submission)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "submit"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    if args.action == "plan":
        result = plan(args.data_root, args.manifest)
    else:
        if args.manifest is None:
            parser.error("--manifest is required for run and submit")
        if args.action == "run":
            if args.task_id is None:
                parser.error("--task-id is required for run")
            result = run(args.manifest, args.task_id)
        else:
            result = submit(args.manifest, concurrency=args.concurrency,
                            repo_root=ROOT)
    print(json.dumps(result, indent=2, default=str))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
