#!/usr/bin/env python3
"""Prepare or submit one generation's checked screening array (48-job cap).

Only one experiment array can be live; two slots remain for acquisition or
materialization. A plan is cheap and does not authorize source semantics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.experiments.nextgen_contract import parse_fingerprint_id
from gridiron_ml.experiments.nextgen_screening import checked_fingerprint, generation_barrier, run_task
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file


def retry_jobs(root, manifest):
    """Return only failed jobs in terminal architecture cells below three successes."""
    groups = {}
    for job in manifest["jobs"]:
        groups.setdefault((job["fingerprint"], job["model"]), []).append(job)
    selected = []
    decisions = []
    for (fingerprint, model), jobs in groups.items():
        if len(jobs) != 10 or len({j["setpoint"] for j in jobs}) != 10:
            raise ValueError("Retry planning requires the original ten-setpoint matrix")
        results = []
        for job in jobs:
            run_id = f"{fingerprint}__{model}__{job['setpoint']}"
            path = root / "experiments" / manifest["generation"] / run_id / "result.json"
            if not path.exists():
                raise ValueError("An initial result is missing; verify scheduler completion before recovery")
            result = json.loads(path.read_text())
            if result.get("status") not in {"success", "failed"}:
                raise ValueError("Nonterminal screening result")
            results.append((job, result))
        successes = sum(r["status"] == "success" for _, r in results)
        retry = [j for j, r in results if successes < 3 and r["status"] == "failed"
                 and int(r.get("attempt_number", 1)) < 4]
        selected.extend(retry)
        decisions.append({"fingerprint": fingerprint, "model": model, "successes": successes,
                          "retry_count": len(retry), "incomplete_coverage": successes < 10,
                          "proceed": successes >= 3})
    return [{**j, "task_id": i+1} for i, j in enumerate(selected)], decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "retry-plan", "submit", "run"])
    parser.add_argument("--fingerprints", nargs="+")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/experiments/nextgen_fingerprints_v1.json").read_text())
    root = Path(cfg["artifact_root"])
    if args.action == "plan":
        if not args.fingerprints:
            raise ValueError("Fingerprints are required")
        if len(set(args.fingerprints)) != len(args.fingerprints):
            raise ValueError("Duplicate fingerprints")
        generations = {parse_fingerprint_id(f)[0] for f in args.fingerprints}
        if len(generations) != 1:
            raise ValueError("One generation per training array")
        points_path = ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json"
        points = json.loads(points_path.read_text())
        jobs = [{"task_id": i+1, "fingerprint": f, "model": m, "setpoint": p["id"]}
                for i, (f, m, p) in enumerate((f, m, p) for f in args.fingerprints for m in ("M2", "M4") for p in points[m])]
        manifest = {"generation": next(iter(generations)), "jobs": jobs, "maximum_running_array_tasks": 48,
                    "seed": 1701, "setpoints_sha256": sha256_file(points_path), "repo_root": str(ROOT),
                    "created_at_utc": datetime.now(timezone.utc).isoformat()}
        if args.manifest.exists() and json.loads(args.manifest.read_text()).get("jobs") != jobs:
            raise ValueError("Existing manifest has a different task mapping; use a new path")
        atomic_json(args.manifest, manifest)
        print(json.dumps({"manifest": str(args.manifest), "tasks": len(jobs)}))
        return
    manifest = json.loads(args.manifest.read_text())
    if manifest["setpoints_sha256"] != sha256_file(ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json"):
        raise ValueError("Frozen model setpoints changed since array preparation")
    if args.action == "retry-plan":
        if args.output_manifest is None or args.output_manifest.resolve() == args.manifest.resolve():
            raise ValueError("Use a separate output path for the retry manifest")
        jobs, decisions = retry_jobs(root, manifest)
        if args.output_manifest.exists():
            raise ValueError("Retry plan already exists; use a new path")
        if jobs:
            atomic_json(args.output_manifest, {**manifest, "jobs": jobs, "retry_decisions": decisions,
                        "original_matrix_manifest": str(args.manifest.resolve()),
                        "created_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"retry_tasks": len(jobs), "decisions": decisions}, indent=2))
        return
    generation_barrier(root, manifest["generation"])
    if args.action == "run":
        selected = [j for j in manifest["jobs"] if j["task_id"] == args.task_id]
        if len(selected) != 1:
            raise ValueError("Unknown task index")
        job = selected[0]
        result = run_task(root, job["fingerprint"], job["model"], job["setpoint"])
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "success" else 1)
    # Fail before submission if any requested source is unapproved or corrupted.
    for fingerprint in sorted({j["fingerprint"] for j in manifest["jobs"]}):
        checked_fingerprint(root, fingerprint)
    xml = subprocess.check_output(["qstat", "-xml", "-u", os.environ["USER"]], text=True)
    active = [j for j in ET.fromstring(xml).iter("job_list") if (j.findtext("JB_name") or "").startswith("tdng_")]
    if active:
        raise ValueError("Another nextgen scheduler job is live; do not overlap experiment arrays")
    logs = root / "scratch/scheduler_logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = ["qsub", "-terse", "-N", "tdng_"+manifest["generation"], "-t", f"1-{len(manifest['jobs'])}",
           "-tc", "48", "-o", str(logs), "-v", f"REPO_ROOT={ROOT},NEXTGEN_JOB_MANIFEST={args.manifest.resolve()}",
           str(ROOT / "scripts/sge/nextgen_screening_task.sge")]
    receipt = subprocess.check_output(cmd, text=True).strip()
    atomic_json(args.manifest.with_suffix(".submission.json"), {
        "receipt": receipt, "command": cmd, "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(args.manifest)})
    print(receipt)


if __name__ == "__main__":
    main()
