#!/usr/bin/env python3
"""Accept the five CPU smoke candidates only after every fold and binding passes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RESULTS = Path("/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1/results")
STUDY = RESULTS.parent
TASK_IDS = (1, 3701, 5701, 11001, 11801)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def accept() -> dict:
    preflight = json.loads((STUDY / "input_preflight.json").read_text())
    if preflight["status"] != "ready_for_cpu_smoke" or len(preflight["tiers"]) != 20:
        raise ValueError("Twenty-tier source preflight is not complete")
    snapshot = STUDY / "code_snapshot"
    manifest = snapshot / "publication/2026/research/scientific_120_hps_v1/candidate_tasks.jsonl"
    plan = json.loads((snapshot / "publication/2026/research/scientific_120_hps_v1/plan.json").read_text())
    if sha256(manifest) != plan["manifest_sha256"]:
        raise ValueError("Frozen smoke manifest differs from plan")
    worker = snapshot / "scripts/publication/run_scientific_120_hps_task.py"
    results = []
    for task_id in TASK_IDS:
        path = RESULTS / f"task_{task_id:05d}.json"
        item = json.loads(path.read_text())
        if (item["status"] != "success" or item["task_id"] != task_id or
                item["cpu_slots"] > 2 or item["gpu_visible"] not in ("", "-1") or
                item["checkpoints_written"] != 0 or len(item["fold_metrics"]) != 5 or
                item["binding"]["manifest_sha256"] != plan["manifest_sha256"] or
                item["binding"]["worker_sha256"] != sha256(worker)):
            raise ValueError(f"CPU smoke task {task_id} failed its acceptance contract")
        if sorted(row["test_season"] for row in item["fold_metrics"]) != list(range(2021, 2026)):
            raise ValueError(f"Smoke task {task_id} is missing a historical fold")
        if any("ats_accuracy" not in row or "brier" not in row for row in item["fold_metrics"]):
            raise ValueError(f"Smoke task {task_id} is missing a secondary metric")
        if item["binding"]["task"]["architecture"] == "M5":
            history = RESULTS / f"task_{task_id:05d}_history.csv"
            if not history.exists() or sha256(history) != item["history_sha256"]:
                raise ValueError("M5 smoke loss history is missing or changed")
        results.append({"task_id": task_id, "result_sha256": sha256(path),
                        "runtime_seconds": item["runtime_seconds"]})
    receipt = {"status": "accepted", "source_commit": (snapshot / "source_commit.txt").read_text().strip(),
               "manifest_sha256": plan["manifest_sha256"], "smoke_tasks": results,
               "input_preflight_sha256": sha256(STUDY / "input_preflight.json")}
    tmp = RESULTS / "smoke_passed.tmp.json"
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(RESULTS / "smoke_passed.json")
    return receipt


if __name__ == "__main__":
    print(json.dumps(accept(), indent=2))
