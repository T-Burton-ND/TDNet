#!/usr/bin/env python3
"""Collect and issue F20 research forecasts when the weekly pipeline is invoked."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from issue_f20_forecasts import ROOT, issue, write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--phase", choices=("research",), default="research")
    parser.add_argument("--run-label", default="on_demand")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/private/f20/prospective")
    parser.add_argument("--schedule", type=Path, default=ROOT / "data/raw/cfbd/v2/games/2026.parquet")
    parser.add_argument("--models", type=Path, default=ROOT / "data/private/f20/models/baseline_2017_2024_v1")
    args = parser.parse_args()
    if not 8 <= args.week <= 15:
        raise ValueError("Scheduled 2026 F20 week is outside the remaining regular season")
    run_at = datetime.now(UTC)
    run_id = f"2026_w{args.week:02d}_{args.run_label}_{run_at:%Y%m%dT%H%M%S%fZ}"
    run_path = args.archive / "runs" / f"{run_id}.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    status = {"run_id": run_id, "phase": args.phase, "week": args.week,
              "started_at_utc": run_at.isoformat(), "schedule_status": "not_started",
              "capture_status": "not_started",
              "issuance_status": "not_started"}
    try:
        schedule_path = args.schedule
        schedule_command = [sys.executable, str(ROOT / "scripts/publication/refresh_f20_schedule.py"),
                            "--archive", str(args.archive), "--week", str(args.week)]
        refreshed = subprocess.run(schedule_command, cwd=ROOT, capture_output=True, text=True,
                                   check=False)
        if refreshed.returncode == 0:
            schedule_receipt = json.loads(refreshed.stdout)
            schedule_path = Path(schedule_receipt["schedule_file"])
            status["schedule_status"] = "captured"
            status["schedule_receipt"] = schedule_receipt
        else:
            status["schedule_status"] = "failed_fallback_cached"
            status["schedule_error"] = refreshed.stderr[-1000:]
        command = [sys.executable, str(ROOT / "scripts/publication/capture_f20_experts.py"),
                   "--archive", str(args.archive), "--schedule", str(schedule_path)]
        status["capture_status"] = "attempted"
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        captured = json.loads(result.stdout)
        status["capture_status"] = "succeeded"
        status["source_raw_sha256"] = captured["raw_sha256"]
        status["receipt_file"] = captured["receipt_file"]
        receipt_path = Path(captured["receipt_file"])
        now = datetime.now(UTC)
        issued = issue(args.archive, receipt_path, args.models, now=now,
                       revision_id="on_demand_research", week=args.week,
                       official_enabled=False)
        status["issuance_status"] = "succeeded"
        status["issuance"] = issued
        status["status"] = "captured_with_research_forecasts"
    except Exception as exc:  # noqa: BLE001 - persist every failed run for recovery
        status["status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
    status["finished_at_utc"] = datetime.now(UTC).isoformat()
    write_once(run_path, (json.dumps(status, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"status": status["status"], "run_report": str(run_path),
                      "capture_status": status["capture_status"],
                      "issuance_status": status["issuance_status"],
                      "missing_official_game_ids": status.get("missing_official_game_ids", [])}, indent=2))
    if status["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
