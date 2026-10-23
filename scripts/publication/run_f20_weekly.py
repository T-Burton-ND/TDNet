#!/usr/bin/env python3
"""Collect and issue F20 research forecasts when the weekly pipeline is invoked."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from issue_f20_forecasts import ROOT, issue, sha, write_once


def frozen_target_ids(bundle: Path, *, week: int, deadline_utc: datetime) -> set[int]:
    path = bundle / "public" / "predictions.parquet"
    predictions = pd.read_parquet(path, columns=["season", "week", "game_id",
                                                 "game_start_time_utc", "prediction_deadline_utc"])
    if predictions.empty or not predictions.season.eq(2026).all() or not predictions.week.eq(week).all():
        raise ValueError("Canonical F20 bundle has wrong season or week")
    frozen_cutoffs = pd.to_datetime(predictions.prediction_deadline_utc, utc=True)
    if not frozen_cutoffs.eq(pd.Timestamp(deadline_utc)).all():
        raise ValueError("Canonical F20 cutoff differs from the frozen publication bundle")
    kickoffs = pd.to_datetime(predictions.game_start_time_utc, utc=True)
    if kickoffs.isna().any() or not kickoffs.gt(pd.Timestamp(deadline_utc)).all():
        raise ValueError("Canonical F20 bundle contains a game started by its cutoff")
    return set(predictions.game_id.astype(int))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--phase", choices=("research", "canonical"), default="research")
    parser.add_argument("--deadline-utc", help="Owner-selected cutoff from the frozen weekly bundle")
    parser.add_argument("--bundle", type=Path, help="Frozen publication bundle for exact target game IDs")
    parser.add_argument("--run-label", default="on_demand")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/private/f20/prospective")
    parser.add_argument("--schedule", type=Path, default=ROOT / "data/raw/cfbd/v2/games/2026.parquet")
    parser.add_argument("--models", type=Path, default=ROOT / "data/private/f20/models/f20_scientific_2017_2024_v2_margin")
    args = parser.parse_args()
    if args.phase == "canonical" and (not args.deadline_utc or args.bundle is None):
        parser.error("Canonical phase requires --deadline-utc and --bundle")
    if not 1 <= args.week <= 16:
        raise ValueError("2026 F20 week is outside the regular season")
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
        status["research_issuance"] = issued
        if args.phase == "canonical":
            deadline = datetime.fromisoformat(args.deadline_utc)
            if deadline.tzinfo is None:
                raise ValueError("Canonical deadline must include UTC offset")
            target_ids = frozen_target_ids(args.bundle, week=args.week, deadline_utc=deadline)
            canonical = issue(args.archive, receipt_path, args.models, now=datetime.now(UTC),
                              revision_id="on_demand_f20", week=args.week,
                              official_enabled=True, weekly_cutoff_utc=deadline,
                              target_game_ids=target_ids)
            for game_id in target_ids:
                pointer = json.loads((args.archive / "forecasts" / str(game_id) / "official.json").read_text())
                if pointer.get("canonical_variant") != "F20" or pointer.get("weekly_cutoff_utc") != deadline.isoformat():
                    raise ValueError(f"F20 pointer differs from frozen cutoff: {game_id}")
                record_path = Path(pointer["record_file"])
                record = json.loads(record_path.read_text())
                if (pointer.get("record_sha256") != sha(record_path)
                        or record.get("model_manifest_sha256") != sha(args.models / "manifest.json")
                        or set(record.get("home_win_probability", {})) !=
                        {"M1", "M2", "M3", "M4", "M5", "M10", "F20"}
                        or set(record.get("pred_home_margin", {})) !=
                        {"M1", "M2", "M3", "M4", "M5", "M10", "F20"}):
                    raise ValueError(f"F20 official record is not the frozen six-model panel: {game_id}")
            status["canonical_issuance"] = canonical
            status["official_pointer_count"] = len(target_ids)
            status["status"] = "captured_with_canonical_forecasts"
        else:
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
                      "official_pointer_count": status.get("official_pointer_count", 0),
                      "missing_official_game_ids": status.get("missing_official_game_ids", [])}, indent=2))
    if status["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
