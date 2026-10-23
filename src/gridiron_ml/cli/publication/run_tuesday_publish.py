#!/usr/bin/env python3
"""Generate the Tuesday bundle and X draft after Monday owner approval."""

import hashlib
import json
import subprocess
import sys
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from gridiron_ml.cli._paths import project_root
from gridiron_ml.publication.weekly_rosters import write_weekly_roster_registry

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))
F20_ROOT = Path("/groups/bsavoie2/tburton2/TDNet/f20_2026")


def run_f20_research(root: Path, week: int, operations: Path, *, archive: Path,
                     models: Path) -> dict:
    """Run F20 only when this weekly command is invoked; record failure separately."""
    command = [sys.executable, str(root / "scripts/publication/run_f20_weekly.py"),
               "--week", str(week), "--phase", "research", "--archive", str(archive),
               "--models", str(models)]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {"raw_stdout_tail": completed.stdout[-1000:]}
    receipt = {"season": 2026, "week": week,
               "invoked_at_utc": datetime.now(UTC).isoformat(),
               "status": "succeeded" if completed.returncode == 0 else "failed",
               "exit_code": completed.returncode,
               "runner": "scripts/publication/run_f20_weekly.py",
               "result": result, "stderr_tail": completed.stderr[-1000:]}
    operations.mkdir(parents=True, exist_ok=True)
    path = operations / "f20_research_run.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if completed.returncode or result.get("status") != "captured_with_research_forecasts":
        raise RuntimeError(f"F20 on-demand research run failed; inspect {path}")
    return receipt


def main():
    root = ROOT
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--deadline-local-date", required=True, help="Thursday date in America/New_York.")
    parser.add_argument("--freeze-manifest", type=Path, default=root / "FREEZE_MANIFEST.json")
    parser.add_argument("--schedule-snapshot", type=Path)
    parser.add_argument(
        "--prebuilt-weekly-fingerprint", type=Path,
        help="Use an audited immutable fingerprint state instead of rebuilding the current-week ladder.",
    )
    parser.add_argument(
        "--fingerprint-proof", type=Path,
        help="Required with a prebuilt fingerprint; must bind its SHA-256 and cutoff.",
    )
    parser.add_argument("--ap-top25", type=Path)
    parser.add_argument("--tdnet-top25", type=Path, help="Optional owner/external TDNet poll override; otherwise the frozen roster poll is generated automatically.")
    parser.add_argument("--canonical-poll-objective", choices=["margin"], help="Objective poll to publish as canonical after review.")
    parser.add_argument("--preseason-rankings", type=Path, help="Preseason-frozen historical-performance ranking sidecar.")
    parser.add_argument("--allow-dirty-code", action="store_true", help="Rehearsal only; forward the dirty-tree exception to bundling.")
    parser.add_argument("--kickoff-times-confirmed", action="store_true", help="Confirm the refreshed schedule has no TBD kickoff times.")
    parser.add_argument("--f20-archive", type=Path, default=F20_ROOT / "prospective",
                        help="Owner-only F20 archive for the on-demand research run.")
    parser.add_argument("--f20-models", type=Path,
                        default=F20_ROOT / "models/baseline_2017_2024_v1")
    parser.add_argument(
        "--scientific-inventory",
        type=Path,
        default=Path(
            "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
            "scientific_roster_refits/f0_f8_margin_through_2025_v1/final_model_inventory.csv"
        ),
        help="Frozen 54-cell scientific inventory; its runtime copy uses the refreshed weekly ladder.",
    )
    args = parser.parse_args()
    operations = root / f"data/publication/{args.season}/weekly_operations/week_{args.week:02d}"
    if not (operations / "monday_review.approved").exists():
        raise RuntimeError("Tuesday publication requires monday_review.approved.")
    output = root / f"publication/{args.season}/week_{args.week:02d}/pre_game"
    weekly_inventory = None
    scientific_inventory = None
    if args.season == 2026:
        weekly_fingerprint = (
            args.prebuilt_weekly_fingerprint
            or operations / "fingerprint_ladder_v3" / "canonical_fingerprint.parquet"
        )
        weekly_fingerprint_metadata = operations / "fingerprint_ladder_v3" / "metadata.json"
        if args.prebuilt_weekly_fingerprint is not None:
            if not weekly_fingerprint.is_file():
                raise FileNotFoundError(weekly_fingerprint)
            if args.fingerprint_proof is None:
                raise ValueError("A prebuilt weekly fingerprint requires --fingerprint-proof")
            proof = json.loads(args.fingerprint_proof.read_text())
            content_hash = hashlib.sha256(weekly_fingerprint.read_bytes()).hexdigest()
            games_path = root / f"data/raw/cfbd/v2/games/{args.season}.parquet"
            games_hash = hashlib.sha256(games_path.read_bytes()).hexdigest()
            if (proof.get("status") != "pass"
                or proof.get("cutoff_fingerprint_sha256") != content_hash
                or proof.get("games_sha256") != games_hash
                or int(proof.get("target_week", -1)) != args.week
                or pd.Timestamp(proof.get("cutoff_utc")) != pd.Timestamp(args.deadline_utc)
                or pd.Timestamp(proof.get("latest_source_available_utc")) >= pd.Timestamp(args.deadline_utc)
                or pd.Timestamp(proof.get("generated_at_utc")) > pd.Timestamp(args.deadline_utc)):
                raise ValueError("Prebuilt fingerprint proof does not match the weekly cutoff")
        else:
            subprocess.run([
                "python", "src/gridiron_ml/cli/publication/build_enriched_fingerprint_ladder.py",
                "--source", str(
                    root / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_4/canonical_fingerprint.parquet"
                ),
                "--output", str(weekly_fingerprint),
                "--metadata", str(weekly_fingerprint_metadata),
            ], cwd=root, check=True)
        weekly_inventory = operations / "weekly_learned_model_inventory.csv"
        subprocess.run([
            "python", "src/gridiron_ml/cli/publication/build_weekly_learned_model_inventory.py",
            "--output", str(weekly_inventory),
            "--fingerprint-path", str(weekly_fingerprint),
        ], cwd=root, check=True)
        if args.scientific_inventory is not None:
            scientific = pd.read_csv(args.scientific_inventory)
            if "fingerprint_path" not in scientific:
                raise ValueError("Scientific inventory lacks fingerprint_path.")
            scientific["fingerprint_path"] = str(weekly_fingerprint.resolve())
            scientific_inventory = operations / "scientific_runtime_inventory.csv"
            scientific.to_csv(scientific_inventory, index=False)
    command = [
        "python", "src/gridiron_ml/cli/publication/run_weekly_publication_pipeline.py",
        "--season", str(args.season), "--week", str(args.week),
        "--deadline-utc", args.deadline_utc, "--freeze-manifest", str(args.freeze_manifest),
        "--deadline-local-date", args.deadline_local_date,
        "--output-root", str(output),
    ]
    if weekly_inventory is not None:
        command.extend(["--weekly-inventory", str(weekly_inventory)])
    if scientific_inventory is not None:
        command.extend(["--scientific-inventory", str(scientific_inventory)])
    if args.allow_dirty_code:
        command.append("--allow-dirty-code")
    if args.kickoff_times_confirmed:
        command.append("--kickoff-times-confirmed")
    if args.tdnet_top25:
        command.extend(["--tdnet-top25", str(args.tdnet_top25)])
    if args.canonical_poll_objective:
        command.extend(["--canonical-poll-objective", args.canonical_poll_objective])
    if args.preseason_rankings:
        command.extend(["--preseason-rankings", str(args.preseason_rankings)])
    if args.schedule_snapshot:
        command.extend(["--schedule-snapshot", str(args.schedule_snapshot)])
    if args.ap_top25:
        command.extend(["--ap-top25", str(args.ap_top25), "--top25-label", "AP Top 25"])
    if args.fingerprint_proof:
        command.extend(["--cutoff-proof", str(args.fingerprint_proof)])
    subprocess.run(command, cwd=root, check=True)
    subprocess.run([
        "python", "src/gridiron_ml/cli/publication/build_x_post_package.py", "--weekly-output", str(output),
        "--season", str(args.season), "--week", str(args.week),
    ], cwd=root, check=True)
    if args.season == 2026:
        try:
            research = run_f20_research(root, args.week, operations,
                                        archive=args.f20_archive, models=args.f20_models)
        except RuntimeError:
            receipt = json.loads((operations / "f20_research_run.json").read_text())
            write_weekly_roster_registry(output=output, operations=operations,
                                         season=args.season, week=args.week,
                                         f20_research=receipt)
            raise
        roster_path = write_weekly_roster_registry(output=output, operations=operations,
                                                   season=args.season, week=args.week,
                                                   f20_research=research)
        print(json.dumps({"f20_research": research["result"],
                          "f20_status_file": str(operations / "f20_research_run.json"),
                          "roster_registry": str(roster_path)}))
    print(output)

if __name__ == "__main__":
    main()
