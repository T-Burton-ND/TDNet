#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Generate the Tuesday bundle and X draft after Monday owner approval."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))


def main():
    root = ROOT
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--deadline-local-date", required=True, help="Thursday date in America/New_York.")
    parser.add_argument("--freeze-manifest", type=Path, default=root / "FREEZE_MANIFEST.json")
    parser.add_argument("--schedule-snapshot", type=Path)
    parser.add_argument("--ap-top25", type=Path)
    parser.add_argument("--tdnet-top25", type=Path, help="Optional owner/external TDNet poll override; otherwise the frozen roster poll is generated automatically.")
    parser.add_argument("--canonical-poll-objective", choices=["margin"], help="Objective poll to publish as canonical after review.")
    parser.add_argument("--preseason-rankings", type=Path, help="Preseason-frozen historical-performance ranking sidecar.")
    parser.add_argument("--allow-dirty-code", action="store_true", help="Rehearsal only; forward the dirty-tree exception to bundling.")
    parser.add_argument("--kickoff-times-confirmed", action="store_true", help="Confirm the refreshed schedule has no TBD kickoff times.")
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
        weekly_fingerprint = operations / "fingerprint_ladder_v3" / "canonical_fingerprint.parquet"
        weekly_fingerprint_metadata = operations / "fingerprint_ladder_v3" / "metadata.json"
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
    subprocess.run(command, cwd=root, check=True)
    subprocess.run([
        "python", "src/gridiron_ml/cli/publication/build_x_post_package.py", "--weekly-output", str(output),
        "--season", str(args.season), "--week", str(args.week),
    ], cwd=root, check=True)
    print(output)

if __name__ == "__main__":
    main()
