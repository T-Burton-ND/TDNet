from gridiron_ml.cli._paths import project_root
#!/usr/bin/env python3
"""Generate the Tuesday bundle and X draft after Monday owner approval."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

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
    parser.add_argument("--scientific-inventory", type=Path, help="Runtime-ready frozen F0–F8 scientific roster inventory; market-bearing tiers are excluded at publication time.")
    parser.add_argument("--preseason-rankings", type=Path, help="Preseason-frozen historical-performance ranking sidecar.")
    args = parser.parse_args()
    operations = root / f"data/publication/{args.season}/weekly_operations/week_{args.week:02d}"
    if not (operations / "monday_review.approved").exists():
        raise RuntimeError("Tuesday publication requires monday_review.approved.")
    output = root / f"publication/{args.season}/week_{args.week:02d}"
    if args.season == 2026:
        subprocess.run([
            "python", "src/gridiron_ml/cli/publication/build_weekly_learned_model_inventory.py",
        ], cwd=root, check=True)
    command = [
        "python", "src/gridiron_ml/cli/publication/run_weekly_publication_pipeline.py",
        "--season", str(args.season), "--week", str(args.week),
        "--deadline-utc", args.deadline_utc, "--freeze-manifest", str(args.freeze_manifest),
        "--deadline-local-date", args.deadline_local_date,
        "--output-root", str(output),
    ]
    if args.tdnet_top25:
        command.extend(["--tdnet-top25", str(args.tdnet_top25)])
    if args.canonical_poll_objective:
        command.extend(["--canonical-poll-objective", args.canonical_poll_objective])
    if args.preseason_rankings:
        command.extend(["--preseason-rankings", str(args.preseason_rankings)])
    if args.scientific_inventory:
        command.extend(["--scientific-inventory", str(args.scientific_inventory)])
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
