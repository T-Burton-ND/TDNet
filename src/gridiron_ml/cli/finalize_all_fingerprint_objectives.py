#!/usr/bin/env python
"""Finalize every completed fingerprint-search objective.

This is a thin orchestration layer around
``src/gridiron_ml/cli/finalize_fingerprint_hyperparameter_search.py``. It intentionally
skips objectives whose merge table is not present yet, so it is safe to run
before every SGE array has finished.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = project_root()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("data/experiments/fingerprint_hyperparameter_search"),
    )
    parser.add_argument("--objectives", nargs="*", default=["winner", "balanced", "margin"])
    parser.add_argument("--source-fingerprint-root", type=Path, default=None)
    parser.add_argument("--train-years", nargs="*", type=int, default=list(range(2010, 2026)))
    parser.add_argument("--val-years", nargs="*", type=int, default=[])
    parser.add_argument("--eval-years", nargs="*", type=int, default=[2025])
    parser.add_argument("--poll-2025-weeks", nargs="*", type=int, default=list(range(0, 17)))
    parser.add_argument("--poll-2026-weeks", nargs="*", type=int, default=[0])
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--logo-dir", type=Path, default=Path("data/meta/logos/by_team"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-polls", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.project_root.resolve()
    search_root = resolve_path(root, args.search_root)
    finalize_script = root / "scripts" / "finalize_fingerprint_hyperparameter_search.py"
    skipped = []

    for objective in args.objectives:
        output_root = search_root / str(objective)
        results_path = output_root / "summary" / "tables" / "master_hyperparameter_results.parquet"
        if not results_path.exists():
            skipped.append({"objective": objective, "reason": f"missing {results_path}"})
            print(f"SKIP {objective}: merge table is not available yet")
            continue

        command = [
            sys.executable,
            str(finalize_script),
            "--project-root",
            str(root),
            "--output-root",
            str(output_root),
            "--objective",
            str(objective),
            "--train-years",
            *[str(year) for year in args.train_years],
            "--val-years",
            *[str(year) for year in args.val_years],
            "--eval-years",
            *[str(year) for year in args.eval_years],
            "--poll-2025-weeks",
            *[str(week) for week in args.poll_2025_weeks],
            "--poll-2026-weeks",
            *[str(week) for week in args.poll_2026_weeks],
            "--top-n",
            str(args.top_n),
            "--logo-dir",
            str(args.logo_dir),
            "--skip-completed",
            "--poll-from-completed",
        ]
        if args.source_fingerprint_root is not None:
            command.extend(["--source-fingerprint-root", str(resolve_path(root, args.source_fingerprint_root))])
        if args.models:
            command.extend(["--models", *args.models])
        if args.families:
            command.extend(["--families", *args.families])
        if args.skip_train:
            command.append("--skip-train")
        if args.skip_polls:
            command.append("--skip-polls")

        print("RUN", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)

    if skipped:
        skipped_path = search_root / "finalize_skipped_objectives.csv"
        skipped_path.parent.mkdir(parents=True, exist_ok=True)
        skipped_path.write_text(
            "objective,reason\n"
            + "\n".join(f"{row['objective']},{row['reason']}" for row in skipped)
            + "\n"
        )


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
