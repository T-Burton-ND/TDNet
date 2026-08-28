#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Run prediction, blog rendering, validation, and immutable weekly bundling."""

from argparse import ArgumentParser
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.publication import (
    build_prediction_bundle,
    build_frozen_roster_poll,
    build_weekly_blog_package,
    prepare_public_prediction_table,
)
from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.publication.output_layout import copy_top25_outputs, require_week_directory
from gridiron_ml.publication.scientific_weekly import build_scientific_weekly_outputs
from gridiron_ml.publication.weekly_protocol import validate_deadline_utc


def main():
    root = ROOT
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--deadline-local-date", required=True, help="Thursday date in America/New_York.")
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--weekly-inventory",
        type=Path,
        help="2026 learned-model weekly/paper inventory; KNN is retained and explicit naive baselines are excluded.",
    )
    parser.add_argument(
        "--scientific-inventory",
        type=Path,
        help="Optional scientific F0-F8 inventory; emits paper-only market-free F0-F6 outputs.",
    )
    parser.add_argument("--schedule-snapshot", type=Path)
    parser.add_argument(
        "--market-lines-snapshot", type=Path,
        help="Optional CFBD /lines snapshot; defaults to data/raw/cfbd/v2/lines/<season>.parquet.",
    )
    parser.add_argument("--top25", type=Path)
    parser.add_argument("--top25-label")
    parser.add_argument("--ap-top25", type=Path, help="Official AP/CFBD snapshot; drives ranked games.")
    parser.add_argument("--tdnet-top25", type=Path, help="Optional owner/external TDNet poll override; otherwise the frozen roster poll is generated automatically.")
    parser.add_argument("--canonical-poll-objective", choices=["margin"], help="Optional margin poll to use as the supplied TDNet Top 25 snapshot.")
    parser.add_argument("--preseason-rankings", type=Path, help="Preseason-frozen historical-performance ranking sidecar.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--kickoff-times-confirmed", action="store_true")
    parser.add_argument("--allow-dirty-code", action="store_true", help="Rehearsal only.")
    args = parser.parse_args()
    deadline = validate_deadline_utc(args.deadline_utc, local_date=args.deadline_local_date)

    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    freeze_root = args.freeze_manifest.parent
    if args.season == 2026:
        inventory = args.weekly_inventory or root / "docs/publication_2026/weekly_learned_model_inventory.csv"
        if not inventory.exists():
            raise FileNotFoundError(f"Missing 2026 learned-model weekly inventory: {inventory}")
        weekly_inventory = pd.read_csv(inventory)
        if weekly_inventory.empty or weekly_inventory["model_family"].astype(str).str.lower().eq("naive").any():
            raise ValueError("2026 weekly inventory must contain learned models only; naive baselines are not ballot members.")
        if not weekly_inventory["model_family"].astype(str).str.lower().eq("knn").any():
            raise ValueError("2026 weekly inventory must retain KNN ballot members.")
        if "market_bearing" in weekly_inventory and weekly_inventory["market_bearing"].astype(str).str.lower().isin({"1", "true", "yes", "y"}).any():
            raise ValueError("2026 weekly inventory may not contain market-bearing models.")
        learned = ~weekly_inventory["model_family"].astype(str).str.lower().eq("ensemble")
        if "feature_config" not in weekly_inventory or not weekly_inventory.loc[learned, "feature_config"].astype(str).eq("F6").all():
            raise ValueError("Every learned 2026 wide-margin model must use corrected F6.")
    else:
        inventory = freeze_root / "final_model_inventory.csv"
    ranking = args.preseason_rankings or freeze_root / "preseason_model_rankings.csv"
    if not ranking.exists():
        ranking = inventory.parent / "preseason_model_rankings.csv"
    if not ranking.exists():
        ranking = args.project_root / "models" / f"season_{args.season}_full_roster" / "preseason_model_rankings.csv"
    if not ranking.exists():
        ranking = None
    schedule = args.schedule_snapshot or args.project_root / f"data/raw/cfbd/v2/games/{args.season}.parquet"
    schedule_hash = sha256_file(schedule)
    if "schedule_snapshot_sha256" in freeze and schedule_hash != freeze["schedule_snapshot_sha256"]:
        raise ValueError("Schedule snapshot does not match the preseason freeze manifest.")
    if args.season == 2026:
        feature_hash = sha256_file(args.project_root / "configs/features/feature_registry.yaml")
        data_snapshot = args.project_root / "data/publication/2026/weekly_operations/snapshot_completeness.json"
        if not data_snapshot.exists():
            raise FileNotFoundError(f"Missing 2026 snapshot completeness report: {data_snapshot}")
        snapshot_report = json.loads(data_snapshot.read_text(encoding="utf-8"))
        if snapshot_report.get("status") != "pass" or snapshot_report.get("certification") != "weekly_snapshot_certified":
            raise RuntimeError("2026 weekly publication requires a certified snapshot completeness report")
        data_hash = sha256_file(data_snapshot)
        environment_hash = sha256_file(args.project_root / "configs/environment.gridiron.linux-64.pip.lock.txt")
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.project_root, text=True).strip()
        freeze_version = "canonical-2026-corrected-f6-v2"
        freeze_manifest_hash = sha256_file(args.freeze_manifest)
    else:
        feature_hash = freeze["feature_manifest_sha256"]
        data_hash = freeze["data_snapshot_sha256"]
        environment_hash = freeze["environment_lock_sha256"]
        git_commit = freeze["git_commit"]
        freeze_version = freeze["freeze_version"]
        freeze_manifest_hash = freeze["manifest_sha256"]
    output = args.output_root or (
        args.project_root / "publication" / str(args.season) / f"week_{args.week:02d}" / "pre_game"
    )
    output = require_week_directory(output, "pre_game")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty pre-game publication package: {output}")

    with tempfile.TemporaryDirectory(prefix=f"tdnet-week-{args.week:02d}-") as temporary:
        generated_polls = {}
        for poll_objective in ("margin",):
            generated_polls[poll_objective] = build_frozen_roster_poll(
                inventory,
                season=args.season,
                week=args.week,
                output_dir=Path(temporary) / "tdnet_model_poll" / poll_objective,
                project_root=args.project_root,
                logo_dir=args.project_root / "data/meta/logos/by_team",
                objective=poll_objective,
            )
        if args.tdnet_top25 is not None:
            tdnet_top25_path = args.tdnet_top25
        elif args.canonical_poll_objective is not None:
            tdnet_top25_path = Path(temporary) / "tdnet_model_poll" / args.canonical_poll_objective / "tdnet_top25.csv"
        else:
            tdnet_top25_path = None
        report = build_weekly_blog_package(
            project_root=args.project_root,
            season=args.season,
            week=args.week,
            model_inventory_path=inventory,
            schedule_snapshot_path=schedule,
            market_lines_path=args.market_lines_snapshot,
            top25_path=args.top25,
            top25_label=args.top25_label,
            ap_top25_path=args.ap_top25,
            tdnet_top25_path=tdnet_top25_path,
            output_root=temporary,
            preseason_ranking_path=ranking,
        )
        if report["manifest"]["collapsed_model_count"]:
            raise RuntimeError("Refusing to publish collapsed/intercept-only weekly models.")
        predictions = prepare_public_prediction_table(
            report["all_model_predictions"],
            prediction_deadline_utc=args.deadline_utc,
            feature_manifest_sha256=feature_hash,
            data_snapshot_sha256=data_hash,
            schedule_snapshot_sha256=schedule_hash,
            git_commit=git_commit,
            pipeline_version=freeze_version,
            environment_lock_sha256=environment_hash,
            kickoff_time_confirmed=args.kickoff_times_confirmed,
        )
        support = {
            str(path.relative_to(temporary)).replace("/", "__"): path
            for path in Path(temporary).rglob("*")
            if path.is_file()
        }
        output.mkdir(parents=True, exist_ok=True)
        for package_dir in ("blog", "figures", "tables", "metadata"):
            shutil.copytree(Path(temporary) / package_dir, output / package_dir)
        copy_top25_outputs(
            Path(temporary) / "tdnet_model_poll" / "margin",
            output,
        )
        bundle = build_prediction_bundle(
            predictions,
            output_root=output / "frozen_bundle",
            project_root=args.project_root,
            supporting_files=support,
            metadata={
                "freeze_version": freeze_version,
                "freeze_manifest_sha256": freeze_manifest_hash,
                "week": args.week,
                "season": args.season,
                "roster_type": "wide_margin_corrected_f6",
                "report_manifest": report["manifest"],
                "tdnet_poll_source": (
                    "external_override" if args.tdnet_top25 is not None
                    else "frozen_objective_poll" if args.canonical_poll_objective is not None
                    else "objective_selection_pending"
                ),
                "canonical_poll_objective": args.canonical_poll_objective,
                "deadline": deadline,
                "tdnet_poll_model_count": (
                    int(generated_polls[args.canonical_poll_objective]["ballots"]["ballot_model"].nunique())
                    if args.canonical_poll_objective is not None else None
                ),
                "tdnet_objective_poll_model_counts": {
                    name: int(value["ballots"]["ballot_model"].nunique())
                    for name, value in generated_polls.items()
                },
            },
            allow_dirty_code=args.allow_dirty_code,
        )
    scientific_paths = {}
    if args.scientific_inventory is not None:
        scientific_paths = build_scientific_weekly_outputs(
            project_root=args.project_root,
            inventory_path=args.scientific_inventory,
            schedule_snapshot_path=schedule,
            market_lines_path=args.market_lines_snapshot,
            reference_poll_path=args.ap_top25,
            output_root=output / "scientific",
            season=args.season,
            week=args.week,
            phase="pre_game",
        )
    print(json.dumps({
        "wide_margin_manifest_sha256": bundle["manifest"]["manifest_sha256"],
        "pre_game_output": str(output),
        "scientific_outputs": {name: str(path) for name, path in scientific_paths.items()},
    }))


if __name__ == "__main__":
    main()
