#!/usr/bin/env python3
"""Publish the 2026 preseason package once the official AP Top 25 exists."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import pandas as pd

from gridiron_ml.cli._paths import project_root
from gridiron_ml.cli.publication.run_2026_preseason_rehearsal import materialize_runtime_inventory
from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.publication.polls import load_ap_top25
from gridiron_ml.publication.output_layout import copy_top25_outputs, require_week_directory
from gridiron_ml.publication.roster_poll import build_frozen_roster_poll
from gridiron_ml.publication.scientific_weekly import build_scientific_weekly_outputs
from gridiron_ml.publication.weekly import build_weekly_blog_package


ROOT = project_root()
def _load_ignored_cfbd_key() -> None:
    """Load only CFBD_API_KEY from the ignored repo-local .env when needed."""
    if os.environ.get("CFBD_API_KEY"):
        return
    path = ROOT / ".env"
    if not path.exists():
        raise EnvironmentError("CFBD_API_KEY is unset and the ignored repository .env is missing.")
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "CFBD_API_KEY":
            os.environ["CFBD_API_KEY"] = value.strip().strip("\"'")
            break
    if not os.environ.get("CFBD_API_KEY"):
        raise EnvironmentError("CFBD_API_KEY is not defined in the ignored repository .env.")


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "publication/2026/week_00/pre_game",
    )
    parser.add_argument(
        "--wide-inventory", type=Path,
        default=Path(
            "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
            "corrected_f6_wide_margin_roster/through_2025_v1/final_model_inventory.csv"
        ),
    )
    parser.add_argument(
        "--schedule-snapshot", type=Path,
        default=ROOT / "data/raw/cfbd/v2/games/2026.parquet",
        help="Schedule snapshot to predict; Week 0 releases should pass the date-bounded opening slate.",
    )
    parser.add_argument(
        "--scientific-inventory",
        type=Path,
        default=Path(
            "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
            "scientific_roster_refits/f0_f8_margin_through_2025_v1/final_model_inventory.csv"
        ),
        help="Frozen F0-F8 inventory; weekly output uses market-free F0-F6 cells.",
    )
    parser.add_argument("--skip-scientific", action="store_true")
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.skip_refresh:
        _load_ignored_cfbd_key()
        subprocess.run(
            [
                "python", "-m", "gridiron_ml.pipeline.fetch.cfbd_fetch_v2",
                "--config", "configs/fetch/preseason_2026_watch.yaml", "--refresh",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            check=True,
        )

    rankings_path = ROOT / "data/raw/cfbd/v2/rankings/2026.parquet"
    schedule_path = args.schedule_snapshot.resolve()
    ap = load_ap_top25(rankings_path, season=2026, week=1)
    if len(ap) != 25:
        available = []
        if rankings_path.exists():
            rankings = pd.read_parquet(rankings_path)
            for polls in rankings.get("polls", []):
                values = polls.tolist() if hasattr(polls, "tolist") else polls
                available.extend(str(item.get("poll")) for item in (values or []))
        status = {
            "status": "waiting_for_official_ap_top25",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "available_polls": sorted(set(available)),
        }
        print(json.dumps(status, indent=2, sort_keys=True))
        return 2

    output = args.output_root.resolve()
    output = require_week_directory(output, "pre_game")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Preseason release output already exists: {output}; pass --overwrite explicitly.")
    output.mkdir(parents=True, exist_ok=True)
    input_root = ROOT / "data/publication/2026/week_00/pre_game_inputs"
    ap_path = input_root / "ap_top25.csv"
    ap_path.parent.mkdir(parents=True, exist_ok=True)
    ap.to_csv(ap_path, index=False)

    source = args.wide_inventory.resolve()
    runtime = materialize_runtime_inventory(source, input_root / "margin_wide_f6_runtime_inventory.csv")
    poll_staging = input_root / "generated_top25"
    poll_result = build_frozen_roster_poll(
        runtime,
        season=2026,
        week=0,
        output_dir=poll_staging,
        project_root=ROOT,
        logo_dir=ROOT / "data/meta/logos/by_team",
        objective="margin",
        reference_poll=ap,
        reference_label="AP",
    )
    copy_top25_outputs(poll_staging, output)
    weekly = build_weekly_blog_package(
        project_root=ROOT,
        season=2026,
        week=1,
        model_inventory_path=runtime,
        schedule_snapshot_path=schedule_path,
        output_root=output,
        top25_path=ap_path,
        top25_label="AP Top 25",
        ap_top25_path=ap_path,
        tdnet_top25_path=output / "tables/tdnet_top25.csv",
        logo_dir=ROOT / "data/meta/logos/by_team",
        include_collapsed_models=False,
        schedule_driven_matchups=True,
    )
    scientific_paths = {}
    if not args.skip_scientific:
        scientific_paths = build_scientific_weekly_outputs(
            project_root=ROOT,
            inventory_path=args.scientific_inventory,
            schedule_snapshot_path=schedule_path,
            market_lines_path=ROOT / "data/raw/cfbd/v2/lines/2026.parquet",
            reference_poll_path=ap_path,
            output_root=output / "scientific",
            season=2026,
            week=0,
            poll_week=0,
            prediction_week=1,
            phase="pre_game",
        )
    reports = {
        "margin_wide_f6": {
            "inventory": str(source),
            "runtime_inventory_sha256": sha256_file(runtime),
            "poll_model_count": int(poll_result["ballots"]["ballot_model"].nunique()),
            "poll_failures": int(len(poll_result["failures"])),
            "prediction_model_count": int(weekly["all_model_predictions"]["model_name"].nunique()),
            "predicted_game_count": int(len(weekly["all_games"])),
            "prediction_failures": int(weekly["manifest"]["model_failure_count"]),
        }
    }

    talent_path = ROOT / "data/raw/cfbd/v2/talent/2026.parquet"
    returning_path = ROOT / "data/raw/cfbd/v2/returning/2026.parquet"
    live_talent_rows = len(pd.read_parquet(talent_path)) if talent_path.exists() else 0
    live_returning_rows = len(pd.read_parquet(returning_path)) if returning_path.exists() else 0
    talent_policy = (
        f"live 2026 CFBD talent ({live_talent_rows} rows)"
        if live_talent_rows
        else "2025 carry-forward because 2026 CFBD talent is unavailable"
    )
    returning_policy = (
        f"live 2026 CFBD returning production ({live_returning_rows} rows)"
        if live_returning_rows
        else "unavailable from CFBD at generation time"
    )
    manifest = {
        "status": "ready_for_owner_review_and_publication",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": 2026,
        "ap_poll_rows": 25,
        "ap_poll_sha256": sha256_file(ap_path),
        "schedule_sha256": sha256_file(schedule_path),
        "weekly_roster": "corrected-F6 wide-margin only",
        "scientific_roster_policy": "paper-only weekly package under pre_game/scientific; market-free F0-F6 cells only",
        "wide_margin_fingerprint": "F6",
        "preseason_data_policy": {
            "talent": talent_policy,
            "returning_production": returning_policy,
            "raw_cfbd_cache_modified_by_fallback": False,
        },
        "reports": reports,
        "scientific_outputs": {name: str(path) for name, path in scientific_paths.items()},
    }
    metadata_root = output / "metadata"
    blog_root = output / "blog"
    metadata_root.mkdir(parents=True, exist_ok=True)
    blog_root.mkdir(parents=True, exist_ok=True)
    (metadata_root / "PRESEASON_RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (blog_root / "PRESEASON_BLOG_DRAFT.md").write_text(
        "# TDNet 2026 preseason poll and Week 0 outlook\n\n"
        "TDNet's preseason rankings are produced from models trained only through 2025. "
        "The official AP Top 25 is used as an external comparison, never as a TDNet model input.\n\n"
        "The operational wide-margin roster uses corrected F6, the strongest market-free "
        "fingerprint under the prespecified margin-MAE criterion. The frozen scientific panel "
        "is reserved for postseason comparison and is not regenerated in weekly publication.\n\n"
        "The generated Top 25 table, consensus-spread graphic, descriptive AP-peer fingerprint graphic, "
        "and full ballot matrix appear in that order beside the poll. The fingerprint graphic is "
        "a descriptive comparison aid, not a causal explanation of AP voters or unencoded coaching changes.\n\n"
        "The generated local comparison tables, ballot figures, Week 0 matchups, and model-failure "
        "reports should be reviewed before public posting. Only the reviewed figures and compact "
        "provenance are committed to the public repository.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
