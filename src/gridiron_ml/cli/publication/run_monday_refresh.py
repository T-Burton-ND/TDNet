#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Refresh 2026 data/fingerprints and write a review-gated inspection report."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess

import pandas as pd
import yaml

from gridiron_ml.pipeline.build_full_pipeline import normalize_raw_endpoint_flags
from gridiron_ml.experiments.opponent_adjusted import build_opponent_adjusted_experiment_frames
from gridiron_ml.publication.preseason_states import build_preseason_state_frame
from gridiron_ml.publication.weekly_protocol import build_snapshot_completeness


def main():
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--skip-download", action="store_true", help="Test cached-data rebuild only.")
    args = parser.parse_args()
    if not args.skip_download and not os.environ.get("CFBD_API_KEY"):
        raise EnvironmentError("CFBD_API_KEY is required for the Monday refresh.")
    operations = root / f"data/publication/{args.season}/weekly_operations/week_{args.week:02d}"
    operations.mkdir(parents=True, exist_ok=True)
    approval = operations / "monday_review.approved"
    approval.unlink(missing_ok=True)
    if not args.skip_download:
        subprocess.run(
            ["python", "-m", "gridiron_ml.pipeline.build_full_pipeline", "configs/fetch/weekly_2026_refresh.yaml"],
            cwd=root, check=True,
        )
    adjusted_root = root / "data/experiments/opponent_adjusted_fingerprints"
    artifacts = build_opponent_adjusted_experiment_frames(
        project_root=root, output_root=adjusted_root,
        seasons=tuple(range(2010, args.season + 1)), overwrite=True,
    )
    report = {
        "season": args.season, "week": args.week,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_status": "pending_owner_approval", "checks": {}, "fingerprints": {},
    }
    games_path = root / f"data/raw/cfbd/v2/games/{args.season}.parquet"
    rankings_path = root / f"data/raw/cfbd/v2/rankings/{args.season}.parquet"
    games = pd.read_parquet(games_path)
    report["checks"]["schedule_rows"] = int(len(games))
    report["checks"]["schedule_duplicate_game_ids"] = int(games["id"].duplicated().sum())
    report["checks"]["target_week_games"] = int(pd.to_numeric(games["week"], errors="coerce").eq(args.week).sum())
    report["checks"]["rankings_snapshot_exists"] = rankings_path.exists()
    weekly_config = yaml.safe_load(
        (root / "configs/fetch/weekly_2026_refresh.yaml").read_text(encoding="utf-8")
    ) or {}
    raw_cfg = weekly_config.get("raw_fetch", {})
    raw_cache = root / weekly_config.get("paths", {}).get("raw_cache_dir", "data/raw/cfbd/v2")
    report["checks"]["snapshot_completeness"] = build_snapshot_completeness(
        raw_cache_dir=raw_cache,
        season=args.season,
        endpoints=normalize_raw_endpoint_flags(raw_cfg.get("endpoints")),
        completeness_config=raw_cfg.get("completeness"),
        required_endpoints=raw_cfg.get("required_endpoints"),
    )
    failures = []
    if report["checks"]["schedule_duplicate_game_ids"]:
        failures.append("duplicate_game_ids")
    if not report["checks"]["target_week_games"]:
        failures.append("no_target_week_games")
    if report["checks"]["snapshot_completeness"]["status"] != "pass":
        failures.append("snapshot_completeness")
    for label, artifact in artifacts.items():
        state = build_preseason_state_frame(artifact.frame, season=args.season, project_root=root)
        methods = state["preseason_prior_method"].value_counts().to_dict()
        report["fingerprints"][label] = {
            "rows": int(len(artifact.frame)), "columns": int(artifact.frame.shape[1]),
            "week0_teams": int(state["keys_team"].nunique()), "prior_methods": methods,
            "unfilled_week0_teams": int((~state["preseason_prior_applied"]).sum()),
        }
        if report["fingerprints"][label]["unfilled_week0_teams"]:
            failures.append(f"{label}_unfilled_week0")
    report["automated_status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    (operations / "monday_inspection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (operations / "README.md").write_text(
        f"# {args.season} Week {args.week} Monday review\n\n"
        f"Automated status: **{report['automated_status']}**.\n\n"
        "Inspect `monday_inspection.json` and the refreshed fingerprint diagnostics. "
        "Approve Tuesday generation only after review:\n\n"
        f"```bash\npython src/gridiron_ml/cli/publication/approve_monday_refresh.py --season {args.season} --week {args.week}\n```\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"Monday inspection failed: {failures}")


if __name__ == "__main__":
    main()
