#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Finalize margin-only 2025 artifacts from completed weekly predictions.

This is intentionally separate from prediction generation: it can rebuild
aggregates and individual scorecards after a memory-bound generation process is
terminated during its final aggregation step.
"""

from argparse import ArgumentParser
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import pandas as pd

from scripts.publication.build_2025_roster_outputs import _aggregate_models, _grade_long, _load_games
from gridiron_ml.publication.poll_recaps import plot_full_season_poll_grid
from gridiron_ml.publication.recaps import (
    _build_one_model_recaps,
    plot_objective_weekly_comparison,
    publish_season_champion,
    weekly_recap_metrics,
)


def main() -> int:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    inventory = pd.read_csv(args.inventory)
    if set(inventory["objective"].astype(str)) != {"margin"}:
        raise ValueError("Finalizer accepts only a margin inventory")
    if inventory["model_id"].astype(str).str.startswith("margin_").any():
        raise ValueError("Finalizer refuses margin_-prefixed model IDs")
    games = _load_games(args.schedule, args.season)
    objective_root = args.output_root / "sunday_recaps" / "margin"
    weekly_root = args.output_root / "weekly_predictions" / "margin"
    summaries = []
    polls = []
    prediction_rows = 0
    for week in range(1, 17):
        prediction_path = weekly_root / f"week_{week:02d}" / "blog_preview" / "tables" / "all_game_model_predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        long = pd.read_csv(prediction_path)
        long["model_id"] = long["model_name"].astype(str)
        long["model_slug"] = long["model_id"]
        graded = _grade_long(long, games)
        prediction_rows += len(graded)
        consensus = _aggregate_models(graded)
        week_root = objective_root / f"week_{week:02d}"
        week_root.mkdir(parents=True, exist_ok=True)
        consensus.to_csv(week_root / "prediction_vs_actual.csv", index=False)
        summaries.append({"season": args.season, "week": week, "objective": "margin", **weekly_recap_metrics(consensus)})
        poll_path = week_root / "tdnet_top25.csv"
        if poll_path.exists():
            polls.append(pd.read_csv(poll_path))
    individual_root = objective_root / "individual_models"
    individual_rows = []
    track_frames = []
    first_prediction_path = weekly_root / "week_01" / "blog_preview" / "tables" / "all_game_model_predictions.csv"
    prediction_model_ids = set(pd.read_csv(first_prediction_path, usecols=["model_name"])["model_name"].astype(str))
    inventory_model_ids = set(inventory["model_id"].astype(str))
    if not prediction_model_ids.issubset(inventory_model_ids):
        raise ValueError(f"Prediction files contain models absent from inventory: {sorted(prediction_model_ids - inventory_model_ids)}")
    model_ids = sorted(prediction_model_ids)
    for model_id in model_ids:
        model_parts = []
        for week in range(1, 17):
            prediction_path = weekly_root / f"week_{week:02d}" / "blog_preview" / "tables" / "all_game_model_predictions.csv"
            long = pd.read_csv(prediction_path)
            long = long.loc[long["model_name"].astype(str).eq(model_id)].copy()
            if long.empty:
                continue
            long["model_id"] = model_id
            long["model_slug"] = model_id
            model_parts.append(_grade_long(long, games))
        if not model_parts:
            raise ValueError(f"No replay predictions found for {model_id}")
        model_games = pd.concat(model_parts, ignore_index=True)
        model_games["model_slug"] = str(model_id)
        individual_rows.append(_build_one_model_recaps((str(model_id), model_games, individual_root, args.season, "margin")))
        metrics_path = individual_root / str(model_id) / "weekly_and_cumulative_metrics.csv"
        metrics = pd.read_csv(metrics_path)
        if "model_id" not in metrics:
            metrics.insert(0, "model_id", model_id)
        else:
            metrics["model_id"] = model_id
        track_frames.append(metrics)
        del model_parts, model_games, metrics
        gc.collect()
    leaderboard = pd.DataFrame(individual_rows).sort_values(
        ["su_accuracy", "ats_accuracy_excluding_pushes", "margin_mae"],
        ascending=[False, False, True], ignore_index=True,
    )
    leaderboard.to_csv(individual_root / "season_model_leaderboard.csv", index=False)
    pd.concat(track_frames, ignore_index=True).to_csv(individual_root / "all_models_running_metrics.csv", index=False)
    publish_season_champion(leaderboard, objective_root, objective="margin", season=args.season)
    poll_frame = pd.concat(polls, ignore_index=True) if polls else pd.DataFrame()
    if not poll_frame.empty:
        poll_frame.to_csv(objective_root / "weekly_poll_top25.csv", index=False)
        plot_full_season_poll_grid(poll_frame, objective_root / "full_season_poll_grid.png", objective="margin", season=args.season, logo_dir=root / "data/meta/logos/by_team")
    summary = pd.DataFrame(summaries)
    summary.to_csv(objective_root / "weekly_summary.csv", index=False)
    summary.to_csv(args.output_root / "objective_weekly_comparison.csv", index=False)
    plot_objective_weekly_comparison(summary, args.output_root / "objective_weekly_comparison.png", season=args.season)
    manifest = {
        "season": args.season,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrospective": True,
        "performance_claim_allowed": True,
        "objective": "margin",
        "training_cutoff": 2024,
        "holdout_season": 2025,
        "model_count": int(len(inventory)),
        "prediction_model_count": int(len(model_ids)),
        "prediction_weeks": list(range(1, 17)),
        "prediction_rows": int(prediction_rows),
        "normalized_model_names": True,
        "inventory": str(args.inventory.resolve()),
        "schedule": str(args.schedule.resolve()),
        "outputs": {
            "weekly_predictions": str(weekly_root.resolve()),
            "sunday_recaps": str(objective_root.resolve()),
            "leaderboard": str((individual_root / "season_model_leaderboard.csv").resolve()),
        },
        "note": "Margin-only 2025 holdout replay; every checkpoint was trained through 2024 and no winner objective was used.",
    }
    (args.output_root / "retrospective_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "README.md").write_text(
        "# TDNet 2025 margin-only holdout replay\n\n"
        "This package contains the margin-only roster replay for 2025. All 39 models were trained through 2024; winner-objective models are archived separately. Model IDs intentionally omit the `margin_` prefix.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
