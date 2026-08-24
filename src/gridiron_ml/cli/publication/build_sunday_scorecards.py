#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Build weekly prediction-versus-result scorecards for a completed season."""

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

from gridiron_ml.publication.recaps import (
    build_season_sunday_recaps,
    build_individual_model_recaps,
    build_prediction_set_recaps,
    plot_objective_weekly_comparison,
    weekly_recap_metrics,
)
from gridiron_ml.publication.poll_recaps import build_season_poll_recaps


def main():
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--prediction-root", type=Path, default=root / "data/experiments/fingerprint_hyperparameter_search")
    parser.add_argument("--games", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--objectives", nargs="+", choices=["winner", "margin", "balanced"], default=["winner", "margin"])
    parser.add_argument("--individual-workers", type=int, default=4)
    parser.add_argument("--skip-individual-models", action="store_true")
    parser.add_argument("--ap-rankings", type=Path, default=root / "data/raw/cfbd/v2/rankings/2025.parquet")
    parser.add_argument(
        "--allow-overlapping-train-eval-layout-dry-run", action="store_true",
        help="Render known in-sample legacy artifacts for layout testing only.",
    )
    args = parser.parse_args()
    games = args.games or root / f"data/raw/cfbd/v2/games/{args.season}.parquet"
    output = args.output_root or root / f"publication/{args.season}/sunday_recaps"
    legacy_root = root / "data/experiments/fingerprint_hyperparameter_search"
    if args.season == 2025 and args.prediction_root.resolve() == legacy_root.resolve() and not args.allow_overlapping_train_eval_layout_dry_run:
        raise RuntimeError(
            "The legacy 2025 final_artifact checkpoints trained on 2010-2025 and evaluated on 2025. "
            "They are in-sample and cannot support performance publication. Rebuild honest holdout "
            "artifacts with train=2010-2023, validation=2024, evaluation=2025, or pass "
            "--allow-overlapping-train-eval-layout-dry-run for explicitly non-publishable layout testing."
        )
    comparisons = []
    season_summaries = []
    for objective in args.objectives:
        result = build_season_sunday_recaps(
            args.prediction_root, games, output / objective, season=args.season, objective=objective
        )
        comparisons.append(result["weekly_summary"])
        season_summaries.append({
            "season": args.season, "objective": objective,
            **weekly_recap_metrics(result["games"]),
        })
        prediction_sets = build_prediction_set_recaps(
            args.prediction_root, games, output / objective,
            season=args.season, objective=objective,
            retrospective_warning=args.allow_overlapping_train_eval_layout_dry_run,
        )
        if not args.skip_individual_models:
            individual = build_individual_model_recaps(
                args.prediction_root, games, output / objective,
                season=args.season, objective=objective, workers=args.individual_workers,
            )
            print(individual[["model_id", "su_accuracy", "ats_accuracy_excluding_pushes", "margin_mae"]].to_string(index=False))
        poll_tables = args.prediction_root / objective / "final_artifacts" / "polls" / f"{args.season}_full_season" / "tables"
        build_season_poll_recaps(
            poll_tables, output / objective, objective=objective,
            logo_dir=root / "data/meta/logos/by_team",
            ap_rankings_path=args.ap_rankings, games_path=games, season=args.season,
        )
        print(result["weekly_summary"].to_string(index=False))
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison.to_csv(output / "objective_weekly_comparison.csv", index=False)
    pd.DataFrame(season_summaries).to_csv(output / "objective_season_summary.csv", index=False)
    plot_objective_weekly_comparison(comparison, output / "objective_weekly_comparison.png", season=args.season)
    (output / "README.md").write_text(
        f"# TDNet {args.season} Sunday recap dry run\n\n"
        "**LAYOUT PREVIEW ONLY.** The legacy checkpoints overlap the 2025 evaluation season, "
        "so these results are in-sample and must not be published as performance evidence.\n\n"
        "Each `winner/week_XX` and `margin/week_XX` directory contains the scorecard, "
        "Top-1, Top-3, and all-model scorecards; a consensus Top 25 with AP comparison; "
        "all per-model ballots; receiving votes; and model-consensus disagreement. The "
        "margin directories also contain native 4:5 and 16:9 Top 10 social graphics. "
        "Week 0 contains poll outputs only.\n\n"
        "Projected scores combine TDNet's predicted margin with the captured closing total. "
        "ATS grading uses the captured closing home-team spread and excludes pushes.\n\n"
        "Regenerate from the repository root with:\n\n"
        "```bash\nPYTHONPATH=src python src/gridiron_ml/cli/publication/build_sunday_scorecards.py --season "
        f"{args.season}\n```\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
