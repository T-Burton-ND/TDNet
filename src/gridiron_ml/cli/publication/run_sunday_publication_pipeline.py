#!/usr/bin/env python3
"""Build a reviewable Sunday retrospective publication bundle.

This command scores an already immutable Thursday prediction bundle after the
source completeness report passes. It writes metrics, cumulative summaries,
baseline/Vegas comparison tables, a compact figure, and draft-only copy into a
new output directory. It never edits the prediction bundle and never sends a
public post.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridiron_ml.cli._paths import project_root

ROOT = project_root()
EASTERN = ZoneInfo("America/New_York")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.publication.bundles import (
    score_prediction_bundle,
    sha256_file,
    verify_prediction_bundle,
)
from gridiron_ml.publication.chart_contracts import validate_chart_domains
from gridiron_ml.publication.figure_theme import (
    MODEL_COLORS,
    TDNET_COLORS,
    apply_tdnet_theme,
)
from gridiron_ml.publication.output_layout import require_week_directory
from gridiron_ml.publication.postgame_full import (
    build_full_postgame_package,
)
from gridiron_ml.publication.scientific_postgame import (
    write_scientific_postgame_evaluation,
)
from gridiron_ml.publication.scientific_weekly import (
    build_scientific_weekly_outputs,
)


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    probability = pd.to_numeric(frame.get("pred_home_win_probability"), errors="coerce")
    actual_win = pd.to_numeric(frame.get("actual_home_win"), errors="coerce")
    margin_error = pd.to_numeric(frame.get("absolute_margin_error"), errors="coerce")
    valid_probability = probability.notna() & actual_win.notna()
    valid_margin = margin_error.notna()
    output: dict[str, float | int] = {"games": int(frame["game_id"].nunique())}
    output["winner_accuracy"] = (
        float(frame.loc[valid_probability, "winner_correct"].mean())
        if valid_probability.any()
        else float("nan")
    )
    output["margin_mae"] = (
        float(margin_error.loc[valid_margin].mean())
        if valid_margin.any()
        else float("nan")
    )
    if valid_probability.any():
        clipped = probability.loc[valid_probability].clip(1e-8, 1 - 1e-8)
        y = actual_win.loc[valid_probability]
        output["brier_score"] = float(((clipped - y) ** 2).mean())
        output["log_loss"] = float(
            -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()
        )
    else:
        output["brier_score"] = float("nan")
        output["log_loss"] = float("nan")
    return output


def _weekly_and_cumulative(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = scored.copy()
    if "week" not in frame:
        frame["week"] = 0
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").fillna(0).astype(int)
    weekly_rows = []
    cumulative_rows = []
    for (model, objective), group in frame.groupby(
        ["model_name", "objective"], dropna=False
    ):
        prior = []
        for week, week_frame in group.groupby("week", sort=True):
            prior.append(week_frame)
            weekly_rows.append(
                {
                    "model_name": model,
                    "objective": objective,
                    "week": int(week),
                    **_metrics(week_frame),
                }
            )
            cumulative_rows.append(
                {
                    "model_name": model,
                    "objective": objective,
                    "week": int(week),
                    **_metrics(pd.concat(prior, ignore_index=True)),
                }
            )
    return pd.DataFrame(weekly_rows), pd.DataFrame(cumulative_rows)


def _figure(cumulative: pd.DataFrame, path: Path) -> None:
    apply_tdnet_theme()
    if cumulative.empty:
        raise ValueError("No cumulative metrics are available for the Sunday figure.")
    ranked = (
        cumulative.groupby("model_name", as_index=False)["margin_mae"]
        .last()
        .sort_values("margin_mae")
        .head(12)["model_name"]
    )
    plot = cumulative[cumulative["model_name"].isin(ranked)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    max_mae = (
        float(pd.to_numeric(plot["cumulative_margin_mae"], errors="coerce").max())
        if "cumulative_margin_mae" in plot
        else float(pd.to_numeric(plot["margin_mae"], errors="coerce").max())
    )
    validate_chart_domains(
        chart_kind="margin_error", y_domain=(0.0, max(1.0, max_mae * 1.05))
    )
    for model, group in plot.groupby("model_name"):
        color = MODEL_COLORS.get(str(model), TDNET_COLORS["slate"])
        axes[0].plot(
            group["week"],
            group["margin_mae"],
            marker="o",
            lw=1.5,
            label=str(model),
            color=color,
        )
        axes[1].plot(
            group["week"],
            group["winner_accuracy"],
            marker="o",
            lw=1.5,
            label=str(model),
            color=color,
        )
    axes[0].set(
        xlabel="Week", ylabel="Margin MAE (points)", title="Weekly margin error"
    )
    axes[1].set(
        xlabel="Week",
        ylabel="Winner accuracy",
        title="Weekly winner accuracy",
        ylim=(0, 1),
    )
    axes[1].axhline(0.5, color=TDNET_COLORS["medium_gray"], ls=":", lw=1)
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.4)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Completed CFBD results parquet/CSV.",
    )
    parser.add_argument("--snapshot-completeness", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Canonical week_XX/post_game directory; kept separate from the frozen pre_game package.",
    )
    parser.add_argument("--scientific-inventory", type=Path)
    parser.add_argument("--scientific-schedule-snapshot", type=Path)
    parser.add_argument("--scientific-market-lines-snapshot", type=Path)
    parser.add_argument("--scientific-reference-poll", type=Path)
    parser.add_argument("--scientific-season", type=int)
    parser.add_argument("--scientific-week", type=int)
    parser.add_argument(
        "--scientific-pregame-package",
        type=Path,
        help="Frozen pregame scientific directory to score without rerunning models.",
    )
    parser.add_argument(
        "--full-postgame-inputs",
        type=Path,
        help=(
            "Optional JSON sidecar for the complete scorecard/poll/power package. "
            "This postgame-only step never generates next-week matchup predictions."
        ),
    )
    args = parser.parse_args()
    args.output_root = require_week_directory(args.output_root, "post_game")
    existing = (
        [
            path
            for path in args.output_root.rglob("*")
            if path.is_file() and path.name != "README.md"
        ]
        if args.output_root.exists()
        else []
    )
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite non-empty Sunday output: {args.output_root}"
        )
    snapshot = json.loads(args.snapshot_completeness.read_text(encoding="utf-8"))
    accepted_certifications = {
        "weekly_snapshot_certified",
        "postgame_results_certified",
    }
    if (
        snapshot.get("status") != "pass"
        or snapshot.get("certification") not in accepted_certifications
    ):
        raise RuntimeError("Sunday publication requires a certified source snapshot.")
    verification = verify_prediction_bundle(args.bundle)
    if not verification["valid"]:
        raise RuntimeError(
            f"Cannot score invalid prediction bundle: {verification['failures']}"
        )
    results = (
        pd.read_parquet(args.results)
        if args.results.suffix == ".parquet"
        else pd.read_csv(args.results)
    )
    required_results = {"game_id", "home_points", "away_points"}
    missing_columns = required_results - set(results)
    if missing_columns:
        raise RuntimeError(
            f"Sunday results are missing columns: {sorted(missing_columns)}"
        )
    predictions = pd.read_parquet(args.bundle / "public/predictions.parquet")
    expected_game_ids = set(predictions["game_id"].astype(str))
    result_game_ids = set(results["game_id"].astype(str))
    missing_games = sorted(expected_game_ids - result_game_ids)
    duplicate_games = int(results["game_id"].astype(str).duplicated().sum())
    selected_results = results.loc[
        results["game_id"].astype(str).isin(expected_game_ids)
    ]
    missing_scores = int(
        selected_results[["home_points", "away_points"]].isna().any(axis=1).sum()
    )
    if missing_games or duplicate_games or missing_scores:
        raise RuntimeError(
            "Sunday results are incomplete: "
            f"missing_games={missing_games}, duplicate_games={duplicate_games}, "
            f"missing_scores={missing_scores}."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures_root = args.output_root / "figures"
    tables_root = args.output_root / "tables"
    blog_root = args.output_root / "blog"
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    blog_root.mkdir(parents=True, exist_ok=True)
    scored_root = args.output_root / "scoring"
    tables = score_prediction_bundle(args.bundle, results, output_root=scored_root)
    scored = tables["scored_predictions"]
    weekly, cumulative = _weekly_and_cumulative(scored)
    weekly.to_csv(tables_root / "weekly_metrics.csv", index=False)
    cumulative.to_csv(tables_root / "cumulative_metrics.csv", index=False)
    tables["scorecard"].to_csv(tables_root / "scorecard.csv", index=False)
    model_labels = (
        tables["scorecard"].loc[:, ["model_name", "objective"]].drop_duplicates()
    )
    model_labels["baseline_or_model"] = (
        model_labels["model_name"]
        .astype(str)
        .str.contains("vegas|random|knn|point|prior", case=False, regex=True)
        .map({True: "baseline", False: "model"})
    )
    model_labels.to_csv(tables_root / "baseline_comparison.csv", index=False)
    _figure(weekly, figures_root / "sunday_performance.png")
    scientific_paths = {}
    if args.scientific_pregame_package is not None:
        scientific_season = args.scientific_season or int(
            pd.to_numeric(scored["season"], errors="coerce").dropna().iloc[0]
        )
        scientific_week = (
            args.scientific_week
            if args.scientific_week is not None
            else int(pd.to_numeric(scored["week"], errors="coerce").dropna().iloc[0])
        )
        scientific_paths = write_scientific_postgame_evaluation(
            pregame_package=args.scientific_pregame_package,
            results_path=args.results,
            output_root=args.output_root / "scientific",
            season=scientific_season,
            week=scientific_week,
        )
        full_f0_f8_pregame = args.scientific_pregame_package / "full_f0_f8"
        if (full_f0_f8_pregame / "scientific_all_game_predictions.csv").exists():
            full_f0_f8_paths = write_scientific_postgame_evaluation(
                pregame_package=full_f0_f8_pregame,
                results_path=args.results,
                output_root=args.output_root / "scientific" / "full_f0_f8",
                season=scientific_season,
                week=scientific_week,
            )
            scientific_paths = {
                **{
                    f"market_free_f0_f6_{name}": path
                    for name, path in scientific_paths.items()
                },
                **{
                    f"full_f0_f8_{name}": path
                    for name, path in full_f0_f8_paths.items()
                },
            }
    elif args.scientific_inventory is not None:
        if args.scientific_schedule_snapshot is None:
            raise ValueError(
                "--scientific-schedule-snapshot is required with --scientific-inventory."
            )
        scientific_season = args.scientific_season or int(
            pd.to_numeric(scored["season"], errors="coerce").dropna().iloc[0]
        )
        scientific_week = (
            args.scientific_week
            if args.scientific_week is not None
            else int(pd.to_numeric(scored["week"], errors="coerce").dropna().iloc[0])
        )
        scientific_paths = build_scientific_weekly_outputs(
            project_root=ROOT,
            inventory_path=args.scientific_inventory,
            schedule_snapshot_path=args.scientific_schedule_snapshot,
            market_lines_path=args.scientific_market_lines_snapshot,
            reference_poll_path=args.scientific_reference_poll,
            output_root=args.output_root / "scientific",
            season=scientific_season,
            week=scientific_week,
            phase="post_game",
        )
    (blog_root / "summary.md").write_text(
        f"# TDNet Sunday retrospective — {datetime.now(EASTERN).date()}\n\n"
        "This draft was generated from an immutable prediction bundle after certified source checks. "
        "It is reviewable output, not an automatic publication.\n\n"
        f"Scored games: {int(scored['game_id'].nunique())}.\n",
        encoding="utf-8",
    )
    (blog_root / "post.txt").write_text(
        "TDNet Sunday scorecard draft — review required before posting.\n",
        encoding="utf-8",
    )
    (blog_root / "alt_text.md").write_text(
        "# Alt text\n\nWeekly TDNet margin and winner-accuracy scorecards by model.\n",
        encoding="utf-8",
    )
    files = {}
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and path.name != "sunday_publication_manifest.json":
            files[str(path.relative_to(args.output_root))] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    manifest = {
        "created_at_eastern": datetime.now(EASTERN).isoformat(),
        "status": "sunday_review_bundle_ready",
        "prediction_bundle": str(args.bundle),
        "prediction_bundle_verification": verification,
        "results_path": str(args.results),
        "results_sha256": sha256_file(args.results),
        "snapshot_completeness": snapshot,
        "files": files,
        "scientific_outputs": {
            name: str(path.relative_to(args.output_root))
            for name, path in scientific_paths.items()
        },
        "external_send": "disabled",
    }
    (args.output_root / "sunday_publication_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if args.full_postgame_inputs is not None:
        full = json.loads(args.full_postgame_inputs.read_text(encoding="utf-8"))

        def resolved(name: str) -> Path:
            value = Path(full[name])
            return value if value.is_absolute() else ROOT / value

        build_full_postgame_package(
            scored_predictions_path=scored_root / "scored_predictions.parquet",
            raw_games_path=resolved("raw_games"),
            margin_poll_dir=resolved("margin_poll_dir"),
            scientific_poll_dir=resolved("scientific_poll_dir"),
            margin_inventory_path=resolved("margin_inventory"),
            scientific_inventory_path=resolved("scientific_inventory"),
            reference_poll_path=resolved("reference_poll"),
            api_manifest_path=resolved("api_manifest"),
            output_root=args.output_root,
            season=int(full["season"]),
            completed_week=int(full["completed_week"]),
            state_week=int(full["state_week"]),
            logo_dir=resolved("logo_dir"),
            reference_label=str(
                full.get("reference_label", "Latest available AP Top 25")
            ),
            reference_short_label=str(full.get("reference_short_label", "AP")),
        )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "games": int(scored["game_id"].nunique()),
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
