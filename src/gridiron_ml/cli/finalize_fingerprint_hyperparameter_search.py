#!/usr/bin/env python
"""Retrain best fingerprint-search winners and build final Top 25 polls."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import yaml

REPO_ROOT = project_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gridiron_ml.experiments.hyperparameter_search import (
    always_keep_columns,
    default_source_fingerprint_root,
    safe_label,
)
from gridiron_ml.experiments.opponent_ablation import (
    ablation_spec_from_name,
    apply_ablation_view,
)
from gridiron_ml.experiments.opponent_adjusted import (
    DEFAULT_ALL_YEARS,
    DEFAULT_TRAIN_YEARS,
    DEFAULT_TEST_YEARS,
    DEFAULT_VAL_YEARS,
    StaticFrameFingerprints,
)
from gridiron_ml.models import get_model_class
from gridiron_ml.models.checkpoints import load_model_checkpoint
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.td_run.poll_viz import plot_ballot_logo_grid, plot_weekly_top25_table


DEFAULT_POLL_WEEKS_2025 = tuple(range(0, 17))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--source-fingerprint-root", type=Path, default=None)
    parser.add_argument("--train-years", nargs="*", type=int, default=list(DEFAULT_TRAIN_YEARS))
    parser.add_argument("--val-years", nargs="*", type=int, default=list(DEFAULT_VAL_YEARS))
    parser.add_argument("--eval-years", nargs="*", type=int, default=list(DEFAULT_TEST_YEARS))
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--poll-2025-weeks", nargs="*", type=int, default=list(DEFAULT_POLL_WEEKS_2025))
    parser.add_argument("--poll-2026-weeks", nargs="*", type=int, default=[0])
    parser.add_argument("--logo-dir", type=Path, default=Path("data/meta/logos/by_team"))
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports/final_model_polls"),
        help="Mirror final poll tables and figures into this easier-to-find reports directory.",
    )
    parser.add_argument("--skip-polls", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--models", nargs="*", default=None, help="Optional concrete model names to finalize.")
    parser.add_argument("--families", nargs="*", default=None, help="Optional model families to finalize.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip model directories that already have a checkpoint and eval metrics.")
    parser.add_argument("--poll-from-completed", action="store_true", help="Load completed final artifacts from disk when building polls.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    source_root = (
        args.source_fingerprint_root.resolve()
        if args.source_fingerprint_root
        else default_source_fingerprint_root(root)
    )
    objective = args.objective or output_root.name
    final_root = output_root / "final_artifacts"
    final_root.mkdir(parents=True, exist_ok=True)

    results = load_results_table(output_root)
    selected = select_best_by_model(results)
    selected = filter_selected_models(selected, models=args.models, families=args.families)
    if selected.empty:
        raise ValueError(f"No successful rows found in {output_root}.")
    selected.insert(0, "objective", objective)
    selected.to_csv(final_root / "selected_best_by_model.csv", index=False)

    trained = []
    if not args.skip_train:
        for row in selected.to_dict("records"):
            if args.skip_completed and final_artifact_complete(final_root, objective, row):
                print(
                    f"SKIP completed {objective}/{row['family']}/{row['model']}",
                    flush=True,
                )
                continue
            print(f"TRAIN {objective}/{row['family']}/{row['model']}", flush=True)
            trained.append(
                train_selected_row(
                    row=row,
                    root=root,
                    source_root=source_root,
                    final_root=final_root,
                    objective=objective,
                    train_years=tuple(args.train_years),
                    val_years=tuple(args.val_years),
                    eval_years=tuple(args.eval_years),
                )
            )
    poll_entries = trained
    if args.poll_from_completed:
        poll_entries = load_completed_entries(
            selected=selected,
            source_root=source_root,
            final_root=final_root,
            objective=objective,
        )
    inventory_path = final_root / "final_model_inventory.csv"
    inventory_entries = poll_entries if args.poll_from_completed else trained
    inventory = pd.DataFrame([entry["inventory"] for entry in inventory_entries])
    if inventory_entries or not inventory_path.exists() or inventory_file_empty(inventory_path):
        inventory = merge_inventory(inventory_path, inventory)
        inventory.to_csv(inventory_path, index=False)
    if poll_entries and not args.skip_polls:
        poll_root = final_root / "polls"
        poll_2025 = poll_root / "2025_full_season"
        build_poll_set(
            trained=poll_entries,
            season=2025,
            weeks=tuple(args.poll_2025_weeks),
            top_n=args.top_n,
            output_dir=poll_2025,
            logo_dir=resolve_path(root, args.logo_dir),
        )
        mirror_poll_reports(
            source_dir=poll_2025,
            reports_root=resolve_path(root, args.reports_root),
            objective=objective,
            label="2025_full_season",
        )
        poll_2026 = poll_root / "2026_preseason"
        build_poll_set(
            trained=poll_entries,
            season=2026,
            weeks=tuple(args.poll_2026_weeks),
            top_n=args.top_n,
            output_dir=poll_2026,
            logo_dir=resolve_path(root, args.logo_dir),
        )
        mirror_poll_reports(
            source_dir=poll_2026,
            reports_root=resolve_path(root, args.reports_root),
            objective=objective,
            label="2026_preseason",
        )

    print(f"Objective: {objective}")
    print(f"Selected models: {len(selected)}")
    print(f"Trained models: {len(trained)}")
    print(f"Poll models: {len(poll_entries)}")
    print(f"Final artifacts: {final_root}")


def load_results_table(output_root: Path) -> pd.DataFrame:
    tables_dir = output_root / "summary" / "tables"
    parquet_path = tables_dir / "master_hyperparameter_results.parquet"
    csvgz_path = tables_dir / "master_hyperparameter_results.csv.gz"
    metrics_path = tables_dir / "master_hyperparameter_metrics.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csvgz_path.exists():
        return pd.read_csv(csvgz_path, low_memory=False)
    if metrics_path.exists():
        return pd.read_csv(metrics_path, low_memory=False)
    raise FileNotFoundError(f"No merged search results found under {tables_dir}.")


def select_best_by_model(results: pd.DataFrame) -> pd.DataFrame:
    if "status" not in results.columns:
        return pd.DataFrame()
    success = results.loc[results["status"].astype(str).eq("success")].copy()
    if success.empty:
        return success
    success["tuning_score"] = pd.to_numeric(success.get("tuning_score"), errors="coerce")
    return (
        success.sort_values("tuning_score", ascending=False)
        .groupby(["family", "model"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def filter_selected_models(
    selected: pd.DataFrame,
    *,
    models: list[str] | None,
    families: list[str] | None,
) -> pd.DataFrame:
    """Apply optional finalize-time filters to selected best rows."""

    out = selected
    if families:
        wanted = {str(value) for value in families}
        out = out.loc[out["family"].astype(str).isin(wanted)]
    if models:
        wanted = {str(value) for value in models}
        out = out.loc[out["model"].astype(str).isin(wanted)]
    return out.reset_index(drop=True)


def final_artifact_complete(final_root: Path, objective: str, row: dict) -> bool:
    """Return true when a selected model already has its final checkpoint and metrics."""

    family = str(row["family"])
    model_name = str(row["model"])
    model_label = f"{objective}_{family}_{model_name}"
    out_dir = final_root / safe_label(family) / safe_label(model_name)
    checkpoint_path = out_dir / "checkpoints" / f"{safe_label(model_label)}.pkl"
    return checkpoint_path.exists() and (out_dir / "eval_metrics.csv").exists()


def merge_inventory(path: Path, new_inventory: pd.DataFrame) -> pd.DataFrame:
    """Merge newly trained inventory rows with any existing final inventory."""

    frames = []
    if path.exists():
        try:
            existing = pd.read_csv(path)
            if not existing.empty:
                frames.append(existing)
        except pd.errors.EmptyDataError:
            pass
    if new_inventory is not None and not new_inventory.empty:
        frames.append(new_inventory)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    keys = [col for col in ["objective", "family", "model", "final_model_name"] if col in merged.columns]
    if keys:
        merged = merged.drop_duplicates(subset=keys, keep="last")
    return merged.reset_index(drop=True)


def inventory_file_empty(path: Path) -> bool:
    """Return true when the inventory file is missing columns or rows."""

    if not path.exists():
        return True
    try:
        return pd.read_csv(path).empty
    except pd.errors.EmptyDataError:
        return True


def load_completed_entries(
    *,
    selected: pd.DataFrame,
    source_root: Path,
    final_root: Path,
    objective: str,
) -> list[dict]:
    """Load completed final model checkpoints and frames for poll generation."""

    entries = []
    for row in selected.to_dict("records"):
        if not final_artifact_complete(final_root, objective, row):
            print(
                f"SKIP poll missing checkpoint {objective}/{row['family']}/{row['model']}",
                flush=True,
            )
            continue
        family = str(row["family"])
        model_name = str(row["model"])
        model_label = f"{objective}_{family}_{model_name}"
        out_dir = final_root / safe_label(family) / safe_label(model_name)
        checkpoint_path = out_dir / "checkpoints" / f"{safe_label(model_label)}.pkl"
        print(f"LOAD poll checkpoint {objective}/{family}/{model_name}", flush=True)
        frame = load_selected_frame(row=row, source_root=source_root)
        entries.append(
            {
                "inventory": completed_inventory_row(
                    row=row,
                    objective=objective,
                    label=model_label,
                    checkpoint_path=checkpoint_path,
                    out_dir=out_dir,
                ),
                "model": load_model_checkpoint(checkpoint_path),
                "fingerprints": StaticFrameFingerprints(frame),
                "matchup_builder": MatchupBuilder(representation="unit_matchup"),
                "label": model_label,
            }
        )
    return entries


def completed_inventory_row(*, row: dict, objective: str, label: str, checkpoint_path: Path, out_dir: Path) -> dict:
    """Build an inventory row for a completed model directory."""

    inventory = {
        "objective": objective,
        "family": row.get("family"),
        "model": row.get("model"),
        "final_model_name": label,
        "fingerprint": row.get("fingerprint"),
        "top_k_features": row.get("top_k_features"),
        "trial_index": row.get("trial_index"),
        "tuning_score": row.get("tuning_score"),
        "checkpoint_path": str(checkpoint_path),
        "artifact_root": str(out_dir / "artifacts"),
    }
    metrics_path = out_dir / "eval_metrics.csv"
    if metrics_path.exists():
        try:
            metrics = pd.read_csv(metrics_path)
            if not metrics.empty:
                inventory.update({f"eval_{k}": v for k, v in metrics.iloc[0].to_dict().items()})
        except pd.errors.EmptyDataError:
            pass
    return inventory


def train_selected_row(
    *,
    row: dict,
    root: Path,
    source_root: Path,
    final_root: Path,
    objective: str,
    train_years: tuple[int, ...],
    val_years: tuple[int, ...],
    eval_years: tuple[int, ...],
) -> dict:
    family = str(row["family"])
    model_name = str(row["model"])
    model_label = f"{objective}_{family}_{model_name}"
    out_dir = final_root / safe_label(family) / safe_label(model_name)
    artifacts_dir = out_dir / "artifacts"
    checkpoint_path = out_dir / "checkpoints" / f"{safe_label(model_label)}.pkl"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_selected_frame(row=row, source_root=source_root)
    fingerprints = StaticFrameFingerprints(frame)
    config = parse_json_mapping(row.get("model_config_json"))
    config["model_name"] = model_label
    config["name"] = model_label
    model = get_model_class(family)(config)
    matchup_builder = MatchupBuilder(representation="unit_matchup")
    evaluator = TDEval(
        config={
            "model": {"family": family, **config},
            "eval": {
                "train_years": list(train_years),
                "test_years": list(eval_years),
                "artifact_root": str(artifacts_dir),
            },
        },
        fingerprints=fingerprints,
        matchup_builder=matchup_builder,
        model=model,
    )
    model = evaluator.train(train_years=train_years, val_years=val_years)
    metrics_df = pd.DataFrame()
    if eval_years:
        _, metrics_df = evaluator.evaluate(years=eval_years, label="eval")
    saved_checkpoint = model.save(checkpoint_path)
    artifact_root = evaluator.save_outputs(artifacts_dir)

    write_yaml(out_dir / "model_config.yaml", config)
    pd.DataFrame([row]).to_csv(out_dir / "source_search_row.csv", index=False)
    pd.DataFrame({"feature": selected_features(row)}).to_csv(out_dir / "selected_features.csv", index=False)
    if not metrics_df.empty:
        metrics_df.to_csv(out_dir / "eval_metrics.csv", index=False)

    inventory = {
        "objective": objective,
        "family": family,
        "model": model_name,
        "final_model_name": model_label,
        "fingerprint": row.get("fingerprint"),
        "top_k_features": row.get("top_k_features"),
        "trial_index": row.get("trial_index"),
        "tuning_score": row.get("tuning_score"),
        "checkpoint_path": str(saved_checkpoint),
        "artifact_root": str(artifact_root),
        "train_years_json": json.dumps(list(train_years)),
        "val_years_json": json.dumps(list(val_years)),
        "eval_years_json": json.dumps(list(eval_years)),
    }
    if not metrics_df.empty:
        inventory.update({f"eval_{k}": v for k, v in metrics_df.iloc[0].to_dict().items()})

    return {
        "inventory": inventory,
        "model": model,
        "fingerprints": fingerprints,
        "matchup_builder": matchup_builder,
        "label": model_label,
    }


def load_selected_frame(*, row: dict, source_root: Path) -> pd.DataFrame:
    fp_path = Path(str(row.get("fingerprint_path") or ""))
    if not fp_path.exists():
        fingerprint = str(row["fingerprint"])
        fp_path = source_root / "fingerprints" / safe_label(fingerprint) / "canonical_fingerprint.parquet"
    frame = pd.read_parquet(fp_path)
    frame = apply_ablation_view(frame, ablation_spec_from_name("raw_plus_adjusted_all"))
    features = selected_features(row)
    if features:
        keep = list(dict.fromkeys(always_keep_columns(frame) + [f for f in features if f in frame.columns]))
        frame = frame.loc[:, keep].copy()
    return frame


def selected_features(row: dict) -> list[str]:
    parsed = parse_json_value(row.get("selected_features_json"), default=[])
    return [str(value) for value in parsed] if isinstance(parsed, list) else []


def build_poll_set(*, trained: list[dict], season: int, weeks: tuple[int, ...], top_n: int, output_dir: Path, logo_dir: Path):
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for stale_plot in plots_dir.glob("*.png"):
        stale_plot.unlink()
    weekly_polls = []
    weekly_ballots = []
    skipped = []
    poll_columns = [
        "season",
        "week",
        "rank",
        "keys_team",
        "poll_points",
        "ballots_seen",
        "top25_votes",
        "first_place_votes",
        "average_rank",
        "best_rank",
        "worst_rank",
    ]
    ballot_columns = [
        "season",
        "week",
        "keys_team",
        "ballot_model",
        "ballot_rank",
        "poll_points",
        "top25_vote",
        "first_place_vote",
    ]
    skipped_columns = ["season", "week", "ballot_model", "reason"]

    for week in weeks:
        ballot_frames = []
        for entry in trained:
            try:
                evaluator = TDEval(
                    config={"eval": {"artifact_root": str(output_dir / "_poll_eval")}},
                    fingerprints=entry["fingerprints"],
                    matchup_builder=entry["matchup_builder"],
                    model=entry["model"],
                )
                evaluator.poll(
                    models=[entry["model"]],
                    season=int(season),
                    week=int(week),
                    top_n=top_n,
                    average_scope="season",
                )
                ballot = evaluator.poll_ballots_.copy()
                rank_scores = getattr(evaluator, "_last_poll_rank_scores_", {})
                score_summary = next(iter(rank_scores.values()), None)
                if score_summary and score_summary.get("unique_scores", 0) <= 1:
                    skipped.append(
                        {
                            "season": int(season),
                            "week": int(week),
                            "ballot_model": entry["label"],
                            "reason": (
                                "Poll ballot has no score variance; likely insufficient "
                                "feature coverage for this season/week."
                            ),
                        }
                    )
                    continue
                ballot["ballot_model"] = entry["label"]
                ballot_frames.append(ballot)
            except Exception as exc:
                skipped.append(
                    {
                        "season": int(season),
                        "week": int(week),
                        "ballot_model": entry["label"],
                        "reason": str(exc),
                    }
                )
        if not ballot_frames:
            continue
        ballots = pd.concat(ballot_frames, ignore_index=True)
        poll = aggregate_ballots(ballots, top_n=top_n)
        poll.insert(0, "week", int(week))
        poll.insert(0, "season", int(season))
        ballots.insert(0, "week", int(week))
        ballots.insert(0, "season", int(season))
        weekly_polls.append(poll.head(top_n))
        weekly_ballots.append(ballots)

    weekly_poll = pd.concat(weekly_polls, ignore_index=True) if weekly_polls else pd.DataFrame(columns=poll_columns)
    weekly_ballot = pd.concat(weekly_ballots, ignore_index=True) if weekly_ballots else pd.DataFrame(columns=ballot_columns)
    skipped_df = pd.DataFrame(skipped, columns=skipped_columns)
    weekly_poll.to_csv(tables_dir / "weekly_poll_top25.csv", index=False)
    weekly_ballot.to_csv(tables_dir / "weekly_poll_ballots.csv", index=False)
    skipped_df.to_csv(tables_dir / "weekly_poll_skipped_weeks.csv", index=False)
    season_summary = aggregate_ballots(weekly_ballot, top_n=top_n) if not weekly_ballot.empty else pd.DataFrame()
    season_summary.to_csv(tables_dir / "season_summary_top25.csv", index=False)
    if not weekly_poll.empty:
        plot_weekly_top25_table(weekly_poll, plots_dir / "weekly_poll_top25_table.png", top_n=top_n, logo_dir=logo_dir)
    if not weekly_ballot.empty:
        for week in sorted(pd.to_numeric(weekly_ballot["week"], errors="coerce").dropna().astype(int).unique()):
            plot_ballot_logo_grid(
                weekly_ballot.loc[pd.to_numeric(weekly_ballot["week"], errors="coerce") == week],
                plots_dir / f"weekly_poll_ballot_grid_week_{week:02d}.png",
                top_n=top_n,
                logo_dir=logo_dir,
                title=f"{season} Week {week} Final Model Ballots",
            )
    if not season_summary.empty:
        season_ballot = season_summary.rename(columns={"rank": "ballot_rank"}).copy()
        season_ballot["ballot_model"] = "full_season_summary"
        season_ballot["top25_vote"] = True
        plot_ballot_logo_grid(
            season_ballot,
            plots_dir / "season_summary_top25.png",
            top_n=top_n,
            logo_dir=logo_dir,
            title=f"{season} Full Season Summary Top {top_n}",
        )


def mirror_poll_reports(*, source_dir: Path, reports_root: Path, objective: str, label: str) -> Path:
    """Copy final poll tables and figures into a public-report-friendly folder."""

    destination = reports_root / safe_label(objective) / safe_label(label)
    if not source_dir.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    for subdir in ("tables", "plots"):
        source_subdir = source_dir / subdir
        if not source_subdir.exists():
            continue
        destination_subdir = destination / subdir
        if destination_subdir.exists():
            shutil.rmtree(destination_subdir)
        shutil.copytree(source_subdir, destination_subdir)
    readme = destination / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# {objective.title()} {label.replace('_', ' ').title()} Poll Report",
                "",
                "Mirrored from final model poll artifacts for easier browsing.",
                "",
                f"Source: `{source_dir}`",
                "",
                "Contents:",
                "",
                "- `tables/weekly_poll_top25.csv`",
                "- `tables/weekly_poll_ballots.csv`",
                "- `tables/weekly_poll_skipped_weeks.csv`",
                "- `tables/season_summary_top25.csv`",
                "- `plots/weekly_poll_top25_table.png`",
                "- `plots/weekly_poll_ballot_grid_week_*.png`",
                "- `plots/season_summary_top25.png` when available",
                "",
            ]
        )
    )
    return destination


def aggregate_ballots(ballots: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    poll = (
        ballots.groupby("keys_team", as_index=False)
        .agg(
            poll_points=("poll_points", "sum"),
            ballots_seen=("ballot_model", "nunique"),
            top25_votes=("top25_vote", "sum"),
            first_place_votes=("first_place_vote", "sum"),
            average_rank=("ballot_rank", "mean"),
            best_rank=("ballot_rank", "min"),
            worst_rank=("ballot_rank", "max"),
        )
        .sort_values(["poll_points", "average_rank", "best_rank", "keys_team"], ascending=[False, True, True, True])
        .reset_index(drop=True)
    )
    poll.insert(0, "rank", range(1, len(poll) + 1))
    poll["average_rank"] = poll["average_rank"].astype(float).round(3)
    return poll.head(top_n)


def parse_json_mapping(value) -> dict:
    parsed = parse_json_value(value, default={})
    return dict(parsed or {}) if isinstance(parsed, dict) else {}


def parse_json_value(value, *, default):
    if value is None or pd.isna(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
