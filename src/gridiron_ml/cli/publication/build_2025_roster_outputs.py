#!/usr/bin/env python3
"""Build the complete 2025 holdout example from a frozen model roster."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from gridiron_ml.publication import build_frozen_roster_poll, build_weekly_blog_package
from gridiron_ml.publication.poll_recaps import (
    aggregate_receiving_votes,
    format_receiving_votes,
    model_consensus_disagreement,
    plot_consensus_poll_table,
    plot_full_season_poll_grid,
    plot_model_disagreement,
    plot_tdnet_vs_ap_poll,
)
from gridiron_ml.publication.polls import add_team_records, load_postweek_ap_top25
from gridiron_ml.publication.recaps import (
    add_cumulative_weekly_metrics,
    all_model_cumulative_metrics,
    cumulative_model_metrics,
    grade_postgame_predictions,
    plot_all_model_cumulative_performance,
    plot_model_cumulative_track,
    plot_objective_weekly_comparison,
    plot_sunday_recap_table,
    vegas_recap_metrics,
    weekly_recap_metrics,
    _build_one_model_recaps,
)
from gridiron_ml.publication.weekly import summarize_weekly_predictions
from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.td_run.poll_viz import plot_ballot_logo_grid


OBJECTIVES = ("winner", "margin")
WEEKS = tuple(range(0, 17))
COMPARATIVE_BASELINE_FAMILIES = frozenset({"naive"})
INVALID_PREDICTION_FEATURE_CONFIGS = frozenset({"F7", "F8"})
POLL_EXCLUDED_MODEL_IDS = frozenset(
    {
        "winner_linear_ols",
        "winner_linear_huber",
        "winner_linear_ridge",
        "winner_linear_sgd",
    }
)


def _training_seasons(inventory: pd.DataFrame) -> list[int]:
    seasons: set[int] = set()
    for value in inventory.get("training_seasons", pd.Series(dtype=str)):
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = []
        if isinstance(parsed, list):
            seasons.update(int(season) for season in parsed)
    return sorted(seasons)


def _load_games(path: Path, season: int) -> pd.DataFrame:
    games = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    games = games[pd.to_numeric(games["season"], errors="coerce").eq(season)].copy()
    games["game_id"] = pd.to_numeric(games["id"], errors="coerce").astype("Int64")
    games["week"] = pd.to_numeric(games["week"], errors="coerce").astype(int)
    return games


def _is_comparative_baseline(inventory: pd.DataFrame) -> pd.Series:
    return inventory.get("model_family", pd.Series("", index=inventory.index)).astype(str).str.lower().isin(
        COMPARATIVE_BASELINE_FAMILIES
    )


def _is_invalid_prediction_feature_config(inventory: pd.DataFrame) -> pd.Series:
    return inventory.get("feature_config", pd.Series("", index=inventory.index)).astype(str).isin(
        INVALID_PREDICTION_FEATURE_CONFIGS
    )


def _is_poll_excluded_model(inventory: pd.DataFrame) -> pd.Series:
    excluded_by_id = inventory.get(
        "model_id", pd.Series("", index=inventory.index)
    ).astype(str).isin(POLL_EXCLUDED_MODEL_IDS)
    if "use_in_tdnet_poll" not in inventory:
        return excluded_by_id
    explicitly_disabled = ~inventory["use_in_tdnet_poll"].astype(str).str.lower().isin(
        {"1", "true", "yes", "y"}
    )
    return excluded_by_id | explicitly_disabled


def _assert_no_suspicious_top_rank(poll: pd.DataFrame, *, objective: str, week: int) -> None:
    """Fail closed on the known Air Force #1 poll failure mode."""
    if poll.empty or str(poll.iloc[0].get("keys_team", "")).strip().lower() != "air force":
        return
    raise RuntimeError(
        f"Refusing to publish {objective} week {week}: Air Force is ranked #1. "
        "This is a known invalid-poll signature and requires investigation."
    )


def _grade_long(long: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    actual = games[
        ["game_id", "week", "start_date", "home_team", "away_team", "home_points", "away_points"]
    ].copy()
    actual["game_id"] = pd.to_numeric(actual["game_id"], errors="coerce").astype("Int64")
    out = long.copy()
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
    out = out.merge(actual, on="game_id", how="inner", suffixes=("", "_actual"), validate="many_to_one")
    out["home_team"] = out["home_team"].fillna(out["home_team_actual"])
    out["away_team"] = out["away_team"].fillna(out["away_team_actual"])
    out["week"] = out["week"].fillna(out["week_actual"])
    out["model_count"] = 1
    out["model_agreement"] = 1.0
    if "model_id" in out:
        out["model_slug"] = out["model_id"].astype(str)
    return grade_postgame_predictions(out)


def _aggregate_models(graded: pd.DataFrame) -> pd.DataFrame:
    keys = ["game_id", "week", "home_team", "away_team"]
    first = [
        "start_date", "home_points", "away_points", "market_spread_close", "market_over_under",
        "ap_rank_home", "ap_rank_away",
    ]
    agg = {
        "pred_home_margin": ("pred_home_margin", "mean"),
        "pred_home_win_probability": ("pred_home_win_probability", "mean"),
        "model_count": ("model_name", "nunique"),
        "model_agreement": (
            "pred_home_win_probability",
            lambda values: max(float(pd.Series(values).ge(0.5).mean()), float(pd.Series(values).lt(0.5).mean())),
        ),
    }
    agg.update({column: (column, "first") for column in first if column in graded})
    return grade_postgame_predictions(
        graded.groupby(keys, dropna=False, as_index=False).agg(**agg)
    )


def _weekly_summary_row(*, season: int, week: int, objective: str, games: pd.DataFrame) -> dict:
    return {
        "season": season,
        "week": week,
        "objective": objective,
        **weekly_recap_metrics(games),
        **vegas_recap_metrics(games),
    }


def _weekly_summaries_from_scorecards(*, objective_root: Path, season: int, objective: str) -> pd.DataFrame:
    rows = []
    for week in range(1, 17):
        scorecard = objective_root / f"week_{week:02d}" / "prediction_vs_actual.csv"
        if scorecard.exists():
            rows.append(
                _weekly_summary_row(
                    season=season,
                    week=week,
                    objective=objective,
                    games=pd.read_csv(scorecard),
                )
            )
    if not rows:
        return pd.DataFrame(columns=["season", "week", "objective"])
    return add_cumulative_weekly_metrics(pd.DataFrame(rows))


def _load_all_model_frames_from_weekly_predictions(
    *,
    weekly_root: Path,
    games: pd.DataFrame,
) -> list[pd.DataFrame]:
    frames = []
    for week in range(1, 17):
        predictions = weekly_root / f"week_{week:02d}" / "blog_preview" / "tables" / "all_game_model_predictions.csv"
        if not predictions.exists():
            continue
        long = pd.read_csv(predictions)
        long["model_id"] = long["model_name"].astype(str)
        frames.append(_grade_long(long, games))
    return frames


def _write_all_model_cumulative_artifacts(
    *,
    objective_root: Path,
    weekly_root: Path,
    games: pd.DataFrame,
    all_model_frames: list[pd.DataFrame],
    season: int,
    objective: str,
) -> None:
    frames = all_model_frames or _load_all_model_frames_from_weekly_predictions(
        weekly_root=weekly_root,
        games=games,
    )
    if not frames:
        return
    metrics = all_model_cumulative_metrics(pd.concat(frames, ignore_index=True))
    if metrics.empty:
        return
    output = objective_root / "all_models_cumulative"
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "all_models_cumulative_performance.csv", index=False)
    final = (
        metrics.sort_values(["model_id", "week"], kind="mergesort")
        .groupby("model_id", as_index=False)
        .tail(1)
        .sort_values(
            ["cumulative_su_accuracy", "cumulative_ats_accuracy_excluding_pushes", "cumulative_margin_mae"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    final.to_csv(output / "all_models_final_cumulative_performance.csv", index=False)
    plot_all_model_cumulative_performance(
        metrics,
        output / "all_models_cumulative_performance.png",
        season=season,
        title=f"TDNet {objective.title()}",
    )


def _selection_models(inventory: pd.DataFrame, objective: str, count: int) -> list[str]:
    frame = inventory[inventory["objective"].astype(str).str.lower().eq(objective)].copy()
    frame["preseason_performance_rank"] = pd.to_numeric(
        frame["preseason_performance_rank"], errors="coerce"
    )
    frame = frame.sort_values(["preseason_performance_rank", "model_id"], kind="mergesort")
    return frame.head(count)["model_id"].astype(str).tolist()


def _write_poll_support(
    poll_result: dict[str, pd.DataFrame],
    directory: Path,
    *,
    season: int,
    week: int,
    objective: str,
    ap: pd.DataFrame,
    logo_dir: Path,
    model_order: list[str],
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ballots = poll_result["ballots"].copy()
    # Some ensemble checkpoints expose their constituent models as additional
    # ballots.  Enforce the publication inventory after evaluation so a model
    # that is explicitly disabled for polling cannot re-enter through an
    # ensemble checkpoint.
    allowed_models = {str(model_id).casefold() for model_id in model_order}
    ballots = ballots.loc[
        ballots["ballot_model"].astype(str).str.casefold().isin(allowed_models)
    ].copy()
    if ballots.empty:
        raise RuntimeError(f"No poll-enabled ballots remain for {objective} week {week}.")
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
        .sort_values(
            ["poll_points", "average_rank", "best_rank", "keys_team"],
            ascending=[False, True, True, True],
        )
        .reset_index(drop=True)
    )
    poll.insert(0, "rank", np.arange(1, len(poll) + 1))
    poll.insert(0, "week", int(week))
    poll.insert(0, "season", int(season))
    poll.insert(0, "poll_objective", objective)
    order = {str(model_id).casefold(): index for index, model_id in enumerate(model_order)}
    ballots["_model_order"] = (
        ballots["ballot_model"].astype(str).str.casefold().map(order).fillna(len(order))
    )
    ballots = ballots.sort_values(["_model_order", "ballot_rank"], kind="stable").drop(columns="_model_order")
    if not ap.empty:
        ap_rank = ap.set_index("team")["rank"]
        poll["ap_rank"] = poll["keys_team"].map(ap_rank).astype("Int64")
        poll["tdnet_minus_ap"] = poll["rank"] - poll["ap_rank"]
        poll["ap_snapshot_week"] = int(ap["week"].dropna().iloc[0]) if "week" in ap and ap["week"].notna().any() else int(week) + 1
    else:
        poll["ap_rank"] = pd.Series(pd.NA, index=poll.index, dtype="Int64")
        poll["tdnet_minus_ap"] = pd.Series(pd.NA, index=poll.index, dtype="Int64")
        poll["ap_snapshot_week"] = pd.Series(pd.NA, index=poll.index, dtype="Int64")
    if games is not None:
        poll = add_team_records(poll, games, completed_week=week)
    bad_ballots = ballots.loc[
        ballots["keys_team"].astype(str).str.strip().str.lower().eq("air force")
        & pd.to_numeric(ballots["ballot_rank"], errors="coerce").eq(1)
    ]
    if not bad_ballots.empty:
        raise RuntimeError(
            f"Refusing to publish {objective} week {week}: Air Force is ranked #1 by "
            f"{bad_ballots['ballot_model'].astype(str).tolist()}."
        )
    _assert_no_suspicious_top_rank(poll, objective=objective, week=week)
    receiving = aggregate_receiving_votes(poll, ballots)
    receiving_text = format_receiving_votes(receiving)
    plot_consensus_poll_table(
        poll,
        directory / "tdnet_top25.png",
        title=f"{season} Week {week}: TDNet {objective.title()}-Objective Top 25",
        receiving_votes=receiving_text,
        logo_dir=logo_dir,
    )
    plot_ballot_logo_grid(
        ballots,
        directory / "tdnet_model_ballots.png",
        top_n=25,
        logo_dir=logo_dir,
        title=f"{season} Week {week}: TDNet {objective.title()}-Objective Model Ballots",
    )
    poll.to_csv(directory / "tdnet_top25.csv", index=False)
    ballots.to_csv(directory / "tdnet_model_ballots.csv", index=False)
    receiving.to_csv(directory / "tdnet_receiving_votes.csv", index=False)
    (directory / "receiving_votes.txt").write_text(receiving_text + "\n", encoding="utf-8")
    top_ballots = ballots[pd.to_numeric(ballots["ballot_rank"], errors="coerce").between(1, 25)].copy()
    top_ballots.to_csv(directory / "per_model_top25_long.csv", index=False)
    top_ballots.pivot_table(
        index="ballot_rank", columns="ballot_model", values="keys_team", aggfunc="first"
    ).sort_index().to_csv(directory / "per_model_top25.csv")
    disagreement = model_consensus_disagreement(poll, ballots)
    disagreement.to_csv(directory / "model_consensus_disagreement.csv", index=False)
    plot_model_disagreement(
        disagreement,
        directory / "model_consensus_disagreement.png",
        title=f"{season} Week {week}: Models Farthest from {objective.title()} Consensus",
    )
    if not ap.empty:
        plot_tdnet_vs_ap_poll(
            poll.rename(columns={"keys_team": "team"}),
            ap,
            directory / "tdnet_vs_ap_top25.png",
            title=f"{season} Week {week}: TDNet vs AP Top 25",
            logo_dir=logo_dir,
        )
    return poll


def _build_objective(
    *,
    root: Path,
    season: int,
    objective: str,
    inventory_path: Path,
    ranking_path: Path,
    schedule_path: Path,
    ap_path: Path,
    output_root: Path,
    logo_dir: Path,
    weeks: tuple[int, ...],
    polls_only: bool = False,
    compact_output: bool = True,
) -> pd.DataFrame:
    raw_inventory = pd.read_csv(inventory_path)
    training_seasons = _training_seasons(raw_inventory)
    leakage_rehearsal = bool(training_seasons and max(training_seasons) >= season)
    scorecard_warning = (
        "DRY RUN — roster trained through 2025; evaluation overlaps training"
        if leakage_rehearsal
        else "HOLDOUT EXAMPLE — roster trained through 2024; 2025 is held out"
    )
    inventory = raw_inventory.copy()
    ranking = pd.read_csv(ranking_path)[["model_id", "preseason_performance_rank"]]
    inventory = inventory.merge(ranking, on="model_id", how="left", validate="one_to_one")
    games = _load_games(schedule_path, season)
    objective_root = output_root / "sunday_recaps" / objective
    weekly_root = output_root / "weekly_predictions" / objective
    objective_root.mkdir(parents=True, exist_ok=True)
    weekly_root.mkdir(parents=True, exist_ok=True)
    objective_inventory = inventory.loc[
        inventory["objective"].astype(str).str.lower().eq(objective)
        & ~_is_comparative_baseline(inventory)
        & ~_is_invalid_prediction_feature_config(inventory)
    ].sort_values(["preseason_performance_rank", "model_id"], kind="stable").copy()
    objective_inventory_path = objective_root / "objective_model_inventory.csv"
    objective_inventory.to_csv(objective_inventory_path, index=False)
    poll_inventory = objective_inventory.loc[~_is_poll_excluded_model(objective_inventory)].copy()
    poll_inventory_path = objective_root / "published_poll_model_inventory.csv"
    poll_inventory.to_csv(poll_inventory_path, index=False)
    ap_by_week: dict[int, pd.DataFrame] = {
        week: load_postweek_ap_top25(ap_path, season=season, completed_week=week) for week in range(1, 17)
    }
    weekly_summaries = []
    all_model_frames = []
    polls = []

    # A failed or intentionally partial run can resume without recomputing
    # completed weeks. Existing weekly tables are also the source of the full
    # individual-model tracks when only a suffix is requested.
    for existing_week in WEEKS:
        if existing_week in weeks:
            continue
        existing_poll = objective_root / f"week_{existing_week:02d}" / "tdnet_top25.csv"
        existing_predictions = weekly_root / f"week_{existing_week:02d}" / "blog_preview" / "tables" / "all_game_model_predictions.csv"
        existing_scorecard = objective_root / f"week_{existing_week:02d}" / "prediction_vs_actual.csv"
        if existing_poll.exists():
            existing_poll_frame = pd.read_csv(existing_poll)
            if "season" not in existing_poll_frame:
                existing_poll_frame.insert(0, "season", int(season))
            if "week" not in existing_poll_frame:
                existing_poll_frame.insert(1, "week", int(existing_week))
            polls.append(existing_poll_frame)
        if existing_predictions.exists():
            existing_long = pd.read_csv(existing_predictions)
            existing_long["model_id"] = existing_long["model_name"].astype(str)
            all_model_frames.append(_grade_long(existing_long, games))
        if existing_scorecard.exists():
            existing_games = pd.read_csv(existing_scorecard)
            weekly_summaries.append(
                _weekly_summary_row(season=season, week=existing_week, objective=objective, games=existing_games)
            )
    existing_summary_path = objective_root / "weekly_summary.csv"
    if existing_summary_path.exists():
        try:
            weekly_summaries.extend(pd.read_csv(existing_summary_path).to_dict("records"))
        except pd.errors.EmptyDataError:
            pass

    for week in weeks:
        print(f"{objective}: building week {week:02d} poll", flush=True)
        sunday_week = objective_root / f"week_{week:02d}"
        sunday_week.mkdir(parents=True, exist_ok=True)
        poll_result = build_frozen_roster_poll(
            poll_inventory_path,
            season=season,
            week=week,
            output_dir=sunday_week,
            project_root=root,
            logo_dir=logo_dir,
            objective=objective,
            render_figures=False,
        )
        poll = _write_poll_support(
            poll_result,
            sunday_week,
            season=season,
            week=week,
            objective=objective,
            ap=ap_by_week.get(week, pd.DataFrame()),
            logo_dir=logo_dir,
            model_order=poll_inventory["model_id"].astype(str).tolist(),
            games=games,
        )
        polls.append(poll)
        if polls_only or week == 0:
            continue

        weekly_output = weekly_root / f"week_{week:02d}" / "blog_preview"
        report = build_weekly_blog_package(
            project_root=root,
            season=season,
            week=week,
            model_inventory_path=objective_inventory_path,
            schedule_snapshot_path=schedule_path,
            ap_top25_path=ap_path,
            tdnet_top25_path=sunday_week / "tdnet_top25.csv",
            top25_label="AP Top 25",
            output_root=weekly_output,
            preseason_ranking_path=ranking_path,
            include_collapsed_models=True,
        )
        print(f"{objective}: building week {week:02d} predictions and scorecards", flush=True)
        long = report["all_model_predictions"].copy()
        long["model_id"] = long["model_name"].astype(str)
        graded_long = _grade_long(long, games)
        ap_rank_columns = [
            column for column in ("game_id", "ap_rank_home", "ap_rank_away")
            if column in report["all_games"]
        ]
        if len(ap_rank_columns) == 3:
            ap_ranks = report["all_games"][ap_rank_columns].drop_duplicates("game_id")
            graded_long = graded_long.merge(
                ap_ranks, on="game_id", how="left", validate="many_to_one"
            )
        consensus = _aggregate_models(graded_long)
        consensus.to_csv(sunday_week / "prediction_vs_actual.csv", index=False)
        season_to_date_parts = []
        for completed_week in range(1, week):
            completed_path = objective_root / f"week_{completed_week:02d}" / "prediction_vs_actual.csv"
            if completed_path.exists():
                season_to_date_parts.append(pd.read_csv(completed_path))
        season_to_date_parts.append(consensus)
        plot_sunday_recap_table(
            consensus,
            sunday_week / "prediction_vs_actual.png",
            season=season,
            week=week,
            objective=objective,
            model_label=f"All {len(_selection_models(objective_inventory, objective, 99))} {objective}-objective roster models",
            warning_label=scorecard_warning,
            season_to_date_games=pd.concat(season_to_date_parts, ignore_index=True),
        )
        all_model_frames.append(graded_long)

        if not compact_output:
            for selection_name, count in (("all_models", 99),):
                model_ids = _selection_models(objective_inventory, objective, count)
                selected = graded_long[graded_long["model_id"].isin(model_ids)].copy()
                selected_consensus = _aggregate_models(selected)
                selection_root = objective_root / "prediction_sets" / selection_name
                selection_root.mkdir(parents=True, exist_ok=True)
                selected_consensus.to_csv(selection_root / f"week_{week:02d}_prediction_vs_actual.csv", index=False)
                week_dir = selection_root / f"week_{week:02d}"
                week_dir.mkdir(parents=True, exist_ok=True)
                selected_consensus.to_csv(week_dir / "prediction_vs_actual.csv", index=False)
                plot_sunday_recap_table(
                    selected_consensus,
                    week_dir / "prediction_vs_actual.png",
                    season=season,
                    week=week,
                    objective=objective,
                    model_label=f"{selection_name.replace('_', ' ').title()} · {len(model_ids)} frozen models",
                    warning_label=scorecard_warning,
                )
                shutil.copy2(week_dir / "prediction_vs_actual.png", sunday_week / f"{selection_name}_prediction_vs_actual.png")
                shutil.copy2(week_dir / "prediction_vs_actual.svg", sunday_week / f"{selection_name}_prediction_vs_actual.svg")

        weekly_summaries.append(_weekly_summary_row(season=season, week=week, objective=objective, games=consensus))

    if polls_only:
        poll_frame = pd.concat(polls, ignore_index=True)
        poll_frame.to_csv(objective_root / "weekly_poll_top25.csv", index=False)
        plot_full_season_poll_grid(
            poll_frame,
            objective_root / "full_season_poll_grid.png",
            objective=objective,
            season=season,
            logo_dir=logo_dir,
        )
        _write_all_model_cumulative_artifacts(
            objective_root=objective_root,
            weekly_root=weekly_root,
            games=games,
            all_model_frames=all_model_frames,
            season=season,
            objective=objective,
        )
        summaries = _weekly_summaries_from_scorecards(
            objective_root=objective_root,
            season=season,
            objective=objective,
        )
        if summaries.empty and existing_summary_path.exists():
            try:
                summaries = add_cumulative_weekly_metrics(pd.read_csv(existing_summary_path))
            except pd.errors.EmptyDataError:
                pass
        if not summaries.empty:
            summaries.to_csv(existing_summary_path, index=False)
        return summaries

    # Individual model scorecards and cumulative tracks.
    if not compact_output:
        individual_root = objective_root / "individual_models"
        valid_individual_models = set(objective_inventory["model_id"].astype(str))
        individual_root.mkdir(parents=True, exist_ok=True)
        for stale in individual_root.glob("*"):
            if stale.is_dir() and stale.name not in valid_individual_models:
                shutil.rmtree(stale)
        if all_model_frames:
            grouped_models = list(pd.concat(all_model_frames, ignore_index=True).groupby("model_id", sort=True))
            tasks = [(model_id, model_games, individual_root, season, objective) for model_id, model_games in grouped_models]
            workers = min(4, max(1, len(tasks)))
            with ProcessPoolExecutor(max_workers=workers) as executor:
                individual_rows = list(executor.map(_build_one_model_recaps, tasks))
            individual_leaderboard = pd.DataFrame(individual_rows).sort_values(
                ["su_accuracy", "ats_accuracy_excluding_pushes", "margin_mae"],
                ascending=[False, False, True],
            )
        else:
            individual_leaderboard = pd.DataFrame()
        individual_leaderboard.to_csv(individual_root / "season_model_leaderboard.csv", index=False)

    poll_frame = pd.concat(polls, ignore_index=True)
    poll_frame.to_csv(objective_root / "weekly_poll_top25.csv", index=False)
    plot_full_season_poll_grid(
        poll_frame,
        objective_root / "full_season_poll_grid.png",
        objective=objective,
        season=season,
        logo_dir=logo_dir,
    )
    _write_all_model_cumulative_artifacts(
        objective_root=objective_root,
        weekly_root=weekly_root,
        games=games,
        all_model_frames=all_model_frames,
        season=season,
        objective=objective,
    )
    summaries = (
        pd.DataFrame(weekly_summaries)
        .drop_duplicates(["objective", "week"], keep="last")
        .sort_values(["objective", "week"], kind="mergesort")
        .reset_index(drop=True)
    )
    summaries = add_cumulative_weekly_metrics(summaries)
    summaries.to_csv(objective_root / "weekly_summary.csv", index=False)
    return summaries


def main() -> None:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--season", type=int, default=2025)
    artifact_root = Path("/groups/bsavoie2/tburton2/TDNet/publication_artifacts")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=artifact_root / "scientific_roster_refits/f0_f8_margin_through_2025_v1/final_model_inventory.csv",
    )
    parser.add_argument(
        "--ranking",
        type=Path,
        default=artifact_root / "corrected_f6_wide_margin_roster/holdout_2025_v1/preseason_model_rankings.csv",
    )
    parser.add_argument("--schedule", type=Path, default=root / "data/raw/cfbd/v2/games/2025.parquet")
    parser.add_argument("--ap-rankings", type=Path, default=root / "data/raw/cfbd/v2/rankings/2025.parquet")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=artifact_root / "2025_roster_regenerations/manual",
    )
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES))
    parser.add_argument("--weeks", type=int, nargs="+", default=list(WEEKS), help="Weeks to build; defaults to 0 through 16.")
    parser.add_argument(
        "--polls-only",
        action="store_true",
        help="Refresh poll ballots and Top 25 artifacts without rebuilding weekly prediction scorecards.",
    )
    parser.add_argument(
        "--allow-training-through-holdout",
        action="store_true",
        help=(
            "Allow a rehearsal roster trained through the evaluation season. "
            "This is invalid for a holdout result and is intended only for retrospective dry runs."
        ),
    )
    parser.add_argument(
        "--expanded-output",
        action="store_true",
        help="Also emit legacy selection copies and per-model diagnostic trees; compact output is the default.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    inventory = args.inventory.resolve()
    ranking = args.ranking.resolve()
    schedule = args.schedule.resolve()
    ap = args.ap_rankings.resolve()
    output = args.output_root.resolve()
    logo_dir = root / "data/meta/logos/by_team"
    weeks = tuple(sorted(set(args.weeks)))
    if any(week not in WEEKS for week in weeks):
        raise ValueError(f"weeks must be between 0 and 16, got {weeks}")
    for path in (inventory, ranking, schedule, ap):
        if not path.exists():
            raise FileNotFoundError(path)
    inventory_frame = pd.read_csv(inventory)
    training_seasons = _training_seasons(inventory_frame)
    ranking_frame = pd.read_csv(ranking)
    missing_ranking_models = sorted(
        set(inventory_frame["model_id"].astype(str)) - set(ranking_frame["model_id"].astype(str))
    )
    if missing_ranking_models:
        # Scientific frozen bundles do not carry the operational preseason-CV
        # ranking sidecar.  Give those models a deterministic tie-break order
        # for this rehearsal; it is presentation metadata, not a fitted input.
        current_max = pd.to_numeric(
            ranking_frame["preseason_performance_rank"], errors="coerce"
        ).max()
        current_max = 0 if pd.isna(current_max) else int(current_max)
        ranking_frame = pd.concat(
            [
                ranking_frame[["model_id", "preseason_performance_rank"]],
                pd.DataFrame(
                    {
                        "model_id": missing_ranking_models,
                        "preseason_performance_rank": range(
                            current_max + 1, current_max + 1 + len(missing_ranking_models)
                        ),
                    }
                ),
            ],
            ignore_index=True,
        )
        output.mkdir(parents=True, exist_ok=True)
        generated_ranking = output / "generated_preseason_model_rankings.csv"
        ranking_frame.to_csv(generated_ranking, index=False)
        ranking = generated_ranking
    leakage_rehearsal = bool(training_seasons and max(training_seasons) >= args.season)
    if leakage_rehearsal and not args.allow_training_through_holdout:
        raise ValueError(
            f"Roster is not a holdout roster: training includes {max(training_seasons)} "
            f"for evaluation season {args.season}."
        )

    summaries = []
    for objective in args.objectives:
        summaries.append(
            _build_objective(
                root=root,
                season=args.season,
                objective=objective,
                inventory_path=inventory,
                ranking_path=ranking,
                schedule_path=schedule,
                ap_path=ap,
                output_root=output,
                logo_dir=logo_dir,
                weeks=weeks,
                polls_only=args.polls_only,
                compact_output=not args.expanded_output,
            )
        )
    comparison = add_cumulative_weekly_metrics(pd.concat(summaries, ignore_index=True))
    sunday_root = output / "sunday_recaps"
    comparison.to_csv(sunday_root / "objective_weekly_comparison.csv", index=False)
    plot_objective_weekly_comparison(
        comparison,
        sunday_root / "objective_weekly_comparison.png",
        season=args.season,
    )
    readme_text = (
        f"# TDNet {args.season} frozen-roster retrospective examples\n\n"
        "This directory replays the completed season with a roster frozen before "
        "the 2025 evaluation replay. It includes weekly prediction tables, "
        "blog-preview figures, Sunday scorecards, compact full-roster scorecards, "
        "and objective-specific polls. Explicit naive baselines are retained "
        "only in the comparative Vegas baseline suite; KNN remains a ballot member.\n\n"
        + (
            "This is a leakage rehearsal: the supplied roster includes the evaluation season in its training data. "
            "These outputs are not valid holdout results and are for pipeline dry-run purposes only.\n\n"
            if leakage_rehearsal else
            "The roster was trained through the prior season, so the evaluation season is held out for these examples.\n\n"
        )
        + "All market-bearing F7/F8 models are excluded from predictions and polls; Vegas data is used only "
        "for the separate comparison suite. "
        "Explicit naive baselines remain restricted to the comparative Vegas baseline suite; KNN remains a ballot member.\n\n"
        f"Source inventory: `{inventory}`\n\n"
        + f"Inventory SHA256: `{sha256_file(inventory)}`\n"
    )
    (output / "README.md").write_text(
        readme_text,
        encoding="utf-8",
    )
    available_objectives = [
        objective for objective in OBJECTIVES
        if (output / "sunday_recaps" / objective).exists()
    ]
    poll_model_count_by_objective: dict[str, int] = {}
    for objective in available_objectives:
        poll_inventory_path = output / "sunday_recaps" / objective / "published_poll_model_inventory.csv"
        if poll_inventory_path.exists():
            poll_model_count_by_objective[objective] = int(len(pd.read_csv(poll_inventory_path)))
    manifest = {
        "season": args.season,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrospective": True,
        "performance_claim_allowed": not leakage_rehearsal,
        "leakage_rehearsal": leakage_rehearsal,
        "leakage_warning": (
            "Training includes the evaluation season; do not interpret as a valid holdout result."
            if leakage_rehearsal else None
        ),
        "compact_output": not args.expanded_output,
        "training_seasons": training_seasons,
        "holdout_season": int(args.season),
        "objectives": available_objectives,
        "model_count": int(len(pd.read_csv(inventory))),
        "poll_prediction_model_count": int(sum(poll_model_count_by_objective.values())),
        "poll_prediction_model_count_by_objective": poll_model_count_by_objective,
        "comparative_baseline_families": sorted(COMPARATIVE_BASELINE_FAMILIES),
        "excluded_prediction_feature_configs": sorted(INVALID_PREDICTION_FEATURE_CONFIGS),
        "published_poll_excluded_models": sorted(
            inventory_frame.loc[_is_poll_excluded_model(inventory_frame), "model_id"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "inventory": str(inventory),
        "inventory_sha256": sha256_file(inventory),
        "ranking": str(ranking),
        "ranking_sha256": sha256_file(ranking),
        "schedule": str(schedule),
        "schedule_sha256": sha256_file(schedule),
        "outputs": {
            "weekly_predictions": str(output / "weekly_predictions"),
            "sunday_recaps": str(output / "sunday_recaps"),
        },
    }
    (output / "retrospective_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
