"""Postgame grading tables for Sunday publication recaps."""

from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .team_labels import format_team_with_ap_rank


def build_retrospective_consensus(
    prediction_root: str | Path, games_path: str | Path, *, season: int, objective: str
) -> pd.DataFrame:
    """Combine finalist prediction files and attach authoritative final scores."""
    root = Path(prediction_root)
    frames = []
    objective_root = root / objective / "final_artifacts"
    for path in sorted(objective_root.glob("*/*/artifacts/predictions/predictions.csv")):
        frame = pd.read_csv(path)
        frame = frame[pd.to_numeric(frame["keys_season"], errors="coerce").eq(int(season))].copy()
        if frame.empty:
            continue
        relative = path.relative_to(objective_root).parts
        frame["model_id"] = "/".join((objective, relative[0], relative[1]))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {season} {objective!r} finalist predictions found below {objective_root}.")
    long = pd.concat(frames, ignore_index=True)
    long["next_game_id"] = pd.to_numeric(long["next_game_id"], errors="coerce").astype("Int64")
    keys = ["next_game_id", "keys_season", "next_week", "keys_team_home", "keys_team_away"]
    consensus = (
        long.groupby(keys, dropna=False, as_index=False)
        .agg(
            pred_home_margin=("pred_margin", "mean"),
            pred_home_win_probability=("pred_proba_home_win", "mean"),
            model_count=("model_id", "nunique"),
            model_agreement=("pred_pick_home", lambda value: max(float(pd.Series(value).mean()), 1.0 - float(pd.Series(value).mean()))),
            market_spread_close=("market_spread_close", "median"),
            market_over_under=("market_over_under", "median"),
        )
    )
    games = pd.read_parquet(games_path) if str(games_path).endswith(".parquet") else pd.read_csv(games_path)
    games = games[pd.to_numeric(games["season"], errors="coerce").eq(int(season))].copy()
    actual = games[["id", "week", "start_date", "home_team", "away_team", "home_points", "away_points"]].rename(columns={"id": "next_game_id"})
    actual["next_game_id"] = pd.to_numeric(actual["next_game_id"], errors="coerce").astype("Int64")
    out = consensus.merge(actual, on="next_game_id", how="inner", validate="one_to_one")
    out = out[out["home_points"].notna() & out["away_points"].notna()].copy()
    return grade_postgame_predictions(out)


def load_individual_model_predictions(
    prediction_root: str | Path, games_path: str | Path, *, season: int, objective: str
) -> pd.DataFrame:
    """Load and grade every finalist separately, preserving its stable model ID."""
    objective_root = Path(prediction_root) / objective / "final_artifacts"
    frames = []
    for path in sorted(objective_root.glob("*/*/artifacts/predictions/predictions.csv")):
        frame = pd.read_csv(path)
        frame = frame[pd.to_numeric(frame["keys_season"], errors="coerce").eq(int(season))].copy()
        if frame.empty:
            continue
        family, model = path.relative_to(objective_root).parts[:2]
        frame["model_id"] = f"{objective}/{family}/{model}"
        frame["model_slug"] = f"{family}_{model}"
        frame = frame.rename(columns={
            "pred_margin": "pred_home_margin",
            "pred_proba_home_win": "pred_home_win_probability",
        })
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {objective} finalist predictions found under {objective_root}.")
    long = pd.concat(frames, ignore_index=True)
    long["next_game_id"] = pd.to_numeric(long["next_game_id"], errors="coerce").astype("Int64")
    games = pd.read_parquet(games_path) if str(games_path).endswith(".parquet") else pd.read_csv(games_path)
    actual = games[pd.to_numeric(games["season"], errors="coerce").eq(int(season))][
        ["id", "week", "start_date", "home_team", "away_team", "home_points", "away_points"]
    ].rename(columns={"id": "next_game_id"})
    actual["next_game_id"] = pd.to_numeric(actual["next_game_id"], errors="coerce").astype("Int64")
    out = long.merge(actual, on="next_game_id", how="inner", validate="many_to_one")
    out = out[out["home_points"].notna() & out["away_points"].notna()].copy()
    out["model_count"] = 1
    out["model_agreement"] = 1.0
    return grade_postgame_predictions(out)


def grade_postgame_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Grade straight-up and closing-spread decisions without counting pushes."""
    out = frame.copy()
    out["actual_home_margin"] = pd.to_numeric(out["home_points"], errors="coerce") - pd.to_numeric(out["away_points"], errors="coerce")
    out["pred_winner"] = np.where(out["pred_home_margin"] >= 0, out["home_team"], out["away_team"])
    out["actual_winner"] = np.where(out["actual_home_margin"] > 0, out["home_team"], out["away_team"])
    out["actual_home_win"] = out["actual_home_margin"].gt(0).astype(float)
    out["su_correct"] = out["pred_winner"].eq(out["actual_winner"])
    out["margin_absolute_error"] = (out["pred_home_margin"] - out["actual_home_margin"]).abs()

    spread = pd.to_numeric(out["market_spread_close"], errors="coerce")
    model_edge_home = out["pred_home_margin"] + spread
    actual_ats_home = out["actual_home_margin"] + spread
    out["ats_pick"] = np.where(model_edge_home > 0, out["home_team"], out["away_team"])
    out["ats_edge"] = model_edge_home.abs()
    out["ats_result"] = pd.NA
    available = spread.notna() & model_edge_home.ne(0)
    out.loc[available & actual_ats_home.eq(0), "ats_result"] = "push"
    out.loc[available & actual_ats_home.ne(0), "ats_result"] = np.where(
        np.sign(model_edge_home[available & actual_ats_home.ne(0)])
        == np.sign(actual_ats_home[available & actual_ats_home.ne(0)]),
        "win", "loss",
    )
    out.loc[spread.isna(), "ats_pick"] = pd.NA
    out["projected_home_points"] = (out["market_over_under"] + out["pred_home_margin"]) / 2.0
    out["projected_away_points"] = (out["market_over_under"] - out["pred_home_margin"]) / 2.0
    sort_columns = [column for column in ["week", "start_date", "next_game_id"] if column in out]
    return out.sort_values(sort_columns, ignore_index=True) if sort_columns else out.reset_index(drop=True)


def weekly_recap_metrics(games: pd.DataFrame) -> dict:
    ats = games[games["ats_result"].isin(["win", "loss"])]
    probability = pd.to_numeric(games.get("pred_home_win_probability"), errors="coerce") if "pred_home_win_probability" in games else pd.Series(dtype=float)
    brier = ((probability - games.loc[probability.index, "actual_home_win"]) ** 2).mean() if len(probability) and probability.notna().any() else None
    return {
        "games": int(len(games)),
        "su_wins": int(games["su_correct"].sum()),
        "su_losses": int((~games["su_correct"]).sum()),
        "su_accuracy": float(games["su_correct"].mean()) if len(games) else None,
        "ats_wins": int(games["ats_result"].eq("win").sum()),
        "ats_losses": int(games["ats_result"].eq("loss").sum()),
        "ats_pushes": int(games["ats_result"].eq("push").sum()),
        "ats_ungraded": int(games["ats_result"].isna().sum()),
        "ats_accuracy_excluding_pushes": float(ats["ats_result"].eq("win").mean()) if len(ats) else None,
        "margin_mae": float(games["margin_absolute_error"].mean()) if len(games) else None,
        "brier_score": float(brier) if pd.notna(brier) else None,
        "models": int(games["model_count"].max()) if len(games) and "model_count" in games else 0,
    }


def vegas_recap_metrics(games: pd.DataFrame) -> dict:
    """Grade the closing spread as a home-margin prediction when available."""
    if "market_spread_close" not in games or "actual_home_margin" not in games:
        return {
            "vegas_games": 0,
            "vegas_su_wins": 0,
            "vegas_su_losses": 0,
            "vegas_su_accuracy": None,
            "vegas_margin_mae": None,
        }
    spread = pd.to_numeric(games["market_spread_close"], errors="coerce")
    actual = pd.to_numeric(games["actual_home_margin"], errors="coerce")
    valid = spread.notna() & actual.notna()
    if not valid.any():
        return {
            "vegas_games": 0,
            "vegas_su_wins": 0,
            "vegas_su_losses": 0,
            "vegas_su_accuracy": None,
            "vegas_margin_mae": None,
        }
    vegas_home_margin = -spread.loc[valid]
    actual_home_margin = actual.loc[valid]
    correct = vegas_home_margin.gt(0).eq(actual_home_margin.gt(0))
    return {
        "vegas_games": int(valid.sum()),
        "vegas_su_wins": int(correct.sum()),
        "vegas_su_losses": int((~correct).sum()),
        "vegas_su_accuracy": float(correct.mean()),
        "vegas_margin_mae": float((vegas_home_margin - actual_home_margin).abs().mean()),
    }


def model_vegas_correctness_matrix(games: pd.DataFrame) -> pd.DataFrame:
    """Count Model-correct/incorrect against Vegas-correct/incorrect outcomes."""
    decisions = _model_vegas_decisions(games)
    model_correct = decisions["pred_winner"].eq(decisions["actual_winner"])
    vegas_correct = decisions["vegas_winner"].eq(decisions["actual_winner"])
    counts = pd.crosstab(
        pd.Categorical(
            np.where(model_correct, "Model correct", "Model wrong"),
            categories=["Model correct", "Model wrong"],
        ),
        pd.Categorical(
            np.where(vegas_correct, "Vegas correct", "Vegas wrong"),
            categories=["Vegas correct", "Vegas wrong"],
        ),
        dropna=False,
    )
    counts.index.name = "model_outcome"
    counts.columns.name = None
    return counts.reset_index()


def model_chalk_upset_matrix(games: pd.DataFrame) -> pd.DataFrame:
    """Count Model chalk/upset calls against the realized chalk/upset result."""
    decisions = _model_vegas_decisions(games)
    model_upset = decisions["pred_winner"].ne(decisions["vegas_winner"])
    actual_upset = decisions["actual_winner"].ne(decisions["vegas_winner"])
    counts = pd.crosstab(
        pd.Categorical(
            np.where(model_upset, "Model picks upset", "Model picks chalk"),
            categories=["Model picks chalk", "Model picks upset"],
        ),
        pd.Categorical(
            np.where(actual_upset, "Actual upset", "Actual chalk"),
            categories=["Actual chalk", "Actual upset"],
        ),
        dropna=False,
    )
    counts.index.name = "model_pick"
    counts.columns.name = None
    return counts.reset_index()


def write_model_vegas_confusion_artifacts(
    games: pd.DataFrame,
    output_dir: str | Path,
    *,
    season: int,
    roster_label: str,
    dpi: int = 180,
) -> dict[str, Path]:
    """Write the two season-level Model/Vegas confusion matrices as CSV/PNG/SVG."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = (
        (
            "model_vs_vegas_correctness_confusion_matrix",
            model_vegas_correctness_matrix(games),
            "model_outcome",
            "Model correctness × Vegas correctness",
            "Vegas outcome",
            "Model outcome",
        ),
        (
            "model_chalk_upset_vs_actual_confusion_matrix",
            model_chalk_upset_matrix(games),
            "model_pick",
            "Model chalk/upset call × actual result",
            "Actual result relative to closing-line favorite",
            "Model pick relative to closing-line favorite",
        ),
    )
    written: dict[str, Path] = {}
    for stem, table, row_column, title, xlabel, ylabel in specs:
        csv_path = output / f"{stem}.csv"
        table.to_csv(csv_path, index=False)
        png_path = output / f"{stem}.png"
        _plot_count_matrix(
            table,
            row_column=row_column,
            path=png_path,
            title=f"{season} {roster_label}\n{title}",
            xlabel=xlabel,
            ylabel=ylabel,
            dpi=dpi,
        )
        written[stem] = csv_path
    return written


def _model_vegas_decisions(games: pd.DataFrame) -> pd.DataFrame:
    required = {
        "home_team",
        "away_team",
        "pred_winner",
        "actual_winner",
        "market_spread_close",
    }
    missing = sorted(required - set(games.columns))
    if missing:
        raise ValueError(f"Model/Vegas confusion matrices require columns: {missing}")
    out = games.copy()
    spread = pd.to_numeric(out["market_spread_close"], errors="coerce")
    valid = (
        spread.notna()
        & out["home_team"].notna()
        & out["away_team"].notna()
        & out["pred_winner"].notna()
        & out["actual_winner"].notna()
    )
    out = out.loc[valid, ["home_team", "away_team", "pred_winner", "actual_winner"]].copy()
    # The project convention converts the home-team spread to an implied home
    # margin by negating it; zero therefore follows winner_from_margin's away side.
    out["vegas_winner"] = np.where(
        spread.loc[valid].lt(0), out["home_team"], out["away_team"]
    )
    return out.reset_index(drop=True)


def _plot_count_matrix(
    table: pd.DataFrame,
    *,
    row_column: str,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    dpi: int,
) -> Path:
    rows = table[row_column].astype(str).tolist()
    columns = [column for column in table.columns if column != row_column]
    values = table[columns].to_numpy(dtype=int)
    total = int(values.sum())
    fig, axis = plt.subplots(figsize=(8.2, 6.3))
    image = axis.imshow(values, cmap="Blues", vmin=0)
    threshold = float(values.max()) / 2.0 if values.size else 0.0
    for (row, column), count in np.ndenumerate(values):
        share = count / total if total else 0.0
        axis.text(
            column,
            row,
            f"{count:,}\n({share:.1%})",
            ha="center",
            va="center",
            fontsize=15,
            weight="bold",
            color="white" if count > threshold else "#17263C",
        )
    axis.set_xticks(range(len(columns)), columns)
    axis.set_yticks(range(len(rows)), rows)
    axis.set_xlabel(xlabel, labelpad=12)
    axis.set_ylabel(ylabel, labelpad=12)
    axis.set_title(title, fontsize=15, weight="bold", pad=16)
    fig.colorbar(image, ax=axis, label="Games")
    fig.text(
        0.5,
        0.015,
        f"N = {total:,} games with a captured closing spread. Percentages are shares of all graded games.",
        ha="center",
        fontsize=9,
        color="#555B63",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def add_cumulative_weekly_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    """Add season-to-date objective and Vegas metrics for weekly comparison plots."""
    if comparison.empty:
        return comparison.copy()
    out = comparison.sort_values(["objective", "week"], kind="mergesort").copy()
    frames = []
    for _, frame in out.groupby("objective", sort=False):
        frame = frame.copy()
        games = pd.to_numeric(frame.get("games"), errors="coerce").fillna(0)
        su_wins = pd.to_numeric(frame.get("su_wins"), errors="coerce").fillna(0)
        su_losses = pd.to_numeric(frame.get("su_losses"), errors="coerce").fillna(0)
        su_total = su_wins.cumsum() + su_losses.cumsum()
        frame["cumulative_su_accuracy"] = (su_wins.cumsum() / su_total.replace(0, np.nan)).astype(float)
        ats_wins = pd.to_numeric(frame.get("ats_wins"), errors="coerce").fillna(0)
        ats_losses = pd.to_numeric(frame.get("ats_losses"), errors="coerce").fillna(0)
        ats_total = ats_wins.cumsum() + ats_losses.cumsum()
        frame["cumulative_ats_accuracy_excluding_pushes"] = (
            ats_wins.cumsum() / ats_total.replace(0, np.nan)
        ).astype(float)
        margin_mae = pd.to_numeric(frame.get("margin_mae"), errors="coerce")
        frame["cumulative_margin_mae"] = (
            (margin_mae * games).cumsum() / games.cumsum().replace(0, np.nan)
        ).astype(float)
        if "vegas_games" in frame:
            vegas_games = pd.to_numeric(frame.get("vegas_games"), errors="coerce").fillna(0)
            vegas_wins = pd.to_numeric(frame.get("vegas_su_wins"), errors="coerce").fillna(0)
            vegas_losses = pd.to_numeric(frame.get("vegas_su_losses"), errors="coerce").fillna(0)
            vegas_total = vegas_wins.cumsum() + vegas_losses.cumsum()
            frame["vegas_cumulative_su_accuracy"] = (
                vegas_wins.cumsum() / vegas_total.replace(0, np.nan)
            ).astype(float)
            vegas_mae = pd.to_numeric(frame.get("vegas_margin_mae"), errors="coerce")
            frame["vegas_cumulative_margin_mae"] = (
                (vegas_mae * vegas_games).cumsum() / vegas_games.cumsum().replace(0, np.nan)
            ).astype(float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_sunday_recap_table(
    games: pd.DataFrame, path: str | Path, *, season: int, week: int,
    objective: str = "winner", model_label: str | None = None, dpi: int = 180,
    warning_label: str | None = None,
    season_to_date_games: pd.DataFrame | None = None,
) -> Path:
    """Render a blog-ready results table with SU and ATS grading colors."""
    games = games.copy().reset_index(drop=True)
    metrics = weekly_recap_metrics(games)
    rows = []
    for _, game in games.iterrows():
        away_team = format_team_with_ap_rank(game, "away")
        home_team = format_team_with_ap_rank(game, "home")
        projected = _projected_score(game, away_team=away_team, home_team=home_team)
        final = f"{away_team} {int(game['away_points'])}–{int(game['home_points'])} {home_team}"
        rows.append({
            "Matchup": f"{away_team} at {home_team}",
            "TDNet projected": projected,
            "Final": final,
            "SU": "✓" if game["su_correct"] else "✗",
            "Closing line": _format_home_line(game),
            "TDNet ATS side": _format_ats_pick(game),
            "ATS": {"win": "✓", "loss": "✗", "push": "P"}.get(game["ats_result"], "—"),
        })
    table = pd.DataFrame(rows)
    fig_height = max(4.0, 0.36 * len(table) + 2.25)
    fig, axis = plt.subplots(figsize=(16, fig_height))
    fig.patch.set_facecolor("#F7F4ED")
    axis.axis("off")
    ats_rate = metrics["ats_accuracy_excluding_pushes"]
    weekly_subtitle = (
        f"Week: SU {metrics['su_wins']}–{metrics['su_losses']} ({metrics['su_accuracy']:.1%})"
        f"     •     ATS  {metrics['ats_wins']}–{metrics['ats_losses']}–{metrics['ats_pushes']}"
        + (f"  ({ats_rate:.1%}, pushes excluded)" if ats_rate is not None else "")
        + f"     •     Margin MAE  {metrics['margin_mae']:.1f} pts"
    )
    season_subtitle = ""
    if season_to_date_games is not None:
        season_metrics = weekly_recap_metrics(season_to_date_games)
        season_ats_rate = season_metrics["ats_accuracy_excluding_pushes"]
        season_subtitle = (
            f"\nSeason so far: SU {season_metrics['su_wins']}–{season_metrics['su_losses']} "
            f"({season_metrics['su_accuracy']:.1%})"
            f"     •     ATS {season_metrics['ats_wins']}–{season_metrics['ats_losses']}–"
            f"{season_metrics['ats_pushes']}"
            + (f" ({season_ats_rate:.1%}, pushes excluded)" if season_ats_rate is not None else "")
        )
    objective_label = model_label or ("Winner-trained models" if objective == "winner" else "MAE-trained margin models")
    axis.set_title(
        f"{season} Week {week}: TDNet Sunday Scorecard — {objective_label}\n"
        f"{weekly_subtitle}{season_subtitle}",
        fontsize=17, weight="bold", pad=22, color="#17263C",
    )
    plotted = axis.table(
        cellText=table.values, colLabels=table.columns, loc="center", cellLoc="left", colLoc="left",
        colWidths=[0.22, 0.22, 0.22, 0.055, 0.12, 0.12, 0.055],
        bbox=[0.0, 0.045, 1.0, 0.86],
    )
    plotted.auto_set_font_size(False)
    plotted.set_fontsize(8.2)
    plotted.scale(1, 1.28)
    su_col = table.columns.get_loc("SU")
    ats_col = table.columns.get_loc("ATS")
    for (row, column), cell in plotted.get_celld().items():
        cell.set_edgecolor("#D4D7DB")
        if row == 0:
            cell.set_facecolor("#22324A")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#EEF2F5")
            if column in (su_col, ats_col):
                value = table.iloc[row - 1, column]
                color = {"✓": "#DCEFE1", "✗": "#F6DDDA", "P": "#FFF0C7", "—": "#E8E8E8"}[value]
                cell.set_facecolor(color)
                cell.set_text_props(weight="bold", ha="center", color="#183321" if value == "✓" else "#702820")
    footer = "Projected scores combine TDNet's predicted margin with the captured closing total. ATS uses the captured closing home-team spread; pushes and missing lines are excluded from ATS accuracy."
    fig.text(0.5, 0.015, footer, ha="center", fontsize=8.5, color="#555B63")
    if warning_label:
        fig.text(0.5, 0.002, warning_label, ha="center", fontsize=8.2, weight="bold", color="#9F3A38")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def build_prediction_set_recaps(
    prediction_root: str | Path, games_path: str | Path, output_root: str | Path,
    *, season: int, objective: str, retrospective_warning: bool = False,
) -> dict[str, dict]:
    """Build the all-model weekly scorecard and ranking sidecar.

    The selection is intentionally explicit: models are ranked by full-season
    Brier score.  For a historical layout rehearsal this is useful, but it must
    not be represented as a prospective model selection.
    """
    long = load_individual_model_predictions(
        prediction_root, games_path, season=season, objective=objective
    )
    leaderboard_rows = []
    for model_id, model_games in long.groupby("model_id", sort=True):
        leaderboard_rows.append({"model_id": model_id, **weekly_recap_metrics(model_games)})
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["brier_score", "su_accuracy", "margin_mae"],
        ascending=[True, False, True], ignore_index=True,
    )
    leaderboard.insert(0, "brier_rank", np.arange(1, len(leaderboard) + 1))

    output = Path(output_root) / "prediction_sets"
    output.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output / "legacy_model_brier_ranking.csv", index=False)
    selections = {"all_models": leaderboard["model_id"].tolist()}
    warning = (
        "LAYOUT PREVIEW ONLY — legacy checkpoints overlap the 2025 evaluation season"
        if retrospective_warning else None
    )
    results = {}
    for selection_name, model_ids in selections.items():
        selected = long[long["model_id"].isin(model_ids)].copy()
        graded = _aggregate_graded_models(selected)
        selection_output = output / selection_name
        selection_output.mkdir(parents=True, exist_ok=True)
        graded.to_csv(selection_output / "all_graded_games.csv", index=False)
        metrics = cumulative_model_metrics(graded)
        metrics.to_csv(selection_output / "weekly_and_cumulative_metrics.csv", index=False)
        label = f"All {len(model_ids)} legacy models"
        plot_model_cumulative_track(
            metrics, selection_output / "cumulative_performance.png",
            season=season, model_label=f"{objective.title()} · {label}",
        )
        weekly_rows = []
        for week, week_games in graded.groupby("week", sort=True):
            week = int(week)
            week_dir = selection_output / f"week_{week:02d}"
            week_dir.mkdir(parents=True, exist_ok=True)
            week_games.to_csv(week_dir / "prediction_vs_actual.csv", index=False)
            figure = plot_sunday_recap_table(
                week_games, week_dir / "prediction_vs_actual.png", season=season,
                week=week, objective=objective, model_label=label,
                warning_label=warning,
            )
            # Put the three publishable views together in the ordinary week folder too.
            ordinary_week = Path(output_root) / f"week_{week:02d}"
            ordinary_week.mkdir(parents=True, exist_ok=True)
            shutil.copy2(figure, ordinary_week / f"{selection_name}_prediction_vs_actual.png")
            row = {"season": season, "week": week, "objective": objective,
                   "prediction_set": selection_name, **weekly_recap_metrics(week_games)}
            weekly_rows.append(row)
            (week_dir / "summary.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        weekly_summary = pd.DataFrame(weekly_rows)
        weekly_summary.to_csv(selection_output / "weekly_summary.csv", index=False)
        metadata = {
            "season": season, "objective": objective, "prediction_set": selection_name,
            "selection_metric": "full-season 2025 Brier score", "model_ids": model_ids,
            "retrospective_layout_preview": bool(retrospective_warning),
            "publication_warning": warning,
        }
        (selection_output / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        results[selection_name] = {"games": graded, "weekly_summary": weekly_summary, "metadata": metadata}
    return results


def _aggregate_graded_models(games: pd.DataFrame) -> pd.DataFrame:
    """Average a selected model set into one prediction per game and re-grade it."""
    keys = ["next_game_id", "keys_season", "next_week", "keys_team_home", "keys_team_away"]
    first_columns = [
        "week", "start_date", "home_team", "away_team", "home_points", "away_points",
        "market_spread_close", "market_over_under",
    ]
    aggregation = {
        "pred_home_margin": ("pred_home_margin", "mean"),
        "pred_home_win_probability": ("pred_home_win_probability", "mean"),
        "model_count": ("model_id", "nunique"),
        "model_agreement": ("pred_pick_home", lambda value: max(float(pd.Series(value).mean()), 1.0 - float(pd.Series(value).mean()))),
    }
    aggregation.update({column: (column, "first") for column in first_columns})
    combined = games.groupby(keys, dropna=False, as_index=False).agg(**aggregation)
    return grade_postgame_predictions(combined)


def build_season_sunday_recaps(
    prediction_root: str | Path, games_path: str | Path, output_root: str | Path,
    *, season: int, objective: str,
) -> dict:
    graded = build_retrospective_consensus(prediction_root, games_path, season=season, objective=objective)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    figures = []
    season_to_date = []
    for week, games in graded.groupby("week", sort=True):
        week = int(week)
        directory = output / f"week_{week:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        games.to_csv(directory / "prediction_vs_actual.csv", index=False)
        season_to_date.append(games)
        figure = plot_sunday_recap_table(
            games, directory / "prediction_vs_actual.png", season=season, week=week,
            objective=objective,
            season_to_date_games=pd.concat(season_to_date, ignore_index=True),
        )
        shutil.copy2(figure, directory / "consensus_prediction_vs_actual.png")
        metric = {"season": season, "week": week, "objective": objective, **weekly_recap_metrics(games)}
        summaries.append(metric)
        figures.append(str(figure))
        (directory / "summary.json").write_text(json.dumps(metric, indent=2) + "\n", encoding="utf-8")
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "weekly_summary.csv", index=False)
    graded.to_csv(output / "all_graded_games.csv", index=False)
    manifest = {
        "season": season, "objective": objective, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrospective": True, "prediction_root": str(Path(prediction_root).resolve()),
        "games_path": str(Path(games_path).resolve()), "weeks": len(summaries),
        "games": len(graded), "figures": figures,
        "note": "Retrospective reconstruction from final-artifact predictions; not a timestamped prospective 2025 release.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"games": graded, "weekly_summary": summary, "manifest": manifest}


def cumulative_model_metrics(games: pd.DataFrame) -> pd.DataFrame:
    """Return weekly and through-week records for one already graded model."""
    rows = []
    running = []
    for week, frame in games.groupby("week", sort=True):
        running.append(frame)
        weekly = weekly_recap_metrics(frame)
        cumulative = weekly_recap_metrics(pd.concat(running, ignore_index=True))
        weekly_vegas = vegas_recap_metrics(frame)
        cumulative_vegas = vegas_recap_metrics(pd.concat(running, ignore_index=True))
        rows.append({
            "week": int(week),
            **{f"weekly_{key}": value for key, value in weekly.items()},
            **{f"cumulative_{key}": value for key, value in cumulative.items()},
            **{f"weekly_{key}": value for key, value in weekly_vegas.items()},
            **{f"cumulative_{key}": value for key, value in cumulative_vegas.items()},
            "weekly_vegas_ats_accuracy_excluding_pushes": 0.5,
            "cumulative_vegas_ats_accuracy_excluding_pushes": 0.5,
        })
    return pd.DataFrame(rows)


def all_model_cumulative_metrics(games: pd.DataFrame) -> pd.DataFrame:
    """Return weekly and through-week records for every graded model."""
    if games.empty:
        return pd.DataFrame()
    model_column = "model_id" if "model_id" in games else "model_name"
    rows = []
    for model_id, frame in games.groupby(model_column, sort=True):
        metrics = cumulative_model_metrics(frame.sort_values(["week", "game_id"], kind="mergesort"))
        if metrics.empty:
            continue
        metrics.insert(0, "model_id", str(model_id))
        if "model_family" in frame:
            metrics.insert(1, "model_family", str(frame["model_family"].dropna().iloc[0]) if frame["model_family"].notna().any() else "")
        if "feature_config" in frame:
            metrics.insert(2, "feature_config", str(frame["feature_config"].dropna().iloc[0]) if frame["feature_config"].notna().any() else "")
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["model_id", "week"], kind="mergesort").reset_index(drop=True)


def plot_all_model_cumulative_performance(
    metrics: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    title: str,
    dpi: int = 180,
) -> Path:
    """Plot cumulative SU, ATS, and margin MAE tracks for all models."""
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 11), sharex=True)
    specs = [
        (
            "cumulative_su_accuracy",
            "cumulative_vegas_su_accuracy",
            "Cumulative straight-up accuracy",
            "Vegas SU",
            "below",
        ),
        (
            "cumulative_ats_accuracy_excluding_pushes",
            "cumulative_vegas_ats_accuracy_excluding_pushes",
            "Cumulative ATS accuracy",
            "ATS break-even",
            "below",
        ),
        (
            "cumulative_margin_mae",
            "cumulative_vegas_margin_mae",
            "Cumulative margin MAE (lower is better)",
            "Vegas MAE",
            "above",
        ),
    ]
    model_count = metrics["model_id"].nunique() if "model_id" in metrics else 0
    for axis, (column, reference_column, ylabel, reference_label, worse_side) in zip(axes, specs):
        reference = (
            metrics[["week", reference_column]]
            .dropna()
            .drop_duplicates("week")
            .sort_values("week", kind="mergesort")
            if reference_column in metrics
            else pd.DataFrame()
        )
        for model_id, frame in metrics.groupby("model_id", sort=True):
            axis.plot(frame["week"], frame[column], lw=1.15, alpha=0.52)
        if not reference.empty:
            axis.plot(
                reference["week"],
                reference[reference_column],
                color="#23272F",
                lw=2.3,
                ls="--",
                label=reference_label,
            )
            y_min, y_max = axis.get_ylim()
            if worse_side == "below":
                axis.fill_between(
                    reference["week"],
                    y_min,
                    reference[reference_column],
                    color="#A44A3F",
                    alpha=0.10,
                    label="Worse than reference",
                )
            else:
                axis.fill_between(
                    reference["week"],
                    reference[reference_column],
                    y_max,
                    color="#A44A3F",
                    alpha=0.10,
                    label="Worse than reference",
                )
            axis.set_ylim(y_min, y_max)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), frameon=False, fontsize=8.5, loc="lower right")
    axes[-1].set_xlabel("Week")
    axes[-1].set_xticks(sorted(metrics["week"].dropna().astype(int).unique()))
    fig.suptitle(f"{season} {title}: All Model Cumulative Performance ({model_count} models)", fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.01,
        "Each line is one roster model; table output contains exact per-model values.",
        ha="center",
        fontsize=9,
        color="#555B63",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_cumulative_track(metrics: pd.DataFrame, path: str | Path, *, season: int, model_label: str, dpi: int = 180) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True)
    specs = [
        ("cumulative_su_accuracy", "Cumulative straight-up accuracy", 0.5),
        ("cumulative_ats_accuracy_excluding_pushes", "Cumulative ATS accuracy", 0.5),
        ("cumulative_margin_mae", "Cumulative margin MAE (lower is better)", None),
    ]
    for axis, (column, label, baseline) in zip(axes, specs):
        axis.plot(metrics["week"], metrics[column], color="#274C77", marker="o", lw=2.3)
        if baseline is not None:
            axis.axhline(baseline, color="#777", ls="--", lw=1)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Week")
    axes[-1].set_xticks(metrics["week"])
    fig.suptitle(f"{season} Running Performance — {model_label}", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def build_individual_model_recaps(
    prediction_root: str | Path, games_path: str | Path, output_root: str | Path,
    *, season: int, objective: str, workers: int = 1,
) -> pd.DataFrame:
    """Generate weekly scorecards and cumulative tracks for every finalist."""
    long = load_individual_model_predictions(
        prediction_root, games_path, season=season, objective=objective
    )
    output = Path(output_root) / "individual_models"
    tasks = [(model_id, games.copy(), output, season, objective) for model_id, games in long.groupby("model_id", sort=True)]
    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            leaderboard = list(executor.map(_build_one_model_recaps, tasks))
    else:
        leaderboard = [_build_one_model_recaps(task) for task in tasks]
    leaderboard = pd.DataFrame(leaderboard).sort_values(
        ["su_accuracy", "ats_accuracy_excluding_pushes", "margin_mae"],
        ascending=[False, False, True], ignore_index=True,
    )
    leaderboard.to_csv(output / "season_model_leaderboard.csv", index=False)
    build_model_running_leaderboard(long, output, season=season, objective=objective)
    publish_season_champion(
        leaderboard, Path(output_root), objective=objective, season=season
    )
    return leaderboard


def _build_one_model_recaps(task) -> dict:
    model_id, games, output, season, objective = task
    slug = str(games["model_slug"].iloc[0])
    model_label = model_id.replace("/", " · ")
    directory = Path(output) / slug
    directory.mkdir(parents=True, exist_ok=True)
    games.to_csv(directory / "all_graded_games.csv", index=False)
    metrics = cumulative_model_metrics(games)
    metrics.insert(0, "model_id", model_id)
    metrics.to_csv(directory / "weekly_and_cumulative_metrics.csv", index=False)
    if not (directory / "cumulative_performance.png").exists():
        plot_model_cumulative_track(
            metrics, directory / "cumulative_performance.png", season=season, model_label=model_label
        )
    for week, week_games in games.groupby("week", sort=True):
        week_dir = directory / f"week_{int(week):02d}"
        week_dir.mkdir(parents=True, exist_ok=True)
        week_games.to_csv(week_dir / "prediction_vs_actual.csv", index=False)
        if not (week_dir / "prediction_vs_actual.png").exists():
            plot_sunday_recap_table(
                week_games, week_dir / "prediction_vs_actual.png", season=season,
                week=int(week), objective=objective, model_label=model_label,
            )
    return {"season": season, "objective": objective, "model_id": model_id, "model_slug": slug, **weekly_recap_metrics(games)}


def publish_season_champion(
    leaderboard: pd.DataFrame, objective_output: Path, *, objective: str, season: int
) -> dict:
    """Expose one fixed retrospective champion beside every consensus figure."""
    if objective == "margin":
        ordered = leaderboard.sort_values(
            ["margin_mae", "su_accuracy", "brier_score"], ascending=[True, False, True]
        )
        rule = "lowest season margin MAE; SU accuracy then Brier score as tie-breakers"
    else:
        ordered = leaderboard.sort_values(
            ["su_accuracy", "brier_score", "margin_mae"], ascending=[False, True, True]
        )
        rule = "highest season straight-up accuracy; Brier score then margin MAE as tie-breakers"
    champion = ordered.iloc[0]
    ats_leader = leaderboard.sort_values(
        ["ats_accuracy_excluding_pushes", "su_accuracy"], ascending=[False, False]
    ).iloc[0]
    champion_root = objective_output / "individual_models" / str(champion["model_slug"])
    for week in range(1, 17):
        source = champion_root / f"week_{week:02d}" / "prediction_vs_actual.png"
        if source.exists():
            target_dir = objective_output / f"week_{week:02d}"
            shutil.copy2(source, target_dir / "season_champion_prediction_vs_actual.png")
    cumulative = champion_root / "cumulative_performance.png"
    if cumulative.exists():
        shutil.copy2(cumulative, objective_output / "season_champion_cumulative_performance.png")
    metadata = {
        "season": season,
        "objective": objective,
        "selection_scope": "full-season retrospective dry run",
        "selection_rule": rule,
        "champion_model_id": str(champion["model_id"]),
        "champion_metrics": {key: float(champion[key]) for key in ["su_accuracy", "ats_accuracy_excluding_pushes", "margin_mae", "brier_score"]},
        "ats_leader_model_id": str(ats_leader["model_id"]),
        "ats_leader_accuracy_excluding_pushes": float(ats_leader["ats_accuracy_excluding_pushes"]),
        "publication_warning": "The champion was selected after observing the complete 2025 season and is not a prospective 2025 claim.",
    }
    (objective_output / "season_champion.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def build_model_running_leaderboard(long: pd.DataFrame, output: Path, *, season: int, objective: str) -> None:
    rows = []
    for model_id, games in long.groupby("model_id", sort=True):
        metrics = cumulative_model_metrics(games)
        metrics.insert(0, "model_id", model_id)
        rows.append(metrics)
    track = pd.concat(rows, ignore_index=True)
    track.to_csv(output / "all_models_running_metrics.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    columns = [
        ("cumulative_su_accuracy", "Cumulative SU accuracy"),
        ("cumulative_ats_accuracy_excluding_pushes", "Cumulative ATS accuracy"),
        ("cumulative_margin_mae", "Cumulative margin MAE"),
    ]
    for axis, (column, label) in zip(axes, columns):
        for model_id, frame in track.groupby("model_id"):
            axis.plot(frame["week"], frame[column], lw=1.25, alpha=0.75, label=model_id.split("/", 1)[1])
        axis.set_ylabel(label)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, ncol=3, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[-1].set_xlabel("Week")
    axes[-1].set_xticks(sorted(track["week"].unique()))
    fig.suptitle(f"{season} {objective.title()} Models — Running Performance", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 0.82, 0.97])
    path = output / "all_models_running_performance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_objective_weekly_comparison(comparison: pd.DataFrame, path: str | Path, *, season: int, dpi: int = 180) -> Path:
    """Plot weekly and season-to-date SU, ATS, and margin performance."""
    comparison = add_cumulative_weekly_metrics(comparison)
    colors = {"winner": "#274C77", "margin": "#A44A3F", "balanced": "#5B7F45"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    specs = [
        (
            "su_accuracy",
            "cumulative_su_accuracy",
            "vegas_su_accuracy",
            "vegas_cumulative_su_accuracy",
            "Straight-up accuracy",
            lambda axis: axis.axhline(0.5, color="#777", ls=":", lw=1),
        ),
        (
            "ats_accuracy_excluding_pushes",
            "cumulative_ats_accuracy_excluding_pushes",
            None,
            None,
            "ATS accuracy (pushes excluded)",
            lambda axis: axis.axhline(0.5, color="#777", ls=":", lw=1),
        ),
        (
            "margin_mae",
            "cumulative_margin_mae",
            "vegas_margin_mae",
            "vegas_cumulative_margin_mae",
            "Margin MAE (points; lower is better)",
            lambda axis: None,
        ),
    ]
    for axis, (weekly_col, cumulative_col, vegas_weekly_col, vegas_cumulative_col, ylabel, reference) in zip(axes, specs):
        for objective, frame in comparison.groupby("objective"):
            frame = frame.sort_values("week")
            color = colors.get(objective)
            label = objective.title()
            axis.plot(
                frame["week"],
                frame[weekly_col],
                marker="o",
                lw=1.8,
                alpha=0.45,
                label=f"{label} weekly",
                color=color,
            )
            axis.plot(
                frame["week"],
                frame[cumulative_col],
                marker="",
                lw=2.4,
                ls="-",
                label=f"{label} cumulative",
                color=color,
            )
            if vegas_weekly_col and vegas_weekly_col in frame and frame[vegas_weekly_col].notna().any():
                axis.plot(
                    frame["week"],
                    frame[vegas_weekly_col],
                    marker="s",
                    lw=1.6,
                    alpha=0.45,
                    label="Vegas weekly",
                    color="#555B63",
                )
            if vegas_cumulative_col and vegas_cumulative_col in frame and frame[vegas_cumulative_col].notna().any():
                axis.plot(
                    frame["week"],
                    frame[vegas_cumulative_col],
                    marker="",
                    lw=2.2,
                    ls="--",
                    label="Vegas cumulative",
                    color="#23272F",
                )
        reference(axis)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), frameon=False, ncol=2, fontsize=8.5)
    axes[-1].set_xlabel("Week")
    axes[-1].set_xticks(sorted(comparison["week"].unique()))
    fig.suptitle(f"{season} TDNet Weekly vs Cumulative Objective Comparison", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _projected_score(game, *, away_team: str | None = None, home_team: str | None = None) -> str:
    if pd.isna(game["projected_away_points"]) or pd.isna(game["projected_home_points"]):
        return f"{game['pred_winner']} by {abs(game['pred_home_margin']):.1f}"
    away_team = away_team or str(game["away_team"])
    home_team = home_team or str(game["home_team"])
    return f"{away_team} {game['projected_away_points']:.0f}–{game['projected_home_points']:.0f} {home_team}"


def _format_home_line(game) -> str:
    spread = game["market_spread_close"]
    if pd.isna(spread):
        return "No line"
    return f"{game['home_team']} {float(spread):+g}"


def _format_ats_pick(game) -> str:
    if pd.isna(game["ats_pick"]):
        return "—"
    spread = float(game["market_spread_close"])
    team_spread = spread if game["ats_pick"] == game["home_team"] else -spread
    return f"{game['ats_pick']} {team_spread:+g}"
