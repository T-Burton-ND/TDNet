"""Generate the complete postgame publication package without new picks."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from gridiron_ml.td_run.poll_viz import plot_ballot_logo_grid

from .ats_bankroll import write_consensus_betting_artifacts
from .bundles import sha256_file
from .figure_theme import TDNET_COLORS, apply_tdnet_theme
from .model_calibration import write_cumulative_model_calibration_artifacts
from .poll_explainability import plot_top25_consensus_spread
from .poll_recaps import (
    aggregate_receiving_votes,
    format_receiving_votes,
    model_consensus_disagreement,
    plot_consensus_poll_table,
    plot_model_disagreement,
    plot_tdnet_vs_ap_poll,
)
from .polls import add_team_records
from .recaps import (
    grade_postgame_predictions,
    plot_sunday_recap_table,
    weekly_recap_metrics,
)
from .scientific_performance import write_scientific_cumulative_artifacts
from .scientific_weekly import (
    plot_power_rank_vs_projected_margin,
    plot_scientific_all_team_power_ranking,
    scientific_consensus_power_rankings,
    validate_scientific_ballots,
)
from .social_top10 import render_top10_social

EASTERN = ZoneInfo("America/New_York")


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _lean_layout_metadata(output: Path) -> dict[str, object]:
    main_pngs = list((output / "figures").glob("*.png"))
    scientific_pngs = list((output / "scientific").glob("*.png"))
    return {
        "name": "pregame-parity-lean-v1",
        "directories": ["blog", "figures", "metadata", "scientific", "tables"],
        "png_retention": {
            "figures": len(main_pngs),
            "scientific": len(scientific_pngs),
            "total": len(main_pngs) + len(scientific_pngs),
        },
        "excluded_png_categories": [
            "individual_model_scorecards",
            "frozen_top_n_duplicates",
            "confusion_matrix_diagnostics",
            "scientific_ranking_diagnostics",
        ],
    }


def refresh_postgame_manifest_file_index(output_root: str | Path) -> dict[str, object]:
    """Refresh hashes after a layout-only compaction without rerunning models."""
    output = Path(output_root)
    manifest_path = output / "full_postgame_manifest.json"
    sunday_manifest_path = output / "sunday_publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated_at = datetime.now(EASTERN).isoformat()
    manifest.update(
        {
            "schema": "tdnet-full-postgame-publication-v2",
            "layout_updated_at_eastern": updated_at,
            "artifact_layout": _lean_layout_metadata(output),
        }
    )
    manifest["files"] = {
        str(path.relative_to(output)): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path not in {manifest_path, sunday_manifest_path}
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sunday_manifest = json.loads(sunday_manifest_path.read_text(encoding="utf-8"))
    sunday_manifest.update(
        {
            "layout_updated_at_eastern": updated_at,
            "artifact_layout": manifest["artifact_layout"],
            "full_postgame_manifest": {
                "path": manifest_path.name,
                "sha256": sha256_file(manifest_path),
            },
        }
    )
    sunday_manifest["files"] = {
        str(path.relative_to(output)): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != sunday_manifest_path
    }
    sunday_manifest_path.write_text(
        json.dumps(sunday_manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def _prepare_margin_model_games(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    frame["start_date"] = frame.get("game_start_time_utc")
    frame["market_over_under"] = pd.to_numeric(
        frame.get("vegas_total_close_as_of_prediction"), errors="coerce"
    )
    frame["model_id"] = frame["model_name"].astype(str)
    frame["model_slug"] = frame["model_id"].map(_slug)
    frame["model_count"] = 1
    frame["model_agreement"] = 1.0
    return grade_postgame_predictions(frame)


def _aggregate_models(frame: pd.DataFrame) -> pd.DataFrame:
    first = [
        "season",
        "week",
        "game_id",
        "start_date",
        "home_team",
        "away_team",
        "home_points",
        "away_points",
        "market_spread_close",
        "market_over_under",
    ]
    aggregations = {
        "pred_home_margin": ("pred_home_margin", "mean"),
        "pred_home_win_probability": ("pred_home_win_probability", "mean"),
        "model_count": ("model_id", "nunique"),
        "model_agreement": (
            "pred_home_margin",
            lambda value: max(float(pd.Series(value).ge(0).mean()), float(pd.Series(value).lt(0).mean())),
        ),
    }
    aggregations.update({column: (column, "first") for column in first if column in frame})
    grouped = frame.groupby("game_id", as_index=False).agg(**aggregations)
    return grade_postgame_predictions(grouped)


def _model_scorecard(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, games in frame.groupby("model_id", sort=True):
        row = {
            "model_id": str(model_id),
            "model_family": str(games["model_family"].iloc[0]) if "model_family" in games else "",
            "frozen_roster_rank": int(pd.to_numeric(games.get("roster_rank"), errors="coerce").dropna().iloc[0])
            if "roster_rank" in games and pd.to_numeric(games["roster_rank"], errors="coerce").notna().any()
            else pd.NA,
            **weekly_recap_metrics(games),
        }
        rows.append(row)
    scorecard = pd.DataFrame(rows).sort_values(
        ["margin_mae", "su_accuracy", "ats_accuracy_excluding_pushes", "model_id"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    scorecard.insert(0, "postgame_margin_rank", range(1, len(scorecard) + 1))
    return scorecard


def _plot_model_scorecard(
    scorecard: pd.DataFrame,
    path: Path,
    *,
    season: int,
    completed_week: int,
    title_label: str,
) -> Path:
    frame = scorecard.copy()
    table = pd.DataFrame(
        {
            "Rank": frame.iloc[:, 0].astype(int),
            "Model": frame["model_id"].astype(str).str.replace("margin_", "", regex=False),
            "Family": frame.get("model_family", ""),
            "SU": frame["su_wins"].astype(int).astype(str) + "–" + frame["su_losses"].astype(int).astype(str),
            "SU %": frame["su_accuracy"].map(lambda value: f"{float(value):.1%}"),
            "ATS": frame["ats_wins"].astype(int).astype(str) + "–" + frame["ats_losses"].astype(int).astype(str),
            "ATS %": frame["ats_accuracy_excluding_pushes"].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.1%}"
            ),
            "Margin MAE": frame["margin_mae"].map(lambda value: f"{float(value):.2f}"),
            "Brier": frame["brier_score"].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.3f}"
            ),
        }
    )
    fig, axis = plt.subplots(figsize=(17.5, max(12.0, 0.42 * len(table) + 3.0)))
    fig.patch.set_facecolor("#F7F4ED")
    axis.axis("off")
    plotted = axis.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0.04, 1, 0.89],
        colWidths=[0.055, 0.29, 0.09, 0.08, 0.075, 0.08, 0.075, 0.10, 0.08],
    )
    plotted.auto_set_font_size(False)
    plotted.set_fontsize(10.2)
    plotted.scale(1, 1.35)
    for (row, _), cell in plotted.get_celld().items():
        cell.set_edgecolor("#D4D7DB")
        if row == 0:
            cell.set_facecolor("#22324A")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#EEF2F5")
            if row <= 3:
                cell.set_text_props(weight="bold")
    axis.set_title(
        f"{season} Post–Week {completed_week}: {title_label} Model Scorecard",
        fontsize=21,
        weight="bold",
        pad=22,
        color="#17263C",
    )
    fig.text(
        0.5,
        0.012,
        "Models are ordered by this week's margin MAE; this retrospective ordering does not alter the frozen roster or pregame consensus.",
        ha="center",
        fontsize=10,
        color="#555B63",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _cumulative_performance_metrics(frame: pd.DataFrame) -> dict[str, object]:
    """Score cumulative probability, winner, and spread performance."""
    probability = pd.to_numeric(frame.get("pred_home_win_probability"), errors="coerce")
    actual_home_win = pd.to_numeric(frame.get("actual_home_win"), errors="coerce")
    valid_probability = probability.notna() & actual_home_win.notna()
    brier = (
        float(((probability.loc[valid_probability].clip(1e-8, 1 - 1e-8) - actual_home_win.loc[valid_probability]) ** 2).mean())
        if valid_probability.any()
        else np.nan
    )
    if "su_correct" in frame:
        su_correct = frame["su_correct"].astype("boolean")
    else:
        su_correct = frame["winner_correct"].astype("boolean")
    su_valid = su_correct.notna()
    su_wins = int(su_correct.loc[su_valid].sum())
    su_losses = int(su_valid.sum() - su_wins)

    if "ats_result" in frame:
        ats_result = frame["ats_result"].astype(str).str.casefold()
        ats_wins = int(ats_result.eq("win").sum())
        ats_losses = int(ats_result.eq("loss").sum())
        ats_pushes = int(ats_result.eq("push").sum())
    else:
        spread = pd.to_numeric(frame.get("market_spread_close"), errors="coerce")
        predicted_edge = pd.to_numeric(frame.get("pred_home_margin"), errors="coerce") + spread
        actual_edge = pd.to_numeric(frame.get("actual_home_margin"), errors="coerce") + spread
        graded = spread.notna() & predicted_edge.notna() & actual_edge.notna()
        pushes = graded & actual_edge.eq(0)
        decisions = graded & ~pushes
        correct = predicted_edge.ge(0).eq(actual_edge.gt(0))
        ats_wins = int((decisions & correct).sum())
        ats_losses = int(decisions.sum() - ats_wins)
        ats_pushes = int(pushes.sum())
    ats_decisions = ats_wins + ats_losses
    return {
        "games": int(frame["game_id"].astype(str).nunique()),
        "brier_games": int(valid_probability.sum()),
        "brier_score": brier,
        "su_wins": su_wins,
        "su_losses": su_losses,
        "su_accuracy": su_wins / (su_wins + su_losses) if su_wins + su_losses else np.nan,
        "ats_wins": ats_wins,
        "ats_losses": ats_losses,
        "ats_pushes": ats_pushes,
        "ats_accuracy": ats_wins / ats_decisions if ats_decisions else np.nan,
    }


def _american_odds_probability(odds: object) -> float:
    value = float(odds)
    if value == 0 or not np.isfinite(value):
        return np.nan
    return -value / (-value + 100.0) if value < 0 else 100.0 / (value + 100.0)


def _vegas_no_vig_home_probabilities(lines_path: Path) -> dict[str, float]:
    """Average provider-level, no-vig home probabilities from moneylines."""
    if not lines_path.exists():
        return {}
    lines = pd.read_parquet(lines_path)
    probabilities: dict[str, float] = {}
    for game in lines.itertuples(index=False):
        offers = game.lines.tolist() if hasattr(game.lines, "tolist") else game.lines
        provider_probabilities = []
        for offer in offers or []:
            home_odds = offer.get("homeMoneyline")
            away_odds = offer.get("awayMoneyline")
            if home_odds is None or away_odds is None:
                continue
            home = _american_odds_probability(home_odds)
            away = _american_odds_probability(away_odds)
            if np.isfinite(home) and np.isfinite(away) and home + away > 0:
                provider_probabilities.append(home / (home + away))
        if provider_probabilities:
            probabilities[str(game.id)] = float(np.mean(provider_probabilities))
    return probabilities


def build_cumulative_margin_wide_performance(
    *,
    publication_root: str | Path,
    season: int,
    completed_week: int,
    current_model_games: pd.DataFrame,
) -> pd.DataFrame:
    """Build one cumulative row per model/week plus the TDNet consensus."""
    root = Path(publication_root)
    vegas_probability_by_game = _vegas_no_vig_home_probabilities(
        root.parent.parent / "data" / "raw" / "cfbd" / "v2" / "lines" / f"{season}.parquet"
    )
    current_models = sorted(current_model_games["model_id"].astype(str).unique())
    roster_ranks = (
        current_model_games.assign(
            _rank=pd.to_numeric(current_model_games.get("roster_rank"), errors="coerce")
        )
        .dropna(subset=["_rank"])
        .drop_duplicates("model_id")
        .set_index("model_id")["_rank"]
        .astype(int)
        .to_dict()
    )
    model_weeks: list[pd.DataFrame] = []
    consensus_weeks: list[pd.DataFrame] = []
    vegas_weeks: list[pd.DataFrame] = []
    for week in range(int(completed_week) + 1):
        week_root = root / f"week_{week:02d}" / "post_game"
        scored_path = week_root / "scoring" / "scored_predictions.parquet"
        consensus_path = week_root / "tables" / "margin_wide_prediction_vs_actual.csv"
        if not scored_path.exists() or not consensus_path.exists():
            continue
        scored = pd.read_parquet(scored_path)
        vegas = scored.drop_duplicates("game_id").copy()
        vegas["metric_week"] = int(week)
        frozen_vegas_probability = pd.to_numeric(
            vegas.get("vegas_home_win_probability_as_of_prediction"), errors="coerce"
        )
        moneyline_probability = vegas["game_id"].astype(str).map(vegas_probability_by_game)
        vegas["pred_home_win_probability"] = frozen_vegas_probability.fillna(
            moneyline_probability
        )
        spread = pd.to_numeric(vegas.get("market_spread_close"), errors="coerce")
        actual_home_win = pd.to_numeric(vegas.get("actual_home_win"), errors="coerce")
        vegas_favors_home = spread.lt(0)
        pick_is_valid = spread.notna() & spread.ne(0) & actual_home_win.notna()
        vegas["winner_correct"] = pd.Series(pd.NA, index=vegas.index, dtype="boolean")
        vegas.loc[pick_is_valid, "winner_correct"] = vegas_favors_home.loc[
            pick_is_valid
        ].eq(actual_home_win.loc[pick_is_valid].astype(bool))
        vegas_weeks.append(vegas)
        scored = scored.loc[scored["model_name"].astype(str).isin(current_models)].copy()
        scored["metric_week"] = int(week)
        model_weeks.append(scored)
        consensus = pd.read_csv(consensus_path)
        consensus["metric_week"] = int(week)
        consensus_weeks.append(consensus)
    if not model_weeks or not consensus_weeks or not vegas_weeks:
        raise ValueError("No prior scored margin-wide postgame packages were found.")

    all_models = pd.concat(model_weeks, ignore_index=True)
    all_consensus = pd.concat(consensus_weeks, ignore_index=True)
    all_vegas = pd.concat(vegas_weeks, ignore_index=True)
    available_weeks = sorted(all_models["metric_week"].astype(int).unique())
    rows: list[dict[str, object]] = []
    for week in available_weeks:
        through = all_models.loc[all_models["metric_week"].le(week)]
        for model_id in current_models:
            model = through.loc[through["model_name"].astype(str).eq(model_id)]
            if model.empty:
                continue
            rows.append(
                {
                    "season": int(season),
                    "through_week": int(week),
                    "series_type": "model",
                    "model_id": model_id,
                    "roster_rank": roster_ranks.get(model_id, pd.NA),
                    **_cumulative_performance_metrics(model),
                }
            )
        consensus = all_consensus.loc[all_consensus["metric_week"].le(week)]
        rows.append(
            {
                "season": int(season),
                "through_week": int(week),
                "series_type": "consensus",
                "model_id": "TDNet margin-wide consensus",
                "roster_rank": 0,
                **_cumulative_performance_metrics(consensus),
            }
        )
        vegas = all_vegas.loc[all_vegas["metric_week"].le(week)]
        vegas_metrics = _cumulative_performance_metrics(vegas)
        # A bookmaker line is the indifference point, not a directional ATS
        # selection.  Report the conventional no-vig 50% benchmark explicitly
        # instead of manufacturing wins and losses from the realized games.
        vegas_metrics.update(
            {
                "ats_wins": pd.NA,
                "ats_losses": pd.NA,
                "ats_pushes": pd.NA,
                "ats_accuracy": 0.5,
            }
        )
        rows.append(
            {
                "season": int(season),
                "through_week": int(week),
                "series_type": "vegas",
                "model_id": "Vegas closing-line baseline",
                "roster_rank": -1,
                **vegas_metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["through_week", "series_type", "roster_rank", "model_id"], kind="stable"
    ).reset_index(drop=True)


def plot_cumulative_margin_wide_performance(
    cumulative: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    completed_week: int,
) -> Path:
    """Render a latest cumulative table beside Brier, SU, and ATS trends."""
    latest = cumulative.loc[cumulative["through_week"].eq(int(completed_week))].copy()
    if latest.empty:
        raise ValueError(f"No cumulative rows are available through Week {completed_week}.")
    latest["_display_order"] = latest["series_type"].map(
        {"consensus": 0, "vegas": 1, "model": 2}
    ).fillna(3)
    latest = latest.sort_values(
        ["_display_order", "roster_rank", "model_id"],
        ascending=[True, True, True],
        kind="stable",
    )
    labels = latest["model_id"].astype(str).str.replace("margin_", "", regex=False)
    ats_records = latest.apply(
        lambda row: (
            "No-vig baseline"
            if row["series_type"] == "vegas"
            else f"{int(row['ats_wins'])}–{int(row['ats_losses'])}–{int(row['ats_pushes'])}"
        ),
        axis=1,
    )
    table = pd.DataFrame(
        {
            "Model": labels,
            "G": latest["games"].astype(int),
            "Brier": latest["brier_score"].map(lambda value: f"{float(value):.3f}"),
            "Brier n": latest["brier_games"].astype(int),
            "SU": latest["su_wins"].astype(int).astype(str) + "–" + latest["su_losses"].astype(int).astype(str),
            "SU %": latest["su_accuracy"].map(lambda value: f"{float(value):.1%}"),
            "ATS": ats_records,
            "ATS %": latest["ats_accuracy"].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.1%}"
            ),
        }
    )

    fig = plt.figure(figsize=(22, 18), facecolor="#F7F4ED")
    grid = fig.add_gridspec(3, 2, width_ratios=(1.48, 1.0), hspace=0.34, wspace=0.16)
    table_axis = fig.add_subplot(grid[:, 0])
    table_axis.axis("off")
    plotted = table_axis.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0, 1, 0.965],
        colWidths=[0.38, 0.05, 0.075, 0.065, 0.10, 0.08, 0.13, 0.085],
    )
    plotted.auto_set_font_size(False)
    plotted.set_fontsize(8.7)
    plotted.scale(1, 1.18)
    for (row, _), cell in plotted.get_celld().items():
        cell.set_edgecolor("#D4D7DB")
        if row == 0:
            cell.set_facecolor("#22324A")
            cell.set_text_props(color="white", weight="bold")
        elif row == 1:
            cell.set_facecolor("#FDE2F0")
            cell.set_text_props(weight="bold", color="#17263C")
        elif row == 2:
            cell.set_facecolor("#FFF1C2")
            cell.set_text_props(weight="bold", color="#17263C")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#EEF2F5")
    table_axis.set_title(
        f"Cumulative through Week {completed_week}", loc="left", fontsize=17, weight="bold", color="#17263C", pad=12
    )

    plot_specs = (
        ("brier_score", "Brier score", "Lower is better", None),
        ("su_accuracy", "Straight-up accuracy", "Higher is better", (0, 1)),
        ("ats_accuracy", "Against-the-spread accuracy", "Pushes and unavailable lines excluded", (0, 1)),
    )
    for row_index, (metric, title, subtitle, limits) in enumerate(plot_specs):
        axis = fig.add_subplot(grid[row_index, 1])
        models = cumulative.loc[cumulative["series_type"].eq("model")]
        for _, model in models.groupby("model_id", sort=False):
            axis.plot(
                model["through_week"], model[metric], color="#58718F", alpha=0.22, lw=1.25, marker="o", ms=3
            )
        consensus = cumulative.loc[cumulative["series_type"].eq("consensus")]
        axis.plot(
            consensus["through_week"], consensus[metric], color="#E83E8C", lw=4.0, marker="o", ms=8,
            label="TDNet margin-wide consensus", zorder=5,
        )
        vegas = cumulative.loc[cumulative["series_type"].eq("vegas")]
        axis.plot(
            vegas["through_week"],
            vegas[metric],
            color="#D99A00",
            lw=3.0,
            ls="--",
            marker="s",
            ms=6,
            label="Vegas closing-line baseline",
            zorder=4,
        )
        axis.set_title(
            title, loc="left", fontsize=17, weight="bold", color="#17263C", pad=12
        )
        axis.text(
            0.985,
            0.965,
            subtitle,
            transform=axis.transAxes,
            fontsize=9.5,
            color="#616A75",
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
        )
        axis.set_xlabel("Completed week")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_xticks(sorted(cumulative["through_week"].astype(int).unique()))
        if limits is not None:
            axis.set_ylim(*limits)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.28)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="best", fontsize=9.5)

    fig.suptitle(
        f"TDNet {season} Margin-Wide Cumulative Performance",
        x=0.045,
        y=0.995,
        ha="left",
        fontsize=25,
        weight="bold",
        color="#17263C",
    )
    fig.text(
        0.045,
        0.008,
        "Thin lines are the frozen margin-wide models; pink is TDNet consensus and dashed gold is the Vegas baseline. "
        "Vegas Brier uses no-vig closing-moneyline probabilities (sample size shown), SU uses the closing-line favorite, and ATS is the explicit 50% no-vig benchmark. "
        "Model ATS accuracy excludes pushes and games without a captured spread.",
        fontsize=10.5,
        color="#555B63",
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def _plot_team_margin_parity(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    title_scope: str,
    color_by_week: bool,
) -> dict[str, float | int]:
    """Plot predicted versus actual home margin with labeled decision quadrants."""
    data = frame.copy()
    data["actual_home_margin"] = pd.to_numeric(data["actual_home_margin"], errors="coerce")
    data["pred_home_margin"] = pd.to_numeric(data["pred_home_margin"], errors="coerce")
    data = data.dropna(subset=["actual_home_margin", "pred_home_margin"])
    if data.empty:
        raise ValueError("Margin parity plot requires at least one scored game.")
    residual = data["pred_home_margin"] - data["actual_home_margin"]
    metrics: dict[str, float | int] = {
        "games": len(data),
        "mae": float(residual.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "correlation": float(data["actual_home_margin"].corr(data["pred_home_margin"])),
    }
    limit = max(
        20.0,
        float(np.ceil(max(data["actual_home_margin"].abs().max(), data["pred_home_margin"].abs().max()) / 10.0) * 10.0),
    )
    apply_tdnet_theme()
    fig, axis = plt.subplots(figsize=(13.5, 12.5), facecolor="#F7F4ED")
    axis.set_facecolor("#FFFFFF")
    week_colors = (
        TDNET_COLORS["edge_pink"],
        TDNET_COLORS["ion_blue"],
        TDNET_COLORS["signal_orange"],
        TDNET_COLORS["electric_emerald"],
        TDNET_COLORS["gridiron_violet"],
        TDNET_COLORS["deep_teal"],
    )
    if color_by_week:
        for index, (week, games) in enumerate(data.groupby("week", sort=True)):
            axis.scatter(
                games["actual_home_margin"], games["pred_home_margin"],
                s=42, alpha=0.62, color=week_colors[index % len(week_colors)],
                edgecolor="none", label=f"Week {int(week)}", zorder=3,
            )
    else:
        axis.scatter(
            data["actual_home_margin"], data["pred_home_margin"],
            s=50, alpha=0.65, color=TDNET_COLORS["ion_blue"], edgecolor="none", zorder=3,
        )
    axis.plot([-limit, limit], [-limit, limit], color=TDNET_COLORS["midnight_gridiron"], ls="--", lw=1.8, label="Perfect prediction")
    axis.axhline(0, color=TDNET_COLORS["slate"], lw=1.0, alpha=0.55)
    axis.axvline(0, color=TDNET_COLORS["slate"], lw=1.0, alpha=0.55)
    quadrant_labels = (
        (0.73, 0.95, "TDNet Home Pick · Home Win"),
        (0.27, 0.95, "TDNet Home Pick · Away Win"),
        (0.73, 0.05, "TDNet Away Pick · Home Win"),
        (0.27, 0.05, "TDNet Away Pick · Away Win"),
    )
    for x, y, label in quadrant_labels:
        axis.text(
            x, y, label, transform=axis.transAxes, ha="center",
            va="top" if y > 0.5 else "bottom", fontsize=9.5,
            color=TDNET_COLORS["slate"], alpha=0.58, weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 2.5},
            zorder=2,
        )
    axis.text(
        0.025, 0.86,
        f"{metrics['games']} games\nMAE {metrics['mae']:.2f} · RMSE {metrics['rmse']:.2f} · r {metrics['correlation']:.2f}",
        transform=axis.transAxes, fontsize=12, va="top",
        bbox={"facecolor": "white", "edgecolor": "#D4D7DB", "alpha": 0.92, "boxstyle": "round,pad=0.35"},
    )
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Actual home-team margin (points)")
    axis.set_ylabel("TDNet predicted home-team margin (points)")
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    if color_by_week:
        order = [len(handles) - 1, *range(len(handles) - 1)]
        axis.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, loc="lower right")
    else:
        axis.legend(frameon=False, loc="lower right")
    axis.set_title(
        f"{season} {title_scope}: TDNet Margin Parity\nEach completed game contributes one home-team view",
        fontsize=21, weight="bold", color=TDNET_COLORS["midnight_gridiron"], pad=16,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    metrics["axis_limit"] = limit
    return metrics


def write_team_margin_parity_artifacts(
    *, publication_root: str | Path, output_root: str | Path, season: int, completed_week: int
) -> dict[str, Path]:
    """Write weekly and season-to-date margin parity figures and audit metadata."""
    root = Path(publication_root)
    output = Path(output_root)
    weekly_frames = []
    for week in range(int(completed_week) + 1):
        path = root / f"week_{week:02d}" / "post_game" / "tables" / "margin_wide_prediction_vs_actual.csv"
        if path.exists():
            week_frame = pd.read_csv(path)
            week_frame["week"] = int(week)
            weekly_frames.append(week_frame)
    if not weekly_frames:
        raise ValueError("No season-to-date margin-wide consensus results were found.")
    all_games = pd.concat(weekly_frames, ignore_index=True)
    current = all_games.loc[all_games["week"].eq(int(completed_week))]
    weekly_path = output / "figures" / "team_margin_parity.png"
    cumulative_path = output / "figures" / "cumulative_team_margin_parity.png"
    weekly_metrics = _plot_team_margin_parity(
        current, weekly_path, season=season, title_scope=f"Week {completed_week}", color_by_week=False
    )
    cumulative_metrics = _plot_team_margin_parity(
        all_games, cumulative_path, season=season,
        title_scope=f"Through Week {completed_week}", color_by_week=True,
    )
    metadata_path = output / "metadata" / "margin_parity_manifest.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "tdnet-home-margin-parity-v2",
                "season": int(season),
                "completed_week": int(completed_week),
                "view": "one home-team view per game",
                "quadrant_labels": "TDNet home/away pick crossed with actual home/away winner",
                "weekly": weekly_metrics,
                "cumulative": cumulative_metrics,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return {"weekly_figure": weekly_path, "cumulative_figure": cumulative_path, "metadata": metadata_path}


def _enrich_ballots(ballots: pd.DataFrame, inventory: pd.DataFrame | None) -> pd.DataFrame:
    out = ballots.copy()
    if inventory is None or inventory.empty:
        return out
    metadata = inventory.copy()
    label_column = "final_model_name" if "final_model_name" in metadata else "model_id"
    if "selected_feature_count" not in metadata and "selected_features_json" in metadata:
        metadata["selected_feature_count"] = metadata["selected_features_json"].map(
            lambda value: len(json.loads(value)) if pd.notna(value) and str(value).strip() else 0
        )
    keep = list(
        dict.fromkeys(
            column
            for column in [
                label_column,
                "model_id",
                "model_level",
                "model_family",
                "objective",
                "feature_config",
                "fingerprint",
                "checkpoint_sha256",
                "training_end_season",
                "selected_feature_count",
                "calibration_status",
                "calibrator_sha256",
            ]
            if column in metadata
        )
    )
    metadata = metadata[keep].drop_duplicates(label_column).copy()
    metadata["_ballot_key"] = metadata[label_column].astype(str).str.casefold()
    metadata = metadata.drop(columns=[label_column])
    # The margin poll directory is also the publication destination, so a
    # deliberate rerender may read a previously enriched ballot CSV.  Replace
    # metadata columns deterministically instead of accumulating merge suffixes.
    replace_columns = [
        column for column in metadata.columns if column != "_ballot_key" and column in out
    ]
    if replace_columns:
        out = out.drop(columns=replace_columns)
    out["_ballot_key"] = out["ballot_model"].astype(str).str.casefold()
    out = out.merge(metadata, on="_ballot_key", how="left", validate="many_to_one")
    return out.drop(columns="_ballot_key")


def _write_poll_package(
    *,
    poll: pd.DataFrame,
    ballots: pd.DataFrame,
    output: Path,
    figure_output: Path | None = None,
    table_output: Path | None = None,
    metadata_output: Path | None = None,
    raw_games: pd.DataFrame,
    reference_poll: pd.DataFrame,
    inventory: pd.DataFrame | None,
    season: int,
    completed_week: int,
    state_week: int,
    logo_dir: Path,
    scientific: bool,
    generated_at_eastern: str,
    reference_label: str,
    reference_short_label: str,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    figure_output = figure_output or output
    table_output = table_output or output
    metadata_output = metadata_output or output
    figure_output.mkdir(parents=True, exist_ok=True)
    table_output.mkdir(parents=True, exist_ok=True)
    metadata_output.mkdir(parents=True, exist_ok=True)
    validate_scientific_ballots(ballots)
    frame = add_team_records(poll, raw_games, completed_week=state_week)
    frame["publication_completed_week"] = int(completed_week)
    frame["postgame_state_week"] = int(state_week)
    frame["reference_snapshot_label"] = str(reference_label)
    if not reference_poll.empty:
        reference_rank = reference_poll.set_index("team")["rank"]
        frame["reference_rank"] = frame["keys_team"].map(reference_rank).astype("Int64")
        frame["tdnet_minus_reference"] = frame["rank"] - frame["reference_rank"]

    enriched_ballots = _enrich_ballots(ballots, inventory)
    receiving = aggregate_receiving_votes(frame, ballots)
    receiving_text = format_receiving_votes(receiving)
    disagreement = model_consensus_disagreement(frame, ballots)
    power = scientific_consensus_power_rankings(ballots)
    generic_power = power.rename(columns={"scientific_models": "models_in_consensus"})

    prefix = "scientific_" if scientific else ""
    poll_name = f"{prefix}tdnet_top25.csv"
    ballot_name = "scientific_full_ballots.csv" if scientific else "tdnet_model_ballots.csv"
    power_name = (
        "scientific_consensus_power_rankings.csv"
        if scientific
        else "margin_wide_consensus_power_rankings.csv"
    )
    frame.to_csv(table_output / poll_name, index=False)
    enriched_ballots.to_csv(table_output / ballot_name, index=False)
    generic_power.to_csv(table_output / power_name, index=False)
    receiving.to_csv(table_output / f"{prefix}tdnet_receiving_votes.csv", index=False)
    (table_output / f"{prefix}receiving_votes.txt").write_text(
        receiving_text + "\n", encoding="utf-8"
    )
    top_ballots = ballots.loc[pd.to_numeric(ballots["ballot_rank"], errors="coerce").between(1, 25)].copy()
    top_ballots.to_csv(table_output / f"{prefix}per_model_top25_long.csv", index=False)
    top_ballots.pivot_table(
        index="ballot_rank", columns="ballot_model", values="keys_team", aggfunc="first"
    ).sort_index().to_csv(table_output / f"{prefix}per_model_top25.csv")
    disagreement.to_csv(
        table_output / f"{prefix}model_consensus_disagreement.csv", index=False
    )

    roster_label = "SCIENTIFIC ROSTER" if scientific else "WIDE-MARGIN ROSTER"
    plot_consensus_poll_table(
        frame,
        figure_output / f"{prefix}tdnet_top25.png",
        title=f"{season} Post–Week {completed_week}: TDNet {roster_label.title()} Top 25",
        receiving_votes=receiving_text,
        logo_dir=logo_dir,
        reference_label=reference_short_label,
    )
    plot_ballot_logo_grid(
        top_ballots,
        figure_output
        / ("scientific_top25_ballots.png" if scientific else "tdnet_model_ballots.png"),
        top_n=25,
        logo_dir=logo_dir,
        title=f"{season} Post–Week {completed_week}: {roster_label} Model Ballots",
    )
    # Keep the scientific directory as compact as its pregame counterpart.
    # The main figures directory retains the richer consensus comparisons.
    if not scientific:
        plot_top25_consensus_spread(
            frame,
            ballots,
            figure_output / "top25_consensus_spread.png",
            reference_poll=reference_poll,
            reference_label=reference_short_label,
            title=f"{season} Post–Week {completed_week}: {roster_label} Ballot Spread",
        )
        plot_model_disagreement(
            disagreement,
            figure_output / "model_consensus_disagreement.png",
            title=f"{season} Post–Week {completed_week}: Models Farthest from {roster_label.title()} Consensus",
        )
        if not reference_poll.empty:
            plot_tdnet_vs_ap_poll(
                frame.rename(columns={"keys_team": "team"}),
                reference_poll,
                figure_output / "tdnet_vs_ap_top25.png",
                title=f"{season} Post–Week {completed_week}: TDNet vs {reference_label}",
                logo_dir=logo_dir,
                reference_label=reference_short_label,
            )

    plot_scientific_all_team_power_ranking(
        power,
        figure_output
        / (
            "scientific_all_team_power_rankings.png"
            if scientific
            else "margin_wide_all_team_power_rankings.png"
        ),
        season=season,
        week=state_week,
        bulletin_label=("TDNET RESEARCH BULLETIN" if scientific else "TDNET MODEL BULLETIN"),
        title=(
            "TDNET SCIENTIFIC ALL-TEAM POWER RANKING"
            if scientific
            else "TDNET WIDE-MARGIN ALL-TEAM POWER RANKING"
        ),
        footer_label=(
            "RANK IS DERIVED FROM AGGREGATED SCIENTIFIC BALLOT POINTS"
            if scientific
            else "RANK IS DERIVED FROM AGGREGATED WIDE-MARGIN BALLOT POINTS"
        ),
    )
    if not scientific:
        plot_power_rank_vs_projected_margin(
            power,
            figure_output / "margin_wide_rank_vs_projected_margin.png",
            season=season,
            week=state_week,
            logo_dir=logo_dir,
        )
    social = (
        power.rename(
            columns={
                "poll_points_rank": "rank",
                "keys_team": "team",
                "scientific_models": "ballots_seen",
                "best_ballot_rank": "best_rank",
                "consensus_power_rank": "reference_rank",
            }
        )
        if scientific
        else frame.rename(columns={"keys_team": "team"})
    )
    source_path = table_output / power_name if scientific else table_output / poll_name
    for variant in ("4x5", "16x9"):
        path = figure_output / (
            f"scientific_research_ballot_social_{variant}.png"
            if scientific
            else f"tdnet_top10_social_{variant}.png"
        )
        kwargs = {}
        if scientific:
            kwargs = {
                "reference_label": "POWER",
                "header_title": "TDNet Research Ballot",
                "header_subtitle": "Scientific Model Consensus",
                "header_accent_color": "#FF5A36",
            }
        else:
            kwargs = {"reference_label": reference_short_label.upper()}
        render_top10_social(
            social,
            path,
            season=season,
            week=state_week,
            logo_dir=logo_dir,
            variant=variant,
            generated_at_utc=generated_at_eastern,
            source_sha256=sha256_file(source_path),
            **kwargs,
        )

    poll_teams = set(power.loc[power["poll_points_rank"].le(25), "keys_team"].astype(str))
    power_teams = set(power.loc[power["consensus_power_rank"].le(25), "keys_team"].astype(str))
    divergence = {
        "poll_points_only": sorted(poll_teams - power_teams),
        "power_rating_only": sorted(power_teams - poll_teams),
        "top25_membership_difference_count": len(poll_teams ^ power_teams),
        "ordering_identical": bool(
            power.sort_values("poll_points_rank")["keys_team"].tolist()
            == power.sort_values("consensus_power_rank")["keys_team"].tolist()
        ),
        "note": "Poll points and consensus margin versus average are intentionally independent summaries.",
    }
    (metadata_output / f"{prefix}poll_power_divergence.json").write_text(
        json.dumps(divergence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "poll": frame,
        "ballots": enriched_ballots,
        "power": generic_power,
        "divergence": divergence,
    }


def build_full_postgame_package(
    *,
    scored_predictions_path: str | Path,
    raw_games_path: str | Path,
    margin_poll_dir: str | Path,
    scientific_poll_dir: str | Path,
    margin_inventory_path: str | Path,
    scientific_inventory_path: str | Path,
    reference_poll_path: str | Path,
    api_manifest_path: str | Path,
    output_root: str | Path,
    season: int,
    completed_week: int,
    state_week: int,
    logo_dir: str | Path,
    reference_label: str = "Latest available AP Top 25",
    reference_short_label: str = "AP",
) -> dict[str, object]:
    """Write wide scorecards, two roster polls, power ratings, and social assets."""
    output = Path(output_root)
    scientific_output = output / "scientific"
    figures = output / "figures"
    tables = output / "tables"
    metadata = output / "metadata"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(EASTERN).isoformat()

    scored_path = Path(scored_predictions_path)
    scored = pd.read_parquet(scored_path) if scored_path.suffix == ".parquet" else pd.read_csv(scored_path)
    model_games = _prepare_margin_model_games(scored)
    consensus = _aggregate_models(model_games)
    consensus.to_csv(tables / "margin_wide_prediction_vs_actual.csv", index=False)
    write_team_margin_parity_artifacts(
        publication_root=output.parent.parent,
        output_root=output,
        season=season,
        completed_week=completed_week,
    )
    model_games.to_csv(tables / "margin_wide_model_game_results.csv", index=False)
    scorecard = _model_scorecard(model_games)
    scorecard.to_csv(tables / "margin_wide_model_scorecard.csv", index=False)
    plot_sunday_recap_table(
        consensus,
        figures / "margin_wide_prediction_scorecard.png",
        season=season,
        week=completed_week,
        objective="margin",
        model_label=f"Frozen wide-margin consensus ({model_games['model_id'].nunique()} models)",
    )
    _plot_model_scorecard(
        scorecard,
        figures / "margin_wide_model_scorecard.png",
        season=season,
        completed_week=completed_week,
        title_label="Wide-Margin",
    )
    cumulative_performance = build_cumulative_margin_wide_performance(
        publication_root=output.parent.parent,
        season=season,
        completed_week=completed_week,
        current_model_games=model_games,
    )
    cumulative_performance.to_csv(
        tables / "cumulative_margin_wide_performance.csv", index=False
    )
    plot_cumulative_margin_wide_performance(
        cumulative_performance,
        figures / "cumulative_margin_wide_performance.png",
        season=season,
        completed_week=completed_week,
    )
    raw_games_path = Path(raw_games_path)
    raw_games = pd.read_parquet(raw_games_path) if raw_games_path.suffix == ".parquet" else pd.read_csv(raw_games_path)
    reference = pd.read_csv(reference_poll_path)
    margin_poll_root = Path(margin_poll_dir)
    scientific_poll_root = Path(scientific_poll_dir)
    margin_result = _write_poll_package(
        poll=pd.read_csv(margin_poll_root / "tdnet_top25.csv"),
        ballots=pd.read_csv(margin_poll_root / "tdnet_model_ballots.csv"),
        output=output,
        figure_output=figures,
        table_output=tables,
        metadata_output=metadata,
        raw_games=raw_games,
        reference_poll=reference,
        inventory=pd.read_csv(margin_inventory_path),
        season=season,
        completed_week=completed_week,
        state_week=state_week,
        logo_dir=Path(logo_dir),
        scientific=False,
        generated_at_eastern=generated_at,
        reference_label=reference_label,
        reference_short_label=reference_short_label,
    )
    scientific_result = _write_poll_package(
        poll=pd.read_csv(scientific_poll_root / "tdnet_top25.csv"),
        ballots=pd.read_csv(scientific_poll_root / "tdnet_model_ballots.csv"),
        output=scientific_output,
        raw_games=raw_games,
        reference_poll=reference,
        inventory=pd.read_csv(scientific_inventory_path),
        season=season,
        completed_week=completed_week,
        state_week=state_week,
        logo_dir=Path(logo_dir),
        scientific=True,
        generated_at_eastern=generated_at,
        reference_label=reference_label,
        reference_short_label=reference_short_label,
    )

    (scientific_output / "README.md").write_text(
        "# TDNet scientific postgame package\n\n"
        "The performance files score the immutable 42-model pregame scientific predictions; "
        "no prediction model was rerun. The ranking files use the refreshed postgame team state.\n\n"
        "- `scientific_tdnet_top25.csv` is the model-ballot poll.\n"
        "- `scientific_full_ballots.csv` contains one complete all-team ballot per scientific model.\n"
        "- `scientific_consensus_power_rankings.csv` independently aggregates predicted margin "
        "versus the constructed average FBS team.\n"
        "- `scientific_poll_power_divergence.json` records differences without forcing the two rankings to agree.\n"
        "- No next-week matchup predictions are generated in this directory.\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        f"# TDNet {season} Week {completed_week} postgame package\n\n"
        "This lean Sunday release mirrors the pregame layout: reader-facing wide-margin output "
        "lives in `figures/`, its source tables live in `tables/`, and the paper-oriented roster "
        "lives in `scientific/`. Per-model PNGs and redundant diagnostic renders are not retained.\n\n"
        f"The comparison reference is **{reference_label}** and is labeled that way in every "
        "retained comparison. Poll points and consensus power ratings remain "
        "independent.\n\n"
        "The consensus bankroll figures compare flat $10 ATS and moneyline bets for the "
        "margin-wide, F0–F6 scientific, and full F0–F8 scientific consensuses. ATS uses "
        "an explicit -110 assumption because CFBD does not publish spread-side prices; "
        "moneyline returns use the best available CFBD quote.\n\n"
        "Separate confidence-scaled figures use model support for ATS confidence and "
        "picked-team win probability for moneyline confidence, scaling linearly from "
        "$0 at 49.9% to $25 at 100%.\n\n"
        "Season-to-date confidence threshold sweeps show flat-$10 profit and ROI at "
        "each minimum confidence cutoff. These are descriptive in-sample diagnostics, "
        "not forward-validated betting rules.\n\n"
        "Separate cumulative model-calibration overlays reproduce the historical diagnostic: "
        "binned predicted home-win probability versus observed home-win rate, one line per "
        "individual frozen model, with probability density below.\n\n"
        "Weekly and cumulative margin parity plots label all four TDNet home/away pick "
        "and realized home/away winner quadrants.\n\n"
        f"Week {state_week} matchup predictions are intentionally absent; those belong in "
        f"`publication/{season}/week_{state_week:02d}/pre_game`.\n",
        encoding="utf-8",
    )

    scientific_scorecard_path = scientific_output / "scientific_model_scorecard.csv"
    if scientific_scorecard_path.exists():
        scientific_scorecard = pd.read_csv(scientific_scorecard_path)
        scientific_metadata = pd.read_csv(scientific_inventory_path)
        scientific_metadata["_model_key"] = scientific_metadata[
            "final_model_name" if "final_model_name" in scientific_metadata else "model_id"
        ].astype(str).str.casefold()
        metadata_keep = [
            column
            for column in ("_model_key", "model_family", "feature_config", "model_id")
            if column in scientific_metadata
        ]
        scientific_scorecard["_model_key"] = scientific_scorecard["model_name"].astype(str).str.casefold()
        scientific_scorecard = scientific_scorecard.merge(
            scientific_metadata[metadata_keep].drop_duplicates("_model_key"),
            on="_model_key",
            how="left",
            validate="one_to_one",
            suffixes=("", "_inventory"),
        ).drop(columns="_model_key")
        if {"feature_config", "model_family"}.issubset(scientific_scorecard):
            scientific_scorecard["model_family"] = (
                scientific_scorecard["feature_config"].astype(str)
                + " · "
                + scientific_scorecard["model_family"].astype(str)
            )
        rank_column = next(
            (column for column in ("margin_mae_rank", "postgame_margin_rank", "rank") if column in scientific_scorecard),
            None,
        )
        if rank_column is None:
            scientific_scorecard.insert(0, "postgame_margin_rank", range(1, len(scientific_scorecard) + 1))
        elif scientific_scorecard.columns[0] != rank_column:
            columns = [rank_column] + [column for column in scientific_scorecard if column != rank_column]
            scientific_scorecard = scientific_scorecard[columns]
        rename = {
            "winner_accuracy": "su_accuracy",
            "straight_up_accuracy": "su_accuracy",
            "ats_accuracy": "ats_accuracy_excluding_pushes",
            "absolute_margin_error": "margin_mae",
        }
        scientific_scorecard = scientific_scorecard.rename(columns=rename)
        if "su_wins" not in scientific_scorecard:
            scientific_scorecard["su_wins"] = (
                pd.to_numeric(scientific_scorecard["games"], errors="coerce")
                * pd.to_numeric(scientific_scorecard["su_accuracy"], errors="coerce")
            ).round().astype(int)
        if "su_losses" not in scientific_scorecard:
            scientific_scorecard["su_losses"] = (
                pd.to_numeric(scientific_scorecard["games"], errors="coerce")
                - scientific_scorecard["su_wins"]
            ).astype(int)
        for column in ("ats_wins", "ats_losses"):
            if column not in scientific_scorecard:
                scientific_scorecard[column] = 0
        if "ats_accuracy_excluding_pushes" not in scientific_scorecard:
            ats_decisions = scientific_scorecard["ats_wins"] + scientific_scorecard["ats_losses"]
            scientific_scorecard["ats_accuracy_excluding_pushes"] = (
                scientific_scorecard["ats_wins"] / ats_decisions.replace(0, np.nan)
            )
        if "brier_score" not in scientific_scorecard:
            scientific_scorecard["brier_score"] = np.nan
        if "model_id" not in scientific_scorecard and "model_name" in scientific_scorecard:
            scientific_scorecard["model_id"] = scientific_scorecard["model_name"]
        scientific_scorecard.to_csv(
            scientific_output / "scientific_model_scorecard_enriched.csv", index=False
        )
        _plot_model_scorecard(
            scientific_scorecard,
            scientific_output / "scientific_model_scorecard.png",
            season=season,
            completed_week=completed_week,
            title_label="Scientific Roster",
        )

        write_scientific_cumulative_artifacts(
            publication_root=output.parent.parent,
            season=season,
            completed_week=completed_week,
            output_root=scientific_output,
        )
        full_scientific_output = scientific_output / "full_f0_f8"
        if (full_scientific_output / "scientific_model_game_results.csv").exists():
            write_scientific_cumulative_artifacts(
                publication_root=output.parent.parent,
                season=season,
                completed_week=completed_week,
                output_root=full_scientific_output,
                scientific_subdir="full_f0_f8",
                consensus_label="Full F0–F8 scientific consensus",
                consensus_fingerprint="F0–F8",
            )

    bankroll_sources = (
        tables / "margin_wide_prediction_vs_actual.csv",
        scientific_output / "scientific_consensus_game_results.csv",
        scientific_output / "full_f0_f8" / "scientific_consensus_game_results.csv",
    )
    if all(path.exists() for path in bankroll_sources):
        write_consensus_betting_artifacts(
            publication_root=output.parent.parent,
            output_root=output,
            season=season,
            completed_week=completed_week,
        )
        write_cumulative_model_calibration_artifacts(
            publication_root=output.parent.parent,
            output_root=output,
            season=season,
            completed_week=completed_week,
        )

    api_manifest = json.loads(Path(api_manifest_path).read_text(encoding="utf-8"))
    manifest_path = output / "full_postgame_manifest.json"
    files = {}
    sunday_manifest_path = output / "sunday_publication_manifest.json"
    for path in sorted(output.rglob("*")):
        if path.is_file() and path not in {manifest_path, sunday_manifest_path}:
            files[str(path.relative_to(output))] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    manifest = {
        "schema": "tdnet-full-postgame-publication-v2",
        "generated_at_eastern": generated_at,
        "artifact_layout": _lean_layout_metadata(output),
        "season": int(season),
        "publication_completed_week": int(completed_week),
        "provider_state_week": int(state_week),
        "reference_poll_label": str(reference_label),
        "reference_poll_short_label": str(reference_short_label),
        "week_1_game_predictions_generated": False,
        "week_1_pregame_destination": f"publication/{season}/week_{state_week:02d}/pre_game",
        "margin_wide_prediction_models": int(model_games["model_id"].nunique()),
        "margin_wide_poll_models": int(margin_result["ballots"]["ballot_model"].nunique()),
        "scientific_poll_models": int(scientific_result["ballots"]["ballot_model"].nunique()),
        "api_refresh": api_manifest,
        "sources": {
            "scored_predictions": {"path": str(scored_path.resolve()), "sha256": sha256_file(scored_path)},
            "raw_games": {"path": str(raw_games_path.resolve()), "sha256": sha256_file(raw_games_path)},
            "reference_poll": {"path": str(Path(reference_poll_path).resolve()), "sha256": sha256_file(reference_poll_path)},
            "margin_inventory": {"path": str(Path(margin_inventory_path).resolve()), "sha256": sha256_file(margin_inventory_path)},
            "scientific_inventory": {"path": str(Path(scientific_inventory_path).resolve()), "sha256": sha256_file(scientific_inventory_path)},
        },
        "poll_power_divergence": {
            "margin_wide": margin_result["divergence"],
            "scientific": scientific_result["divergence"],
        },
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sunday_manifest = (
        json.loads(sunday_manifest_path.read_text(encoding="utf-8"))
        if sunday_manifest_path.exists()
        else {}
    )
    sunday_manifest.update(
        {
            "created_at_eastern": generated_at,
            "status": "full_sunday_review_bundle_ready",
            "full_postgame_manifest": {
                "path": manifest_path.name,
                "sha256": sha256_file(manifest_path),
            },
            "week_1_game_predictions_generated": False,
            "external_send": "disabled",
        }
    )
    sunday_manifest["files"] = {
        str(path.relative_to(output)): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != sunday_manifest_path
    }
    sunday_manifest_path.write_text(
        json.dumps(sunday_manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest
