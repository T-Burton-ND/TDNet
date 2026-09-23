"""Season-to-date performance artifacts for the frozen scientific roster."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .bundles import sha256_file
from .figure_theme import apply_tdnet_theme

VEGAS_LABEL = "Vegas closing-line baseline"
CONSENSUS_LABEL = "Scientific consensus"


def _american_probability(odds: object) -> float:
    value = float(odds)
    if value == 0 or not np.isfinite(value):
        return np.nan
    return -value / (-value + 100.0) if value < 0 else 100.0 / (value + 100.0)


def _no_vig_home_probabilities(lines_path: Path) -> dict[str, float]:
    if not lines_path.exists():
        return {}
    lines = pd.read_parquet(lines_path)
    probabilities: dict[str, float] = {}
    for game in lines.itertuples(index=False):
        offers = game.lines.tolist() if hasattr(game.lines, "tolist") else game.lines
        provider_values = []
        for offer in offers or []:
            home_odds = offer.get("homeMoneyline")
            away_odds = offer.get("awayMoneyline")
            if home_odds is None or away_odds is None:
                continue
            home = _american_probability(home_odds)
            away = _american_probability(away_odds)
            if np.isfinite(home) and np.isfinite(away) and home + away > 0:
                provider_values.append(home / (home + away))
        if provider_values:
            probabilities[str(game.id)] = float(np.mean(provider_values))
    return probabilities


def _metrics(
    frame: pd.DataFrame,
    *,
    margin_error: str,
    winner_correct: str,
    probability: str,
    ats_result: str,
) -> dict[str, object]:
    margin = pd.to_numeric(frame.get(margin_error), errors="coerce")
    winner = frame.get(winner_correct, pd.Series(pd.NA, index=frame.index)).astype(
        "boolean"
    )
    probability_values = pd.to_numeric(frame.get(probability), errors="coerce")
    actual_home_win = pd.to_numeric(frame.get("actual_home_win"), errors="coerce")
    probability_valid = probability_values.notna() & actual_home_win.notna()
    ats = (
        frame.get(ats_result, pd.Series("", index=frame.index))
        .astype(str)
        .str.casefold()
    )
    ats_wins = int(ats.eq("win").sum())
    ats_losses = int(ats.eq("loss").sum())
    ats_pushes = int(ats.eq("push").sum())
    winner_valid = winner.notna()
    return {
        "games": int(frame["game_id"].astype(str).nunique()),
        "margin_games": int(margin.notna().sum()),
        "margin_mae": float(margin.mean()) if margin.notna().any() else np.nan,
        "su_wins": int(winner.loc[winner_valid].sum()),
        "su_losses": int(winner_valid.sum() - winner.loc[winner_valid].sum()),
        "su_accuracy": float(winner.loc[winner_valid].mean())
        if winner_valid.any()
        else np.nan,
        "brier_games": int(probability_valid.sum()),
        "brier_score": (
            float(
                (
                    probability_values.loc[probability_valid].clip(1e-8, 1 - 1e-8)
                    - actual_home_win.loc[probability_valid]
                )
                .pow(2)
                .mean()
            )
            if probability_valid.any()
            else np.nan
        ),
        "ats_wins": ats_wins,
        "ats_losses": ats_losses,
        "ats_pushes": ats_pushes,
        "ats_accuracy": ats_wins / (ats_wins + ats_losses)
        if ats_wins + ats_losses
        else np.nan,
    }


def _prepare_week(
    path: Path, *, week: int, vegas_probabilities: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = pd.read_csv(path / "scientific_model_game_results.csv")
    models["metric_week"] = int(week)
    models["actual_home_win"] = (
        pd.to_numeric(models["actual_home_margin"], errors="coerce").gt(0).astype(float)
    )

    consensus = models.drop_duplicates("game_id").copy()
    consensus["consensus_margin_error"] = (
        pd.to_numeric(
            consensus["consensus_predicted_winner_margin"], errors="coerce"
        ).where(
            consensus["consensus_straight_up_pick"].eq(consensus["home_team"]),
            -pd.to_numeric(
                consensus["consensus_predicted_winner_margin"], errors="coerce"
            ),
        )
        - pd.to_numeric(consensus["actual_home_margin"], errors="coerce")
    ).abs()
    consensus["consensus_winner_correct"] = consensus["consensus_straight_up_pick"].eq(
        consensus["actual_winner"]
    )

    vegas = consensus.copy()
    spread = pd.to_numeric(
        vegas.get(
            "consensus_home_team_market_spread", vegas.get("market_spread_close")
        ),
        errors="coerce",
    )
    actual_margin = pd.to_numeric(vegas["actual_home_margin"], errors="coerce")
    vegas["vegas_margin_error"] = (-spread - actual_margin).abs()
    vegas["vegas_probability"] = vegas["game_id"].astype(str).map(vegas_probabilities)
    valid_pick = spread.notna() & spread.ne(0) & actual_margin.ne(0)
    vegas["vegas_winner_correct"] = pd.Series(pd.NA, index=vegas.index, dtype="boolean")
    vegas.loc[valid_pick, "vegas_winner_correct"] = (
        spread.loc[valid_pick].lt(0).eq(actual_margin.loc[valid_pick].gt(0))
    )
    return models, consensus, vegas


def build_scientific_rolling_performance(
    *,
    publication_root: str | Path,
    season: int,
    completed_week: int,
    lines_path: str | Path | None = None,
    scientific_subdir: str | None = None,
    consensus_label: str = CONSENSUS_LABEL,
    consensus_fingerprint: str = "F0–F6",
) -> pd.DataFrame:
    """Build weekly and season-to-date rows for every scientific model and Vegas."""
    root = Path(publication_root)
    line_source = (
        Path(lines_path)
        if lines_path is not None
        else root.parent.parent
        / "data"
        / "raw"
        / "cfbd"
        / "v2"
        / "lines"
        / f"{season}.parquet"
    )
    vegas_probabilities = _no_vig_home_probabilities(line_source)
    model_weeks: list[pd.DataFrame] = []
    consensus_weeks: list[pd.DataFrame] = []
    vegas_weeks: list[pd.DataFrame] = []
    for week in range(int(completed_week) + 1):
        scientific = root / f"week_{week:02d}" / "post_game" / "scientific"
        if scientific_subdir:
            scientific = scientific / scientific_subdir
        if not (scientific / "scientific_model_game_results.csv").exists():
            continue
        models, consensus, vegas = _prepare_week(
            scientific, week=week, vegas_probabilities=vegas_probabilities
        )
        model_weeks.append(models)
        consensus_weeks.append(consensus)
        vegas_weeks.append(vegas)
    if not model_weeks:
        raise ValueError("No scientific postgame model-game files were found.")

    all_models = pd.concat(model_weeks, ignore_index=True)
    all_consensus = pd.concat(consensus_weeks, ignore_index=True)
    all_vegas = pd.concat(vegas_weeks, ignore_index=True)
    current_names = sorted(
        all_models.loc[
            all_models["metric_week"].eq(all_models["metric_week"].max()), "model_name"
        ]
        .astype(str)
        .unique()
    )
    rows: list[dict[str, object]] = []
    for week in sorted(all_models["metric_week"].astype(int).unique()):
        for scope in ("weekly", "cumulative"):
            model_selector = (
                all_models["metric_week"].eq(week)
                if scope == "weekly"
                else all_models["metric_week"].le(week)
            )
            consensus_selector = (
                all_consensus["metric_week"].eq(week)
                if scope == "weekly"
                else all_consensus["metric_week"].le(week)
            )
            vegas_selector = (
                all_vegas["metric_week"].eq(week)
                if scope == "weekly"
                else all_vegas["metric_week"].le(week)
            )
            models = all_models.loc[model_selector]
            for model_name in current_names:
                games = models.loc[models["model_name"].astype(str).eq(model_name)]
                if games.empty:
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "through_week": int(week),
                        "series_type": "model",
                        "model_name": model_name,
                        "model_family": str(games["model_family"].iloc[0]),
                        "fingerprint": str(games["fingerprint"].iloc[0]),
                        **_metrics(
                            games,
                            margin_error="model_absolute_margin_error",
                            winner_correct="model_winner_correct",
                            probability="pred_home_win_probability",
                            ats_result="model_ats_result",
                        ),
                    }
                )
            consensus = all_consensus.loc[consensus_selector]
            rows.append(
                {
                    "scope": scope,
                    "through_week": int(week),
                    "series_type": "consensus",
                    "model_name": consensus_label,
                    "model_family": "consensus",
                    "fingerprint": consensus_fingerprint,
                    **_metrics(
                        consensus,
                        margin_error="consensus_margin_error",
                        winner_correct="consensus_winner_correct",
                        probability="consensus_predicted_home_win_probability",
                        ats_result="consensus_ats_result",
                    ),
                }
            )
            vegas = all_vegas.loc[vegas_selector]
            vegas_metrics = _metrics(
                vegas,
                margin_error="vegas_margin_error",
                winner_correct="vegas_winner_correct",
                probability="vegas_probability",
                ats_result="__no_directional_ats_pick__",
            )
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
                    "scope": scope,
                    "through_week": int(week),
                    "series_type": "vegas",
                    "model_name": VEGAS_LABEL,
                    "model_family": "market",
                    "fingerprint": "evaluation-only",
                    **vegas_metrics,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["scope", "through_week", "series_type", "fingerprint", "model_name"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def cumulative_scientific_scorecard(
    performance: pd.DataFrame, *, completed_week: int
) -> pd.DataFrame:
    """Return one season-to-date scorecard row per model plus consensus and Vegas."""
    latest = performance.loc[
        performance["scope"].eq("cumulative")
        & performance["through_week"].eq(int(completed_week))
    ].copy()
    if latest.empty:
        raise ValueError(
            f"No cumulative scientific rows exist through Week {completed_week}."
        )
    latest["scorecard_rank"] = latest["margin_mae"].rank(method="first").astype("Int64")
    latest["display_order"] = latest["series_type"].map(
        {"consensus": 0, "vegas": 1, "model": 2}
    )
    return (
        latest.sort_values(
            ["display_order", "scorecard_rank", "model_name"], kind="stable"
        )
        .drop(columns="display_order")
        .reset_index(drop=True)[
            [
                "scorecard_rank",
                "series_type",
                "model_name",
                "model_family",
                "fingerprint",
                "games",
                "margin_games",
                "margin_mae",
                "su_wins",
                "su_losses",
                "su_accuracy",
                "brier_games",
                "brier_score",
                "ats_wins",
                "ats_losses",
                "ats_pushes",
                "ats_accuracy",
            ]
        ]
    )


def _plot_metric_lines(
    axis, cumulative: pd.DataFrame, metric: str, *, show_models: bool
) -> None:
    if show_models:
        for _, model in cumulative.loc[cumulative["series_type"].eq("model")].groupby(
            "model_name", sort=False
        ):
            axis.plot(
                model["through_week"],
                model[metric],
                color="#58718F",
                alpha=0.24,
                lw=1.15,
            )
    consensus = cumulative.loc[cumulative["series_type"].eq("consensus")]
    vegas = cumulative.loc[cumulative["series_type"].eq("vegas")]
    axis.plot(
        consensus["through_week"],
        consensus[metric],
        color="#E83E8C",
        lw=3.5,
        marker="o",
        label=CONSENSUS_LABEL,
        zorder=4,
    )
    axis.plot(
        vegas["through_week"],
        vegas[metric],
        color="#D99A00",
        lw=2.8,
        ls="--",
        marker="s",
        label=VEGAS_LABEL,
        zorder=3,
    )


def plot_scientific_cumulative_all_models(
    performance: pd.DataFrame, path: str | Path, *, season: int
) -> Path:
    """Plot every scientific model, consensus, and the applicable Vegas baselines."""
    apply_tdnet_theme()
    cumulative = performance.loc[performance["scope"].eq("cumulative")]
    specs = (
        ("margin_mae", "Margin MAE", False),
        ("su_accuracy", "Straight-up accuracy", True),
        ("brier_score", "Brier score", False),
        ("ats_accuracy", "ATS accuracy", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), facecolor="#F7F4ED")
    for axis, (metric, title, percent) in zip(axes.flat, specs):
        _plot_metric_lines(axis, cumulative, metric, show_models=True)
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel("Completed week")
        axis.set_xticks(sorted(cumulative["through_week"].unique()))
        if percent:
            axis.set_ylim(0, 1)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle(
        f"TDNet {season} Scientific Roster · Cumulative Performance",
        fontsize=23,
        weight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Thin blue lines are frozen scientific models. Vegas MAE/SU use the frozen closing spread; Vegas Brier uses available no-vig moneylines; ATS is the explicit 50% no-vig benchmark.",
        ha="center",
        fontsize=9.5,
        color="#555B63",
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def plot_scientific_rolling_summary(
    performance: pd.DataFrame, path: str | Path, *, season: int
) -> Path:
    """Compare weekly and cumulative consensus with the roster and Vegas."""
    apply_tdnet_theme()
    specs = (
        ("margin_mae", "Margin MAE", False),
        ("su_accuracy", "Straight-up accuracy", True),
        ("brier_score", "Brier score", False),
        ("ats_accuracy", "ATS accuracy", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), facecolor="#F7F4ED")
    for axis, (metric, title, percent) in zip(axes.flat, specs):
        cumulative = performance.loc[performance["scope"].eq("cumulative")]
        weekly_consensus = performance.loc[
            performance["scope"].eq("weekly")
            & performance["series_type"].eq("consensus")
        ]
        weekly_vegas = performance.loc[
            performance["scope"].eq("weekly")
            & performance["series_type"].eq("vegas")
        ]
        _plot_metric_lines(axis, cumulative, metric, show_models=False)
        axis.plot(
            weekly_consensus["through_week"],
            weekly_consensus[metric],
            color="#E83E8C",
            lw=1.5,
            ls=":",
            marker="o",
            alpha=0.7,
            label="Scientific consensus · weekly",
        )
        axis.plot(
            weekly_vegas["through_week"],
            weekly_vegas[metric],
            color="#D99A00",
            lw=1.4,
            ls=":",
            marker="s",
            alpha=0.7,
            label="Vegas baseline · weekly",
        )
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel("Completed week")
        axis.set_xticks(sorted(cumulative["through_week"].unique()))
        if percent:
            axis.set_ylim(0, 1)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8.5)
    fig.suptitle(
        f"TDNet {season} Scientific Roster · Weekly and Cumulative",
        fontsize=23,
        weight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Solid/dotted pink are cumulative/weekly scientific consensus; dashed/dotted gold are cumulative/weekly Vegas baselines.",
        ha="center",
        fontsize=9.5,
        color="#555B63",
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def plot_cumulative_scientific_scorecard(
    scorecard: pd.DataFrame, path: str | Path, *, season: int, completed_week: int
) -> Path:
    """Render the season-to-date scientific scorecard including market baselines."""

    def record(row: pd.Series, prefix: str) -> str:
        if row["series_type"] == "vegas" and prefix == "ats":
            return "No-vig baseline"
        values = [row[f"{prefix}_wins"], row[f"{prefix}_losses"]]
        if prefix == "ats":
            values.append(row["ats_pushes"])
        return "–".join(str(int(value)) for value in values)

    table = pd.DataFrame(
        {
            "Rank": scorecard["scorecard_rank"].map(
                lambda value: "—" if pd.isna(value) else str(int(value))
            ),
            "Model / comparator": scorecard["model_name"],
            "Tier": scorecard["fingerprint"].where(
                ~scorecard["series_type"].eq("vegas"), "Market"
            ),
            "G": scorecard["games"].astype(int),
            "Margin MAE": scorecard["margin_mae"].map(
                lambda x: "—" if pd.isna(x) else f"{x:.2f}"
            ),
            "SU": scorecard.apply(lambda row: record(row, "su"), axis=1),
            "SU %": scorecard["su_accuracy"].map(
                lambda x: "—" if pd.isna(x) else f"{x:.1%}"
            ),
            "Brier": scorecard["brier_score"].map(
                lambda x: "—" if pd.isna(x) else f"{x:.3f}"
            ),
            "ATS": scorecard.apply(lambda row: record(row, "ats"), axis=1),
            "ATS %": scorecard["ats_accuracy"].map(
                lambda x: "—" if pd.isna(x) else f"{x:.1%}"
            ),
        }
    )
    fig, axis = plt.subplots(
        figsize=(18, max(13, 0.43 * len(table) + 3)), facecolor="#F7F4ED"
    )
    axis.axis("off")
    plotted = axis.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0.035, 1, 0.91],
        colWidths=[0.05, 0.25, 0.08, 0.045, 0.09, 0.08, 0.07, 0.07, 0.12, 0.07],
    )
    plotted.auto_set_font_size(False)
    plotted.set_fontsize(9.2)
    for (row, _), cell in plotted.get_celld().items():
        cell.set_edgecolor("#D4D7DB")
        if row == 0:
            cell.set_facecolor("#22324A")
            cell.set_text_props(color="white", weight="bold")
        elif row == 1:
            cell.set_facecolor("#FDE2F0")
            cell.set_text_props(weight="bold")
        elif row == 2:
            cell.set_facecolor("#FFF1C2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#EEF2F5")
    axis.set_title(
        f"TDNet {season} Scientific Models · Cumulative through Week {completed_week}",
        fontsize=21,
        weight="bold",
        pad=18,
    )
    fig.text(
        0.5,
        0.008,
        "Rank is by cumulative margin MAE. Vegas is an evaluation-only comparator; its ATS entry is the 50% no-vig benchmark, not a directional pick record.",
        ha="center",
        fontsize=9.5,
        color="#555B63",
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def write_scientific_cumulative_artifacts(
    *,
    publication_root: str | Path,
    season: int,
    completed_week: int,
    output_root: str | Path,
    scientific_subdir: str | None = None,
    consensus_label: str = CONSENSUS_LABEL,
    consensus_fingerprint: str = "F0–F6",
) -> dict[str, Path]:
    """Write the cumulative data, scorecard, figures, and refreshed scientific manifest."""
    output = Path(output_root)
    performance = build_scientific_rolling_performance(
        publication_root=publication_root,
        season=season,
        completed_week=completed_week,
        scientific_subdir=scientific_subdir,
        consensus_label=consensus_label,
        consensus_fingerprint=consensus_fingerprint,
    )
    # F7 is a learned market-only benchmark.  In the full scientific cohort it
    # would otherwise occupy six visually prominent rows/lines while repeating
    # information already represented by the single Vegas comparator.  Keep its
    # metrics in the audit CSV, but omit it from ranked/display artifacts.
    excluded_display_fingerprints = {"F7"} if scientific_subdir == "full_f0_f8" else set()
    display_performance = performance.loc[
        ~(
            performance["series_type"].eq("model")
            & performance["fingerprint"].isin(excluded_display_fingerprints)
        )
    ].copy()
    scorecard = cumulative_scientific_scorecard(
        display_performance, completed_week=completed_week
    )
    paths = {
        "rolling_csv": output / "scientific_rolling_cumulative_performance.csv",
        "scorecard_csv": output / "scientific_cumulative_model_scorecard.csv",
        "scorecard_png": output / "scientific_cumulative_model_scorecard.png",
        "rolling_png": output / "scientific_rolling_cumulative_roster_performance.png",
        "all_models_png": output / "scientific_cumulative_all_models_performance.png",
    }
    performance.to_csv(paths["rolling_csv"], index=False)
    scorecard.to_csv(paths["scorecard_csv"], index=False)
    plot_cumulative_scientific_scorecard(
        scorecard, paths["scorecard_png"], season=season, completed_week=completed_week
    )
    plot_scientific_rolling_summary(
        display_performance, paths["rolling_png"], season=season
    )
    plot_scientific_cumulative_all_models(
        display_performance, paths["all_models_png"], season=season
    )

    manifest_path = output / "scientific_postgame_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.setdefault("files", {})
        consensus_png = output / "scientific_consensus_game_results.png"
        if consensus_png.exists():
            files["consensus_results_png"] = {
                "path": consensus_png.name,
                "sha256": sha256_file(consensus_png),
                "size_bytes": consensus_png.stat().st_size,
            }
        for name, path in paths.items():
            files[name] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        manifest["cumulative_through_week"] = int(completed_week)
        manifest["vegas_baselines_included"] = True
        manifest["cumulative_display_excluded_fingerprints"] = sorted(
            excluded_display_fingerprints
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    readme_path = output / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        note = (
            "- `scientific_cumulative_model_scorecard.csv` and `.png` rank the "
            "season-to-date scientific roster and include the evaluation-only Vegas "
            "baselines. In the full cohort, F7 remains in the rolling audit CSV but is "
            "omitted from scorecards and figures because it is the market-only tier.\n"
        )
        if "scientific_cumulative_model_scorecard.csv" not in readme:
            marker = "- No next-week matchup predictions are generated in this directory."
            if marker in readme:
                readme = readme.replace(marker, note + marker)
            else:
                readme = readme.rstrip() + "\n\n" + note
            readme_path.write_text(readme, encoding="utf-8")
    return paths
