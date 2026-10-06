"""Build honest 2026 season-to-date curves for the available scientific roster."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from gridiron_ml.publication.figure_theme import TDNET_COLORS, apply_tdnet_theme

STAGES = [*(f"F{i}" for i in range(17)), "F17-market"]
ARCHITECTURES = ["M1", "M2", "M3", "M4", "M5", "M10"]
ARCHITECTURE_LABELS = {
    "M1": "Linear",
    "M2": "Spline",
    "M3": "Tree",
    "M4": "Boosted",
    "M5": "Neural",
    "M10": "KNN",
}
LINESTYLES = {
    "M1": "-",
    "M2": "--",
    "M3": ":",
    "M4": "-.",
    "M5": (0, (5, 1.5, 1, 1.5)),
    "M10": (0, (1, 1)),
}
MODEL_PATTERN = re.compile(r"scientific_(F\d+)_M(\d+)$")
FAMILY_TO_MODEL_ID = {
    "linear": "M1",
    "spline": "M2",
    "tree": "M3",
    "boosted": "M4",
    "neural": "M5",
    "knn": "M10",
}
METRIC_SPECS = [
    ("brier_score", "Brier score", False),
    ("su_accuracy", "Straight-up winner accuracy", True),
    ("ats_accuracy", "Against-the-spread accuracy", True),
]
SCORECARD_COLUMNS = [
    "season",
    "through_week",
    "series_type",
    "model_name",
    "model_family",
    "fingerprint",
    "coverage_status",
    "games",
    "margin_mae",
    "upset_games",
    "upset_recall",
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


def _stage_number(stage: str) -> int:
    if stage == "F17-market":
        return 17
    return int(stage[1:])


def _model_details(name: str) -> tuple[str, str] | None:
    match = MODEL_PATTERN.fullmatch(str(name))
    if not match:
        return None
    return match.group(1), f"M{match.group(2)}"


def _full_scorecard(latest: pd.DataFrame, *, season: int, through_week: int) -> pd.DataFrame:
    values = latest.loc[latest["series_type"].isin(["model", "consensus", "vegas"])].copy()
    values["season"] = season
    values["coverage_status"] = "scored_2026"
    present = {
        (str(row.fingerprint), FAMILY_TO_MODEL_ID.get(str(row.model_family), ""))
        for row in values.loc[values["series_type"].eq("model")].itertuples()
    }
    missing: list[dict[str, object]] = []
    for stage in STAGES:
        for model_id in ARCHITECTURES:
            if (stage, model_id) in present:
                continue
            model_family = {
                "M1": "linear",
                "M2": "spline",
                "M3": "tree",
                "M4": "boosted",
                "M5": "neural",
                "M10": "knn",
            }[model_id]
            row = {column: np.nan for column in SCORECARD_COLUMNS}
            row.update(
                {
                    "season": season,
                    "through_week": through_week,
                    "series_type": "model",
                    "model_name": f"scientific_{stage}_{model_id}",
                    "model_family": model_family,
                    "fingerprint": stage,
                    "coverage_status": "no_2026_predictions",
                }
            )
            missing.append(row)
    values["through_week"] = through_week
    scorecard = pd.concat([values, pd.DataFrame(missing)], ignore_index=True)
    scorecard = scorecard.reindex(columns=SCORECARD_COLUMNS)
    scorecard["_stage_order"] = scorecard["fingerprint"].map(
        lambda value: _stage_number(value) if value in STAGES else 99
    )
    return scorecard.sort_values(
        ["series_type", "_stage_order", "model_family", "model_name"],
        kind="stable",
    ).drop(columns="_stage_order")


def _season_upset_recall(
    season_root: Path, *, through_week: int, scorecard: pd.DataFrame
) -> pd.DataFrame:
    """Add canonical upset recall from the immutable, week-by-week game records."""
    week_frames = []
    for week in range(through_week + 1):
        path = (
            season_root
            / f"week_{week:02d}"
            / "post_game"
            / "scientific"
            / "full_f0_f8"
            / "scientific_model_game_results.csv"
        )
        if path.exists():
            week_frames.append(pd.read_csv(path))
    if not week_frames:
        raise ValueError(f"No game-level 2026 records found under {season_root}.")
    games = pd.concat(week_frames, ignore_index=True)
    games["actual_home_margin"] = pd.to_numeric(games["actual_home_margin"], errors="coerce")
    games["market_spread_close"] = pd.to_numeric(games["market_spread_close"], errors="coerce")
    valid_market = games["market_spread_close"].notna() & games["market_spread_close"].ne(0)
    actual_home = games["actual_home_margin"].gt(0)
    favorite_home = games["market_spread_close"].lt(0)
    favorite_won = (favorite_home & actual_home) | (~favorite_home & ~actual_home)
    actual_upset = valid_market & ~favorite_won

    rows: list[dict[str, object]] = []
    for model_name, frame in games.groupby("model_name", sort=False):
        probability = pd.to_numeric(frame["pred_home_win_probability"], errors="coerce")
        picked_home = probability.ge(0.5)
        picked_favorite = (favorite_home.loc[frame.index] & picked_home) | (
            ~favorite_home.loc[frame.index] & ~picked_home
        )
        upset_rows = actual_upset.loc[frame.index] & probability.notna()
        rows.append(
            {
                "model_name": model_name,
                "upset_games": int(upset_rows.sum()),
                "upset_recall": float((~picked_favorite.loc[upset_rows]).mean())
                if upset_rows.any()
                else np.nan,
            }
        )

    unique_games = games.drop_duplicates("game_id").copy()
    consensus_probability = pd.to_numeric(
        unique_games["consensus_predicted_home_win_probability"], errors="coerce"
    )
    consensus_home = consensus_probability.ge(0.5)
    consensus_upset_rows = actual_upset.loc[unique_games.index] & consensus_probability.notna()
    consensus_favorite = (
        favorite_home.loc[unique_games.index] & consensus_home
    ) | (~favorite_home.loc[unique_games.index] & ~consensus_home)
    rows.append(
        {
            "model_name": "Full F0–F8 scientific consensus",
            "upset_games": int(consensus_upset_rows.sum()),
            "upset_recall": float((~consensus_favorite.loc[consensus_upset_rows]).mean())
            if consensus_upset_rows.any()
            else np.nan,
        }
    )
    vegas_favorite = pd.Series(True, index=unique_games.index)
    vegas_upset_rows = actual_upset.loc[unique_games.index]
    rows.append(
        {
            "model_name": "Vegas closing-line baseline",
            "upset_games": int(vegas_upset_rows.sum()),
            "upset_recall": float((~vegas_favorite.loc[unique_games.index][vegas_upset_rows]).mean())
            if vegas_upset_rows.any()
            else np.nan,
        }
    )
    upset = pd.DataFrame(rows).drop_duplicates("model_name", keep="last")
    return scorecard.drop(columns=["upset_games", "upset_recall"], errors="ignore").merge(
        upset, on="model_name", how="left", validate="many_to_one"
    )


def _plot(
    trajectory: pd.DataFrame,
    output: Path,
    *,
    season: int,
    through_week: int,
    reconstructed_games: int = 0,
) -> None:
    apply_tdnet_theme()
    fig, axes = plt.subplots(1, 3, figsize=(19, 8), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.82, bottom=0.30, wspace=0.22)
    generation_cmap = LinearSegmentedColormap.from_list(
        "tdnet_fingerprint_generation",
        [
            TDNET_COLORS["edge_pink"],
            TDNET_COLORS["gridiron_violet"],
            TDNET_COLORS["figure_primary"],
        ],
    )
    stage_colors = {
        stage: generation_cmap(_stage_number(stage) / (len(STAGES) - 1))
        for stage in STAGES
    }

    cumulative = trajectory.loc[trajectory["scope"].eq("cumulative")].copy()
    models = cumulative.loc[cumulative["series_type"].eq("model")].copy()
    models["details"] = models["model_name"].map(_model_details)
    models = models.loc[models["details"].notna()].copy()
    models[["parsed_stage", "model_id"]] = pd.DataFrame(
        models["details"].tolist(), index=models.index
    )
    max_scored_generation = max(_stage_number(stage) for stage in models.parsed_stage.unique())

    consensus = cumulative.loc[cumulative["series_type"].eq("consensus")]
    vegas = cumulative.loc[cumulative["series_type"].eq("vegas")]
    for ax, (metric, title, percent) in zip(axes, METRIC_SPECS):
        for (stage, model_id), group in models.groupby(["parsed_stage", "model_id"], sort=False):
            group = group.sort_values("through_week")
            ax.plot(
                group["through_week"],
                group[metric],
                color=stage_colors[stage],
                linestyle=LINESTYLES[model_id],
                linewidth=1.35,
                alpha=0.73,
                marker="o",
                markersize=2.8,
                zorder=2,
            )
        for frame, color, linestyle, label, marker in (
            (consensus, TDNET_COLORS["edge_pink"], "-", f"F0–F{max_scored_generation} scientific consensus", "o"),
            (vegas, TDNET_COLORS["figure_highlight"], "--", "Vegas closing-line baseline", "s"),
        ):
            frame = frame.sort_values("through_week")
            ax.plot(
                frame["through_week"],
                frame[metric],
                color=color,
                linestyle=linestyle,
                linewidth=2.7,
                marker=marker,
                markersize=5,
                label=label,
                zorder=5,
            )
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        ax.set_xlabel("Completed 2026 week")
        ax.set_xlim(-0.15, through_week + 0.3)
        ax.set_xticks(sorted(cumulative["through_week"].astype(int).unique()))
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            low = max(0.0, float(np.nanmin(cumulative[metric])) - 0.05)
            high = min(1.0, float(np.nanmax(cumulative[metric])) + 0.05)
            ax.set_ylim(low, high)
        else:
            low = max(0.0, float(np.nanmin(cumulative[metric])) - 0.015)
            high = float(np.nanmax(cumulative[metric])) + 0.015
            ax.set_ylim(low, high)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", alpha=0.12)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"TDNet {season} scientific roster · cumulative season performance",
        fontsize=22,
        fontweight="bold",
        color=TDNET_COLORS["midnight_gridiron"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.905,
        f"Games completed through Week {through_week}  ·  "
        f"{models.loc[models.through_week.eq(through_week), ['parsed_stage', 'model_id']].drop_duplicates().shape[0]} "
        "of 108 model × fingerprint cells scored for 2026",
        ha="center",
        fontsize=12,
        color=TDNET_COLORS["figure_axis"],
    )

    arch_handles = [
        Line2D([0], [0], color=TDNET_COLORS["figure_axis"], lw=1.8,
               linestyle=LINESTYLES[mid], label=f"{mid} · {ARCHITECTURE_LABELS[mid]}")
        for mid in ARCHITECTURES
    ]
    comparator_handles = [
        Line2D([0], [0], color=TDNET_COLORS["edge_pink"], lw=2.7, marker="o",
               label=f"F0–F{max(_stage_number(s) for s in models.parsed_stage.unique())} scientific consensus"),
        Line2D([0], [0], color=TDNET_COLORS["figure_highlight"], lw=2.7,
               linestyle="--", marker="s", label="Vegas closing-line baseline"),
    ]
    fig.legend(
        handles=arch_handles + comparator_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.145),
        ncol=4,
        frameon=False,
        fontsize=10,
        columnspacing=1.8,
        handlelength=2.8,
    )
    color_map = plt.cm.ScalarMappable(
        cmap=generation_cmap, norm=plt.Normalize(vmin=0, vmax=17)
    )
    color_map.set_array([])
    colorbar = fig.colorbar(
        color_map,
        ax=axes,
        orientation="horizontal",
        fraction=0.065,
        pad=0.19,
        aspect=45,
    )
    colorbar.set_ticks([0, 4, 8, 12, 17])
    colorbar.set_ticklabels(["F0", "F4", "F8", "F12", "F17-market"])
    colorbar.set_label("Fingerprint generation (red → blue)", labelpad=7)
    fig.text(
        0.5,
        0.018,
        (
            f"{reconstructed_games} Week 1 games reconstructed on Oct 6 from Week 0 inputs; "
            "Vegas closing lines were refreshed on Oct 6. "
            if reconstructed_games
            else "Curves use the archived 2026 pregame predictions. "
        )
        + f"F0–F{max(_stage_number(s) for s in models.parsed_stage.unique())} have 2026 predictions; "
          "F17-market remains unscored because quote timestamps are unavailable; no retrospective scores are substituted.",
        ha="center",
        fontsize=9.5,
        color=TDNET_COLORS["figure_axis"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build(
    *, source: Path, output_dir: Path, season: int, through_week: int,
    reconstructed_games: int = 0,
) -> None:
    trajectory = pd.read_csv(source)
    expected = {"scope", "through_week", "series_type", "model_name", "model_family", "fingerprint", *[m for m, _, _ in METRIC_SPECS]}
    missing_columns = expected.difference(trajectory.columns)
    if missing_columns:
        raise ValueError(f"Source is missing required columns: {sorted(missing_columns)}")
    trajectory = trajectory.loc[
        trajectory["scope"].eq("cumulative")
        & pd.to_numeric(trajectory["through_week"], errors="coerce").le(through_week)
    ].copy()
    if trajectory.empty or int(trajectory["through_week"].max()) != through_week:
        raise ValueError(f"Source has no cumulative records through week {through_week}.")
    latest = trajectory.loc[pd.to_numeric(trajectory["through_week"]).eq(through_week)].copy()
    scorecard = _full_scorecard(latest, season=season, through_week=through_week)
    season_root = source.parents[4]
    scorecard = _season_upset_recall(
        season_root, through_week=through_week, scorecard=scorecard
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "scientific_2026_cumulative_trajectory.csv"
    scorecard_path = output_dir / "scientific_2026_current_season_scorecard.csv"
    figure_path = output_dir / "scientific_2026_full_f0_f17_market_cumulative_performance.png"
    trajectory.to_csv(trajectory_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    _plot(
        trajectory, figure_path, season=season, through_week=through_week,
        reconstructed_games=reconstructed_games,
    )

    model_cells = scorecard.loc[scorecard["series_type"].eq("model")]
    coverage = {
        "season": season,
        "through_week": through_week,
        "scored_model_fingerprint_cells": int(model_cells["coverage_status"].eq("scored_2026").sum()),
        "expected_model_fingerprint_cells": len(STAGES) * len(ARCHITECTURES),
        "scored_fingerprints": sorted(model_cells.loc[model_cells["coverage_status"].eq("scored_2026"), "fingerprint"].unique().tolist(), key=_stage_number),
        "not_scored_fingerprints": sorted(model_cells.loc[model_cells["coverage_status"].eq("no_2026_predictions"), "fingerprint"].unique().tolist(), key=_stage_number),
        "source_trajectory": str(source),
        "figure": figure_path.name,
        "scorecard": scorecard_path.name,
        "trajectory": trajectory_path.name,
        "market_fingerprint_label": "F17-market",
    }
    (output_dir / "coverage_manifest.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )
    readme = (
        "# 2026 scientific fingerprint performance\n\n"
        f"This season-to-date view scores immutable 2026 predictions through Week {through_week}. "
        "It is not the retrospective 2024–25 development evaluation.\n\n"
        f"The source contains {int(model_cells['coverage_status'].eq('scored_2026').sum())} scored model × fingerprint cells across "
        f"{len(coverage['scored_fingerprints'])} generations out of 108 possible cells (F0–F17-market × six architectures). "
        "F17-market is left blank because the archived feature quotes lack verifiable pregame timestamps; no retrospective values are substituted.\n\n"
        f"The figure shows cumulative Brier score, straight-up accuracy, and ATS accuracy by completed week. Line color maps fingerprint generation from red (earlier) to blue (later); line style maps architecture. Thick pink is the equal-weight F0–F{max((_stage_number(s) for s in coverage['scored_fingerprints']), default=0)} scientific consensus, and brass is the Vegas closing-line baseline.\n\n"
        "`scientific_2026_current_season_scorecard.csv` has one row per model/fingerprint pair plus consensus and Vegas. `scientific_2026_cumulative_trajectory.csv` has the week-by-week cumulative series used to draw the curves. Margin MAE, upset recall, and underlying counts are included in the scorecard.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Scientific cumulative performance CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--through-week", type=int, required=True)
    parser.add_argument("--reconstructed-games", type=int, default=0)
    args = parser.parse_args()
    build(
        source=args.source, output_dir=args.output_dir, season=args.season,
        through_week=args.through_week, reconstructed_games=args.reconstructed_games,
    )


if __name__ == "__main__":
    main()
