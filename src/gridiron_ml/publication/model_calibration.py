"""Season-to-date individual-model home-win calibration overlays."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .figure_theme import TDNET_COLORS, apply_tdnet_theme

ROSTERS = (
    (
        "margin_wide",
        "Margin-wide roster",
        "tables/margin_wide_model_game_results.csv",
        "figures/margin_wide_cumulative_model_calibration.png",
        "tables/margin_wide_cumulative_model_calibration.csv",
    ),
    (
        "scientific_f0_f6",
        "Scientific F0–F6 roster",
        "scientific/scientific_model_game_results.csv",
        "scientific/scientific_cumulative_model_calibration.png",
        "scientific/scientific_cumulative_model_calibration.csv",
    ),
    (
        "scientific_full_f0_f8",
        "Full scientific F0–F8 roster (F7 omitted)",
        "scientific/full_f0_f8/scientific_model_game_results.csv",
        "scientific/full_f0_f8/scientific_cumulative_model_calibration.png",
        "scientific/full_f0_f8/scientific_cumulative_model_calibration.csv",
    ),
)


def _short_model_label(model: object, fingerprint: object, family: object) -> str:
    text = str(model)
    scientific = re.fullmatch(r"scientific_(F\d+)_(M\d+)", text)
    if scientific:
        return f"{scientific.group(1)}/{scientific.group(2)} {family}"
    return text.removeprefix("margin_").replace("_", " ")


def load_cumulative_model_predictions(
    *,
    publication_root: str | Path,
    completed_week: int,
    relative_path: str,
    omit_fingerprints: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Load one prediction per model/game from every completed publication week."""
    root = Path(publication_root)
    frames = []
    for week in range(int(completed_week) + 1):
        path = root / f"week_{week:02d}" / "post_game" / relative_path
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = {
            "game_id",
            "model_name",
            "pred_home_win_probability",
            "actual_home_margin",
        }
        if missing := required - set(frame):
            raise ValueError(f"{path} is missing calibration columns {sorted(missing)}.")
        frame["publication_week"] = int(week)
        frames.append(frame)
    if not frames:
        raise ValueError(f"No cumulative calibration inputs found for {relative_path}.")
    output = pd.concat(frames, ignore_index=True)
    current_roster = set(
        output.loc[
            output["publication_week"].eq(int(completed_week)), "model_name"
        ].astype(str)
    )
    if not current_roster:
        raise ValueError(
            f"No Week {completed_week} roster was found for {relative_path}."
        )
    output = output.loc[output["model_name"].astype(str).isin(current_roster)].copy()
    output["fingerprint"] = output.get("fingerprint", pd.Series("", index=output.index)).astype(str)
    if omit_fingerprints:
        output = output.loc[~output["fingerprint"].isin(omit_fingerprints)].copy()
    output["model_family"] = output.get(
        "model_family", pd.Series("model", index=output.index)
    ).astype(str)
    output["predicted_home_win_probability"] = pd.to_numeric(
        output["pred_home_win_probability"], errors="coerce"
    )
    output["actual_home_win"] = (
        pd.to_numeric(output["actual_home_margin"], errors="coerce") > 0
    ).astype(float)
    output = output.dropna(
        subset=["predicted_home_win_probability", "actual_home_win"]
    )
    output = output.loc[output["predicted_home_win_probability"].between(0, 1)]
    return output.drop_duplicates(
        ["publication_week", "game_id", "model_name"], keep="last"
    ).reset_index(drop=True)


def build_model_calibration_table(
    predictions: pd.DataFrame, *, bins: int = 5
) -> pd.DataFrame:
    """Build equal-count reliability points for every individual model."""
    rows: list[dict[str, object]] = []
    for model_name, frame in predictions.groupby("model_name", sort=True):
        work = frame.copy()
        unique = work["predicted_home_win_probability"].nunique()
        quantiles = min(int(bins), len(work), int(unique))
        if quantiles < 2:
            continue
        work["probability_bin"] = pd.qcut(
            work["predicted_home_win_probability"], q=quantiles, duplicates="drop"
        )
        fingerprint = str(work["fingerprint"].iloc[0])
        family = str(work["model_family"].iloc[0])
        label = _short_model_label(model_name, fingerprint, family)
        points = (
            work.groupby("probability_bin", observed=True)
            .agg(
                predicted_home_win_probability=(
                    "predicted_home_win_probability",
                    "mean",
                ),
                observed_home_win_rate=("actual_home_win", "mean"),
                games=("actual_home_win", "size"),
            )
            .reset_index(drop=True)
        )
        points.insert(0, "model_label", label)
        points.insert(0, "model_family", family)
        points.insert(0, "fingerprint", fingerprint)
        points.insert(0, "model_name", str(model_name))
        points["total_model_games"] = len(work)
        rows.extend(points.to_dict("records"))
    if not rows:
        raise ValueError("No individual model had enough predictions for calibration.")
    return pd.DataFrame(rows)


def plot_model_calibration_overlay(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    completed_week: int,
    roster_label: str,
) -> Path:
    """Render the historical two-panel reliability overlay for a current roster."""
    apply_tdnet_theme()
    fig, (curve_axis, density_axis) = plt.subplots(
        2,
        1,
        figsize=(22, 14),
        height_ratios=(4.2, 1.15),
        sharex=True,
        facecolor="#F7F4ED",
    )
    fig.subplots_adjust(left=0.07, right=0.75, top=0.90, bottom=0.08, hspace=0.08)
    for axis in (curve_axis, density_axis):
        axis.set_facecolor("#FFFFFF")
        axis.spines[["top", "right"]].set_visible(False)
    models = sorted(calibration["model_name"].unique())
    color_map = plt.get_cmap("turbo")
    bins = np.linspace(0, 1, 21)
    for index, model_name in enumerate(models):
        color = color_map(index / max(1, len(models) - 1))
        points = calibration.loc[calibration["model_name"].eq(model_name)]
        label = str(points["model_label"].iloc[0])
        curve_axis.plot(
            points["predicted_home_win_probability"],
            points["observed_home_win_rate"],
            marker="o",
            ms=3.8,
            lw=1.35,
            alpha=0.82,
            color=color,
            label=label,
        )
        probabilities = predictions.loc[
            predictions["model_name"].eq(model_name),
            "predicted_home_win_probability",
        ]
        density_axis.hist(
            probabilities,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.05,
            alpha=0.72,
            color=color,
        )
    curve_axis.plot(
        [0, 1],
        [0, 1],
        color=TDNET_COLORS["slate"],
        ls="--",
        lw=2.0,
        label="Perfect calibration",
        zorder=0,
    )
    curve_axis.set_xlim(0, 1)
    curve_axis.set_ylim(0, 1)
    curve_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    curve_axis.set_ylabel("Observed home-win rate")
    curve_axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=7.4,
        ncol=2 if len(models) > 36 else 1,
        title=f"Individual models ({len(models)})",
        title_fontsize=9,
    )
    density_axis.set_xlim(0, 1)
    density_axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    density_axis.set_xlabel("Predicted home-win probability")
    density_axis.set_ylabel("Density")
    curve_axis.set_title(
        f"TDNet {season} Through Week {completed_week}: {roster_label} Calibration\n"
        f"{int(calibration.groupby('model_name').size().median())} equal-count probability bins · "
        "one line per individual frozen model",
        fontsize=22,
        weight="bold",
        color=TDNET_COLORS["midnight_gridiron"],
        pad=18,
    )
    fig.text(
        0.41,
        0.018,
        "The dashed diagonal is perfect calibration. The lower panel shows each model's season-to-date probability distribution.",
        ha="center",
        fontsize=10,
        color=TDNET_COLORS["slate"],
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def write_cumulative_model_calibration_artifacts(
    *,
    publication_root: str | Path,
    output_root: str | Path,
    season: int,
    completed_week: int,
) -> dict[str, Path]:
    """Write current-season calibration overlays for all three frozen rosters."""
    output = Path(output_root)
    paths: dict[str, Path] = {}
    audit: dict[str, object] = {
        "schema": "tdnet-cumulative-individual-model-calibration-v1",
        "season": int(season),
        "completed_week": int(completed_week),
        "definition": {
            "x": "mean predicted home-win probability within equal-count model-specific bins",
            "y": "observed home-win rate within the same bins",
            "bins_per_model": "adaptive: max(3, min(10, completed games // 30))",
            "bin_rationale": "Keep roughly 30 games per point while growing toward the historical ten-bin format.",
            "density": "individual model predicted home-win probability distribution",
        },
        "rosters": {},
    }
    for key, label, relative_source, relative_figure, relative_table in ROSTERS:
        omit = ("F7",) if key == "scientific_full_f0_f8" else ()
        predictions = load_cumulative_model_predictions(
            publication_root=publication_root,
            completed_week=completed_week,
            relative_path=relative_source,
            omit_fingerprints=omit,
        )
        minimum_model_games = int(predictions.groupby("model_name").size().min())
        bin_count = max(3, min(10, minimum_model_games // 30))
        calibration = build_model_calibration_table(predictions, bins=bin_count)
        figure_path = output / relative_figure
        table_path = output / relative_table
        table_path.parent.mkdir(parents=True, exist_ok=True)
        calibration.to_csv(table_path, index=False)
        plot_model_calibration_overlay(
            predictions,
            calibration,
            figure_path,
            season=season,
            completed_week=completed_week,
            roster_label=label,
        )
        paths[f"{key}_figure"] = figure_path
        paths[f"{key}_table"] = table_path
        if key == "margin_wide":
            family_groups = {
                "linear": {"linear"},
                "neighbors_and_kernels": {"knn", "kernel"},
                "trees_and_temporal": {"tree", "boosted", "temporal"},
                "ensembles_and_other": {
                    "ensemble",
                    "neural",
                    "spline",
                    "stat",
                    "structured_neural",
                },
            }
            detail_groups = [
                (group_name, predictions.loc[predictions["model_family"].isin(families)])
                for group_name, families in family_groups.items()
            ]
            detail_root = output / "figures" / "calibration_by_model_group"
        else:
            detail_groups = list(predictions.groupby("fingerprint", sort=True))
            detail_root = (
                output
                / Path(relative_figure).parent
                / "calibration_by_fingerprint"
            )
        detail_paths = []
        for group_name, group_predictions in detail_groups:
            if group_predictions.empty:
                continue
            group_calibration = calibration.loc[
                calibration["model_name"].isin(group_predictions["model_name"].unique())
            ]
            detail_path = detail_root / f"{str(group_name).casefold()}_calibration.png"
            plot_model_calibration_overlay(
                group_predictions,
                group_calibration,
                detail_path,
                season=season,
                completed_week=completed_week,
                roster_label=f"{label} · {group_name}",
            )
            paths[f"{key}_{str(group_name).casefold()}_figure"] = detail_path
            detail_paths.append(str(detail_path.relative_to(output)))
        audit["rosters"][key] = {
            "label": label,
            "models": int(predictions["model_name"].nunique()),
            "games_per_model_min": int(predictions.groupby("model_name").size().min()),
            "games_per_model_max": int(predictions.groupby("model_name").size().max()),
            "bins_per_model": int(bin_count),
            "fingerprints": sorted(predictions["fingerprint"].unique().tolist()),
            "omitted_fingerprints": list(omit),
            "source": relative_source,
            "figure": relative_figure,
            "table": relative_table,
            "readable_detail_figures": detail_paths,
        }
    metadata_path = output / "metadata" / "cumulative_model_calibration.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["metadata"] = metadata_path
    return paths
