#!/usr/bin/env python3
"""Plot all completed model/fingerprint results in MAE–accuracy–Brier space."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/nextgen_fingerprints/figures"
DATA = FIGURES / "all_architectures_scatter_data.csv"
MODEL_LABELS = {
    "M1": "M1 · Linear", "M2": "M2 · Spline", "M3": "M3 · Random forest",
    "M4": "M4 · Boosted trees", "M5": "M5 · Neural net", "M10": "M10 · KNN",
}
MODEL_COLORS = {
    "M1": "Brass", "M2": "Ion Blue", "M3": "Electric Emerald",
    "M4": "Gridiron Violet", "M5": "Soft Mint", "M10": "Slate Line",
}
SYMBOLS = {"historical": "o", "broad": "D", "narrow": "s", "market": "*"}


def main() -> None:
    palette = pd.read_csv(ROOT / "docs/style/color_palettes/tdnet_palette.csv")
    colors = dict(zip(palette.name, palette.hex))
    frame = pd.read_csv(DATA)
    pair_counts = frame.groupby(["stage", "model"]).size()
    if len(frame) != 110 or not pair_counts.eq(1).all():
        raise ValueError("Expected one plotted summary for each of the 110 model/fingerprint pairs")

    metric_columns = ["mae", "winner_accuracy", "brier_score"]
    metric_values = frame[metric_columns].to_numpy(float)
    metric_mean = metric_values.mean(axis=0)
    metric_sd = metric_values.std(axis=0, ddof=1)
    standardized = (metric_values - metric_mean) / metric_sd
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(standardized, rowvar=False))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    pc1 = eigenvectors[:, 0]
    if not (pc1[0] < 0 and pc1[1] > 0 and pc1[2] < 0):
        pc1 *= -1
    frame["pc1_better_sd"] = standardized @ pc1 / np.sqrt(eigenvalues[0])

    labels = {
        "M1": "Linear", "M2": "Spline", "M3": "Random forest",
        "M4": "Boosted trees", "M5": "Neural net", "M10": "KNN",
    }
    full_scores = frame.groupby("model").pc1_better_sd.median().rename("full_range_median_score_sd")
    matched_scores = (frame.loc[frame.generation.ge(9)].groupby("model").pc1_better_sd
                     .median().rename("F09_F17_median_score_sd"))
    ranking = pd.concat([full_scores, matched_scores], axis=1).reset_index()
    ranking = ranking.sort_values("full_range_median_score_sd", ascending=False)
    ranking.insert(0, "rank_full_range", range(1, len(ranking) + 1))
    ranking["rank_F09_F17"] = ranking.F09_F17_median_score_sd.rank(
        method="min", ascending=False).astype(int)
    ranking["model_name"] = ranking.model.map(labels)
    ranking.to_csv(FIGURES / "all_fingerprints_f0_f17_pca_model_ranking.csv", index=False,
                   float_format="%.6f")

    fingerprint_ranking = (frame.groupby(["stage", "generation", "setting"], as_index=False)
                           .agg(score_sd=("pc1_better_sd", "median"),
                                models_in_generation=("model", "nunique")))
    fingerprint_ranking = fingerprint_ranking.sort_values("score_sd", ascending=False)
    fingerprint_ranking.insert(0, "rank", range(1, len(fingerprint_ranking) + 1))
    fingerprint_ranking = fingerprint_ranking.rename(columns={"score_sd": "median_pc1_score_sd"})
    fingerprint_ranking.to_csv(
        FIGURES / "all_fingerprints_f0_f17_pca_fingerprint_ranking.csv", index=False,
        float_format="%.6f")

    pair_ranking = frame[["stage", "generation", "setting", "model", "n_games",
                          "mae", "winner_accuracy", "brier_score", "pc1_better_sd"]].copy()
    pair_ranking = pair_ranking.sort_values("pc1_better_sd", ascending=False)
    pair_ranking.insert(0, "rank", range(1, len(pair_ranking) + 1))
    pair_ranking = pair_ranking.rename(columns={"pc1_better_sd": "pc1_score_sd"})
    pair_ranking.to_csv(FIGURES / "all_fingerprints_f0_f17_pca_pair_ranking.csv", index=False,
                        float_format="%.6f")

    def marker(row: pd.Series) -> str:
        if row.stage == "F17_market":
            return SYMBOLS["market"]
        if row.setting == "historical rolling folds":
            return SYMBOLS["historical"]
        if row.setting == "nextgen 2025 broad A":
            return SYMBOLS["broad"]
        return SYMBOLS["narrow"]

    frame["marker"] = frame.apply(marker, axis=1)
    frame["color"] = frame.model.map(lambda model: colors[MODEL_COLORS[model]])

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": colors["Midnight Gridiron"],
        "text.color": colors["Midnight Gridiron"],
    })
    fig = plt.figure(figsize=(13.8, 9.2), facecolor=colors["Mist Panel"])
    ax = fig.add_subplot(111, projection="3d", facecolor=colors["Mist Panel"])
    ax.view_init(elev=23, azim=-55)
    ax.set_xlim(11.4, 17.5)
    ax.set_ylim(57, 76)
    ax.set_zlim(.164, .251)
    ax.set_xticks([12, 13, 14, 15, 16, 17])
    ax.set_yticks([60, 65, 70, 75])
    ax.set_zticks([.17, .19, .21, .23, .25])
    ax.set_xlabel("Margin MAE · points (lower is better)", labelpad=14, fontsize=11)
    ax.set_ylabel("Winner accuracy · % (higher is better)", labelpad=14, fontsize=11)
    ax.set_zlabel("Brier score (lower is better)", labelpad=3, fontsize=11)
    ax.tick_params(colors=colors["Slate Line"], labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(colors["Mist Panel"])
        axis.pane.set_edgecolor(colors["Polar Mist"])
        axis._axinfo["grid"]["color"] = colors["Polar Mist"]
        axis._axinfo["grid"]["linewidth"] = .8

    # Plot each cohort symbol separately so the shared 3D axes remain comparable.
    for symbol, subset in frame.groupby("marker", sort=False):
        for model, points in subset.groupby("model", sort=False):
            ax.scatter(
                points.mae, points.winner_accuracy * 100, points.brier_score,
                color=colors[MODEL_COLORS[model]], marker=symbol, s=66 if symbol != "*" else 140,
                alpha=.88, edgecolors="white" if symbol != "*" else colors["Midnight Gridiron"],
                linewidths=.55, depthshade=False,
            )

    # Arrow for a one-standard-deviation move along the favorable standardized PC1.
    vector = pc1 * np.sqrt(eigenvalues[0]) * metric_sd
    vector[1] *= 100  # plot winner accuracy in percentage points
    start = [metric_mean[0], metric_mean[1] * 100, metric_mean[2]]
    pc1_arrow = ax.quiver(*start, *vector, color=colors["Edge Pink"], linewidth=2.8,
                          arrow_length_ratio=.16, normalize=False)

    model_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[MODEL_COLORS[m]],
               markeredgecolor="white", markersize=8, label=MODEL_LABELS[m])
        for m in MODEL_LABELS
    ]
    cohort_handles = [
        Line2D([0], [0], marker=symbol, linestyle="", color=colors["Slate Line"],
               markersize=8, label=label)
        for label, symbol in [
            ("F0–F8 · historical folds", "o"),
            ("F06 · broad 2025 A", "D"),
            ("F09–F16 · narrow 2025 A", "s"),
            ("F17-market", "*"),
        ]
    ]
    pc1_handle = Line2D([0], [0], color=colors["Edge Pink"], marker=">", linewidth=2,
                        label="Favorable PC1 · +1 SD")
    legend = ax.legend(
        handles=model_handles + cohort_handles + [pc1_handle],
        loc="upper left", bbox_to_anchor=(1.22, .98), frameon=True,
        facecolor="white", edgecolor=colors["Polar Mist"], fontsize=9,
        title="Model color · cohort marker", title_fontsize=10,
    )
    legend.get_title().set_color(colors["Midnight Gridiron"])
    fig.suptitle("Fingerprint landscape in three metrics", x=.46, y=.96,
                 fontsize=19, fontweight="bold", color=colors["Midnight Gridiron"])
    subtitle = fig.text(
        .46, .915,
        "One marker per model × fingerprint pair (110 total). Pink arrow: +1 SD along the favorable PCA direction.",
        ha="center", fontsize=9, color=colors["Slate Line"])
    fig.text(.46, .045,
             "Lower MAE and Brier are better; higher winner accuracy is better.",
             ha="center", fontsize=9, color=colors["Slate Line"])
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        output_path = FIGURES / f"all_fingerprints_f0_f17_3d.{suffix}"
        fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
        if suffix == "svg":
            output_path.write_text("\n".join(line.rstrip() for line in output_path.read_text().splitlines()) + "\n")
    # Keep the PC1 arrow on the still figure, but omit it from the rotating GIF.
    pc1_arrow.remove()
    subtitle.set_text("One marker per model × fingerprint pair (110 total).")
    legend.remove()
    legend = ax.legend(
        handles=model_handles + cohort_handles,
        loc="upper left", bbox_to_anchor=(1.22, .98), frameon=True,
        facecolor="white", edgecolor=colors["Polar Mist"], fontsize=9,
        title="Model color · cohort marker", title_fontsize=10,
    )
    legend.get_title().set_color(colors["Midnight Gridiron"])
    ax.set_position([.05, .10, .69, .78])
    legend.set_bbox_to_anchor((1.10, .95))
    animation = FuncAnimation(
        fig,
        lambda frame: ax.view_init(elev=23, azim=-55 + 360 * frame / 96),
        frames=96,
        interval=125,
        blit=False,
    )
    animation.save(
        FIGURES / "all_fingerprints_f0_f17_3d_rotation.gif",
        writer=PillowWriter(fps=8),
        dpi=90,
        savefig_kwargs={"facecolor": fig.get_facecolor()},
    )
    plt.close(fig)

    # Alternate view: generation controls the red-to-blue gradient; marker shape is model.
    model_symbols = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "P", "M10": "X"}
    generation_map = LinearSegmentedColormap.from_list(
        "fingerprint_generation_red_blue", ["#D73027", "#2166AC"])
    generation_norm = Normalize(vmin=0, vmax=17)
    fig2 = plt.figure(figsize=(13.8, 9.2), facecolor=colors["Mist Panel"])
    ax2 = fig2.add_axes([.05, .12, .68, .76], projection="3d", facecolor=colors["Mist Panel"])
    ax2.view_init(elev=23, azim=-55)
    ax2.set_xlim(11.4, 17.5)
    ax2.set_ylim(57, 76)
    ax2.set_zlim(.164, .251)
    ax2.set_xticks([12, 13, 14, 15, 16, 17])
    ax2.set_yticks([60, 65, 70, 75])
    ax2.set_zticks([.17, .19, .21, .23, .25])
    ax2.set_xlabel("Margin MAE · points (lower is better)", labelpad=14, fontsize=11)
    ax2.set_ylabel("Winner accuracy · % (higher is better)", labelpad=14, fontsize=11)
    ax2.set_zlabel("Brier score (lower is better)", labelpad=3, fontsize=11)
    ax2.tick_params(colors=colors["Slate Line"], labelsize=8)
    for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
        axis.pane.set_facecolor(colors["Mist Panel"])
        axis.pane.set_edgecolor(colors["Polar Mist"])
        axis._axinfo["grid"]["color"] = colors["Polar Mist"]
        axis._axinfo["grid"]["linewidth"] = .8
    for model, points in frame.groupby("model", sort=False):
        ax2.scatter(
            points.mae, points.winner_accuracy * 100, points.brier_score,
            c=points.generation, cmap=generation_map, norm=generation_norm,
            marker=model_symbols[model], s=72, alpha=.94,
            edgecolors="white", linewidths=.55, depthshade=False,
        )
    model_handles2 = [
        Line2D([0], [0], marker=model_symbols[model], linestyle="",
               color=colors["Slate Line"], markeredgecolor="white", markersize=8,
               label=MODEL_LABELS[model])
        for model in MODEL_LABELS
    ]
    legend2 = fig2.legend(
        handles=model_handles2, loc="upper right", bbox_to_anchor=(.985, .94),
        frameon=True, facecolor="white", edgecolor=colors["Polar Mist"],
        fontsize=9, title="Marker · model", title_fontsize=10,
    )
    legend2.get_title().set_color(colors["Midnight Gridiron"])
    scalar = ScalarMappable(norm=generation_norm, cmap=generation_map)
    scalar.set_array([])
    colorbar = fig2.colorbar(scalar, cax=fig2.add_axes([.78, .24, .025, .50]))
    colorbar.set_ticks([0, 3, 6, 9, 12, 15, 17])
    colorbar.set_ticklabels(["F0", "F3", "F6", "F9", "F12", "F15", "F17"])
    colorbar.set_label("Fingerprint generation · red → blue", color=colors["Midnight Gridiron"])
    colorbar.ax.tick_params(colors=colors["Slate Line"], labelsize=8)
    fig2.suptitle("Fingerprint landscape · color by generation", x=.43, y=.96,
                  fontsize=19, fontweight="bold", color=colors["Midnight Gridiron"])
    fig2.text(.43, .915,
              "One marker per model × fingerprint pair (110 total). Later generations are bluer.",
              ha="center", fontsize=9, color=colors["Slate Line"])
    fig2.text(.43, .045,
              "Lower MAE and Brier are better; higher winner accuracy is better.",
              ha="center", fontsize=9, color=colors["Slate Line"])
    generation_animation = FuncAnimation(
        fig2,
        lambda frame_idx: ax2.view_init(elev=23, azim=-55 + 360 * frame_idx / 96),
        frames=96, interval=125, blit=False,
    )
    generation_animation.save(
        FIGURES / "all_fingerprints_f0_f17_3d_rotation_by_generation.gif",
        writer=PillowWriter(fps=8), dpi=90,
        savefig_kwargs={"facecolor": fig2.get_facecolor()},
    )
    plt.close(fig2)


if __name__ == "__main__":
    main()
