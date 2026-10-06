#!/usr/bin/env python3
"""Render four rotating 3D metric surfaces for the 2026 scientific what-if."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from gridiron_ml.publication.figure_theme import TDNET_COLORS, apply_tdnet_theme

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication/2026/week_05/post_game/scientific/current_season_f0_f17_market"
FIGURES = PUB / "figures"
STAGES = [*(f"F{i}" for i in range(17)), "F17-market"]
MODELS = ["M1", "M2", "M3", "M4", "M5", "M10"]
FAMILY_TO_MODEL = {
    "linear": "M1", "spline": "M2", "tree": "M3",
    "boosted": "M4", "neural": "M5", "knn": "M10",
}
MODEL_LABELS = {
    "M1": "M1 · Linear", "M2": "M2 · Spline", "M3": "M3 · Tree",
    "M4": "M4 · Boosted", "M5": "M5 · Neural", "M10": "M10 · KNN",
}
METRICS = [
    ("ats_accuracy", "Against-the-spread accuracy", "ATS accuracy", True),
    ("su_accuracy", "Straight-up winner accuracy", "Winner accuracy", True),
    ("margin_mae", "Margin mean absolute error", "MAE · points", False),
    ("brier_score", "Brier score", "Brier score", False),
]


def score_grid(scorecard: pd.DataFrame, metric: str) -> tuple[np.ndarray, pd.DataFrame]:
    frame = scorecard.loc[scorecard.series_type.eq("model")].copy()
    frame["model_id"] = frame.model_family.map(FAMILY_TO_MODEL)
    frame = frame.loc[frame.coverage_status.isin(
        ["scored_2026", "scored_2026_partial_unverified_market_inputs",
         "scored_2026_unverified_quote_time"]
    )].copy()
    expected = pd.MultiIndex.from_product([STAGES, MODELS], names=["fingerprint", "model_id"])
    actual = frame.set_index(["fingerprint", "model_id"])
    values = actual[metric].reindex(expected).to_numpy(dtype=float).reshape(len(STAGES), len(MODELS)).T
    status = actual["coverage_status"].reindex(expected).rename("coverage_status").reset_index()
    if len(frame) != 108 or frame[["fingerprint", "model_id"]].duplicated().any():
        raise ValueError(f"Expected one score per each of the 108 2026 model × fingerprint cells; found {len(frame)}")
    if np.isnan(values[:, :17]).any() or np.isnan(values[:, 17]).any():
        raise ValueError(f"Incomplete metric surface for {metric}")
    return values, status


def metric_cmap(higher_is_better: bool) -> LinearSegmentedColormap:
    colors = TDNET_COLORS
    low, mid, high = (
        (colors["edge_pink"], colors["signal_orange"], colors["electric_emerald"])
        if higher_is_better else
        (colors["electric_emerald"], colors["signal_orange"], colors["edge_pink"])
    )
    return LinearSegmentedColormap.from_list("tdnet_surface_metric", [low, mid, high])


def render_surface(metric: str, title: str, zlabel: str, higher_is_better: bool,
                   values: np.ndarray, output_png: Path, output_gif: Path) -> dict:
    apply_tdnet_theme()
    colors = TDNET_COLORS
    x = np.arange(len(STAGES), dtype=float)
    y = np.arange(len(MODELS), dtype=float)
    X, Y = np.meshgrid(x, y)
    Z = values
    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    pad = max((zmax - zmin) * 0.08, 0.01 if "accuracy" in metric else 0.1)
    norm = Normalize(vmin=zmin, vmax=zmax)
    cmap = metric_cmap(higher_is_better)

    fig = plt.figure(figsize=(12, 9), facecolor=colors["parchment"])
    ax = fig.add_subplot(111, projection="3d", facecolor=colors["parchment_panel"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(colors["parchment_panel"])
        axis.pane.set_edgecolor(colors["polar_mist"])
        axis._axinfo["grid"]["color"] = colors["polar_mist"]
        axis._axinfo["grid"]["linewidth"] = 0.75
    surface = ax.plot_surface(
        X, Y, Z, cmap=cmap, norm=norm, rstride=1, cstride=1,
        linewidth=0.48, edgecolor=colors["parchment"], antialiased=True,
        alpha=0.94, shade=True,
    )
    # Discrete stage/model measurements remain visible at every surface vertex.
    ax.scatter(X, Y, Z, c=Z, cmap=cmap, norm=norm, s=13,
               edgecolors=colors["midnight_gridiron"], linewidths=0.32,
               depthshade=False, zorder=5)
    # F7/F8 include four Week 1 lines from a postgame refreshed snapshot.
    partial = np.array([7, 8])
    ax.scatter(np.repeat(partial, len(MODELS)), np.tile(y, len(partial)),
               Z[:, partial].T.reshape(-1), marker="s", s=44,
               facecolors="none", edgecolors=colors["signal_orange"],
               linewidths=1.7, depthshade=False, zorder=8)
    # The final column is exploratory because its target-game quote times are unknown.
    ax.scatter(np.full(len(MODELS), 17.0), y, Z[:, 17], marker="x", s=76,
               color=colors["midnight_gridiron"], linewidths=1.8,
               depthshade=False, zorder=8)
    ax.plot([16.5, 16.5], [-0.5, len(MODELS) - 0.5], [zmax + pad] * 2,
            color=colors["midnight_gridiron"], linestyle="--", linewidth=1.4)
    ax.set_xlim(-0.5, 17.5)
    ax.set_ylim(-0.5, len(MODELS) - 0.5)
    ax.set_zlim(zmin - pad, zmax + pad)
    ax.set_xticks([0, 4, 8, 12, 16, 17])
    ax.set_xticklabels(["F0", "F4", "F8", "F12", "F16", "F17M"])
    ax.set_yticks(y)
    ax.set_yticklabels(MODELS)
    ax.set_xlabel("Fingerprint generation", labelpad=14)
    ax.set_ylabel("Scientific model", labelpad=12)
    ax.set_zlabel(zlabel, labelpad=10)
    ax.tick_params(colors=colors["midnight_gridiron"], labelsize=9)
    ax.set_box_aspect((18, 7, 7))
    ax.view_init(elev=27, azim=-58)
    direction = "Higher is better" if higher_is_better else "Lower is better"
    fig.suptitle(f"2026 scientific roster · {title}", y=.96, fontsize=18,
                 fontweight="bold", color=colors["midnight_gridiron"])
    fig.text(.5, .915,
             "90 cutoff-checked · 12 partial-market-timing · 6 quote-time-unverified cells",
             ha="center", fontsize=10.5, color=colors["slate"])
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, shrink=.67, pad=.11, aspect=25)
    cbar.set_label(zlabel, color=colors["midnight_gridiron"])
    uncertainty = Line2D([0], [0], color=colors["midnight_gridiron"], marker="x",
                         linestyle="--", label="F17-market · quote time unverified")
    partial_marker = Line2D([0], [0], color=colors["signal_orange"], marker="s",
                            markerfacecolor="none", linestyle="",
                            label="F7/F8 · 4/271 Week 1 lines from Oct 6 refresh")
    fig.legend(handles=[partial_marker, uncertainty], loc="lower center", bbox_to_anchor=(.5, .07),
               frameon=False, fontsize=10)
    fig.text(.5, .025, f"{direction} · F0–F6/F9–F16 cutoff-checked; F7/F8 and F17-market have market-time caveats.",
             ha="center", fontsize=9, color=colors["slate"])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    animation = FuncAnimation(
        fig, lambda k: ax.view_init(elev=27, azim=-58 + 360 * k / 72),
        frames=72, interval=167, blit=False,
    )
    animation.save(output_gif, writer=PillowWriter(fps=6), dpi=90,
                   savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    return {"metric": metric, "png": output_png.name, "gif": output_gif.name,
            "cells": int(Z.size), "cutoff_checked_cells": 90,
            "partial_market_timing_cells": 12, "unverified_quote_time_cells": 6,
            "z_min": zmin, "z_max": zmax, "rotation_frames": 72, "rotation_fps": 6}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    scorecard = pd.read_csv(PUB / "scientific_2026_current_season_scorecard.csv")
    outputs = []
    for metric, title, zlabel, higher in METRICS:
        values, _ = score_grid(scorecard, metric)
        outputs.append(render_surface(
            metric, title, zlabel, higher, values,
            FIGURES / f"scientific_2026_{metric}_3d_surface.png",
            FIGURES / f"scientific_2026_{metric}_3d_surface_rotation.gif",
        ))
    manifest = {
        "scope": "2026 current-season what-if through Week 5",
        "target_games": 271,
        "grid": {"fingerprints": STAGES, "models": MODELS},
        "cutoff_verified_cells": 90,
        "partial_market_timing_cells": 12,
        "exploratory_quote_time_unverified_cells": 6,
        "f17_market_quote_timestamp_available": False,
        "surface_points": "one measured scorecard value per model × fingerprint cell; no interpolation beyond adjacent grid cells",
        "rotations": outputs,
        "vegas_included": False,
    }
    (FIGURES / "scientific_2026_3d_surface_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (FIGURES / "README.md").write_text(
        "# 2026 scientific 3D views\n\n"
        "Four true mesh surfaces use fingerprint generation and model architecture as the x/y axes, with ATS accuracy, straight-up accuracy, margin MAE, or Brier on z. Each static PNG has a companion slow 72-frame GIF. Mesh vertices correspond to measured model × fingerprint cells; the surface only joins adjacent cells for viewing.\n\n"
        "The grid includes 90 cutoff-checked cells (F0–F6 and F9–F16), 12 partially unverified cells (F7/F8 each include four of 271 Week 1 line inputs from an October 6 refresh), and six F17-market exploratory cells. Orange outlined squares mark F7/F8; crosses mark F17-market, whose target-game quote times are unavailable. Vegas is omitted from the surfaces.\n"
    )
    print(json.dumps({"surfaces": outputs, "manifest": str(FIGURES / "scientific_2026_3d_surface_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
