#!/usr/bin/env python3
"""Rotate one final cumulative SU/ATS/Brier point for each scientific cell."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.publication.figure_theme import TDNET_COLORS, apply_tdnet_theme

SOURCE = ROOT / "publication/2026/research/scientific_f0_f20_posthoc_scorecards_week01_07"
OUTPUT = ROOT / "publication/2026/research/scientific_f0_f20_cumulative_3d_week01_07"
FINGERPRINTS = [*(f"F{i}" for i in range(17)), "F17-market", "F18", "F19", "F20"]
MODELS = ["M1", "M2", "M3", "M4", "M5", "M10"]
MARKERS = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "P", "M10": "X"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(frame: pd.DataFrame, *, paired: bool) -> None:
    expected = {(fingerprint, model) for fingerprint in FINGERPRINTS for model in MODELS}
    if set(zip(frame.fingerprint, frame.model_id)) != expected:
        raise ValueError("The 126-cell scientific roster is incomplete")
    if frame.duplicated(["fingerprint", "model_id", "week"]).any():
        raise ValueError("Duplicate cumulative cell-week")
    counts = frame.groupby(["fingerprint", "model_id"]).week.agg(["min", "max", "count"])
    if paired:
        if not (counts["min"].eq(1) & counts["max"].eq(6) & counts["count"].eq(6)).all():
            raise ValueError("Paired cohort must cover Weeks 1–6 for all 126 cells")
        if not frame.loc[frame.week.eq(6), "games"].eq(304).all():
            raise ValueError("Paired endpoints must use the same 304 games")
    else:
        old = counts.loc[counts.index.get_level_values("fingerprint") != "F20"]
        f20 = counts.loc[counts.index.get_level_values("fingerprint") == "F20"]
        if not (old["min"].eq(1) & old["max"].eq(7) & old["count"].eq(7)).all():
            raise ValueError("F0–F19 must have seven cumulative weeks")
        if not (f20["min"].eq(1) & f20["max"].eq(6) & f20["count"].eq(6)).all():
            raise ValueError("F20 must end at Week 6; Week 7 expert rows are absent")
        if not frame.loc[frame.week.eq(7), "games"].eq(380).all():
            raise ValueError("F0–F19 Week 7 endpoints must have 380 games")
        if not frame.loc[frame.fingerprint.eq("F20") & frame.week.eq(6), "games"].eq(320).all():
            raise ValueError("F20 Week 6 endpoints must have 320 games")
    if len(frame) != (756 if paired else 876):
        raise ValueError("Unexpected number of cumulative score rows")
    if not frame.su_wins.add(frame.su_losses).eq(frame.games).all():
        raise ValueError("SU counts do not reconcile")
    if not frame[["ats_wins", "ats_losses", "ats_pushes", "ats_no_pick"]].sum(axis=1).eq(frame.ats_line_games).all():
        raise ValueError("ATS counts do not reconcile")
    if not frame.brier_games.eq(frame.games).all():
        raise ValueError("Brier and SU denominators differ")
    if not np.allclose(frame.su_accuracy, frame.su_wins / frame.games):
        raise ValueError("SU rates do not reconcile")
    if not np.allclose(frame.ats_accuracy, frame.ats_wins / (frame.ats_wins + frame.ats_losses)):
        raise ValueError("ATS rates do not reconcile")
    if not (frame.brier.between(0, 1).all() and np.isfinite(frame.brier).all()):
        raise ValueError("Invalid Brier scores")


def load_full() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / "cumulative_scorecard.csv")
    validate(frame, paired=False)
    return frame


def load_paired() -> pd.DataFrame:
    weekly = pd.read_csv(SOURCE / "f20_paired_weekly_scorecard.csv")
    if len(weekly) != 126 * 6 or weekly.duplicated(["fingerprint", "model_id", "week"]).any():
        raise ValueError("Paired weekly scorecard is incomplete")
    if not weekly.groupby("week").games.nunique().eq(1).all():
        raise ValueError("Paired weekly game counts differ across cells")
    if weekly.groupby("week").games.first().sum() != 304:
        raise ValueError("Paired weeks do not total 304 games")
    if not weekly.ats_line_games.eq(weekly.games).all():
        raise ValueError("Paired cohort has an unexpected missing spread")
    frame = weekly.sort_values(["fingerprint", "model_id", "week"]).copy()
    groups = frame.groupby(["fingerprint", "model_id"], sort=False)
    for column in ("games", "su_wins", "su_losses", "brier_games", "ats_line_games",
                   "ats_wins", "ats_losses", "ats_pushes", "ats_no_pick", "ats_no_line"):
        frame[column] = groups[column].cumsum()
    frame["brier_numerator"] = (weekly.loc[frame.index, "brier"] * weekly.loc[frame.index, "brier_games"])
    frame["brier_numerator"] = frame.groupby(["fingerprint", "model_id"]).brier_numerator.cumsum()
    frame["brier"] = frame.brier_numerator / frame.brier_games
    frame = frame.drop(columns="brier_numerator")
    frame["su_accuracy"] = frame.su_wins / frame.games
    frame["ats_accuracy"] = frame.ats_wins / (frame.ats_wins + frame.ats_losses)
    validate(frame, paired=True)
    return frame


def render(frame: pd.DataFrame, *, paired: bool) -> dict:
    apply_tdnet_theme()
    colors = TDNET_COLORS
    label = "paired_304" if paired else "latest_available"
    fig = plt.figure(figsize=(12.5, 9.1), facecolor=colors["parchment"])
    ax = fig.add_axes([.055, .15, .77, .73], projection="3d", facecolor=colors["parchment_panel"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(colors["parchment_panel"])
        axis.pane.set_edgecolor(colors["polar_mist"])
        axis._axinfo["grid"]["color"] = colors["polar_mist"]
        axis._axinfo["grid"]["linewidth"] = .7
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=0, vmax=20)
    endpoints = frame.sort_values("week").groupby(["fingerprint", "model_id"], sort=False).tail(1)
    if len(endpoints) != 126:
        raise ValueError("Expected one endpoint per scientific cell")
    for point in endpoints.itertuples():
        fingerprint, model = point.fingerprint, point.model_id
        color = cmap(norm(FINGERPRINTS.index(fingerprint)))
        ax.scatter([point.su_accuracy * 100], [point.ats_accuracy * 100], [point.brier],
                   marker=MARKERS[model], s=105, color=color,
                   edgecolors=colors["parchment"], linewidths=.8, alpha=.88,
                   depthshade=False, zorder=4)
    ax.set_xlim(61, 81)
    ax.set_ylim(42.5, 55.5)
    ax.set_zlim(.145, .24)
    ax.set_xticks([62, 66, 70, 74, 78])
    ax.set_yticks([44, 48, 52, 55])
    ax.set_zticks([.15, .18, .21, .24])
    ax.set_xlabel("Straight-up accuracy · % ↑", labelpad=14, fontsize=13)
    ax.set_ylabel("ATS accuracy · % ↑", labelpad=14, fontsize=13)
    ax.set_zlabel("Brier score ↓", labelpad=12, fontsize=13)
    ax.tick_params(colors=colors["midnight_gridiron"], labelsize=11)
    ax.set_box_aspect((1.15, 1.15, .85), zoom=1.14)
    ax.view_init(elev=23, azim=-55)
    fig.suptitle("126 scientific cells · cumulative 3D performance", x=.48, y=.97,
                 fontsize=21, fontweight="bold", color=colors["midnight_gridiron"])
    subtitle = ("All 126 cells on the same 304 games · cumulative CFBD Weeks 1–6"
                if paired else "Latest available · F0–F19 through Week 7 (380 games); F20 through Week 6 (320)")
    fig.text(.48, .922, subtitle, ha="center", fontsize=12, color=colors["slate"])
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar_ax = fig.add_axes([.88, .29, .017, .48])
    cbar = fig.colorbar(scalar, cax=cbar_ax)
    cbar.set_ticks([0, 4, 8, 12, 16, 20])
    cbar.set_ticklabels(["F0", "F4", "F8", "F12", "F16", "F20"])
    cbar.set_label("Fingerprint", color=colors["midnight_gridiron"], fontsize=12)
    cbar.ax.tick_params(labelsize=10.5)
    handles = [Line2D([0], [0], marker=MARKERS[m], linestyle="", color=colors["gridiron_violet"],
                      markeredgecolor=colors["parchment"], markersize=11, label=m) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.48, .065),
               ncol=6, frameon=False, fontsize=10.5)
    note = ("One final cumulative point per F×M pair. Paired cohort supports direct comparison; F20 was fit after these games."
            if paired else "One final cumulative point per F×M pair. Cohorts differ; F20 Week 7 expert inputs are unavailable.")
    fig.text(.48, .023, note, ha="center", fontsize=10.5, color=colors["slate"])
    png = OUTPUT / f"cumulative_su_ats_brier_3d_{label}.png"
    gif = OUTPUT / f"cumulative_su_ats_brier_3d_{label}.gif"
    fig.savefig(png, dpi=170, facecolor=fig.get_facecolor())
    animation = FuncAnimation(fig, lambda step: ax.view_init(elev=23, azim=-55 + 360 * step / 48),
                              frames=48, interval=125, blit=False)
    animation.save(gif, writer=PillowWriter(fps=8), dpi=85,
                   savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    return {"png": png.name, "gif": gif.name, "frames": 48, "fps": 8,
            "cell_count": 126, "plotted_points": len(endpoints), "source_cell_week_rows": len(frame),
            "endpoints_week_6": 126 if paired else 6, "endpoints_week_7": 0 if paired else 120}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full = load_full()
    paired = load_paired()
    plotted = []
    for label, frame in (("latest_available", full), ("paired_304", paired)):
        path = OUTPUT / f"plotted_cumulative_{label}.csv"
        endpoints = frame.sort_values("week").groupby(["fingerprint", "model_id"], sort=False).tail(1)
        endpoints.to_csv(path, index=False, float_format="%.12g")
        plotted.append(path)
    results = [render(full, paired=False), render(paired, paired=True)]
    files = [*plotted, *(OUTPUT / result[key] for result in results for key in ("png", "gif"))]
    manifest = {
        "scope": "2026 retrospective cumulative scientific F0–F20, M1–M5 and M10",
        "source_sha256": {name: sha256(SOURCE / name) for name in
                          ("cumulative_scorecard.csv", "f20_paired_weekly_scorecard.csv")},
        "f20_original_expert_publication_timestamps_verified": False,
        "f20_models_fitted_before_2026_games": False,
        "latest_available_cohorts": {"F0–F19": "380 games through Week 7", "F20": "320 games through Week 6"},
        "paired_cohort": "304 identical games through Week 6 for all 126 cells",
        "views": results,
        "output_sha256": {path.name: sha256(path) for path in files},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "views": results}, indent=2))


if __name__ == "__main__":
    main()
