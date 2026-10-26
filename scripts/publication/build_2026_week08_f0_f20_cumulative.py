#!/usr/bin/env python3
"""Extend archived scientific cumulative scorecards with certified Week 8 scores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.publication.figure_theme import TDNET_COLORS, apply_tdnet_theme


ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "publication/2026/research/scientific_f0_f20_posthoc_scorecards_week01_07"
WEEK = ROOT / "publication/2026/week_08/post_game/scientific/expanded_f0_f20"
OUTPUT = WEEK / "cumulative"
FINGERPRINTS = [*(f"F{i}" for i in range(17)), "F17-market", "F18", "F19", "F20"]
MODELS = ["M1", "M2", "M3", "M4", "M5", "M10"]
MARKERS = dict(zip(MODELS, ["o", "s", "^", "D", "P", "X"]))


def combine(prior: pd.DataFrame, week: pd.DataFrame) -> pd.DataFrame:
    expected = {(fp, model) for fp in FINGERPRINTS for model in MODELS}
    if set(zip(prior.fingerprint, prior.model_id)) != expected:
        raise ValueError("Prior cumulative roster is incomplete")
    if set(zip(week.fingerprint, week.model_id)) != expected:
        raise ValueError("Week 8 roster is incomplete")
    paired = prior.merge(week, on=["fingerprint", "model_id"], validate="one_to_one", suffixes=("_prior", "_week08"))
    rows = []
    for row in paired.itertuples(index=False):
        old_games = int(row.games_prior)
        new_games = int(row.games_week08)
        total = old_games + new_games
        old_su = int(row.su_wins)
        week_su = round(float(row.su_accuracy_week08) * new_games)
        if not np.isclose(week_su / new_games, row.su_accuracy_week08):
            raise ValueError("Week 8 straight-up counts do not reconcile")
        wins = int(row.ats_wins_prior) + int(row.ats_wins_week08)
        losses = int(row.ats_losses_prior) + int(row.ats_losses_week08)
        pushes = int(row.ats_pushes_prior) + int(row.ats_pushes_week08)
        no_pick = int(row.ats_no_pick_prior) + int(row.ats_no_pick_week08)
        no_line = int(row.ats_no_line_prior) + int(row.ats_no_line_week08)
        if wins + losses + pushes + no_pick + no_line != total:
            raise ValueError("Cumulative ATS counts do not reconcile")
        rows.append({
            "fingerprint": row.fingerprint, "model_id": row.model_id,
            "games": total, "su_wins": old_su + week_su,
            "su_losses": total - old_su - week_su,
            "su_accuracy": (old_su + week_su) / total,
            "brier": (float(row.brier_prior) * old_games + float(row.brier_week08) * new_games) / total,
            "margin_mae": (float(row.margin_mae_prior) * old_games + float(row.margin_mae_week08) * new_games) / total,
            "ats_wins": wins, "ats_losses": losses, "ats_pushes": pushes,
            "ats_no_pick": no_pick, "ats_no_line": no_line,
            "ats_accuracy": wins / (wins + losses),
            "week08_games": new_games,
            "last_scored_week": 8,
            "week7_included": row.fingerprint != "F20" if old_games != 304 else False,
        })
    return pd.DataFrame(rows).sort_values(["fingerprint", "model_id"]).reset_index(drop=True)


def render(frame: pd.DataFrame, *, label: str, subtitle: str, note: str) -> list[Path]:
    apply_tdnet_theme()
    colors = TDNET_COLORS
    fig = plt.figure(figsize=(13.2, 9.4), facecolor=colors["parchment"])
    ax = fig.add_axes([.055, .15, .77, .73], projection="3d", facecolor=colors["parchment_panel"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(colors["parchment_panel"])
        axis.pane.set_edgecolor(colors["polar_mist"])
        axis._axinfo["grid"]["color"] = colors["polar_mist"]
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=0, vmax=20)
    for point in frame.itertuples(index=False):
        color = cmap(norm(FINGERPRINTS.index(point.fingerprint)))
        ax.scatter([point.su_accuracy * 100], [point.ats_accuracy * 100], [point.brier],
                   marker=MARKERS[point.model_id], s=116, color=color,
                   edgecolors=colors["parchment"], linewidths=.8, alpha=.9,
                   depthshade=False)
    values = [(frame.su_accuracy * 100, ax.set_xlim),
              (frame.ats_accuracy * 100, ax.set_ylim), (frame.brier, ax.set_zlim)]
    for series, setter in values:
        low, high = float(series.min()), float(series.max())
        pad = max((high - low) * .09, .005 if setter == ax.set_zlim else 1.0)
        setter(low - pad, high + pad)
    ax.set_xlabel("Straight-up accuracy · % ↑", labelpad=16, fontsize=14)
    ax.set_ylabel("ATS accuracy · % ↑", labelpad=16, fontsize=14)
    ax.set_zlabel("Brier score ↓", labelpad=14, fontsize=14)
    ax.tick_params(colors=colors["midnight_gridiron"], labelsize=12)
    ax.set_box_aspect((1.15, 1.15, .85), zoom=1.14)
    ax.view_init(elev=23, azim=-55)
    fig.suptitle("126 scientific cells · cumulative SU, ATS, Brier", x=.48, y=.97,
                 fontsize=21, fontweight="bold", color=colors["midnight_gridiron"])
    fig.text(.48, .922, subtitle, ha="center", fontsize=12, color=colors["slate"])
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, cax=fig.add_axes([.88, .29, .017, .48]))
    cbar.set_ticks([0, 4, 8, 12, 16, 20])
    cbar.set_ticklabels(["F0", "F4", "F8", "F12", "F16", "F20"])
    cbar.set_label("Fingerprint", color=colors["midnight_gridiron"], fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    handles = [Line2D([0], [0], marker=MARKERS[model], linestyle="", color=colors["gridiron_violet"],
                      markeredgecolor=colors["parchment"], markersize=11, label=model) for model in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.48, .065),
               ncol=6, frameon=False, fontsize=11)
    fig.text(.48, .023, note, ha="center", fontsize=10.5, color=colors["slate"])
    png = OUTPUT / f"cumulative_su_ats_brier_3d_{label}.png"
    gif = OUTPUT / f"cumulative_su_ats_brier_3d_{label}.gif"
    fig.savefig(png, dpi=170, facecolor=fig.get_facecolor())
    animation = FuncAnimation(fig, lambda step: ax.view_init(elev=23, azim=-55 + 360 * step / 48),
                              frames=48, interval=125, blit=False)
    animation.save(gif, writer=PillowWriter(fps=8), dpi=85,
                   savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)
    return [png, gif]


def main() -> None:
    latest_prior = pd.read_csv(PRIOR / "latest_scorecard.csv")
    paired_prior = pd.read_csv(PRIOR / "f20_paired_304_game_scorecard.csv")
    full_week = pd.read_csv(WEEK / "week08_f0_f19_120_model_scorecard.csv")
    matched_week = pd.read_csv(WEEK / "week08_f0_f20_126_model_scorecard_matched.csv")
    latest_week = pd.concat([full_week, matched_week[matched_week.fingerprint.eq("F20")]], ignore_index=True)
    latest = combine(latest_prior, latest_week)
    paired = combine(paired_prior, matched_week)
    if not (len(latest) == len(paired) == 126):
        raise ValueError("Expected 126 cumulative cells")
    if set(latest.loc[latest.fingerprint.ne("F20"), "games"]) != {433} or set(latest.loc[latest.fingerprint.eq("F20"), "games"]) != {369}:
        raise ValueError("Unexpected latest-available game counts")
    if set(paired.games) != {353}:
        raise ValueError("Unexpected matched cumulative game count")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    latest_path = OUTPUT / "latest_available_126_model_scorecard.csv"
    paired_path = OUTPUT / "matched_353_game_126_model_scorecard.csv"
    latest.to_csv(latest_path, index=False)
    paired.to_csv(paired_path, index=False)
    figures = []
    figures += render(latest, label="latest_available_week08",
                      subtitle="F0–F19: Weeks 1–8, 433 games · F20: Weeks 1–6 and 8, 369 games",
                      note="One point per F×M cell. Denominators differ; F20 Week 7 expert inputs are absent.")
    figures += render(paired, label="matched_353_games_week08",
                      subtitle="All 126 cells on the same 353 games · Weeks 1–6 and 8",
                      note="One point per F×M cell. Week 7 is excluded because F20 forecasts are unavailable.")
    manifest = {"schema": "tdnet-f0-f20-cumulative-week08-v1",
                "latest_available": {"F0–F19_games": 433, "F20_games": 369},
                "paired_games": 353, "week7_f20_missing": True,
                "sources": {str(p): sha256_file(p) for p in [PRIOR / "latest_scorecard.csv",
                           PRIOR / "f20_paired_304_game_scorecard.csv",
                           WEEK / "week08_f0_f19_120_model_scorecard.csv",
                           WEEK / "week08_f0_f20_126_model_scorecard_matched.csv"]},
                "files": {p.name: sha256_file(p) for p in [latest_path, paired_path, *figures]}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cells": 126, "latest_games": [433, 369], "paired_games": 353,
                      "figures": [p.name for p in figures]}))


if __name__ == "__main__":
    main()
