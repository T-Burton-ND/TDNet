#!/usr/bin/env python3
"""Rebuild the scientific architecture and fingerprint violins with F20."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "publication/2026/research/scientific_f0_f20_posthoc_scorecards_week01_07"
F20_REPLAY = ROOT / "data/private/f20/replay_2026_week01_07_v3_margin/f20_replay.parquet"
MARKET = ROOT / "publication/2026/week_07/post_game/retrospective_replay/cumulative/scientific_f0_f19/common_cohort_vegas_games.csv"
OUTPUT = ROOT / "publication/2026/research/scientific_f0_f20_violin_week01_06"
FINGERPRINTS = [*(f"F{i}" for i in range(17)), "F17-market", "F18", "F19", "F20"]
MODELS = ("M1", "M2", "M3", "M4", "M5", "M10")
MARKERS = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "P", "M10": "X"}
NAMES = {"M1": "Linear", "M2": "Spline", "M3": "Tree", "M4": "Boosted",
         "M5": "Neural", "M10": "KNN"}
PALETTE = json.loads((ROOT / "gridiron.palette.json").read_text())["colors"]
PAPER = PALETTE["parchment"]
PANEL = "#FFFCF7"
INK = PALETTE["midnightGridiron"]
MUTED = PALETTE["steelGrey"]
GOLD = PALETTE["brass"]
COLORS = {"M1": PALETTE["projectorBlue"], "M2": PALETTE["projectorGreen"],
          "M3": PALETTE["gridironViolet"], "M4": "#008F93",
          "M5": "#495C77", "M10": "#9A513D"}
FORMULA = "mean(100*(vegas_mae-model_mae)/vegas_mae, 100*(vegas_brier-model_brier)/vegas_brier, 100*(model_su-vegas_su)/vegas_su)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[pd.DataFrame, dict[str, float]]:
    scores = pd.read_csv(SCORES / "f20_paired_304_game_scorecard.csv")
    expected = {(f, m) for f in FINGERPRINTS for m in MODELS}
    if (len(scores) != 126 or scores.duplicated(["fingerprint", "model_id"]).any()
            or set(zip(scores.fingerprint, scores.model_id)) != expected
            or not scores.games.eq(304).all()
            or not scores.margin_games.eq(304).all()
            or not scores.brier_games.eq(304).all()):
        raise ValueError("Expected 126 F×M scores on the same 304 games")
    replay = pd.read_parquet(F20_REPLAY)
    market = pd.read_csv(MARKET)
    common = market.loc[market.game_id.isin(replay.game_id) &
                        market.market_home_probability.notna()].copy()
    if (len(replay) != 320 or len(common) != 304 or common.game_id.duplicated().any()
            or common.market_home_spread.isna().any()
            or not common.actual_home_margin.eq(common.game_id.map(
                replay.set_index("game_id").actual_home_margin)).all()):
        raise ValueError("Vegas baseline is not on the exact F20 paired cohort")
    baseline = {"games": 304, "margin_mae": float(common.margin_error.mean()),
                "su_accuracy": float(common.winner_correct.astype(float).mean()),
                "brier": float(common.brier_error.mean())}
    if any(not np.isfinite(baseline[key]) or baseline[key] <= 0
           for key in ("margin_mae", "su_accuracy", "brier")):
        raise ValueError("Invalid Vegas baseline")
    scores["fingerprint_number"] = scores.fingerprint.map(dict(zip(FINGERPRINTS, range(21))))
    scores["mae_skill_percent"] = 100 * (baseline["margin_mae"] - scores.margin_mae) / baseline["margin_mae"]
    scores["brier_skill_percent"] = 100 * (baseline["brier"] - scores.brier) / baseline["brier"]
    scores["su_skill_percent"] = 100 * (scores.su_accuracy - baseline["su_accuracy"]) / baseline["su_accuracy"]
    scores["balanced_skill_percent"] = scores[["mae_skill_percent", "brier_skill_percent",
                                                 "su_skill_percent"]].mean(axis=1)
    if not np.isfinite(scores.balanced_skill_percent).all():
        raise ValueError("Invalid balanced relative skill")
    return scores, baseline


def swarm_offsets(values: np.ndarray, min_x: float = .65, step: float = .16) -> np.ndarray:
    offsets = np.full(len(values), np.nan)
    for index in np.argsort(values):
        for candidate in [0.0, *(v for level in range(1, 18) for v in (level * step, -level * step))]:
            if all(abs(values[index] - values[j]) >= min_x or
                   abs(candidate - offsets[j]) >= step - 1e-8
                   for j in range(len(values)) if np.isfinite(offsets[j])):
                offsets[index] = candidate
                break
        if not np.isfinite(offsets[index]):
            raise ValueError("Violin point packing failed")
    return offsets


def violin(ax, values: np.ndarray, position: int, color: str, width: float) -> None:
    body = ax.violinplot([values], positions=[position], orientation="horizontal", widths=width,
                         showmeans=False, showmedians=False, showextrema=False)["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(.20)
    body.set_linewidth(1.5)
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    ax.plot([q1, q3], [position, position], color=INK, lw=4.3,
            solid_capstyle="round", zorder=5)
    ax.scatter([median], [position], marker="|", color=PANEL, s=260,
               linewidth=2.4, zorder=8)
    ax.scatter([values.mean()], [position], marker="o", facecolors=PANEL,
               edgecolors=INK, s=55, linewidths=1.4, zorder=9)


def legend(ax, publication: bool) -> None:
    handles = [Line2D([0], [0], marker=MARKERS[m], linestyle="", color="none",
                      markerfacecolor=COLORS[m], markeredgecolor="white",
                      markersize=10, label=f"{m} · {NAMES[m]}") for m in MODELS]
    handles.extend([
        Line2D([0], [0], color=GOLD, lw=2.6, label="Vegas · 0%"),
        Line2D([0], [0], color=INK, lw=4.3, label="Interquartile range"),
        Line2D([0], [0], marker="|", linestyle="", color=INK,
               markersize=17, markeredgewidth=2.4, label="Median"),
        Line2D([0], [0], marker="o", linestyle="", color=INK,
               markerfacecolor=PANEL, markersize=9, label="Mean"),
    ])
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False, labelspacing=.78, fontsize=13 if publication else 10)


def make_figure(scores: pd.DataFrame, *, fingerprint: bool, publication: bool) -> plt.Figure:
    tall = fingerprint
    fig, ax = plt.subplots(figsize=(15, 15.5) if tall else (15, 8.5))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PANEL)
    fig.subplots_adjust(left=.15 if publication else .11, right=.76 if publication else .77,
                        top=.95 if publication else .82, bottom=.085 if tall else .14)
    if not publication:
        fig.text(.11, .963, "TDNET  /  2026 SCIENTIFIC ROSTER", color=MUTED, size=10, weight="bold")
        title = "Fingerprint spread across architectures" if tall else "Architecture distributions"
        subtitle = ("Six models per fingerprint. Gold line = Vegas."
                    if tall else "Twenty-one fingerprints per family. Gold line = Vegas.")
        fig.text(.11, .910, title, color=INK, size=23, weight="bold")
        fig.text(.11, .865, subtitle, color=MUTED, size=11)
        fig.text(.11, .037, "Paired 304 games, CFBD Weeks 1–6 · retrospective F20 fit · source times unverified",
                 color=MUTED, size=9)
    if tall:
        for number, name in enumerate(FINGERPRINTS):
            group = scores.loc[scores.fingerprint.eq(name)].set_index("model_id").loc[list(MODELS)].reset_index()
            values = group.balanced_skill_percent.to_numpy()
            violin(ax, values, number, PALETTE["projectorBlue"], .65)
            offsets = swarm_offsets(values)
            for index, row in group.iterrows():
                ax.scatter(row.balanced_skill_percent, number + offsets[index] * .30,
                           marker=MARKERS[row.model_id], color=COLORS[row.model_id], s=58,
                           edgecolor="white", linewidth=.7, zorder=7)
        ax.set_yticks(range(21), FINGERPRINTS)
        ax.set_ylim(20.7, -.7)
        ax.set_ylabel("Fingerprint", fontsize=18 if publication else 12)
    else:
        for number, model in enumerate(MODELS):
            group = scores.loc[scores.model_id.eq(model)].set_index("fingerprint").loc[FINGERPRINTS].reset_index()
            values = group.balanced_skill_percent.to_numpy()
            violin(ax, values, number, COLORS[model], .68)
            offsets = swarm_offsets(values)
            for index, row in group.iterrows():
                ax.scatter(row.balanced_skill_percent, number + offsets[index] * .30,
                           marker=MARKERS[model], color=COLORS[model], s=46,
                           edgecolor="white", linewidth=.75, zorder=7)
        ax.set_yticks(range(6), [f"{m} · {NAMES[m]}" for m in MODELS])
        ax.set_ylim(5.7, -.7)
    ax.axvline(0, color=GOLD, lw=2.6, alpha=.95, zorder=2)
    ax.set_xlim(-46, 5)
    ax.set_xlabel("Balanced relative skill versus Vegas · % ↑", fontsize=18 if publication else 12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=MUTED, alpha=.18, zorder=0)
    ax.tick_params(colors=INK, labelsize=15 if publication else 10)
    legend(ax, publication)
    return fig


def save(fig: plt.Figure, stem: str, publication: bool) -> dict[str, str]:
    names = {}
    suffix = "_publication" if publication else ""
    for extension in ("png", "svg"):
        path = OUTPUT / f"{stem}{suffix}.{extension}"
        kwargs = {"dpi": 300 if publication else 200, "facecolor": PAPER}
        if extension == "svg":
            kwargs["metadata"] = {"Date": "2026-10-23"}
        fig.savefig(path, **kwargs)
        names[path.name] = sha256(path)
    plt.close(fig)
    return names


def distribution(scores: pd.DataFrame, key: str) -> pd.DataFrame:
    result = scores.groupby(key, sort=False).balanced_skill_percent.agg(
        count="count", minimum="min", mean="mean", q1=lambda values: values.quantile(.25),
        median="median", q3=lambda values: values.quantile(.75), maximum="max").reset_index()
    result["iqr"] = result.q3 - result.q1
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.style.use(ROOT / "docs/style/color_palettes/gridiron_light.mplstyle")
    plt.rcParams.update({"figure.autolayout": False, "font.family": "DejaVu Sans",
                         "svg.hashsalt": "tdnet-2026-f0-f20-violin"})
    scores, baseline = load()
    ledger = OUTPUT / "paired_304_balanced_skill.csv"
    scores[["fingerprint", "fingerprint_number", "model_id", "games", "margin_mae",
            "su_accuracy", "brier", "mae_skill_percent", "su_skill_percent",
            "brier_skill_percent", "balanced_skill_percent"]].to_csv(
                ledger, index=False, float_format="%.12g")
    summary_files = []
    for key, name in (("model_id", "architecture_distribution.csv"),
                      ("fingerprint", "fingerprint_distribution.csv")):
        path = OUTPUT / name
        distribution(scores, key).to_csv(path, index=False, float_format="%.12g")
        summary_files.append(path)
    figures = {}
    for fingerprint, stem in ((False, "architecture_violin"), (True, "fingerprint_violin")):
        for publication in (False, True):
            figures.update(save(make_figure(scores, fingerprint=fingerprint,
                                            publication=publication), stem, publication))
    sources = (SCORES / "f20_paired_304_game_scorecard.csv", F20_REPLAY, MARKET)
    outputs = [ledger, *summary_files, *(OUTPUT / name for name in figures)]
    manifest = {
        "scope": "F0–F20 M1–M5/M10 retrospective balanced relative skill on 304 identical CFBD Weeks 1–6 games",
        "model_cells": 126, "paired_games": 304, "fingerprints": FINGERPRINTS,
        "architectures": MODELS, "vegas_baseline": baseline,
        "balanced_skill_percent": FORMULA,
        "f20_fitted_after_2026_games": True,
        "expert_source_publication_times_verified": False,
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "cells": len(scores),
                      "figures": len(figures), "vegas_baseline": baseline}, indent=2))


if __name__ == "__main__":
    main()
