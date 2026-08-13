#!/usr/bin/env python3
"""Render explanatory figures for the scientific matrix and weekly roster."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


BG = "#07101d"
PANEL = "#0d1a2b"
INK = "#f1f6ff"
MUTED = "#9aadc5"
GRID = "#2b405b"
FINGERPRINT = ["#4fc3d9", "#61b6e8", "#7899f2", "#9a7be8", "#c878ca", "#e77ca5", "#f29a74", "#f2c85b"]
MODEL = {"M1": "#55c4d8", "M2": "#78a0f5", "M3": "#a77cf0", "M4": "#e37ab4", "M5": "#f39b64", "M10": "#f1c95b"}
MODEL_NAME = {"M1": "linear", "M2": "spline", "M3": "tree", "M4": "boosted", "M5": "MLP", "M10": "KNN"}
TIER_NAME = {
    "F0": "structural\nbaseline",
    "F1": "raw\nbox scores",
    "F2": "efficiency\n& rates",
    "F3": "opponent\nadjusted",
    "F4": "temporal\n& situational",
    "F5": "football\ninformation",
    "F6": "complete\nmarket-free",
    "F7": "market\ncomparator",
}


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, facecolor=BG, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(path.with_suffix(".svg"), facecolor=BG, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def _rounded(ax, x, y, w, h, color, edge=GRID, alpha=1.0, radius=0.08):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}", facecolor=color, edgecolor=edge, linewidth=1.0, alpha=alpha)
    ax.add_patch(patch)


def _glyph(ax, level, x, y, scale=1.0):
    """Draw a tiny visual signature for each model architecture."""
    c = MODEL[level]
    if level == "M1":
        ax.plot([x - .16*scale, x + .16*scale], [y - .10*scale, y + .11*scale], color=c, lw=2.3, solid_capstyle="round")
    elif level == "M2":
        xx = np.linspace(x - .17*scale, x + .17*scale, 20)
        ax.plot(xx, y + .12*scale*np.sin((xx-x)*10/scale), color=c, lw=2.2)
    elif level in {"M3", "M4"}:
        ax.plot([x, x, x-.14*scale], [y+.13*scale, y, y-.12*scale], color=c, lw=1.6)
        ax.plot([x, x+.14*scale], [y, y-.12*scale], color=c, lw=1.6)
        ax.scatter([x, x-.14*scale, x+.14*scale], [y+.13*scale, y-.12*scale, y-.12*scale], s=10*scale, color=c, zorder=4)
        if level == "M4":
            ax.plot([x-.18*scale, x+.18*scale], [y+.18*scale, y+.18*scale], color=c, lw=1.1, alpha=.6)
    elif level == "M5":
        for dx in (-.14, 0, .14):
            ax.scatter([x+dx*scale], [y+.14*scale], s=8*scale, color=c)
            ax.scatter([x+dx*scale], [y-.14*scale], s=8*scale, color=c)
        for dx in (-.14, 0, .14):
            ax.plot([x+dx*scale, x], [y+.14*scale, y], color=c, lw=.7, alpha=.75)
            ax.plot([x, x+dx*scale], [y, y-.14*scale], color=c, lw=.7, alpha=.75)
    else:
        ax.scatter([x-.12*scale, x+.02*scale, x+.15*scale], [y+.05*scale, y-.09*scale, y+.13*scale], s=10*scale, color=c)
        ax.scatter([x-.01*scale], [y+.02*scale], s=25*scale, facecolors="none", edgecolors=c, lw=1.5)


def scientific_matrix(output: Path) -> None:
    tiers = [f"F{i}" for i in range(8)]
    levels = ["M1", "M2", "M3", "M4", "M5", "M10"]
    fig = plt.figure(figsize=(15.2, 9.0), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.3, 1.65], hspace=.06)
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(BG)
    for x, tier in enumerate(tiers):
        ax.axvspan(x-.48, x+.48, color=FINGERPRINT[x], alpha=.06 if tier != "F7" else .15, zorder=0)
        if tier == "F7":
            ax.axvspan(x-.48, x+.48, color="#ef6f88", alpha=.12, zorder=0)
    for y, level in enumerate(levels):
        ax.plot([-.48, 7.48], [y, y], color=GRID, lw=.8, alpha=.65, zorder=0)
        ax.text(-.72, y, f"{level}\n{MODEL_NAME[level]}", ha="right", va="center", color=INK, fontsize=10, weight="bold", linespacing=1.0)
        for x, tier in enumerate(tiers):
            _rounded(ax, x-.34, y-.30, .68, .60, PANEL, edge=MODEL[level], alpha=.96, radius=.07)
            _glyph(ax, level, x, y+.07, .82)
            ax.text(x, y-.21, tier, ha="center", va="center", color=FINGERPRINT[x] if tier != "F7" else "#ffabb8", fontsize=7.4, weight="bold")
    for x in range(7):
        ax.annotate("", xy=(x+1-.43, 5.92), xytext=(x+.43, 5.92), arrowprops={"arrowstyle": "-|>", "color": "#607690", "lw": 1.1})
    ax.text(3.5, 6.95, "THE SAME QUESTION, CHANGED IN TWO DIRECTIONS", color=INK, fontsize=21, weight="bold", ha="center")
    ax.text(3.5, 6.55, "Rows change how the model learns. Columns change what football information it is allowed to see.", color=MUTED, fontsize=11, ha="center")
    ax.text(3.5, -0.72, "Each cell is one frozen scientific model: 6 architectures × 8 fingerprints = 48 experiments", color=MUTED, fontsize=10, ha="center")
    ax.set_xlim(-1.2, 7.72); ax.set_ylim(-.95, 7.2); ax.axis("off")

    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor(PANEL)
    increments = ["structure", "+ box scores", "+ rates", "+ opponent context", "+ time / situation", "+ football context", "+ all non-market", "+ market"]
    for i, (tier, label) in enumerate(zip(tiers, increments)):
        ax2.plot([i, i], [0, 1], color=FINGERPRINT[i], lw=7, alpha=.95, solid_capstyle="round")
        ax2.scatter([i], [1], s=80, color=FINGERPRINT[i], edgecolor=INK, linewidth=.8, zorder=3)
        ax2.text(i, -.18, tier, color=INK, ha="center", va="top", fontsize=9, weight="bold")
        ax2.text(i, -.43, label, color=MUTED, ha="center", va="top", fontsize=8)
    ax2.plot(range(7), [1]*7, color="#91a6bd", lw=1.5, alpha=.45)
    ax2.text(6.9, .93, "market enters", color="#ffabb8", fontsize=8.5, ha="left", va="center")
    ax2.set_xlim(-.55, 7.85); ax2.set_ylim(-.72, 1.18); ax2.axis("off")
    save(fig, output / "scientific_model_garden.png")


def roster_composition(output: Path, wide_inventory: Path, scientific_inventory: Path) -> None:
    wide = pd.read_csv(wide_inventory)
    scientific = pd.read_csv(scientific_inventory)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.2, 7.4), gridspec_kw={"width_ratios": [1.25, 1]}, facecolor=BG)
    for a in (ax, ax2): a.set_facecolor(BG)

    # Left: the scientific factorial design as six horizontal tracks.
    tiers = [f"F{i}" for i in range(8)]
    levels = ["M1", "M2", "M3", "M4", "M5", "M10"]
    for y, level in enumerate(levels):
        ax.text(-.6, y, f"{level}  {MODEL_NAME[level]}", color=INK, ha="right", va="center", fontsize=9.5, weight="bold")
        ax.plot([0, 7], [y, y], color=GRID, lw=1)
        for x, tier in enumerate(tiers):
            ax.scatter([x], [y], s=250, color=FINGERPRINT[x], edgecolor=MODEL[level], linewidth=2.2, zorder=3)
            _glyph(ax, level, x, y, .55)
    ax.axvspan(6.55, 7.45, color="#ef6f88", alpha=.12)
    ax.text(3.5, 6.0, "SCIENTIFIC PANEL", color=INK, fontsize=17, weight="bold", ha="center")
    ax.text(3.5, 5.63, "controlled factorial design", color=MUTED, fontsize=10, ha="center")
    for x, tier in enumerate(tiers):
        ax.text(x, -0.55, tier, color="#ffabb8" if tier == "F7" else FINGERPRINT[x], ha="center", fontsize=9, weight="bold")
    ax.text(7.05, 2.5, "F7\ncomparative", color="#ffabb8", fontsize=9, ha="center", va="center", weight="bold")
    ax.set_xlim(-1.8, 8.1); ax.set_ylim(-.95, 6.4); ax.axis("off")

    # Right: actual wide roster composition; inventory rows are de-duplicated
    # by family/fingerprint for a readable count, while the declared poll size
    # remains 33 informative automated members.
    family_col = "model_family" if "model_family" in wide else "family"
    counts = wide[family_col].astype(str).str.lower().value_counts()
    order = [name for name in ["linear", "spline", "tree", "boosted", "neural", "knn", "stat", "temporal", "kernel", "ensemble"] if name in counts.index]
    order += [name for name in counts.index if name not in order]
    y = np.arange(len(order))[::-1]
    colors = [MODEL.get({"linear":"M1", "spline":"M2", "tree":"M3", "boosted":"M4", "neural":"M5", "knn":"M10"}.get(name, "M1"), "#86a0bb") for name in order]
    bars = ax2.barh(y, [counts[name] for name in order], color=colors, alpha=.9, height=.58)
    for bar, name, count in zip(bars, order, [counts[name] for name in order]):
        ax2.text(bar.get_width()+.35, bar.get_y()+bar.get_height()/2, str(count), color=INK, va="center", fontsize=10, weight="bold")
        ax2.text(-.35, bar.get_y()+bar.get_height()/2, name, color=INK, va="center", ha="right", fontsize=9.5, weight="bold")
    ax2.axvline(33, color="#f2c85b", lw=1.4, ls="--")
    ax2.text(33, len(order)-.05, "33 poll members", color="#f2c85b", fontsize=9, ha="right", va="bottom")
    ax2.text(max(counts.max()+4, 12), len(order)+1.2, "WIDE-MARGIN INVENTORY", color=INK, fontsize=17, weight="bold", ha="center")
    ax2.text(max(counts.max()+4, 12), len(order)+.78, "39 retained rows; 33 informative poll members", color=MUTED, fontsize=10, ha="center")
    ax2.set_xlim(-5, max(counts.max()+8, 18)); ax2.set_ylim(-1, len(order)+1.7); ax2.set_xticks([]); ax2.set_yticks([])
    for spine in ax2.spines.values(): spine.set_visible(False)
    fig.suptitle("TWO ROSTERS, TWO JOBS", color=INK, fontsize=22, weight="bold", y=.98)
    fig.text(.5, .015, "Scientific models isolate information effects. The wide roster maximizes useful weekly coverage. F7 is never interpreted as confirmatory.", color=MUTED, ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .05, 1, .94])
    save(fig, output / "roster_constellation.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wide-inventory", type=Path, required=True)
    parser.add_argument("--scientific-inventory", type=Path, required=True)
    args = parser.parse_args()
    scientific_matrix(args.output)
    roster_composition(args.output, args.wide_inventory, args.scientific_inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
