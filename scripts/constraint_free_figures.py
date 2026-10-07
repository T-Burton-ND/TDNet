#!/usr/bin/env python3
"""Render the measured F0–F19 scientific figure suite after 2026 scoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from constraint_free_score_2026 import DEFAULT_OUTPUT as SCORE_DIR
from constraint_free_search import digest, write_json
from gridiron_ml.experiments.constraint_free import ROSTER

REPO = Path(__file__).resolve().parents[1]
PRIOR = REPO / "publication/2026/week_05/post_game/scientific/current_season_f0_f17_market/scientific_2026_current_season_scorecard.csv"
SEARCH = SCORE_DIR.parent.parent.parent
FINALISTS = SEARCH / "finalists.json"
DEFAULT_OUTPUT = SCORE_DIR.parent / "figures"
FAMILY = {"linear": "M1", "spline": "M2", "tree": "M3",
          "boosted": "M4", "neural": "M5", "knn": "M10"}
LABELS = {"M1": "Linear", "M2": "Spline", "M3": "Random forest",
          "M4": "Boosted trees", "M5": "Neural net", "M10": "KNN"}
COLORS = {"M1": "#B99748", "M2": "#477FAF", "M3": "#258F70",
          "M4": "#7557A4", "M5": "#75BDA8", "M10": "#586B7A"}
METRICS = {
    "mae": ("Margin MAE", "Points · lower is better"),
    "brier": ("Brier score", "Lower is better"),
    "winner_accuracy": ("Winner accuracy", "Fraction · higher is better"),
    "upset_recall": ("Upset recall", "Fraction · higher is better"),
}


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "#F8F4EA",
                         "axes.facecolor": "#FFFCF5", "savefig.facecolor": "#F8F4EA"})


def _full_ladder(metrics: pd.DataFrame) -> pd.DataFrame:
    prior = pd.read_csv(PRIOR)
    prior = prior.loc[prior.series_type.eq("model")].copy()
    prior["model_id"] = prior.model_family.map(FAMILY)
    prior["generation"] = prior.fingerprint.map(
        lambda value: 17 if value == "F17-market" else int(value.removeprefix("F")))
    if len(prior) != 108 or prior.duplicated(["generation", "model_id"]).any():
        raise ValueError("Prior F0–F17 scientific ladder is incomplete")
    old = prior.rename(columns={"margin_mae": "mae", "brier_score": "brier",
                                "su_accuracy": "winner_accuracy"})[
        ["generation", "fingerprint", "model_id", "games", *METRICS]]
    new = metrics.loc[metrics.scope.eq("available") & metrics.tier.isin(["F18", "F19"])
                      & metrics.model_id.isin(ROSTER)].copy()
    new["generation"] = new.tier.str.removeprefix("F").astype(int)
    new["fingerprint"] = new.tier
    new = new.rename(columns={"model_id": "model_id"})[
        ["generation", "fingerprint", "model_id", "games", *METRICS]]
    ladder = pd.concat([old, new], ignore_index=True).sort_values(["generation", "model_id"])
    if len(ladder) != 120 or ladder.duplicated(["generation", "model_id"]).any():
        raise ValueError("Expected all 120 F0–F19 model × fingerprint cells")
    return ladder


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _metric_curves(ladder: pd.DataFrame, output: Path) -> list[Path]:
    paths = []
    for metric, (title, ylabel) in METRICS.items():
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
        for ax, model in zip(axes.flat, ROSTER, strict=True):
            data = ladder.loc[ladder.model_id.eq(model)].sort_values("generation")
            interpretable = data.loc[data.generation.le(16)]
            ax.plot(interpretable.generation, interpretable[metric], color=COLORS[model],
                    linewidth=1.6, marker="o", markersize=3.5)
            for generation, symbol in ((17, "D"), (18, "s"), (19, "*")):
                point = data.loc[data.generation.eq(generation)]
                ax.scatter(point.generation, point[metric], color=COLORS[model],
                           marker=symbol, s=72 if generation < 19 else 110,
                           edgecolors="#172634", linewidths=.6, zorder=5)
            ax.axvspan(16.5, 19.5, color="#E9DDD0", alpha=.32)
            ax.set_title(f"{model} · {LABELS[model]}")
            ax.set_xticks([0, 4, 8, 12, 16, 17, 18, 19])
            ax.set_xticklabels(["F0", "F4", "F8", "F12", "F16", "F17M", "F18", "F19"],
                               fontsize=8)
            ax.grid(alpha=.2)
        for ax in axes[-1]:
            ax.set_xlabel("Fingerprint")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        fig.suptitle(f"F0–F19 scientific roster · {title}", fontsize=18, y=.985)
        fig.text(.5, .015, "2026 archived-pregame replay · F18: 271 games · F19: 263 market-covered games · F17-market: partial archived market inputs",
                 ha="center", fontsize=9, color="#52606B")
        fig.tight_layout(rect=[0, .04, 1, .96])
        path = output / f"f0_f19_{metric}.png"
        _save(fig, path)
        paths.append(path)
    return paths


def _tier_comparison(metrics: pd.DataFrame, comparisons: pd.DataFrame,
                     output: Path) -> list[Path]:
    common = metrics.loc[metrics.scope.eq("common_market_263")]
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(ROSTER))
    width = .23
    for offset, tier, color in ((-1, "F16", "#778691"),
                                (0, "F18", "#257E83"),
                                (1, "F19", "#BD7A44")):
        values = [common.loc[common.tier.eq(tier) & common.model_id.eq(model), "mae"].iloc[0]
                  for model in ROSTER]
        ax.bar(x + offset * width, values, width, label=tier, color=color)
    market = common.loc[common.tier.eq("market") & common.model_id.eq("baseline"), "mae"]
    if len(market):
        ax.axhline(float(market.iloc[0]), linestyle="--", color="#AD8C40",
                   label="Archived market spread")
    ax.set_xticks(x, [f"{model}\n{LABELS[model]}" for model in ROSTER])
    ax.set_ylabel("Margin MAE · 263 common games")
    ax.set_title("Interpretable F16 → unconstrained F18 → market-eligible F19")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.text(.5, .015, "All bars use the same 263 market-covered 2026 games; F19 historical quote timing remains unverified.",
             ha="center", fontsize=9, color="#52606B")
    fig.tight_layout(rect=[0, .035, 1, 1])
    path = output / "f16_f18_f19_common_cohort.png"
    _save(fig, path)

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(ROSTER) + 1)
    labels = [*ROSTER, "Equal consensus"]
    pair = comparisons.loc[comparisons.a.eq("F18") & comparisons.b.eq("F19")].set_index("model_id")
    keys = [*ROSTER, "equal"]
    estimate = pair.loc[keys, "mae_a_minus_b"].to_numpy(float)
    low = pair.loc[keys, "ci95_low"].to_numpy(float)
    high = pair.loc[keys, "ci95_high"].to_numpy(float)
    ax.errorbar(estimate, y, xerr=[estimate - low, high - estimate], fmt="o",
                color="#9C623A", ecolor="#52606B", capsize=4, markersize=7)
    ax.axvline(0, color="#172634", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("F18 MAE − F19 MAE · positive favors F19")
    ax.set_title("Market contribution after matched optimization")
    ax.grid(axis="x", alpha=.2)
    fig.text(.5, .015, "Paired week-stratified 95% bootstrap intervals · 263 common 2026 games",
             ha="center", fontsize=9, color="#52606B")
    fig.tight_layout(rect=[0, .035, 1, 1])
    path2 = output / "f18_f19_paired_market_contribution.png"
    _save(fig, path2)
    return [path, path2]


def _prospective_consensus(metrics: pd.DataFrame, output: Path) -> Path:
    common = metrics.loc[metrics.scope.eq("common_market_263")]
    cells = [("F16", "equal", "F16"), ("F18", "equal", "F18"),
             ("F17-market", "equal", "F17M"), ("F19", "equal", "F19"),
             ("market", "baseline", "Market")]
    values = [common.loc[common.tier.eq(tier) & common.model_id.eq(model)].iloc[0]
              for tier, model, _ in cells]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#778691", "#257E83", "#A47A79", "#BD7A44", "#AD8C40"]
    for ax, metric, title in zip(axes, ("mae", "brier"),
                                 ("Margin MAE", "Win-probability Brier"), strict=True):
        ax.bar([name for _, _, name in cells], [row[metric] for row in values], color=colors)
        ax.set_title(title)
        ax.set_ylabel("Points" if metric == "mae" else "Brier score")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("2026 common-market consensus comparison", fontsize=17)
    fig.text(.5, .015, "263 common games for MAE; market Brier has 259 valid archived probabilities. F17M uses partial archived market inputs.",
             ha="center", fontsize=9, color="#52606B")
    fig.tight_layout(rect=[0, .04, 1, .93])
    path = output / "prospective_common_market_consensus.png"
    _save(fig, path)
    return path


def _representation_search(output: Path) -> Path:
    selected = json.loads(FINALISTS.read_text())
    root = FINALISTS.parent
    rows = []
    for tier in ("F18", "F19"):
        for model in ROSTER:
            sources = {}
            for stage in ("retry1/stage1", "retry1/stage2", "extra_reducers/stage2"):
                folder = root / stage / "results"
                for path in folder.glob("task_*.json"):
                    row = json.loads(path.read_text())
                    if row.get("status") == "success" and row["tier"] == tier and row["architecture"] == model:
                        kind = row["representation"]
                        category = ("Raw" if kind == "raw" else
                                    "Family PCA" if kind.startswith("family") else
                                    "Selection" if kind.startswith("select") or kind == "mi128" else
                                    "PCA / reducer")
                        sources.setdefault(category, []).append(row["mean_mae"])
            for track in ("ga", "bo"):
                folder = root / "advanced" / track / f"{tier}_{model}" / "candidates"
                values = [json.loads(path.read_text())["mean_mae"] for path in folder.glob("*.json")
                          if json.loads(path.read_text()).get("status") == "success"]
                sources[track.upper()] = values
            winner = next(row for row in selected["winners"]
                          if row["tier"] == tier and row["architecture"] == model)
            sources["Final"] = [winner["mean_mae"]]
            for category, values in sources.items():
                if values:
                    rows.append({"tier": tier, "model": model, "category": category,
                                 "best_development_mae": min(values)})
    table = pd.DataFrame(rows)
    categories = ["Raw", "PCA / reducer", "Family PCA", "Selection", "GA", "BO", "Final"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ax, model in zip(axes.flat, ROSTER, strict=True):
        for tier, color, marker in (("F18", "#257E83", "o"), ("F19", "#BD7A44", "s")):
            cell = table.loc[table.model.eq(model) & table.tier.eq(tier)].set_index("category")
            ax.plot(categories, [cell.best_development_mae.get(cat, np.nan) for cat in categories],
                    color=color, marker=marker, linewidth=1.5, label=tier)
        ax.set_title(f"{model} · {LABELS[model]}")
        ax.tick_params(axis="x", rotation=38, labelsize=8)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(frameon=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean 2022–2025 MAE")
    fig.suptitle("Historical representation search · matched F18/F19", fontsize=18)
    fig.text(.5, .015, "Best observed candidate in each category; selection makes these development scores optimistic. Final is the selected architecture configuration.",
             ha="center", fontsize=9, color="#52606B")
    fig.tight_layout(rect=[0, .04, 1, .95])
    path = output / "representation_search_development.png"
    _save(fig, path)
    table.to_csv(output / "representation_search_development.csv", index=False)
    return path


def _metric_landscape(ladder: pd.DataFrame, output: Path) -> list[Path]:
    data = ladder[["mae", "winner_accuracy", "brier"]].to_numpy(float)
    standardized = StandardScaler().fit_transform(data)
    pca = PCA(n_components=3).fit(standardized)
    favorable = np.array([-1., 1., -1.])
    direction = pca.components_[0].copy()
    if np.dot(direction, favorable) < 0:
        direction *= -1
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    symbols = {"prior": "o", "F17M": "D", "F18": "s", "F19": "*"}
    for model in ROSTER:
        cell = ladder.loc[ladder.model_id.eq(model)]
        for group, subset in (("prior", cell.loc[cell.generation.le(16)]),
                              ("F17M", cell.loc[cell.generation.eq(17)]),
                              ("F18", cell.loc[cell.generation.eq(18)]),
                              ("F19", cell.loc[cell.generation.eq(19)])):
            ax.scatter(subset.mae, subset.winner_accuracy, subset.brier,
                       marker=symbols[group], s=35 if group == "prior" else 100,
                       color=COLORS[model], alpha=.85, edgecolors="#172634",
                       linewidths=.4, depthshade=False)
    center = data.mean(axis=0)
    span = data.std(axis=0)
    ax.quiver(*center, *(direction * span), color="#D24F88", linewidth=2.5,
              arrow_length_ratio=.15)
    ax.set_xlabel("Margin MAE · lower")
    ax.set_ylabel("Winner accuracy · higher")
    ax.set_zlabel("Brier · lower")
    ax.set_title("F0–F19 measured metric landscape · descriptive PCA direction", pad=22)
    model_handles = [Line2D([0], [0], color=COLORS[m], marker="o", linestyle="",
                            label=f"{m} · {LABELS[m]}") for m in ROSTER]
    tier_handles = [Line2D([0], [0], color="#52606B", marker=marker, linestyle="",
                           label=label) for label, marker in (("F0–F16", "o"),
                                                                ("F17-market", "D"),
                                                                ("F18", "s"), ("F19", "*"))]
    ax.legend(handles=model_handles + tier_handles, bbox_to_anchor=(1.1, 1), frameon=False)
    ax.view_init(elev=22, azim=-58)
    fig.text(.5, .02, "Each point is a measured model × fingerprint score; the pink arrow is PCA of the three displayed metrics, never predictor PCA.",
             ha="center", fontsize=9, color="#52606B")
    png = output / "f0_f19_metric_landscape_3d.png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    animation = FuncAnimation(fig, lambda n: ax.view_init(elev=22, azim=-58 + 360 * n / 48),
                              frames=48, interval=150, blit=False)
    gif = output / "f0_f19_metric_landscape_rotation.gif"
    animation.save(gif, writer=PillowWriter(fps=8), dpi=80)
    plt.close(fig)
    generation_cmap = LinearSegmentedColormap.from_list(
        "generation_red_blue", ["#B64639", "#D6AA62", "#4B83B1"])
    generation_norm = Normalize(vmin=0, vmax=19)
    model_markers = {"M1": "o", "M2": "s", "M3": "^", "M4": "D",
                     "M5": "P", "M10": "X"}
    fig2 = plt.figure(figsize=(13, 9))
    ax2 = fig2.add_subplot(111, projection="3d")
    for model in ROSTER:
        cell = ladder.loc[ladder.model_id.eq(model)]
        ax2.scatter(cell.mae, cell.winner_accuracy, cell.brier,
                    c=cell.generation, cmap=generation_cmap, norm=generation_norm,
                    marker=model_markers[model], s=65, alpha=.9,
                    edgecolors="#172634", linewidths=.4, depthshade=False)
    colorbar = fig2.colorbar(plt.cm.ScalarMappable(cmap=generation_cmap,
                                                   norm=generation_norm),
                            ax=ax2, shrink=.55, pad=.1)
    colorbar.set_label("Fingerprint generation · F0–F19")
    ax2.set_xlabel("Margin MAE · lower")
    ax2.set_ylabel("Winner accuracy · higher")
    ax2.set_zlabel("Brier · lower")
    ax2.set_title("F0–F19 metric landscape · generation color", pad=22)
    handles = [Line2D([0], [0], color="#52606B", marker=model_markers[m],
                      linestyle="", label=f"{m} · {LABELS[m]}") for m in ROSTER]
    ax2.legend(handles=handles, bbox_to_anchor=(1.12, 1), frameon=False)
    ax2.view_init(elev=22, azim=-58)
    fig2.text(.5, .02, "Red → blue follows fingerprint generation; each marker is one measured architecture × fingerprint score.",
              ha="center", fontsize=9, color="#52606B")
    generation_png = output / "f0_f19_generation_colored_3d.png"
    fig2.savefig(generation_png, dpi=220, bbox_inches="tight")
    animation2 = FuncAnimation(fig2, lambda n: ax2.view_init(
        elev=22, azim=-58 + 360 * n / 48), frames=48, interval=150, blit=False)
    generation_gif = output / "f0_f19_generation_colored_rotation.gif"
    animation2.save(generation_gif, writer=PillowWriter(fps=8), dpi=80)
    plt.close(fig2)
    write_json(output / "metric_landscape_pca.json", {
        "metric_columns": ["mae", "winner_accuracy", "brier"],
        "standardized_component_1_favorable": direction.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "description": "descriptive PCA of measured metric coordinates; not a predictor representation"})
    return [png, gif, generation_png, generation_gif]


def build(score_dir: Path = SCORE_DIR, output: Path = DEFAULT_OUTPUT) -> dict:
    score_receipt_path = score_dir / "score_receipt.json"
    score_receipt = json.loads(score_receipt_path.read_text())
    for name in ("metrics.csv", "paired_comparisons.csv"):
        path = score_dir / name
        if digest(path) != score_receipt["outputs"][str(path)]:
            raise ValueError(f"Scored figure input changed: {path}")
    if (output / "figure_manifest.json").exists():
        raise FileExistsError("F0–F19 figure suite is immutable")
    output.mkdir(parents=True, exist_ok=True)
    _style()
    metrics = pd.read_csv(score_dir / "metrics.csv")
    comparisons = pd.read_csv(score_dir / "paired_comparisons.csv")
    ladder = _full_ladder(metrics)
    ladder_path = output / "f0_f19_metric_ladder.csv"
    ladder.to_csv(ladder_path, index=False)
    figures = [*_metric_curves(ladder, output),
               *_tier_comparison(metrics, comparisons, output),
               _prospective_consensus(metrics, output),
               _representation_search(output),
               *_metric_landscape(ladder, output)]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_receipt": str(score_receipt_path),
        "score_receipt_sha256": digest(score_receipt_path),
        "figures": {str(path): digest(path) for path in figures},
        "ladder_table": str(ladder_path), "ladder_table_sha256": digest(ladder_path),
        "model_fingerprint_cells": len(ladder),
        "new_tier_cohorts": {"F18": 271, "F19": 263},
        "prior_market_caveat": "F17-market 271-game what-if imputes eight games without pregame snapshots; F19 requires 263 covered games",
        "data_interpretation": "archived-pregame 2026 replay with prior project exposure to 2026 outcomes",
    }
    write_json(output / "figure_manifest.json", manifest)
    return {"manifest": str(output / "figure_manifest.json"),
            "figure_count": len(figures), "cells": len(ladder)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, default=SCORE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.score_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
