"""Nine paired 2026 scientific-roster views and two publication violin variants.

The default cohort is retrospective through Week 7; supply a later common-cohort
directory and market provenance note to rebuild the gallery after more games.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "publication/2026/week_07/post_game/retrospective_replay/cumulative/scientific_f0_f19"
DEFAULT_OUTPUT = ROOT / "publication/2026/research/scientific_total_quality"
SOURCE = DEFAULT_SOURCE
OUTPUT = DEFAULT_OUTPUT
RENDER_DATE = "2026-10-21"
MARKET_NOTE = "Week 7 market quotes were retrieved after games; quote times unverified."
FOOTER = "Paired through Week 7: 379 spread games; 372 moneyline games. " + MARKET_NOTE
COHORT = {"spread_games": 379, "moneyline_games": 372, "max_week": 7}
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
COLORS = {
    "M1": PALETTE["projectorBlue"], "M2": PALETTE["projectorGreen"],
    "M3": PALETTE["gridironViolet"], "M4": "#008F93",
    "M5": "#495C77", "M10": "#9A513D",
}


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE,
                        help="Directory containing the three common-cohort source files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory for the figures, score tables, and manifests")
    parser.add_argument("--render-date", default="2026-10-21",
                        help="ISO date embedded in SVG metadata; set explicitly for a later snapshot")
    parser.add_argument("--market-note", default=None,
                        help="Provenance sentence printed in every figure footer")
    return parser


def configure(args: argparse.Namespace) -> None:
    global SOURCE, OUTPUT, RENDER_DATE, MARKET_NOTE
    SOURCE = args.source_dir.resolve()
    OUTPUT = args.output_dir.resolve()
    RENDER_DATE = args.render_date
    MARKET_NOTE = args.market_note or (
        "Week 7 market quotes were retrieved after games; quote times unverified."
        if SOURCE == DEFAULT_SOURCE else
        "Inspect market_provenance and quote times before prospective claims."
    )


def set_cohort(spread_games: int, moneyline_games: int, max_week: int) -> None:
    global COHORT, FOOTER
    COHORT = {"spread_games": spread_games, "moneyline_games": moneyline_games,
              "max_week": max_week}
    FOOTER = (f"Paired through Week {max_week}: {spread_games} spread games; "
              f"{moneyline_games} moneyline games. {MARKET_NOTE}")


def source_label() -> str:
    try:
        return str(SOURCE.relative_to(ROOT))
    except ValueError:
        return str(SOURCE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_scores() -> pd.DataFrame:
    model_source = SOURCE / "common_120_model_game_results.parquet"
    market_source = SOURCE / "common_cohort_vegas_games.csv"
    model = pd.read_parquet(model_source)
    market = pd.read_csv(market_source)
    if market.empty or market.game_id.duplicated().any():
        raise ValueError("Market source needs one row per game")
    if len(model) != 120 * len(market) or model.duplicated(
        ["fingerprint", "model_id", "game_id"]
    ).any() or set(model.game_id) != set(market.game_id):
        raise ValueError("Expected one prediction per roster cell and market game")
    if model.actual_home_margin.isna().any() or market.actual_home_margin.isna().any():
        raise ValueError("Every included game must have a final margin")
    actual = market.set_index("game_id").actual_home_margin
    if not np.allclose(model.actual_home_margin, model.game_id.map(actual)):
        raise ValueError("Model and market final margins disagree")
    spread = market.dropna(subset=["market_home_spread"])
    moneyline = market.dropna(subset=["market_home_probability"])
    if spread.empty or moneyline.empty:
        raise ValueError("Need at least one spread and moneyline game")
    if moneyline.market_home_probability.lt(0).any() or moneyline.market_home_probability.gt(1).any():
        raise ValueError("Market probabilities must be between zero and one")
    model_probabilities = model.loc[model.game_id.isin(moneyline.game_id),
                                    "pred_home_win_probability"]
    if (model_probabilities.isna().any() or model_probabilities.lt(0).any()
            or model_probabilities.gt(1).any()):
        raise ValueError("Model probabilities must be present and between zero and one")
    set_cohort(len(spread), len(moneyline), int(market.week.max()))
    vegas_mae = float(spread.margin_error.mean())
    vegas_su = float(spread.winner_correct.astype(float).mean())
    vegas_brier = float(moneyline.brier_error.mean())
    rows = []
    for (fingerprint, model_id), group in model.groupby(["fingerprint", "model_id"]):
        margin = group.loc[group.game_id.isin(spread.game_id)]
        probability = group.loc[group.game_id.isin(moneyline.game_id)]
        if (len(group) != len(market) or set(group.game_id) != set(market.game_id)
                or set(margin.game_id) != set(spread.game_id)
                or set(probability.game_id) != set(moneyline.game_id)):
            raise ValueError(f"Incomplete paired games for {fingerprint}/{model_id}")
        rows.append({
            "fingerprint": fingerprint, "fingerprint_number": int(fingerprint[1:].split("-")[0]),
            "model_id": model_id, "label": f"{fingerprint}/{model_id}",
            "kind": "model", "mae_games": len(margin), "brier_games": len(probability),
            "mae": float(margin.margin_error.mean()),
            "su_correct": int(margin.winner_correct.sum()),
            "su_accuracy": float(margin.winner_correct.mean()),
            "brier": float(probability.brier_error.mean()),
        })
    if len(rows) != 120 or set(zip((r["fingerprint"] for r in rows), (r["model_id"] for r in rows))) != {
        ("F17-market" if i == 17 else f"F{i}", model_id)
        for i in range(20) for model_id in MODELS
    }:
        raise ValueError("Expected all 120 F0–F19 / M1–M5,M10 cells")
    rows.append({
        "fingerprint": "Vegas", "fingerprint_number": 20, "model_id": "Vegas", "label": "Vegas",
        "kind": "market", "mae_games": len(spread), "brier_games": len(moneyline),
        "mae": vegas_mae, "su_correct": int(spread.winner_correct.astype(float).sum()),
        "su_accuracy": vegas_su, "brier": vegas_brier,
    })
    frame = pd.DataFrame(rows)
    frame["mae_skill_percent"] = 100 * (vegas_mae - frame.mae) / vegas_mae
    frame["brier_skill_percent"] = 100 * (vegas_brier - frame.brier) / vegas_brier
    frame["su_skill_percent"] = 100 * (frame.su_accuracy - vegas_su) / vegas_su
    frame["balanced_skill_percent"] = frame[
        ["mae_skill_percent", "brier_skill_percent", "su_skill_percent"]
    ].mean(axis=1)
    for metric, ascending in (("mae", True), ("brier", True), ("su_accuracy", False)):
        frame[f"rank_{metric}"] = frame[metric].rank(method="min", ascending=ascending).astype(int)
    frame["mean_metric_rank"] = frame[["rank_mae", "rank_brier", "rank_su_accuracy"]].mean(axis=1)
    frame["worst_metric_rank"] = frame[["rank_mae", "rank_brier", "rank_su_accuracy"]].max(axis=1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["kind", "fingerprint_number", "model_id"]).to_csv(OUTPUT / "paired_quality.csv", index=False)
    return frame


def figure(title: str, subtitle: str, size: tuple[float, float] = (16, 9.5),
           *, publication: bool = False):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PANEL)
    if publication:
        fig.subplots_adjust(left=.15, right=.75, top=.96,
                            bottom=.14 if size[1] < 10 else .085)
    else:
        fig.subplots_adjust(left=.105, right=.88, top=.80, bottom=.15)
        fig.text(.105, .957, "TDNET  /  2026 SCIENTIFIC ROSTER", color=MUTED, size=10, weight="bold")
        fig.text(.105, .900, title, color=INK, size=23, weight="bold")
        fig.text(.105, .850, subtitle, color=MUTED, size=11)
        fig.text(.105, .052, FOOTER, color=MUTED, size=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=MUTED, alpha=.13, zorder=0)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    if publication:
        ax.tick_params(axis="both", labelsize=15, length=6, width=1.2)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
    return fig, ax


def legend(ax, *, loc="upper left", anchor=(1.01, 1.0), neutral=False,
           vegas_threshold=False, publication=False):
    handles = [Line2D([0], [0], marker=MARKERS[m], linestyle="", color="none",
                      markerfacecolor="#54718F" if neutral else COLORS[m],
                      markeredgecolor="white", markersize=9,
                      label=f"{m} · {NAMES[m]}") for m in MODELS]
    if vegas_threshold:
        handles.append(Line2D([0], [0], color=GOLD, lw=2.3,
                              label="Vegas · 0% threshold"))
    else:
        handles.append(Line2D([0], [0], marker="*", linestyle="", color="none",
                              markerfacecolor=GOLD, markeredgecolor=INK,
                              markersize=14, label="Vegas"))
    if publication:
        handles.extend([
            Line2D([0], [0], color=INK, lw=4.3, label="Interquartile range"),
            Line2D([0], [0], marker="|", linestyle="", color=INK,
                   markersize=17, markeredgewidth=2.4, label="Median"),
            Line2D([0], [0], marker="o", linestyle="", color=INK,
                   markerfacecolor=PANEL, markersize=9, label="Mean"),
        ])
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, frameon=False,
              labelspacing=.9, fontsize=14 if publication else 9,
              markerscale=1.25 if publication else 1,
              handlelength=2.1 if publication else 2)


def vegas_line(ax, *, horizontal=False):
    draw = ax.axhline if horizontal else ax.axvline
    draw(0, color=GOLD, lw=2.3, alpha=.92, zorder=2)


def plot_points(ax, frame: pd.DataFrame, x: str, y: str, *, size=72, brier_color=False):
    models = frame.loc[frame.kind.eq("model")]
    brier_min = .155 if SOURCE == DEFAULT_SOURCE else min(.155, float(frame.brier.min()))
    brier_max = .24 if SOURCE == DEFAULT_SOURCE else max(.24, float(frame.brier.max()))
    for model_id in MODELS:
        group = models.loc[models.model_id.eq(model_id)]
        if brier_color:
            ax.scatter(group[x], group[y], c=group.brier, cmap="Blues_r", vmin=brier_min,
                       vmax=brier_max, marker=MARKERS[model_id], s=size, alpha=.82,
                       edgecolors="white", linewidths=.6, zorder=3)
        else:
            ax.scatter(group[x], group[y], color=COLORS[model_id], marker=MARKERS[model_id],
                       s=size, alpha=.84, edgecolors="white", linewidths=.6, zorder=3)
    vegas = frame.loc[frame.kind.eq("market")].iloc[0]
    ax.scatter([vegas[x]], [vegas[y]], color=GOLD, marker="*", s=size * 3.2,
               edgecolors=INK, linewidths=1.1, zorder=7)
    return vegas


def annotate_market(ax, vegas, x, y, offset=(12, 10)):
    ax.annotate("Vegas", (vegas[x], vegas[y]), xytext=offset,
                textcoords="offset points", color=INK, fontsize=10,
                weight="bold", bbox={"boxstyle": "round,pad=.25",
                                     "facecolor": PANEL, "edgecolor": GOLD,
                                     "linewidth": 1.2})


def metric_landscape(frame):
    fig, ax = figure("Error landscape", "Each mark is one model. Left and down are better; size shows straight-up wins.")
    for model_id in MODELS:
        group = frame.loc[frame.model_id.eq(model_id)]
        ax.scatter(group.mae, group.brier, marker=MARKERS[model_id], color=COLORS[model_id],
                   s=42 + (group.su_correct - 230) * 1.6, alpha=.82,
                   edgecolors="white", linewidth=.6, zorder=3)
    vegas = frame.loc[frame.kind.eq("market")].iloc[0]
    ax.scatter(vegas.mae, vegas.brier, s=300, marker="*", color=GOLD, edgecolor=INK, zorder=7)
    annotate_market(ax, vegas, "mae", "brier", (17, -25))
    ax.set(xlabel="Margin MAE · points ↓", ylabel="Brier score ↓")
    if SOURCE == DEFAULT_SOURCE:
        ax.set(xlim=(11.1, 17.1), ylim=(.151, .243))
    legend(ax)
    return fig


def accuracy_landscape(frame):
    fig, ax = figure("Winner versus margin quality", "Each mark is one model. Left and up are better; darker blue means lower Brier score.")
    vegas = plot_points(ax, frame, "mae", "su_accuracy", brier_color=True)
    annotate_market(ax, vegas, "mae", "su_accuracy", (18, -25))
    ax.set(xlabel="Margin MAE · points ↓", ylabel="Straight-up accuracy ↑")
    if SOURCE == DEFAULT_SOURCE:
        ax.set(xlim=(11.1, 17.1), ylim=(.605, .795))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    legend(ax, loc="upper right", anchor=(.99, .99), neutral=True)
    brier_min = .155 if SOURCE == DEFAULT_SOURCE else min(.155, float(frame.brier.min()))
    brier_max = .24 if SOURCE == DEFAULT_SOURCE else max(.24, float(frame.brier.max()))
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(brier_min, brier_max), cmap="Blues_r"),
                 ax=ax, fraction=.025, pad=.015, label="Brier score ↓")
    return fig


def ranked_ladder(frame):
    ordered = frame.loc[frame.kind.eq("model")].sort_values(
        "balanced_skill_percent", ascending=False).reset_index(drop=True)
    fig, ax = figure("Balanced quality ladder", "120 model cells ranked by mean relative skill; the gold line is the Vegas threshold.", (16, 22))
    for item, y in zip(fig.texts, (.989, .967, .948, .018)):
        item.set_y(y)
    fig.subplots_adjust(left=.12, right=.86, top=.927, bottom=.055)
    for pos, row in ordered.iterrows():
        ax.scatter(row.balanced_skill_percent, pos, marker=MARKERS[row.model_id],
                   color=COLORS[row.model_id], s=35, edgecolor="white",
                   linewidth=.75, zorder=3)
    vegas_line(ax)
    ax.set_ylim(len(ordered) + .5, -1.5)
    ax.set_yticks(range(0, len(ordered), 10))
    ax.set_yticklabels([str(i + 1) for i in range(0, len(ordered), 10)])
    ax.set(xlabel="Balanced relative skill versus Vegas · % ↑", ylabel="Rank among 120 models")
    legend(ax, vegas_threshold=True)
    return fig


def roster_matrix(frame):
    fig, ax = figure("The complete 20 × 6 roster", "Every matrix cell is a distinct model; color shows balanced relative skill versus Vegas.", (14, 12))
    fig.subplots_adjust(left=.12, right=.82, top=.82, bottom=.12)
    models = frame.loc[frame.kind.eq("model")]
    norm = Normalize(vmin=min(-36, models.balanced_skill_percent.min()),
                     vmax=max(0, models.balanced_skill_percent.max()))
    cmap = plt.get_cmap("Blues")
    for _, row in models.iterrows():
        x = MODELS.index(row.model_id)
        y = row.fingerprint_number
        ax.scatter(x, y, s=250, marker="o", c=[row.balanced_skill_percent],
                   cmap=cmap, norm=norm, edgecolor="white", linewidth=.8, zorder=3)
    ax.set_xticks(range(6), MODELS)
    ax.set_yticks(range(20), ["F17-market" if i == 17 else f"F{i}" for i in range(20)])
    ax.set_xlim(-.6, 5.6)
    ax.set_ylim(19.8, -.8)
    ax.set(xlabel="Model architecture", ylabel="Fingerprint")
    ax.grid(alpha=.09)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            fraction=.035, pad=.04,
                            label="Balanced relative skill · % (0 = Vegas)")
    colorbar.ax.axhline(0, color=GOLD, lw=2.3, alpha=.92, clip_on=False)
    return fig


def _swarm_offsets(xs: np.ndarray, min_x: float = .65, step: float = .16) -> np.ndarray:
    """Deterministic vertical packing with exact horizontal metric positions."""
    positions = np.full(len(xs), np.nan)
    for index in np.argsort(xs):
        candidates = [0.0]
        for level in range(1, 11):
            candidates.extend((level * step, -level * step))
        for candidate in candidates:
            if all(abs(xs[index] - xs[j]) >= min_x or abs(candidate - positions[j]) >= step - 1e-8
                   for j in range(len(xs)) if not np.isnan(positions[j])):
                positions[index] = candidate
                break
        if np.isnan(positions[index]):
            raise ValueError("Swarm packing failed")
    return positions


def architecture_swarm(frame):
    fig, ax = figure("Which architecture holds up?", "Twenty fingerprints per row; horizontal position is exact. Gold line = Vegas threshold.")
    for i, model_id in enumerate(MODELS):
        group = frame.loc[frame.model_id.eq(model_id)].sort_values("fingerprint_number").reset_index(drop=True)
        xs = group.balanced_skill_percent.to_numpy()
        offsets = _swarm_offsets(xs)
        for j, row in group.iterrows():
            ax.scatter(row.balanced_skill_percent, i + offsets[j] * .45, marker=MARKERS[model_id],
                       color=COLORS[model_id], s=80, edgecolor="white",
                       linewidth=.8, zorder=4)
    vegas_line(ax)
    ax.set_yticks(range(6), [f"{m} · {NAMES[m]}" for m in MODELS])
    ax.set_ylim(5.7, -.7)
    ax.set(xlabel="Balanced relative skill versus Vegas · % ↑")
    legend(ax, vegas_threshold=True)
    return fig


def _violin(ax, values: np.ndarray, position: float, color: str, width: float = .68):
    result = ax.violinplot([values], positions=[position], vert=False, widths=width,
                           showmeans=False, showmedians=False, showextrema=False)
    body = result["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(.17)
    body.set_linewidth(1.4)
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    ax.plot([q1, q3], [position, position], color=INK, lw=4.3,
            solid_capstyle="round", zorder=5)
    ax.scatter([median], [position], marker="|", color=PANEL, s=260,
               linewidth=2.4, zorder=8)
    ax.scatter([values.mean()], [position], marker="o", facecolors=PANEL,
               edgecolors=INK, s=50, linewidths=1.4, zorder=9)


def architecture_violin(frame, *, publication=False):
    fig, ax = figure("Architecture distributions", "Twenty fingerprints per family. Bar = IQR; tick = median; open circle = mean; gold line = Vegas.",
                     (15, 8.5) if publication else (16, 9.5), publication=publication)
    for i, model_id in enumerate(MODELS):
        group = frame.loc[frame.model_id.eq(model_id)].sort_values("fingerprint_number").reset_index(drop=True)
        xs = group.balanced_skill_percent.to_numpy()
        _violin(ax, xs, i, COLORS[model_id])
        offsets = _swarm_offsets(xs)
        for j, row in group.iterrows():
            ax.scatter(row.balanced_skill_percent, i + offsets[j] * .30,
                       marker=MARKERS[model_id], color=COLORS[model_id],
                       s=42, edgecolor="white", linewidth=.8, zorder=7)
    vegas_line(ax)
    ax.set_yticks(range(6), [f"{m} · {NAMES[m]}" for m in MODELS])
    ax.set_ylim(5.7, -.7)
    ax.set(xlabel="Balanced relative skill versus Vegas · % ↑")
    legend(ax, vegas_threshold=True, publication=publication)
    return fig


def fingerprint_swarm(frame):
    fig, ax = figure("Fingerprint distributions", "Six architectures per fingerprint. Vertical offsets separate neighbors; gold line = Vegas threshold.", (16, 15))
    for number in range(20):
        group = frame.loc[frame.kind.eq("model") & frame.fingerprint_number.eq(number)].copy()
        group["model_order"] = group.model_id.map({model_id: i for i, model_id in enumerate(MODELS)})
        group = group.sort_values("model_order").reset_index(drop=True)
        xs = group.balanced_skill_percent.to_numpy()
        offsets = _swarm_offsets(xs)
        for j, row in group.iterrows():
            ax.scatter(row.balanced_skill_percent, number + offsets[j] * .36,
                       marker=MARKERS[row.model_id], color=COLORS[row.model_id],
                       s=67, edgecolor="white", linewidth=.75, zorder=5)
    vegas_line(ax)
    ax.set_yticks(range(20),
                  ["F17-market" if i == 17 else f"F{i}" for i in range(20)])
    ax.set_ylim(19.7, -.7)
    ax.set(xlabel="Balanced relative skill versus Vegas · % ↑", ylabel="Fingerprint")
    legend(ax, vegas_threshold=True)
    return fig


def fingerprint_violin(frame, *, publication=False):
    fig, ax = figure("Fingerprint spread across architectures", "Six models per row. Bar = IQR; tick = median; open circle = mean; gold line = Vegas.",
                     (15, 15) if publication else (16, 15), publication=publication)
    models = frame.loc[frame.kind.eq("model")]
    for number in range(20):
        group = models.loc[models.fingerprint_number.eq(number)].copy()
        group["model_order"] = group.model_id.map({model_id: i for i, model_id in enumerate(MODELS)})
        group = group.sort_values("model_order").reset_index(drop=True)
        xs = group.balanced_skill_percent.to_numpy()
        _violin(ax, xs, number, PALETTE["projectorBlue"], width=.65)
        offsets = _swarm_offsets(xs)
        for j, row in group.iterrows():
            ax.scatter(row.balanced_skill_percent, number + offsets[j] * .30,
                       marker=MARKERS[row.model_id], color=COLORS[row.model_id],
                       s=48, edgecolor="white", linewidth=.65, zorder=7)
    vegas_line(ax)
    ax.set_yticks(range(20),
                  ["F17-market" if i == 17 else f"F{i}" for i in range(20)])
    ax.set_ylim(19.7, -.7)
    ax.set(xlabel="Balanced relative skill versus Vegas · % ↑", ylabel="Fingerprint")
    legend(ax, vegas_threshold=True, publication=publication)
    return fig


def fingerprint_progression(frame):
    fig, ax = figure("Fingerprint progression", "Six architectures per fingerprint; gold line is the zero-skill Vegas threshold.")
    fig.subplots_adjust(bottom=.20)
    for model_id in MODELS:
        group = frame.loc[frame.model_id.eq(model_id)]
        offset = (MODELS.index(model_id) - 2.5) * .11
        ax.scatter(group.fingerprint_number + offset, group.balanced_skill_percent,
                   marker=MARKERS[model_id], color=COLORS[model_id], s=75,
                   edgecolor="white", linewidth=.7, zorder=3)
    vegas_line(ax, horizontal=True)
    ax.set_xticks(range(20),
                  ["F17\nmarket" if i == 17 else f"F{i}" for i in range(20)], rotation=45)
    ax.set_xlim(-.6, 19.6)
    ax.set(xlabel="Fingerprint generation", ylabel="Balanced relative skill versus Vegas · % ↑")
    legend(ax, vegas_threshold=True)
    return fig


PLOTS = {
    "01_error_landscape": metric_landscape,
    "02_accuracy_landscape": accuracy_landscape,
    "03_balanced_quality_ladder": ranked_ladder,
    "04_roster_matrix": roster_matrix,
    "05_architecture_swarm": architecture_swarm,
    "06_fingerprint_progression": fingerprint_progression,
    "07_architecture_violin": architecture_violin,
    "08_fingerprint_swarm": fingerprint_swarm,
    "09_fingerprint_violin": fingerprint_violin,
}
PUBLICATION_PLOTS = {
    "07_architecture_violin_publication": lambda frame: architecture_violin(frame, publication=True),
    "09_fingerprint_violin_publication": lambda frame: fingerprint_violin(frame, publication=True),
}


def main() -> None:
    configure(argument_parser().parse_args())
    plt.style.use(ROOT / "docs/style/color_palettes/gridiron_light.mplstyle")
    plt.rcParams.update({"figure.autolayout": False,
                         "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
                         "axes.facecolor": PANEL, "font.family": "DejaVu Sans",
                         "svg.hashsalt": "tdnet-2026-scientific-quality"})
    frame = load_scores()
    model_rows = frame.loc[frame.kind.eq("model")]
    for name, keys in (("architecture_distribution", ["model_id"]),
                       ("fingerprint_distribution", ["fingerprint", "fingerprint_number"])):
        distribution = model_rows.groupby(keys).balanced_skill_percent.agg(
            count="count", minimum="min", mean="mean", q1=lambda values: values.quantile(.25),
            median="median", q3=lambda values: values.quantile(.75), maximum="max",
            standard_deviation="std"
        ).reset_index()
        distribution["iqr"] = distribution.q3 - distribution.q1
        distribution.to_csv(OUTPUT / f"{name}.csv", index=False)
    for name, plotter in {**PLOTS, **PUBLICATION_PLOTS}.items():
        fig = plotter(frame)
        for extension in ("png", "svg"):
            path = OUTPUT / f"{name}.{extension}"
            options = {"metadata": {"Date": RENDER_DATE}} if extension == "svg" else {}
            fig.savefig(path, dpi=300 if name in PUBLICATION_PLOTS else 200,
                        facecolor=PAPER, **options)
            if extension == "svg":
                # Matplotlib emits trailing spaces inside path-data lines.
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)
    generated = [OUTPUT / f"{name}.{extension}" for name in (*PLOTS, *PUBLICATION_PLOTS)
                 for extension in ("png", "svg")]
    generated += [OUTPUT / f"{name}.csv" for name in
                  ("paired_quality", "architecture_distribution", "fingerprint_distribution")]
    manifest = {
        "source": {path.name: sha256(path) for path in
                   (SOURCE / "common_120_model_game_results.parquet",
                    SOURCE / "common_cohort_vegas_games.csv")},
        "source_directory": source_label(),
        "render_date": RENDER_DATE,
        "market_note": MARKET_NOTE,
        "cohort": {**COHORT, "model_points": 120, "vegas_baselines": 1},
        "balanced_skill_percent": "mean(100*(vegas_mae-model_mae)/vegas_mae, 100*(vegas_brier-model_brier)/vegas_brier, 100*(model_su-vegas_su)/vegas_su)",
        "outputs": {path.name: sha256(path) for path in generated},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(PLOTS)} gallery figures and {len(PUBLICATION_PLOTS)} publication variants from 120 model cells and one Vegas baseline to {OUTPUT}")


if __name__ == "__main__":
    main()
