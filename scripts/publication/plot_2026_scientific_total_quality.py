"""Six one-point-per-model views of paired 2026 scientific roster quality.

The paired market cohort is retrospective through Week 7. In particular, the
Week 7 market quotes were retrieved after games and quote times are unverified.
"""

from __future__ import annotations

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
SOURCE = ROOT / "publication/2026/week_07/post_game/retrospective_replay/cumulative/scientific_f0_f19"
OUTPUT = ROOT / "publication/2026/research/scientific_total_quality"
MODELS = ("M1", "M2", "M3", "M4", "M5", "M10")
MARKERS = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "P", "M10": "X"}
NAMES = {"M1": "Linear", "M2": "Spline", "M3": "Tree", "M4": "Boosted",
         "M5": "Neural", "M10": "KNN"}
PALETTE = json.loads((ROOT / "gridiron.palette.json").read_text())["colors"]
PAPER = PALETTE["parchment"]
PANEL = "#FFFCF7"
INK = PALETTE["midnightGridiron"]
MUTED = PALETTE["steelGrey"]
PINK = PALETTE["edgePink"]
GOLD = PALETTE["brass"]
COLORS = {
    "M1": PALETTE["projectorBlue"], "M2": PALETTE["projectorGreen"],
    "M3": PALETTE["gridironViolet"], "M4": "#008F93",
    "M5": "#495C77", "M10": "#9A513D",
}


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
    if len(model) != 45_600 or model.game_id.nunique() != 380:
        raise ValueError("Scientific model-game cohort changed")
    spread = market.dropna(subset=["market_home_spread"])
    moneyline = market.dropna(subset=["market_home_probability"])
    if len(spread) != 379 or len(moneyline) != 372:
        raise ValueError("Paired market cohort changed")
    if not set(moneyline.game_id).issubset(spread.game_id):
        raise ValueError("Moneyline games must be part of the spread cohort")
    vegas_mae = float(spread.margin_error.mean())
    vegas_su = float(spread.winner_correct.astype(float).mean())
    vegas_brier = float(moneyline.brier_error.mean())
    rows = []
    for (fingerprint, model_id), group in model.groupby(["fingerprint", "model_id"]):
        margin = group.loc[group.game_id.isin(spread.game_id)]
        probability = group.loc[group.game_id.isin(moneyline.game_id)]
        if set(margin.game_id) != set(spread.game_id) or set(probability.game_id) != set(moneyline.game_id):
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


def figure(title: str, subtitle: str, size: tuple[float, float] = (16, 9.5)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PANEL)
    fig.subplots_adjust(left=.105, right=.88, top=.80, bottom=.15)
    fig.text(.105, .957, "TDNET  /  2026 SCIENTIFIC ROSTER", color=MUTED, size=10, weight="bold")
    fig.text(.105, .900, title, color=INK, size=23, weight="bold")
    fig.text(.105, .850, subtitle, color=MUTED, size=11)
    fig.text(.105, .052,
             "Paired through Week 7: 379 spread games; 372 moneyline games. Week 7 market quotes were retrieved after games; quote times unverified.",
             color=MUTED, size=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=MUTED, alpha=.13, zorder=0)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    return fig, ax


def legend(ax, *, loc="upper left", anchor=(1.01, 1.0), neutral=False):
    handles = [Line2D([0], [0], marker=MARKERS[m], linestyle="", color="none",
                      markerfacecolor="#54718F" if neutral else COLORS[m],
                      markeredgecolor="white", markersize=9,
                      label=f"{m} · {NAMES[m]}") for m in MODELS]
    handles += [Line2D([0], [0], marker="D", linestyle="", color="none",
                       markerfacecolor=PINK, markeredgecolor=INK, markersize=10, label="F19/M4"),
                Line2D([0], [0], marker="*", linestyle="", color="none",
                       markerfacecolor=GOLD, markeredgecolor=INK, markersize=14, label="Vegas")]
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, frameon=False,
              labelspacing=.9, fontsize=9)


def plot_points(ax, frame: pd.DataFrame, x: str, y: str, *, size=72, brier_color=False):
    models = frame.loc[frame.kind.eq("model")]
    for model_id in MODELS:
        group = models.loc[models.model_id.eq(model_id) & models.label.ne("F19/M4")]
        if brier_color:
            ax.scatter(group[x], group[y], c=group.brier, cmap="Blues_r", vmin=.155,
                       vmax=.24, marker=MARKERS[model_id], s=size, alpha=.82,
                       edgecolors="white", linewidths=.6, zorder=3)
        else:
            ax.scatter(group[x], group[y], color=COLORS[model_id], marker=MARKERS[model_id],
                       s=size, alpha=.84, edgecolors="white", linewidths=.6, zorder=3)
    pink = frame.loc[frame.label.eq("F19/M4")].iloc[0]
    vegas = frame.loc[frame.kind.eq("market")].iloc[0]
    ax.scatter([pink[x]], [pink[y]], color=PINK, marker="D", s=size * 2.0,
               edgecolors=INK, linewidths=1.2, zorder=6)
    ax.scatter([vegas[x]], [vegas[y]], color=GOLD, marker="*", s=size * 3.2,
               edgecolors=INK, linewidths=1.1, zorder=7)
    return pink, vegas


def annotate_special(ax, pink, vegas, x, y, pink_offset=(12, -23), vegas_offset=(12, 10)):
    for row, label, offset, color in ((pink, "F19/M4", pink_offset, PINK),
                                      (vegas, "Vegas", vegas_offset, GOLD)):
        ax.annotate(label, (row[x], row[y]), xytext=offset, textcoords="offset points",
                    color=INK, fontsize=10, weight="bold",
                    bbox={"boxstyle": "round,pad=.25", "facecolor": PANEL,
                          "edgecolor": color, "linewidth": 1.2})


def metric_landscape(frame):
    fig, ax = figure("Error landscape", "Each mark is one model. Left and down are better; size shows straight-up wins.")
    for model_id in MODELS:
        group = frame.loc[frame.model_id.eq(model_id) & frame.label.ne("F19/M4")]
        ax.scatter(group.mae, group.brier, marker=MARKERS[model_id], color=COLORS[model_id],
                   s=42 + (group.su_correct - 230) * 1.6, alpha=.82,
                   edgecolors="white", linewidth=.6, zorder=3)
    pink = frame.loc[frame.label.eq("F19/M4")].iloc[0]
    vegas = frame.loc[frame.kind.eq("market")].iloc[0]
    ax.scatter(pink.mae, pink.brier, s=170, marker="D", color=PINK, edgecolor=INK, zorder=6)
    ax.scatter(vegas.mae, vegas.brier, s=300, marker="*", color=GOLD, edgecolor=INK, zorder=7)
    annotate_special(ax, pink, vegas, "mae", "brier", (18, 13), (17, -25))
    ax.set(xlabel="Margin MAE · points ↓", ylabel="Brier score ↓", xlim=(11.1, 17.1), ylim=(.151, .243))
    legend(ax)
    return fig


def accuracy_landscape(frame):
    fig, ax = figure("Winner versus margin quality", "Each mark is one model. Left and up are better; darker blue means lower Brier score.")
    pink, vegas = plot_points(ax, frame, "mae", "su_accuracy", brier_color=True)
    annotate_special(ax, pink, vegas, "mae", "su_accuracy", (18, 12), (18, -25))
    ax.set(xlabel="Margin MAE · points ↓", ylabel="Straight-up accuracy ↑",
           xlim=(11.1, 17.1), ylim=(.605, .795))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    legend(ax, loc="upper right", anchor=(.99, .99), neutral=True)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(.155, .24), cmap="Blues_r"),
                 ax=ax, fraction=.025, pad=.015, label="Brier score ↓")
    return fig


def ranked_ladder(frame):
    ordered = frame.sort_values("balanced_skill_percent", ascending=False).reset_index(drop=True)
    fig, ax = figure("Balanced quality ladder", "One point per model, ordered by the mean of three relative skill percentages; higher is better.", (16, 22))
    for item, y in zip(fig.texts, (.989, .967, .948, .018)):
        item.set_y(y)
    fig.subplots_adjust(left=.12, right=.86, top=.927, bottom=.055)
    for pos, row in ordered.iterrows():
        color = GOLD if row.kind == "market" else PINK if row.label == "F19/M4" else COLORS[row.model_id]
        marker = "*" if row.kind == "market" else MARKERS[row.model_id]
        ax.scatter(row.balanced_skill_percent, pos, marker=marker, color=color,
                   s=165 if row.kind == "market" else 95 if row.label == "F19/M4" else 35,
                   edgecolor=INK if row.kind == "market" or row.label == "F19/M4" else "white",
                   linewidth=.75, zorder=3)
    ax.axvline(0, color=GOLD, lw=1.5, alpha=.65, zorder=1)
    ax.set_ylim(len(ordered) + .5, -1.5)
    ax.set_yticks(range(0, len(ordered), 10))
    ax.set_yticklabels([str(i + 1) for i in range(0, len(ordered), 10)])
    ax.set(xlabel="Balanced relative skill versus Vegas · % ↑", ylabel="Rank among 121 points")
    for label in ("Vegas", "F19/M4"):
        idx = int(ordered.index[ordered.label.eq(label)][0])
        row = ordered.iloc[idx]
        ax.annotate(f"{label}  {row.balanced_skill_percent:+.2f}%",
                    (row.balanced_skill_percent, idx), xytext=(-75, -5), textcoords="offset points",
                    fontsize=10, weight="bold", color=INK)
    legend(ax)
    return fig


def roster_matrix(frame):
    fig, ax = figure("The complete 20 × 6 roster", "Every matrix cell is a distinct model; color shows balanced relative skill versus Vegas.", (14, 12))
    fig.subplots_adjust(left=.12, right=.82, top=.82, bottom=.12)
    models = frame.loc[frame.kind.eq("model")]
    norm = Normalize(vmin=min(-36, models.balanced_skill_percent.min()), vmax=0)
    cmap = plt.get_cmap("Blues")
    for _, row in models.iterrows():
        x = MODELS.index(row.model_id)
        y = row.fingerprint_number
        if row.label == "F19/M4":
            ax.scatter(x, y, s=260, marker="D", color=PINK, edgecolor=INK, linewidth=1, zorder=6)
        else:
            ax.scatter(x, y, s=250, marker="o", c=[row.balanced_skill_percent],
                       cmap=cmap, norm=norm, edgecolor="white", linewidth=.8, zorder=3)
    ax.scatter(6.45, 19, s=400, marker="*", color=GOLD, edgecolor=INK, zorder=7)
    ax.set_xticks([*range(6), 6.45], [*MODELS, "Vegas"])
    ax.set_yticks(range(20), ["F17-market" if i == 17 else f"F{i}" for i in range(20)])
    ax.set_xlim(-.6, 7.0)
    ax.set_ylim(19.8, -.8)
    ax.set(xlabel="Model architecture", ylabel="Fingerprint")
    ax.grid(alpha=.09)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 fraction=.035, pad=.04, label="Balanced relative skill · %")
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
    fig, ax = figure("Which architecture holds up?", "Twenty fingerprints per row; horizontal position is exact, vertical offset separates nearby points.")
    for i, model_id in enumerate(MODELS):
        group = frame.loc[frame.model_id.eq(model_id)].sort_values("fingerprint_number").reset_index(drop=True)
        xs = group.balanced_skill_percent.to_numpy()
        offsets = _swarm_offsets(xs)
        for j, row in group.iterrows():
            color = PINK if row.label == "F19/M4" else COLORS[model_id]
            ax.scatter(row.balanced_skill_percent, i + offsets[j] * .45, marker=MARKERS[model_id],
                       color=color, s=155 if row.label == "F19/M4" else 80,
                       edgecolor=INK if row.label == "F19/M4" else "white", linewidth=.8, zorder=4)
    vegas = frame.loc[frame.kind.eq("market")].iloc[0]
    ax.scatter(vegas.balanced_skill_percent, 6.25, marker="*", color=GOLD,
               s=350, edgecolor=INK, zorder=7)
    ax.axvline(0, color=GOLD, lw=1.4, alpha=.65)
    ax.set_yticks([*range(6), 6.25], [f"{m} · {NAMES[m]}" for m in MODELS] + ["Vegas"])
    ax.set_ylim(6.75, -.7)
    ax.set(xlabel="Balanced relative skill versus Vegas · % ↑")
    legend(ax)
    return fig


def fingerprint_progression(frame):
    fig, ax = figure("Fingerprint progression", "Six architectures per fingerprint; horizontal offsets separate their points. Higher is better.")
    fig.subplots_adjust(bottom=.20)
    for model_id in MODELS:
        group = frame.loc[frame.model_id.eq(model_id) & frame.label.ne("F19/M4")]
        offset = (MODELS.index(model_id) - 2.5) * .11
        ax.scatter(group.fingerprint_number + offset, group.balanced_skill_percent,
                   marker=MARKERS[model_id], color=COLORS[model_id], s=75,
                   edgecolor="white", linewidth=.7, zorder=3)
    pink = frame.loc[frame.label.eq("F19/M4")].iloc[0]
    vegas = frame.loc[frame.kind.eq("market")].iloc[0]
    pink_x = 19 + (MODELS.index("M4") - 2.5) * .11
    ax.scatter(pink_x, pink.balanced_skill_percent, marker="D", color=PINK,
               s=175, edgecolor=INK, zorder=6)
    ax.scatter(20.3, vegas.balanced_skill_percent, marker="*", color=GOLD,
               s=320, edgecolor=INK, zorder=7)
    ax.annotate("F19/M4", (pink_x, pink.balanced_skill_percent), xytext=(-65, 16),
                textcoords="offset points", weight="bold", color=INK)
    ax.annotate("Vegas", (20.3, 0), xytext=(-18, -30), textcoords="offset points", weight="bold", color=INK)
    ax.set_xticks([*range(20), 20.3],
                  [*["F17\nmarket" if i == 17 else f"F{i}" for i in range(20)], "Vegas"], rotation=45)
    ax.set_xlim(-.6, 20.9)
    ax.set(xlabel="Fingerprint generation", ylabel="Balanced relative skill versus Vegas · % ↑")
    legend(ax)
    return fig


PLOTS = {
    "01_error_landscape": metric_landscape,
    "02_accuracy_landscape": accuracy_landscape,
    "03_balanced_quality_ladder": ranked_ladder,
    "04_roster_matrix": roster_matrix,
    "05_architecture_swarm": architecture_swarm,
    "06_fingerprint_progression": fingerprint_progression,
}


def main() -> None:
    plt.style.use(ROOT / "docs/style/color_palettes/gridiron_light.mplstyle")
    plt.rcParams.update({"figure.autolayout": False,
                         "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
                         "axes.facecolor": PANEL, "font.family": "DejaVu Sans"})
    frame = load_scores()
    for name, plotter in PLOTS.items():
        fig = plotter(frame)
        for extension in ("png", "svg"):
            path = OUTPUT / f"{name}.{extension}"
            fig.savefig(path, dpi=200, facecolor=PAPER)
            if extension == "svg":
                # Matplotlib emits trailing spaces inside path-data lines.
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)
    manifest = {
        "source": {path.name: sha256(path) for path in
                   (SOURCE / "common_120_model_game_results.parquet",
                    SOURCE / "common_cohort_vegas_games.csv")},
        "cohort": {"spread_games": 379, "moneyline_games": 372, "model_points": 120, "vegas_points": 1},
        "balanced_skill_percent": "mean(100*(vegas_mae-model_mae)/vegas_mae, 100*(vegas_brier-model_brier)/vegas_brier, 100*(model_su-vegas_su)/vegas_su)",
        "outputs": {p.name: sha256(p) for p in sorted(OUTPUT.iterdir()) if p.suffix in (".csv", ".png", ".svg")},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(PLOTS)} figures with {len(frame)} points each to {OUTPUT}")


if __name__ == "__main__":
    main()
