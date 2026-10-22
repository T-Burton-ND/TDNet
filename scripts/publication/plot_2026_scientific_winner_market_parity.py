"""Categorical winner-accuracy heatmap focused on exact market parity.

Colors intentionally reserve blue for a model ahead of Vegas and brass for an
exact tie. Every cell prints the integer difference in correct winner calls,
so coarse color bands do not hide the measured result.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

import plot_2026_scientific_total_quality as style


STEM = "winner_accuracy_market_parity_heatmap"
LEDGER = "fingerprint_architecture_heatmap_scores.csv"
SCORE = "winner_accuracy_market_parity_scores.csv"
MANIFEST = "winner_accuracy_market_parity_manifest.json"
FINGERPRINTS = tuple([f"F{i}" for i in range(17)] + ["F17-market", "F18", "F19"])
COLORS = ("#8F4950", "#C37B6C", "#E3BDA9", "#F3E8D8",
          style.GOLD, style.PALETTE["projectorBlue"])
GROUPS = ("At least 10 fewer", "5–9 fewer", "2–4 fewer", "1 fewer",
          "Equal to Vegas", "Ahead of Vegas")


def scored_cells(ledger_path: Path, market_path: Path,
                 model_path: Path) -> tuple[pd.DataFrame, dict]:
    cells = pd.read_csv(ledger_path)
    market = pd.read_csv(market_path)
    shared = market.dropna(subset=["market_home_spread", "market_home_probability"])
    if (len(shared) == 0 or shared.game_id.duplicated().any() or len(cells) != 120 or
            cells.duplicated(["fingerprint", "model_id"]).any() or
            set(cells.fingerprint) != set(FINGERPRINTS) or
            set(cells.model_id) != set(style.MODELS) or
            cells.games.ne(len(shared)).any()):
        raise ValueError("Winner parity heatmap requires the complete paired 120-cell cohort")
    raw_wins = cells.winner_advantage_pp.to_numpy(float) * len(shared) / 100
    rounded_wins = np.rint(raw_wins)
    if not np.allclose(raw_wins, rounded_wins, atol=1e-8):
        raise ValueError("Winner advantage is not an integer difference in correct games")
    cells = cells[["fingerprint", "model_id", "games", "winner_advantage_pp"]].copy()
    cells["extra_correct_winners"] = rounded_wins.astype(int)
    market_wins = int(shared.winner_correct.sum())
    cells["model_correct_winners"] = market_wins + cells.extra_correct_winners
    raw = pd.read_parquet(model_path, columns=["game_id", "fingerprint",
                                                "model_id", "winner_correct"])
    raw = raw.loc[raw.game_id.isin(shared.game_id)]
    if (len(raw) != 120 * len(shared) or
            raw.duplicated(["fingerprint", "model_id", "game_id"]).any() or
            raw.groupby(["fingerprint", "model_id"]).game_id.nunique().ne(len(shared)).any()):
        raise ValueError("Raw model winner calls differ from the shared game cohort")
    observed = raw.groupby(["fingerprint", "model_id"]).winner_correct.sum()
    expected = cells.set_index(["fingerprint", "model_id"]).model_correct_winners
    if not observed.reindex(expected.index).equals(expected.astype(observed.dtype)):
        raise ValueError("Winner-parity ledger disagrees with archived model calls")
    delta = cells.extra_correct_winners.to_numpy()
    cells["color_band"] = np.select(
        [delta <= -10, delta <= -5, delta <= -2, delta == -1, delta == 0, delta > 0],
        np.arange(6), default=-1).astype(int)
    if (cells.color_band.lt(0).any() or cells.model_correct_winners.lt(0).any() or
            cells.model_correct_winners.gt(len(shared)).any()):
        raise ValueError("Invalid winner parity score or color band")
    evidence = {"paired_games": len(shared), "vegas_correct_winners": market_wins,
                "max_week": int(shared.week.max()),
                "spread_games": int(market.market_home_spread.notna().sum()),
                "ahead_cells": int((delta > 0).sum()), "equal_cells": int((delta == 0).sum()),
                "one_win_behind_cells": int((delta == -1).sum())}
    return cells, evidence


def _legend_label(cells: pd.DataFrame, band: int) -> str:
    selected = cells.loc[cells.color_band.eq(band)]
    if band in (4, 5) and len(selected) == 1:
        row = selected.iloc[0]
        name = f"{row.fingerprint}/{row.model_id}"
        if band == 5:
            return f"Ahead · {name} (+{row.extra_correct_winners})"
        return f"Equal · {name} (0)"
    return f"{GROUPS[band]} · {len(selected)} model{'s' if len(selected) != 1 else ''}"


def plot(cells: pd.DataFrame, evidence: dict) -> None:
    table = cells.pivot(index="model_id", columns="fingerprint",
                        values="extra_correct_winners").reindex(
                            index=style.MODELS, columns=FINGERPRINTS)
    bands = cells.pivot(index="model_id", columns="fingerprint", values="color_band").reindex(
        index=style.MODELS, columns=FINGERPRINTS)
    if table.isna().any().any() or bands.isna().any().any():
        raise ValueError("Winner parity heatmap has an empty roster cell")
    fig, ax = style.figure(
        "How close are winner calls to Vegas?",
        (f"Each cell is the extra number of correct winners on {evidence['paired_games']} paired games; "
         "brass marks a tie and blue marks a model ahead."), size=(27, 10.5))
    fig.subplots_adjust(left=.14, right=.72, top=.79, bottom=.16)
    ax.imshow(bands.to_numpy(dtype=int), cmap=ListedColormap(COLORS),
              vmin=-.5, vmax=5.5, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(FINGERPRINTS)),
                  ["F17\nmarket" if value == "F17-market" else value for value in FINGERPRINTS])
    ax.set_yticks(np.arange(len(style.MODELS)),
                  [f"{model} · {style.NAMES[model]}" for model in style.MODELS])
    ax.set_xticks(np.arange(-.5, len(FINGERPRINTS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(style.MODELS), 1), minor=True)
    ax.grid(which="minor", color=style.PAPER, linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set(xlabel="Fingerprint generation", ylabel="Model architecture")
    values = table.to_numpy(dtype=int)
    colors = bands.to_numpy(dtype=int)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            difference = int(values[row, column])
            band = int(colors[row, column])
            label = f"+{difference}" if difference > 0 else str(difference).replace("-", "−")
            ax.text(column, row, label, ha="center", va="center",
                    color="white" if band == 0 else style.INK,
                    fontsize=10 if band >= 4 else 8.6,
                    weight="bold" if band >= 3 else "normal", zorder=5)
            if band >= 4:
                ax.add_patch(Rectangle((column - .48, row - .47), .96, .94,
                                       fill=False, edgecolor=style.INK,
                                       linewidth=2.7, zorder=6))
    order = (5, 4, 3, 2, 1, 0)
    handles = [Patch(facecolor=COLORS[band], edgecolor=style.INK if band >= 4 else style.PAPER,
                     linewidth=1.5, label=_legend_label(cells, band)) for band in order]
    legend = ax.legend(handles=handles, title=("Extra correct winners vs Vegas\n"
                        f"Vegas: {evidence['vegas_correct_winners']}/{evidence['paired_games']}"),
                       loc="upper left", bbox_to_anchor=(1.025, 1), frameon=False,
                       fontsize=11, title_fontsize=12, labelspacing=1.1,
                       handlelength=1.5, handleheight=1.2)
    legend.set_alignment("left")
    style.save_pair(fig, STEM)
    style.publicationize(fig, STEM)
    legend.get_title().set_fontsize(16)
    for item in ax.texts:
        item.set_fontsize(11 if item.get_fontweight() == "bold" else 9.5)
    style.save_pair(fig, STEM + "_publication", dpi=300)
    plt.close(fig)


def main() -> None:
    style.configure(style.argument_parser().parse_args())
    plt.style.use(style.ROOT / "docs/style/color_palettes/gridiron_light.mplstyle")
    plt.rcParams.update({"figure.autolayout": False,
                         "figure.facecolor": style.PAPER, "savefig.facecolor": style.PAPER,
                         "axes.facecolor": style.PANEL, "font.family": "DejaVu Sans",
                         "svg.hashsalt": "tdnet-2026-scientific-quality"})
    style.OUTPUT.mkdir(parents=True, exist_ok=True)
    ledger_path = style.OUTPUT / LEDGER
    market_path = style.SOURCE / "common_cohort_vegas_games.csv"
    model_path = style.SOURCE / "common_120_model_game_results.parquet"
    cells, evidence = scored_cells(ledger_path, market_path, model_path)
    style.set_cohort(evidence["spread_games"], evidence["paired_games"], evidence["max_week"])
    plot(cells, evidence)
    cells.sort_values(["fingerprint", "model_id"]).to_csv(style.OUTPUT / SCORE, index=False)
    outputs = [style.OUTPUT / f"{STEM}{suffix}.{extension}"
               for suffix in ("", "_publication") for extension in ("png", "svg")]
    outputs.append(style.OUTPUT / SCORE)
    manifest = {"source": {str(ledger_path): style.sha256(ledger_path),
                           str(market_path): style.sha256(market_path),
                           str(model_path): style.sha256(model_path),
                           str(Path(__file__)): style.sha256(Path(__file__))},
                "render_date": style.RENDER_DATE, "market_note": style.MARKET_NOTE,
                "evidence": evidence,
                "color_rule": {GROUPS[index]: COLORS[index] for index in range(6)},
                "outputs": {path.name: style.sha256(path) for path in outputs}}
    (style.OUTPUT / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stem": STEM, **evidence, "output": str(style.OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
