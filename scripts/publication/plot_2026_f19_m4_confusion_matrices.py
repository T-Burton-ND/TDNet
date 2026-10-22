"""Rebuild F19/M4 winner-pick and paired-Vegas correctness matrices.

Uses the exact spread-and-moneyline intersection behind the scientific
winner-parity heatmap. The archived straight-up grading uses model home-win
probability and the market home spread for its two picks, respectively.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import plot_2026_scientific_total_quality as style


STEM = "f19_m4_winner_confusion_matrices"
GAME_TABLE = "f19_m4_paired_winner_picks.csv"
MATRIX_TABLE = "f19_m4_winner_confusion_counts.csv"
MANIFEST = "f19_m4_winner_confusion_manifest.json"


def build_tables(model_path: Path, market_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model = pd.read_parquet(model_path)
    market = pd.read_csv(market_path)
    if market.empty or market.game_id.duplicated().any():
        raise ValueError("Market export needs one row per game")
    model = model.loc[model.fingerprint.eq("F19") & model.model_id.eq("M4")]
    market = market.dropna(subset=["market_home_spread", "market_home_probability"])
    if (market.empty or model.duplicated("game_id").any() or
            market.market_home_spread.eq(0).any() or
            market.market_home_probability.lt(0).any() or
            market.market_home_probability.gt(1).any()):
        raise ValueError("Invalid paired market/model export")
    joined = market.merge(model, on=["game_id", "week"], how="left",
                          validate="one_to_one", suffixes=("_vegas", "_model"))
    if (len(joined) != len(market) or joined.pred_home_win_probability.isna().any() or
            joined.pred_home_win_probability.lt(0).any() or
            joined.pred_home_win_probability.gt(1).any() or
            joined.actual_home_margin_model.isna().any() or
            joined.winner_correct_model.isna().any() or
            joined.winner_correct_vegas.isna().any() or
            not np.allclose(joined.actual_home_margin_model,
                            joined.actual_home_margin_vegas)):
        raise ValueError("F19/M4 predictions or final scores differ from the paired cohort")
    actual_home = joined.actual_home_margin_vegas.gt(0)
    if joined.actual_home_margin_vegas.eq(0).any():
        raise ValueError("Tied final scores need a separate winner rule")
    model_home = joined.pred_home_win_probability.ge(.5)
    vegas_home = joined.market_home_spread.lt(0)
    model_correct = model_home.eq(actual_home)
    vegas_correct = vegas_home.eq(actual_home)
    if (not model_correct.eq(joined.winner_correct_model.astype(bool)).all() or
            not vegas_correct.eq(joined.winner_correct_vegas.astype(bool)).all()):
        raise ValueError("Derived picks disagree with archived straight-up grading")
    games = pd.DataFrame({
        "week": joined.week.astype(int), "game_id": joined.game_id,
        "home_team": joined.home_team, "away_team": joined.away_team,
        "actual_home_margin": joined.actual_home_margin_vegas,
        "actual_winner": np.where(actual_home, "home", "away"),
        "actual_winner_team": np.where(actual_home, joined.home_team, joined.away_team),
        "model_home_win_probability": joined.pred_home_win_probability,
        "model_pick": np.where(model_home, "home", "away"),
        "model_pick_team": np.where(model_home, joined.home_team, joined.away_team),
        "market_home_spread": joined.market_home_spread,
        "market_home_win_probability": joined.market_home_probability,
        "vegas_pick": np.where(vegas_home, "home", "away"),
        "vegas_pick_team": np.where(vegas_home, joined.home_team, joined.away_team),
        "model_correct": model_correct, "vegas_correct": vegas_correct,
        "market_provenance": joined.market_provenance,
        "model_provenance": joined.provenance_class,
    }).sort_values(["week", "game_id"]).reset_index(drop=True)
    records = []
    for outcome in ("home", "away"):
        for pick in ("home", "away"):
            records.append({"matrix": "model_pick_vs_actual_winner", "row": outcome,
                            "column": pick, "games": int((games.actual_winner.eq(outcome) &
                                                         games.model_pick.eq(pick)).sum())})
    for vegas in (True, False):
        for model_result in (True, False):
            records.append({"matrix": "vegas_correct_vs_model_correct", "row": str(vegas),
                            "column": str(model_result),
                            "games": int((games.vegas_correct.eq(vegas) &
                                          games.model_correct.eq(model_result)).sum())})
    counts = pd.DataFrame(records)
    n = len(games)
    if any(group.games.sum() != n for _, group in counts.groupby("matrix")):
        raise ValueError("Confusion-matrix counts do not sum to paired games")
    evidence = {"paired_games": n, "max_week": int(games.week.max()),
                "model_correct": int(model_correct.sum()),
                "vegas_correct": int(vegas_correct.sum()),
                "model_only_correct": int((model_correct & ~vegas_correct).sum()),
                "vegas_only_correct": int((~model_correct & vegas_correct).sum()),
                "same_pick": int(model_home.eq(vegas_home).sum())}
    return games, counts, evidence


def draw_matrix(ax, values, rows, columns, colors, *, xlabel, ylabel, fontsize):
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_aspect("equal")
    ax.set_facecolor(style.PAPER)
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((c + .025, r + .025), .95, .95,
                                   facecolor=colors[r][c], edgecolor=style.PAPER,
                                   linewidth=4))
            ax.text(c + .5, r + .5, str(values[r][c]), ha="center", va="center",
                    fontsize=fontsize + 8, weight="bold", color=style.INK)
    ax.set_xticks([.5, 1.5], columns)
    ax.set_yticks([.5, 1.5], rows)
    ax.tick_params(length=0, labelsize=fontsize, pad=9)
    ax.set_xlabel(xlabel, fontsize=fontsize + 1, labelpad=15, weight="bold")
    ax.set_ylabel(ylabel, fontsize=fontsize + 1, labelpad=16, weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot(counts: pd.DataFrame, evidence: dict, *, publication: bool) -> None:
    first = counts.loc[counts.matrix.eq("model_pick_vs_actual_winner")]
    second = counts.loc[counts.matrix.eq("vegas_correct_vs_model_correct")]
    a = first.pivot(index="row", columns="column", values="games").loc[
        ["home", "away"], ["home", "away"]].to_numpy(int)
    b = second.pivot(index="row", columns="column", values="games").loc[
        ["True", "False"], ["True", "False"]].to_numpy(int)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.4))
    fig.patch.set_facecolor(style.PAPER)
    fig.subplots_adjust(left=.10, right=.96, wspace=.34,
                        top=.78 if not publication else .96,
                        bottom=.22 if not publication else .12)
    draw_matrix(axes[0], a, ["Home won", "Away won"],
                ["Pick home", "Pick away"],
                [["#A9CEDA", "#E6BAAD"], ["#E6BAAD", "#A9CEDA"]],
                xlabel="F19/M4 pick", ylabel="Actual winner",
                fontsize=15 if publication else 13)
    draw_matrix(axes[1], b, ["Vegas correct", "Vegas wrong"],
                ["Model correct", "Model wrong"],
                [["#D4E1E2", "#E5C77F"], ["#91BED1", "#E6BAAD"]],
                xlabel="F19/M4 result", ylabel="Vegas result",
                fontsize=15 if publication else 13)
    if not publication:
        for ax, label in zip(axes, ("A  ·  F19/M4 pick versus winner",
                                    "B  ·  F19/M4 versus Vegas")):
            ax.set_title(label, loc="left", color=style.INK,
                         fontsize=15, weight="bold", pad=23)
    if not publication:
        fig.text(.09, .96, "TDNET  /  2026 SCIENTIFIC ROSTER", color=style.MUTED,
                 size=10, weight="bold")
        fig.text(.09, .91, "Where does F19/M4 gain two winners?", color=style.INK,
                 size=23, weight="bold")
        fig.text(.09, .86,
                 (f"{evidence['paired_games']} paired games through Week {evidence['max_week']}  ·  "
                  f"F19/M4 {evidence['model_correct']} correct  ·  "
                  f"Vegas {evidence['vegas_correct']} correct  ·  "
                  f"{evidence['same_pick']} matching picks"),
                 color=style.MUTED, size=11)
        fig.text(.09, .055, ("Model pick: home-win probability ≥ 50%. "
                  "Vegas pick: negative home spread. " + style.MARKET_NOTE),
                 color=style.MUTED, size=9)
    style.save_pair(fig, STEM + ("_publication" if publication else ""),
                    dpi=300 if publication else 200)
    plt.close(fig)


def main() -> None:
    style.configure(style.argument_parser().parse_args())
    plt.style.use(style.ROOT / "docs/style/color_palettes/gridiron_light.mplstyle")
    plt.rcParams.update({"figure.autolayout": False,
                         "figure.facecolor": style.PAPER,
                         "savefig.facecolor": style.PAPER,
                         "font.family": "DejaVu Sans",
                         "svg.hashsalt": "tdnet-2026-scientific-quality"})
    style.OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = style.SOURCE / "common_120_model_game_results.parquet"
    market_path = style.SOURCE / "common_cohort_vegas_games.csv"
    games, counts, evidence = build_tables(model_path, market_path)
    style.set_cohort(int(pd.read_csv(market_path).market_home_spread.notna().sum()),
                     evidence["paired_games"], evidence["max_week"])
    games.to_csv(style.OUTPUT / GAME_TABLE, index=False)
    counts.to_csv(style.OUTPUT / MATRIX_TABLE, index=False)
    plot(counts, evidence, publication=False)
    plot(counts, evidence, publication=True)
    outputs = [style.OUTPUT / name for name in (GAME_TABLE, MATRIX_TABLE)]
    outputs += [style.OUTPUT / f"{STEM}{suffix}.{ext}"
                for suffix in ("", "_publication") for ext in ("png", "svg")]
    manifest = {"source": {str(path): style.sha256(path) for path in
                           (model_path, market_path, Path(__file__))},
                "render_date": style.RENDER_DATE,
                "market_note": style.MARKET_NOTE,
                "pick_rules": {"model": "pred_home_win_probability >= 0.5",
                               "vegas": "market_home_spread < 0"},
                "evidence": evidence,
                "outputs": {path.name: style.sha256(path) for path in outputs}}
    (style.OUTPUT / MANIFEST).write_text(json.dumps(manifest, indent=2,
                                                 sort_keys=True) + "\n")
    print(json.dumps({"stem": STEM, **evidence,
                      "output": str(style.OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
