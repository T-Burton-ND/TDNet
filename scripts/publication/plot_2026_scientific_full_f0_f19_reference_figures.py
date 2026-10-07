#!/usr/bin/env python3
"""Rebuild the two requested 2026 scientific figures for all 120 cells."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "publication/2026/week_05/post_game/scientific/current_season_f0_f19"
MODELS = ("M1", "M2", "M3", "M4", "M5", "M10")
STAGES = (*[f"F{i}" for i in range(17)], "F17-market", "F18", "F19")
FAMILIES = {"M1": "Linear", "M2": "Spline", "M3": "Tree", "M4": "Boosted", "M5": "Neural", "M10": "KNN"}
STYLES = {"M1": "-", "M2": "--", "M3": ":", "M4": "-.", "M5": (0, (5, 1.5, 1, 1.5)), "M10": (0, (1, 1))}
NAVY = "#15294C"
PAPER = "#F7F3E9"
PANEL = "#FFFCF6"
PINK = "#D1478D"
PLUM = "#913F78"
GOLD = "#AA853A"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _verify_source(manifest: dict, name: str) -> Path:
    source = ROOT / manifest["outputs"][name]["path"]
    if digest(source) != manifest["outputs"][name]["sha256"]:
        raise ValueError(f"Source differs from F0–F19 manifest: {name}")
    return source


def _curve_figure(models: pd.DataFrame, consensus: pd.DataFrame, path: Path) -> None:
    if len(models) != 120 * 5 or models.groupby("series_id").week.nunique().ne(5).any():
        raise ValueError("Expected five cumulative points for every one of 120 models")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "figure.facecolor": PAPER, "axes.facecolor": PANEL,
                         "axes.spines.top": False, "axes.spines.right": False})
    cmap = LinearSegmentedColormap.from_list("fingerprints", ["#EA5A9B", "#8753AE", "#2272A7", "#0F638B"])
    fig, axes = plt.subplots(1, 3, figsize=(23, 9), facecolor=PAPER)
    specs = (("brier_score", "Brier score", False),
             ("winner_accuracy", "Straight-up winner accuracy", True),
             ("ats_accuracy", "Against-the-spread accuracy", True))
    for ax, (metric, title, percent) in zip(axes, specs, strict=True):
        for (stage, model_id), group in models.groupby(["fingerprint", "model_id"], sort=False):
            ax.plot(group.week, group[metric], color=cmap(STAGES.index(stage) / 19),
                    linestyle=STYLES[model_id], linewidth=1.15, alpha=.63,
                    marker="o", markersize=2.6, zorder=2)
        lines = (
            ("full_available_120", PINK, "-", "Full F0–F19 available-vote consensus", "o"),
            ("full_common_120", PLUM, "-", "Full F0–F19 common-game consensus", "D"),
            ("archived_pregame_market", GOLD, "--", "Archived pregame market", "s"),
        )
        for series, color, style, label, marker in lines:
            group = consensus.loc[consensus.series_id.eq(series)].sort_values("week")
            if series == "archived_pregame_market" and metric == "ats_accuracy":
                continue  # The market spread has no directional ATS wager.
            if len(group) != 5:
                raise ValueError(f"Missing five cumulative points for {series}")
            ax.plot(group.week, group[metric], color=color, linestyle=style,
                    linewidth=3, marker=marker, markersize=6, label=label, zorder=6)
        ax.set_title(title, loc="left", color=NAVY, fontweight="bold", pad=14)
        ax.set_xticks(range(1, 6))
        ax.set_xlim(.75, 5.25)
        ax.set_xlabel("Completed 2026 week")
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.16)
    fig.suptitle("TDNet 2026 scientific roster · cumulative season performance",
                 color=NAVY, fontsize=22, fontweight="bold", y=.97)
    fig.text(.5, .914, "120 F0–F19 model × fingerprint curves · Weeks 1–5 · M1, M2, M3, M4, M5, M10",
             ha="center", color="#465266", fontsize=12)
    arch = [Line2D([0], [0], color="#3D4A5D", lw=1.7, linestyle=STYLES[mid],
                   label=f"{mid} · {FAMILIES[mid]}") for mid in MODELS]
    comparators = [Line2D([0], [0], color=color, lw=3, linestyle=style, marker=marker, label=label)
                   for _, color, style, label, marker in lines]
    fig.legend(handles=arch + comparators, loc="lower center", bbox_to_anchor=(.5, .075),
               ncol=5, frameon=False, fontsize=10, columnspacing=1.3)
    scalar = plt.cm.ScalarMappable(norm=Normalize(0, 19), cmap=cmap)
    scalar.set_array([])
    color_axis = fig.add_axes([.16, .215, .68, .025])
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_ticks([0, 4, 8, 12, 16, 19])
    colorbar.set_ticklabels(["F0", "F4", "F8", "F12", "F16", "F19"])
    colorbar.set_label("Fingerprint generation (pink → blue)")
    fig.text(.5, .025,
             "F19 has 263 archived-market games; other model cells have 271. The available-vote consensus has 114 votes on eight F19-missing games. "
             "Market Brier uses 259 probabilities; a market ATS accuracy is undefined.",
             ha="center", color="#465266", fontsize=9.5)
    fig.subplots_adjust(left=.045, right=.985, top=.82, bottom=.325, wspace=.22)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def _record(group: pd.DataFrame, column: str) -> str:
    counts = group.ats_result.value_counts() if column == "ats" else None
    if column == "ats":
        return f"{int(counts.get('win', 0))}–{int(counts.get('loss', 0))}–{int(counts.get('push', 0))}"
    wins = int(group.winner_correct.fillna(False).astype(bool).sum())
    return f"{wins}–{int(group.winner_correct.notna().sum()) - wins}"


def _scorecard(models: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    eligible = set(models.loc[models.fingerprint.eq("F19") & models.forecast_status.eq("available"), "game_id"])
    if len(eligible) != 263:
        raise ValueError("Expected 263 F19-eligible games")
    rows: list[dict] = []
    for series, group in models.groupby("series_id", sort=False):
        common = group.loc[group.game_id.isin(eligible)]
        valid = common.loc[common.forecast_status.eq("available")]
        if len(group) != 271 or len(valid) != 263:
            raise ValueError(f"Incomplete common 263-game model cohort: {series}")
        available = group.loc[group.forecast_status.eq("available")]
        ats = valid.ats_result.value_counts()
        rows.append({"row_type": "model", "series_id": series,
                     "fingerprint": str(group.fingerprint.iloc[0]),
                     "model_id": str(group.model_id.iloc[0]),
                     "available_games": len(available), "common_games": len(valid),
                     "margin_mae_common": valid.absolute_margin_error.mean(),
                     "margin_mae_available": available.absolute_margin_error.mean(),
                     "su_record": _record(valid, "su"),
                     "winner_accuracy_common": valid.winner_correct.mean(),
                     "brier_score_common": valid.brier_error.mean(),
                     "brier_games_common": valid.brier_error.notna().sum(),
                     "ats_record": _record(valid, "ats"),
                     "ats_accuracy_common": ats.get("win", 0) / max(1, ats.get("win", 0) + ats.get("loss", 0))})
    table = pd.DataFrame(rows).sort_values(["margin_mae_common", "fingerprint", "model_id"]).reset_index(drop=True)
    table.insert(0, "rank_common_263", np.arange(1, 121))
    if len(table) != 120 or table.series_id.duplicated().any():
        raise ValueError("Expected 120 ranked cells")
    for series, label in (("full_available_120", "Full available-vote consensus"),
                          ("full_common_120", "Full common-game consensus"),
                          ("fingerprint_F18", "F18 equal consensus"),
                          ("fingerprint_F19", "F19 equal consensus")):
        group = consensus.loc[consensus.series_id.eq(series)]
        valid = group.loc[group.game_id.isin(eligible)]
        if len(valid) != 263:
            raise ValueError(f"Incomplete consensus common cohort: {series}")
        ats = valid.ats_result.value_counts()
        row = {"rank_common_263": np.nan, "row_type": "consensus", "series_id": label,
               "fingerprint": "F0–F19" if series.startswith("full") else series[-3:],
               "model_id": "Equal", "available_games": len(group), "common_games": 263,
               "margin_mae_common": valid.absolute_margin_error.mean(),
               "margin_mae_available": group.absolute_margin_error.mean(),
               "su_record": _record(valid, "su"), "winner_accuracy_common": valid.winner_correct.mean(),
               "brier_score_common": valid.brier_error.mean(), "brier_games_common": valid.brier_error.notna().sum(),
               "ats_record": _record(valid, "ats"),
               "ats_accuracy_common": ats.get("win", 0) / max(1, ats.get("win", 0) + ats.get("loss", 0))}
        table = pd.concat([pd.DataFrame([row]), table], ignore_index=True)
    market = models.drop_duplicates("game_id").loc[lambda frame: frame.game_id.isin(eligible)]
    if len(market) != 263:
        raise ValueError("Market comparator needs 263 eligible games")
    spread = market.market_home_spread.astype(float)
    margin = market.actual_home_margin.astype(float)
    probability = market.market_home_probability.astype(float)
    truth = margin.gt(0)
    prob = probability.notna()
    market_row = {"rank_common_263": np.nan, "row_type": "market", "series_id": "Archived pregame market",
                  "fingerprint": "Market", "model_id": "—", "available_games": 263,
                  "common_games": 263, "margin_mae_common": (margin + spread).abs().mean(),
                  "margin_mae_available": (margin + spread).abs().mean(),
                  "su_record": f"{int((probability[prob].ge(.5) == truth[prob]).sum())}–{int(prob.sum() - (probability[prob].ge(.5) == truth[prob]).sum())}",
                  "winner_accuracy_common": (probability[prob].ge(.5) == truth[prob]).mean(),
                  "brier_score_common": ((probability[prob] - truth[prob].astype(float)) ** 2).mean(),
                  "brier_games_common": int(prob.sum()), "ats_record": "No bet", "ats_accuracy_common": np.nan}
    table = pd.concat([pd.DataFrame([market_row]), table], ignore_index=True)
    if len(table) != 125:
        raise ValueError("Expected 120 models and five comparators")
    return table


def _scorecard_figure(scorecard: pd.DataFrame, path: Path) -> None:
    def number(value: float, digits: int = 2) -> str:
        return "—" if pd.isna(value) else f"{value:.{digits}f}"

    display = pd.DataFrame({
        "Rank": scorecard.rank_common_263.map(lambda value: "—" if pd.isna(value) else str(int(value))),
        "Model / comparator": scorecard.series_id,
        "Tier": scorecard.fingerprint,
        "G avail": scorecard.available_games.astype(int),
        "MAE 263": scorecard.margin_mae_common.map(number),
        "MAE avail": scorecard.margin_mae_available.map(number),
        "SU 263": scorecard.su_record,
        "SU %": scorecard.winner_accuracy_common.map(lambda value: f"{value:.1%}"),
        "Brier 263": scorecard.brier_score_common.map(lambda value: number(value, 3)),
        "ATS 263": scorecard.ats_record,
        "ATS %": scorecard.ats_accuracy_common.map(lambda value: "—" if pd.isna(value) else f"{value:.1%}"),
    })
    fig, ax = plt.subplots(figsize=(22, 50), facecolor=PAPER)
    ax.axis("off")
    ax.set_position([.025, .028, .95, .91])
    table = ax.table(cellText=display.values, colLabels=display.columns,
                     cellLoc="left", colLoc="left", bbox=[0, .01, 1, .98],
                     colWidths=[.043, .245, .074, .057, .07, .072, .075, .065, .077, .115, .07])
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D3D8DC")
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif row <= 5:
            cell.set_facecolor("#FFF0C4" if row == 1 else "#F9DDEB")
            cell.set_text_props(weight="bold")
        else:
            stage = str(scorecard.iloc[row - 1].fingerprint)
            base = "#FAEEE2" if stage == "F19" else "#E9F4F2" if stage == "F18" else "#FFFFFF" if row % 2 else "#EFF2F5"
            cell.set_facecolor(base)
    ax.set_title("TDNet 2026 Scientific Models · F0–F19 cumulative through Week 5",
                 color=NAVY, fontsize=22, fontweight="bold", pad=22)
    fig.text(.5, .008,
             "All model ranks and 263 columns use the same 263 market-eligible games. Available MAE uses 271 games except F19 (263). "
             "Market SU/Brier use 259 valid probabilities. ATS excludes pushes, zero spreads and exact predicted cover ties; market makes no ATS pick.",
             ha="center", color="#525B68", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    base_manifest = json.loads((OUTPUT / "manifest.json").read_text())
    model_curve = pd.read_csv(_verify_source(base_manifest, "model_cumulative"))
    consensus_curve = pd.read_csv(_verify_source(base_manifest, "consensus_cumulative"))
    model_games = pd.read_parquet(_verify_source(base_manifest, "model_games_parquet"))
    consensus_games = pd.read_parquet(_verify_source(base_manifest, "consensus_games_parquet"))
    outputs = {
        "full_roster_cumulative_figure": OUTPUT / "scientific_2026_full_f0_f19_cumulative_performance.png",
        "full_roster_common_scorecard": OUTPUT / "scientific_2026_full_f0_f19_common_scorecard.csv",
        "full_roster_scorecard_figure": OUTPUT / "scientific_2026_full_f0_f19_cumulative_model_scorecard.png",
    }
    _curve_figure(model_curve, consensus_curve, outputs["full_roster_cumulative_figure"])
    scorecard = _scorecard(model_games, consensus_games)
    scorecard.to_csv(outputs["full_roster_common_scorecard"], index=False, float_format="%.8f")
    _scorecard_figure(scorecard, outputs["full_roster_scorecard_figure"])
    readme = OUTPUT / "reference_figures_README.md"
    readme.write_text(
        "# F0–F19 versions of the requested scientific figures\n\n"
        "`scientific_2026_full_f0_f19_cumulative_performance.png` follows the earlier "
        "three-panel cumulative Brier, straight-up winner accuracy, and ATS accuracy "
        "figure, now with all 120 scientific model × fingerprint curves. The two full "
        "consensuses and archived pregame market are highlighted. F19 covers 263 games; "
        "the other model cells cover 271.\n\n"
        "`scientific_2026_full_f0_f19_cumulative_model_scorecard.png` follows the "
        "published scientific scorecard table style for all 120 cells. Its model ranks, "
        "MAE 263, winner, Brier, and ATS columns use the same 263 market-eligible games. "
        "MAE avail shows each model's available 271 or 263 games. The unrounded "
        "`scientific_2026_full_f0_f19_common_scorecard.csv` accompanies the figure. "
        "The published operational scorecard remains separate.\n\n"
        "The `provenance/` subdirectory retains seven hash-checked earlier-roster audit "
        "files after the F0–F17-only output directories were retired.\n"
    )
    manifest = {
        "scope": "Two F0-F19 reproductions of the requested 2026 scientific figure styles",
        "season": 2026, "through_week": 5, "model_cells": 120,
        "common_comparison_games": 263,
        "source_manifest_sha256": digest(OUTPUT / "manifest.json"),
        "script_sha256": digest(Path(__file__)),
        "readme_sha256": digest(readme),
        "outputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
                    for name, path in outputs.items()},
    }
    (OUTPUT / "reference_figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "model_rows": int(scorecard.row_type.eq("model").sum()),
                      "common_games": 263, "figures": [str(path) for name, path in outputs.items() if name.endswith("figure")]}, indent=2))


if __name__ == "__main__":
    main()
