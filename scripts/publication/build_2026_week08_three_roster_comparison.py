#!/usr/bin/env python3
"""Compare the three Week 8 roster consensuses on F20's 49-game cohort."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.publication.figure_theme import TDNET_COLORS, apply_tdnet_theme


ROOT = Path(__file__).resolve().parents[2]
POST = ROOT / "publication/2026/week_08/post_game"
PRIVATE = ROOT / "data/private/postgame_2026_week08"
OUTPUT = POST / "three_roster_comparison"
ROSTERS = ("Margin-wide", "Scientific F0–F6", "Scientific F0–F20")
COLORS = ("#173D60", "#3C8D93", "#D16D3A")


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite comparison: {OUTPUT}")
    wide_path = POST / "tables/margin_wide_prediction_vs_actual.csv"
    original_path = POST / "scientific/scientific_model_game_results.csv"
    expanded_path = POST / "scientific/expanded_f0_f20/week08_f0_f20_126_model_games_matched.csv"
    odds_path = PRIVATE / "frozen_bundle/private/metadata__cfbd_moneyline_snapshot.csv"
    freeze_path = PRIVATE / "frozen_bundle/public/predictions.parquet"
    wide = pd.read_csv(wide_path)
    original = pd.read_csv(original_path)
    expanded = pd.read_csv(expanded_path)
    matched_ids = set(expanded.game_id)
    if len(matched_ids) != 49 or not matched_ids.issubset(set(wide.game_id)):
        raise ValueError("Expanded roster is not the expected 49-game subset")
    original_consensus = original.groupby("game_id", as_index=False).agg(
        pred_home_margin=("pred_home_margin", "mean"),
        pred_home_win_probability=("pred_home_win_probability", "mean"),
        model_count=("model_name", "nunique"),
    )
    expanded_consensus = expanded.groupby("game_id", as_index=False).agg(
        pred_home_margin=("pred_home_margin", "mean"),
        pred_home_win_probability=("pred_home_win_probability", "mean"),
        model_count=("model_name", "nunique"),
    )
    if not original_consensus.model_count.eq(42).all() or not expanded_consensus.model_count.eq(126).all():
        raise ValueError("One scientific roster lacks model votes")
    freeze = pd.read_parquet(freeze_path)[["game_id", "game_start_time_utc"]].drop_duplicates("game_id")
    odds = pd.read_csv(odds_path)[["game_id", "home_moneyline", "away_moneyline"]]
    games = wide[["game_id", "home_team", "away_team", "home_points", "away_points", "market_spread_close"]]
    games = games.merge(freeze, on="game_id", validate="one_to_one").merge(odds, on="game_id", validate="one_to_one")
    games = games[games.game_id.isin(matched_ids)]
    if len(games) != 49 or games.duplicated("game_id").any():
        raise ValueError("Three-roster comparison does not have 49 unique games")
    rows = []
    sources = [wide[["game_id", "pred_home_margin", "pred_home_win_probability"]],
               original_consensus, expanded_consensus]
    for roster, source in zip(ROSTERS, sources):
        frame = games.merge(source[["game_id", "pred_home_margin", "pred_home_win_probability"]],
                            on="game_id", validate="one_to_one")
        frame["roster"] = roster
        rows.append(frame)
    scored = pd.concat(rows, ignore_index=True)
    scored["actual_home_margin"] = scored.home_points - scored.away_points
    actual_home = scored.actual_home_margin.gt(0)
    scored["su_correct"] = scored.pred_home_win_probability.ge(.5).eq(actual_home)
    scored["brier"] = (scored.pred_home_win_probability - actual_home.astype(int)) ** 2
    scored["margin_absolute_error"] = (scored.pred_home_margin - scored.actual_home_margin).abs()
    pred_cover = scored.pred_home_margin + scored.market_spread_close
    actual_cover = scored.actual_home_margin + scored.market_spread_close
    scored["ats_result"] = np.select(
        [scored.market_spread_close.isna(), actual_cover.eq(0), pred_cover.eq(0),
         pred_cover.gt(0).eq(actual_cover.gt(0))],
        ["No line", "Push", "No pick", "Win"], default="Loss")
    scored["ats_profit_10"] = np.select(
        [scored.ats_result.eq("Win"), scored.ats_result.eq("Loss")],
        [100 / 11, -10.0], default=0.0)
    home_pick = scored.pred_home_win_probability.ge(.5)
    selected_odds = scored.home_moneyline.where(home_pick, scored.away_moneyline)
    win = scored.su_correct
    scored["moneyline_quoted"] = selected_odds.notna()
    profit_win = np.where(selected_odds.gt(0), 10 * selected_odds / 100,
                          1000 / selected_odds.abs())
    scored["moneyline_profit_10"] = np.where(scored.moneyline_quoted,
                                              np.where(win, profit_win, -10), 0.0)
    scored = scored.sort_values(["roster", "game_start_time_utc", "game_id"])
    summary = scored.groupby("roster").agg(
        games=("game_id", "nunique"), su_wins=("su_correct", "sum"),
        su_accuracy=("su_correct", "mean"), brier=("brier", "mean"),
        margin_mae=("margin_absolute_error", "mean"),
        ats_wins=("ats_result", lambda values: int(values.eq("Win").sum())),
        ats_losses=("ats_result", lambda values: int(values.eq("Loss").sum())),
        ats_pushes=("ats_result", lambda values: int(values.eq("Push").sum())),
        ats_profit_10=("ats_profit_10", "sum"),
        moneyline_quoted_games=("moneyline_quoted", "sum"),
        moneyline_profit_10=("moneyline_profit_10", "sum"),
    ).reset_index()
    summary["ats_accuracy"] = summary.ats_wins / (summary.ats_wins + summary.ats_losses)
    summary["roster"] = pd.Categorical(summary.roster, categories=ROSTERS, ordered=True)
    summary = summary.sort_values("roster")
    OUTPUT.mkdir(parents=True)
    scored.to_csv(OUTPUT / "matched_49_game_roster_consensus_results.csv", index=False)
    summary.to_csv(OUTPUT / "matched_49_game_roster_scorecard.csv", index=False)
    apply_tdnet_theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, metric, title, format_value in zip(
        axes.flat,
        ["su_accuracy", "ats_accuracy", "brier", "margin_mae"],
        ["Straight-up accuracy", "ATS win rate", "Brier score · lower is better", "Margin MAE · lower is better"],
        [lambda value: f"{value:.1%}", lambda value: f"{value:.1%}",
         lambda value: f"{value:.3f}", lambda value: f"{value:.2f}"]):
        vals = summary[metric].to_numpy()
        bars = ax.bar(np.arange(3), vals, color=COLORS, width=.64)
        ax.set_xticks(np.arange(3), ROSTERS, rotation=12)
        ax.set_title(title, fontsize=15)
        ax.bar_label(bars, labels=[format_value(value) for value in vals], padding=4, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("2026 Week 8 · three roster consensuses · same 49 games", fontsize=20)
    fig.savefig(OUTPUT / "three_roster_matched_scorecard.png", dpi=210)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
    for roster, color in zip(ROSTERS, COLORS):
        part = scored[scored.roster.eq(roster)].sort_values(["game_start_time_utc", "game_id"])
        for ax, column in zip(axes, ["ats_profit_10", "moneyline_profit_10"]):
            ax.plot(np.arange(1, len(part) + 1), part[column].cumsum(),
                    color=color, lw=2.5, label=roster)
    for ax, title in zip(axes, ["Flat $10 ATS · -110 assumed", "Flat $10 moneyline · saved pregame quote"]):
        ax.axhline(0, color=TDNET_COLORS["slate"], lw=1)
        ax.set(xlabel="Completed game in kickoff order", ylabel="Cumulative profit ($)", title=title)
        ax.grid(alpha=.25)
    axes[0].legend(frameon=False)
    fig.suptitle("2026 Week 8 · matched-cohort consensus betting diagnostics", fontsize=19)
    fig.savefig(OUTPUT / "three_roster_matched_bankroll.png", dpi=210)
    plt.close(fig)
    manifest = {"schema": "tdnet-week08-three-roster-matched-comparison-v1",
                "games": 49, "rosters": list(ROSTERS),
                "ats_pricing": "flat $10 stake at assumed -110; push returns zero",
                "moneyline_pricing": "flat $10 at saved pregame quote; unquoted games excluded",
                "f20_margin_status": "postgame application of pregame-frozen margin model to archived pregame features",
                "sources": {str(p): sha256_file(p) for p in [wide_path, original_path, expanded_path, odds_path, freeze_path]},
                "files": {p.name: sha256_file(p) for p in OUTPUT.iterdir() if p.is_file()}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUTPUT / "README.md").write_text(
        "# Week 8 three-roster comparison\n\n"
        "All three consensus scorecards use the same 49 games with F20 forecasts. "
        "The margin-wide and F0–F6 rosters also cover all 53 frozen Week 8 games "
        "in their primary scorecards. The expanded F0–F20 roster has 126 cells; "
        "F20 probabilities were issued as late pregame research forecasts, while "
        "F20 margins and ATS are postgame applications of pregame-frozen models "
        "to archived pregame expert inputs. ATS assumes -110 and the moneyline "
        "uses saved pregame quotes. These bankroll curves are descriptive.\n",
        encoding="utf-8")
    print(summary.to_json(orient="records"))


if __name__ == "__main__":
    main()
