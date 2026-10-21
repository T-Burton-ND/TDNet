#!/usr/bin/env python3
"""Render the scored Week 7 F0–F19 equal-vote consensus game card."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "publication/2026/week_07/post_game/retrospective_replay"
SCORED = REPLAY / "cumulative/scientific_f0_f19/common_120_consensus_game_results.csv"
GAMES = REPLAY / "week07_replay_consensus_games.csv"
OUTPUT = REPLAY / "scientific/full_f0_f19"
MARKET = OUTPUT / "f17_f19_market_snapshot.parquet"


def main() -> None:
    scored = pd.read_csv(SCORED).query("week == 7").copy()
    games = pd.read_csv(GAMES)[
        ["game_id", "game_start_time_utc", "home_team", "away_team", "home_points", "away_points"]
    ]
    card = scored.merge(games, on="game_id", validate="one_to_one")
    card = card.sort_values(["game_start_time_utc", "game_id"]).reset_index(drop=True)
    if len(card) != 59 or card.model_votes.ne(120).any():
        raise ValueError("Expected 59 Week 7 games with 120 votes each")
    if card.game_id.eq(401856809).sum() != 1:
        raise ValueError("Notre Dame at BYU is missing")
    if not (card.actual_home_margin == card.home_points - card.away_points).all():
        raise ValueError("Stored outcomes and game results disagree")
    lines = pd.read_parquet(MARKET)[
        ["target_game_id", "market_home_spread", "market_provenance"]
    ].rename(columns={"target_game_id": "game_id"})
    lines = lines.loc[lines.market_provenance.eq("ex_post_oct21_cache_unverified_quote_time")]
    card = card.merge(lines, on="game_id", validate="one_to_one")
    if len(card) != 59 or card.market_home_spread.isna().any():
        raise ValueError("The 59 Week 7 retrospective market spreads are incomplete")

    card["predicted_winner"] = card.home_team.where(card.pred_home_margin.ge(0), card.away_team)
    card["predicted_winner_probability"] = card.pred_home_win_probability.where(
        card.pred_home_margin.ge(0), 1 - card.pred_home_win_probability
    )
    card["actual_winner"] = card.home_team.where(card.actual_home_margin.gt(0), card.away_team)
    card["matchup"] = card.away_team + " at " + card.home_team
    card["prediction"] = card.predicted_winner + " by " + card.pred_home_margin.abs().map("{:.1f}".format)
    card["final"] = (
        card.away_team + " " + card.away_points.astype(int).astype(str) + " – "
        + card.home_points.astype(int).astype(str) + " " + card.home_team
    )
    card["su"] = card.winner_correct.map({True: "W", False: "L"})
    card["margin_error_points"] = card.margin_error.map("{:.1f}".format)
    card["pick_confidence"] = card.predicted_winner_probability.map("{:.0%}".format)
    model_cover_margin = card.pred_home_margin + card.market_home_spread
    actual_cover_margin = card.actual_home_margin + card.market_home_spread
    card["ats_side"] = card.home_team.where(model_cover_margin.gt(0), card.away_team)
    card.loc[model_cover_margin.eq(0), "ats_side"] = "—"
    card["ats_result"] = "—"
    decided = model_cover_margin.ne(0) & actual_cover_margin.ne(0)
    card.loc[decided, "ats_result"] = (
        model_cover_margin.loc[decided].gt(0).eq(actual_cover_margin.loc[decided].gt(0))
        .map({True: "W", False: "L"})
    )
    card.loc[model_cover_margin.ne(0) & actual_cover_margin.eq(0), "ats_result"] = "P"
    card["market_line"] = card.home_team + " " + card.market_home_spread.map("{:+.1f}".format)
    card.loc[card.market_home_spread.eq(0), "market_line"] = "Pick'em"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    card.to_csv(OUTPUT / "week07_120_model_consensus_scorecard.csv", index=False)

    columns = ["matchup", "prediction", "pick_confidence", "final", "su", "market_line",
               "ats_side", "ats_result", "margin_error_points"]
    display = card[columns].rename(columns={
        "matchup": "Matchup", "prediction": "120-model pick", "pick_confidence": "Pick p",
        "final": "Final", "su": "SU", "market_line": "Home line", "ats_side": "ATS side",
        "ats_result": "ATS", "margin_error_points": "MAE pts",
    })
    wins = int(card.winner_correct.sum())
    ats_wins = int(card.ats_result.eq("W").sum())
    ats_losses = int(card.ats_result.eq("L").sum())
    ats_pushes = int(card.ats_result.eq("P").sum())
    fig, ax = plt.subplots(figsize=(23, 23))
    fig.patch.set_facecolor("#f7f4ee")
    ax.axis("off")
    ax.set_position([.025, .03, .95, .9])
    table = ax.table(
        cellText=display.values, colLabels=display.columns, cellLoc="left", colLoc="left",
        colWidths=[0.19, 0.18, 0.06, 0.265, 0.04, 0.105, 0.095, 0.04, 0.065],
        bbox=[0.0, 0.01, 1.0, 0.98],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d4d7db")
        if row == 0:
            cell.set_facecolor("#14294f")
            cell.set_text_props(color="white", weight="bold")
        else:
            highlighted = card.iloc[row - 1].game_id == 401856809
            cell.set_facecolor("#fff0c7" if highlighted else ("#ffffff" if row % 2 else "#edf1f4"))
            if col in (4, 7):
                result = card.iloc[row - 1].su if col == 4 else card.iloc[row - 1].ats_result
                cell.set_facecolor({"W": "#ccebd4", "L": "#f4d5d0", "P": "#fff0c7", "—": "#e8e8e8"}[result])
                cell.set_text_props(weight="bold", ha="center", color="#183321" if result == "W" else "#702820")
    ax.set_title(
        "2026 Week 7 | F0–F19 scientific consensus (120 models)\n"
        f"SU {wins}–{len(card)-wins} ({wins/len(card):.1%})   •   "
        f"ATS {ats_wins}–{ats_losses}–{ats_pushes} ({ats_wins/(ats_wins+ats_losses):.1%}, pushes excluded)   •   "
        f"Margin MAE {card.margin_error.mean():.2f} pts   •   Brier {card.brier_error.mean():.4f}",
        fontsize=18, weight="bold", color="#14294f", pad=22,
    )
    fig.text(
        0.5, 0.012,
        "Retrospective scoring after the games; not a timely pregame forecast. "
        "ATS uses October 21 after-game spreads with unverified quote times; W/L is not a backtest of pregame bets. "
        "Notre Dame at BYU is highlighted.",
        ha="center", fontsize=10, color="#555b63",
    )
    path = OUTPUT / "week07_120_model_consensus_scorecard.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"{path}: {len(card)} games, SU {wins}-{len(card)-wins}, ATS {ats_wins}-{ats_losses}-{ats_pushes}, "
          f"MAE {card.margin_error.mean():.4f}, Brier {card.brier_error.mean():.4f}")


if __name__ == "__main__":
    main()
