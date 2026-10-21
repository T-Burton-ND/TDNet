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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    card.to_csv(OUTPUT / "week07_120_model_consensus_scorecard.csv", index=False)

    columns = ["matchup", "prediction", "pick_confidence", "final", "su", "margin_error_points"]
    display = card[columns].rename(columns={
        "matchup": "Matchup", "prediction": "120-model pick", "pick_confidence": "Pick p",
        "final": "Final", "su": "SU", "margin_error_points": "MAE pts",
    })
    wins = int(card.winner_correct.sum())
    fig, ax = plt.subplots(figsize=(19, 27))
    fig.patch.set_facecolor("#f7f4ee")
    ax.axis("off")
    table = ax.table(
        cellText=display.values, colLabels=display.columns, cellLoc="left", colLoc="left",
        colWidths=[0.23, 0.22, 0.07, 0.32, 0.055, 0.105], bbox=[0.0, 0.035, 1.0, 0.91],
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
            if col == 4:
                correct = bool(card.iloc[row - 1].winner_correct)
                cell.set_facecolor("#ccebd4" if correct else "#f4d5d0")
                cell.set_text_props(weight="bold", ha="center", color="#183321" if correct else "#702820")
    ax.set_title(
        "2026 Week 7 | F0–F19 scientific consensus (120 models)\n"
        f"SU {wins}–{len(card)-wins} ({wins/len(card):.1%})   •   "
        f"Margin MAE {card.margin_error.mean():.2f} pts   •   Brier {card.brier_error.mean():.4f}",
        fontsize=18, weight="bold", color="#14294f", pad=22,
    )
    fig.text(
        0.5, 0.012,
        "Retrospective scoring after the games; not a timely pregame forecast. "
        "Week 7 F7/F17/F19 use ex-post market inputs. Notre Dame at BYU is highlighted.",
        ha="center", fontsize=10, color="#555b63",
    )
    path = OUTPUT / "week07_120_model_consensus_scorecard.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"{path}: {len(card)} games, {wins} correct, MAE {card.margin_error.mean():.4f}, Brier {card.brier_error.mean():.4f}")


if __name__ == "__main__":
    main()
