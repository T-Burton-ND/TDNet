#!/usr/bin/env python3
"""Evaluate an equal-weight, retrospectively selected five-cell 2026 consensus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from build_2026_week08_scientific_full_scorecards import (
    MODEL_COLUMNS, _summarize, _vegas, load_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
WEEK7_F20 = ROOT / "data/private/f20/week07_backfill_20261025/week07_f20_59_game_model_results.parquet"
WEEK7_MANIFEST = ROOT / "publication/2026/week_07/post_game/retrospective_replay/scientific/expanded_f0_f20/manifest.json"
OUTPUT = ROOT / "publication/2026/research/targeted_five_consensus_week01_08"
CELLS = (("F20", "M1"), ("F7", "M1"), ("F19", "M1"), ("F19", "M2"), ("F19", "M10"))
DISPLAY = {"targeted_consensus": "Targeted five", "f19_consensus": "F19 consensus",
           "vegas_reference": "Vegas"}
COLORS = {"targeted_consensus": "#b94172", "f19_consensus": "#207a65",
          "vegas_reference": "#d4a928"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forecast(frame: pd.DataFrame, game_ids: set[int], expected_cells: int) -> pd.DataFrame:
    selected = frame.loc[frame.game_id.isin(game_ids)]
    counts = selected.groupby("game_id").size()
    if len(counts) != len(game_ids) or not counts.eq(expected_cells).all():
        raise ValueError(f"Expected {expected_cells} cells on every game")
    return selected.groupby("game_id", as_index=False).agg(
        pred_home_margin=("pred_home_margin", "mean"),
        pred_home_win_probability=("pred_home_win_probability", "mean"),
        model_count=("model_id", "size"),
    )


def scorecard(forecasts: dict[str, pd.DataFrame], games: pd.DataFrame) -> pd.DataFrame:
    ids = set(games.game_id)
    rows = []
    for kind, (label, frame) in forecasts.items():
        rows.append(_summarize(frame.loc[frame.game_id.isin(ids)], games, kind=kind, label=label))
    rows.append(_vegas(games))
    card = pd.DataFrame(rows)
    if len(card) != 9 or not card.games.eq(len(games)).all():
        raise ValueError("Targeted five scorecard has incomplete coverage")
    return card


def cumulative_cards(forecasts: dict[str, pd.DataFrame], games: pd.DataFrame,
                     *, cumulative: bool) -> pd.DataFrame:
    rows = []
    for week in range(1, 9):
        week_games = games.loc[games.week.le(week) if cumulative else games.week.eq(week)]
        card = scorecard(forecasts, week_games)
        card = card.loc[card.row_type.isin(DISPLAY)].copy()
        card.insert(0, "through_week" if cumulative else "week", week)
        rows.append(card)
    result = pd.concat(rows, ignore_index=True)
    if len(result) != 24 or result.groupby("row_type").size().to_dict() != {key: 8 for key in DISPLAY}:
        raise ValueError("Cumulative or weekly target/Vegas comparison is incomplete")
    return result


def render_cumulative(card: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#f7f3eb", constrained_layout=True)
    panels = (("su_accuracy", "Straight-up accuracy"), ("ats_accuracy", "ATS accuracy"),
              ("brier", "Brier · lower is better"), ("margin_mae", "Margin MAE · points"))
    for ax, (field, title) in zip(axes.flat, panels):
        ax.set_facecolor("#f7f3eb")
        for kind, label in DISPLAY.items():
            if kind == "vegas_reference" and field == "ats_accuracy":
                continue
            series = card.loc[card.row_type.eq(kind)].sort_values("through_week")
            if len(series) != 8 or not series.through_week.eq(range(1, 9)).all():
                raise ValueError(f"Missing cumulative {kind} values")
            ax.plot(series.through_week, series[field], marker="s" if kind == "targeted_consensus" else "o",
                    lw=2.6, color=COLORS[kind], label=label)
        if field == "ats_accuracy":
            ax.axhline(110 / 210, color="#6f7780", ls="--", lw=1.4, label="-110 break-even")
        ax.set_title(title, fontsize=14)
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("Through CFBD week")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("2026 targeted five · cumulative paired results", fontsize=20,
                 weight="bold", color="#20324b")
    fig.get_layout_engine().set(rect=(.02, .085, .98, .91))
    fig.text(.5, .018, "412 shared games with saved Vegas spread and probability · five cells selected after observing results · Week 7 retrospective",
             ha="center", fontsize=9, color="#526173")
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Experiment output is immutable; choose a new version")
    backfill = json.loads(WEEK7_MANIFEST.read_text())
    if sha(WEEK7_F20) != backfill["private_predictions_sha256"]:
        raise ValueError("Week 7 F20 predictions differ from the audited backfill")
    previous, games, sources = load_inputs()
    predictions = pd.concat([previous, pd.read_parquet(WEEK7_F20)[MODEL_COLUMNS]], ignore_index=True)
    if predictions.duplicated(["game_id", "fingerprint", "model_id"]).any():
        raise ValueError("Duplicate game × scientific cell")
    chosen = predictions.set_index(["fingerprint", "model_id"]).loc[list(CELLS)].reset_index()
    counts = chosen.groupby("game_id").size()
    if counts.value_counts().to_dict() != {5: 420, 4: 13, 1: 8}:
        raise ValueError("Five-cell coverage differs from the audited 2026 cohort")
    common = set(counts[counts.eq(5)].index)
    all_games = games.loc[games.game_id.isin(common)].copy()
    spread_games = all_games.loc[all_games.market_spread.notna()].copy()
    paired_games = spread_games.loc[spread_games.market_probability.notna()].copy()
    if (len(all_games), len(spread_games), len(paired_games)) != (420, 419, 412):
        raise ValueError("2026 full/spread/probability cohorts changed")
    forecasts = {
        "targeted_consensus": ("F20/M1 + F7/M1 + F19/M1 + F19/M2 + F19/M10", forecast(chosen, common, 5)),
        "f19_consensus": ("F19 six-model consensus", forecast(predictions.loc[predictions.fingerprint.eq("F19")], common, 6)),
        "f20_consensus": ("F20 six-model consensus", forecast(predictions.loc[predictions.fingerprint.eq("F20")], common, 6)),
    }
    for fp, model in CELLS:
        key = f"{fp.lower()}_{model.lower()}"
        cell = chosen.loc[chosen.fingerprint.eq(fp) & chosen.model_id.eq(model)]
        forecasts[key] = (f"{fp}/{model}", forecast(cell, common, 1))
    cards = {
        "scorecard_full_420.csv": scorecard(forecasts, all_games),
        "scorecard_spread_paired_419.csv": scorecard(forecasts, spread_games),
        "scorecard_fully_paired_412.csv": scorecard(forecasts, paired_games),
        "cumulative_fully_paired_412.csv": cumulative_cards(forecasts, paired_games, cumulative=True),
        "weekly_fully_paired_412.csv": cumulative_cards(forecasts, paired_games, cumulative=False),
    }
    OUTPUT.mkdir(parents=True)
    for name, card in cards.items():
        card.to_csv(OUTPUT / name, index=False, float_format="%.10f")
    render_cumulative(cards["cumulative_fully_paired_412.csv"], OUTPUT / "cumulative_fully_paired_412.png")
    spread = cards["scorecard_spread_paired_419.csv"].set_index("row_type")
    paired = cards["scorecard_fully_paired_412.csv"].set_index("row_type")
    full = cards["scorecard_full_420.csv"].set_index("row_type")
    target_spread, vegas_spread = spread.loc["targeted_consensus"], spread.loc["vegas_reference"]
    target_paired, vegas_paired = paired.loc["targeted_consensus"], paired.loc["vegas_reference"]
    f19_spread, f19_paired = spread.loc["f19_consensus"], paired.loc["f19_consensus"]
    readme = f"""# Retrospective targeted five-cell 2026 consensus

Equal weights on F20/M1, F7/M1, F19/M1, F19/M2, and F19/M10. For each game, the target's home-win probability is the arithmetic mean of the five saved probabilities and its predicted home margin is the arithmetic mean of the five saved margins. Straight-up picks use probability ≥ 0.5; ATS uses the mean margin against the saved Vegas spread. No models were refitted or reweighted.

All five forecasts exist on 420 completed CFBD Weeks 1–8 games. The primary four-metric view uses the **same 412 games** with both a saved Vegas spread and probability. The 419-game spread-paired scorecard adds seven games lacking a saved Vegas probability; the full 420-game scorecard adds one game lacking a spread. Use the separate metric denominators in those files when comparing to Vegas.

| Metric | Targeted five | F19 six-model consensus | Vegas |
| --- | ---: | ---: | ---: |
| SU, 419 spread-paired games | {int(target_spread.su_wins)}/419 | {int(f19_spread.su_wins)}/419 | {int(vegas_spread.su_wins)}/419 |
| ATS, 419 spread-paired games | {int(target_spread.ats_wins)}–{int(target_spread.ats_losses)}–{int(target_spread.ats_pushes)} ({target_spread.ats_accuracy:.2%}) | {int(f19_spread.ats_wins)}–{int(f19_spread.ats_losses)}–{int(f19_spread.ats_pushes)} ({f19_spread.ats_accuracy:.2%}) | N/A |
| Brier, 412 fully paired games ↓ | {target_paired.brier:.6f} | {f19_paired.brier:.6f} | {vegas_paired.brier:.6f} |
| Margin MAE, 419 spread-paired games ↓ | {target_spread.margin_mae:.6f} | {f19_spread.margin_mae:.6f} | {vegas_spread.margin_mae:.6f} |

The target is {int(full.loc['targeted_consensus'].su_wins)}/420 straight up on its full available cohort. Its paired Brier is {target_paired.brier - vegas_paired.brier:+.6f} relative to Vegas (positive is worse), and its spread-paired margin MAE is {target_spread.margin_mae - vegas_spread.margin_mae:+.6f} points relative to Vegas (positive is worse). Vegas has no ATS record against its own line. At hypothetical -110 odds, ATS needs 52.38% wins to break even; the target's {target_spread.ats_accuracy:.2%} is below that threshold.

`cumulative_fully_paired_412.csv` and the PNG track the same 412-game universe through each week, comparing the target, F19 consensus, and Vegas. `weekly_fully_paired_412.csv` gives the separate weekly increments. The 9-row scope scorecards also include F20 consensus and each of the five selected cells. These are aggregate public results; source game-level predictions remain private.

**Interpretation:** The five cells were selected after inspecting their 2026 results, so this is an in-sample exploratory combination. F20 Week 7 was reconstructed with models fitted after those games; the source archive was obtained after games and original expert and market quote times remain unverified. Some F0–F19 weeks and Week 8 F20 margins also have retrospective timing limits. This result cannot establish prospective superiority or a realizable betting edge.
"""
    (OUTPUT / "README.md").write_text(readme)
    public_files = sorted(path for path in OUTPUT.iterdir() if path.is_file())
    manifest = {
        "schema": "tdnet-targeted-five-consensus-2026-week01-08-v1",
        "season": 2026, "through_cfbd_week": 8, "selected_cells": [f"{fp}/{model}" for fp, model in CELLS],
        "weights": "equal_probability_and_margin_means",
        "cohorts": {"full": 420, "spread_paired": 419, "fully_paired": 412},
        "selection_timing": "posthoc_after_observing_2026_results",
        "f20_week7_prospective": False,
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in [*sources.values(), WEEK7_F20, WEEK7_MANIFEST]},
        "files": {path.name: sha(path) for path in public_files},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "paired_games": 412,
                      "target_paired_su": int(target_paired.su_wins),
                      "vegas_paired_su": int(vegas_paired.su_wins),
                      "target_paired_brier": target_paired.brier,
                      "vegas_paired_brier": vegas_paired.brier,
                      "target_spread_ats": [int(target_spread.ats_wins), int(target_spread.ats_losses), int(target_spread.ats_pushes)],
                      "target_spread_mae": target_spread.margin_mae,
                      "vegas_spread_mae": vegas_spread.margin_mae}))


if __name__ == "__main__":
    main()
