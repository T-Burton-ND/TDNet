#!/usr/bin/env python3
"""Chart F0–F20 with matched 2026 retrospective F20 replay and coverage notes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "publication/2026/week_07/post_game/retrospective_replay/cumulative/scientific_f0_f19"
OUTPUT = ROOT / "publication/2026/research/scientific_f0_f20_cumulative_week01_07"
F20_ARCHIVE = Path("/groups/bsavoie2/tburton2/TDNet/f20_2026/prospective")
F20_REPLAY = ROOT / "data/private/f20/replay_2026_week01_07_v2"
MODELS = {"M1", "M2", "M3", "M4", "M5", "M10"}
TIERS = {*(f"F{i}" for i in range(17)), "F17-market", "F18", "F19"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_matched(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = pd.read_parquet(source / "common_120_model_game_results.parquet")
    market = pd.read_csv(source / "common_cohort_vegas_games.csv")
    if market.game_id.duplicated().any() or set(market.week) != set(range(1, 8)):
        raise ValueError("Market common cohort has duplicate games or unexpected weeks")
    if len(models) != 120 * len(market) or models.duplicated(["game_id", "fingerprint", "model_id"]).any():
        raise ValueError("Scientific common cohort is not one row per game/cell")
    if set(models.game_id) != set(market.game_id):
        raise ValueError("Scientific and market game sets differ")
    if set(models.fingerprint) != TIERS or set(models.model_id) != MODELS:
        raise ValueError("Scientific tier/architecture contract changed")
    cells = models.groupby(["fingerprint", "model_id"]).game_id.nunique()
    if len(cells) != 120 or not cells.eq(len(market)).all():
        raise ValueError("A scientific cell is missing a common-cohort game")
    reference = market.set_index("game_id")
    if not models.week.eq(models.game_id.map(reference.week)).all():
        raise ValueError("Scientific and market weeks differ")
    if not np.allclose(models.actual_home_margin, models.game_id.map(reference.actual_home_margin)):
        raise ValueError("Scientific and market outcomes differ")
    if not np.isfinite(models.pred_home_margin).all() or not models.pred_home_win_probability.between(0, 1).all():
        raise ValueError("Invalid scientific model prediction")
    if not market.market_home_probability.dropna().between(0, 1).all():
        raise ValueError("Invalid market probability")
    return models, market


def audit_f20(archive: Path) -> dict:
    records = []
    for path in (archive / "forecasts").glob("*/*.json"):
        if path.name == "official.json":
            continue
        record = json.loads(path.read_text())
        if record.get("season") != 2026:
            continue
        records.append((int(record["week"]), int(record["game_id"])))
    early = sorted({game for week, game in records if 1 <= week <= 7})
    if early:
        raise ValueError(
            "F20 Week 1–7 records now exist; audit their point-in-time eligibility before charting: "
            + str(early[:10])
        )
    return {"season_2026_forecast_records": len(records),
            "week_01_07_forecast_records": 0,
            "week_01_07_unique_games": 0,
            "other_weeks": sorted({week for week, _ in records}),
            "status": "no_week_01_07_f20_forecasts"}


def cumulative_scores(models: pd.DataFrame, market: pd.DataFrame,
                      f20: pd.DataFrame) -> pd.DataFrame:
    spread_ids = set(market.loc[market.market_home_spread.notna(), "game_id"])
    probability_ids = set(market.loc[market.market_home_probability.notna(), "game_id"])
    if not spread_ids or not probability_ids:
        raise ValueError("Both matched market cohorts must be nonempty")
    grouped = [
        (fingerprint, frame.groupby("game_id", as_index=False).agg(
            pred_home_margin=("pred_home_margin", "mean"),
            pred_home_win_probability=("pred_home_win_probability", "mean")))
        for fingerprint, frame in models.groupby("fingerprint")
    ]
    grouped += [(f"{fingerprint}/{model_id}", frame[["game_id", "pred_home_margin",
                                                        "pred_home_win_probability"]].copy())
                for (fingerprint, model_id), frame in models.groupby(["fingerprint", "model_id"])]
    grouped += [
        ("Frozen F0–F6", models.loc[models.fingerprint.isin({f"F{i}" for i in range(7)})]
         .groupby("game_id", as_index=False).agg(
             pred_home_margin=("pred_home_margin", "mean"),
             pred_home_win_probability=("pred_home_win_probability", "mean"))),
        ("Expanded F0–F19", models.groupby("game_id", as_index=False).agg(
            pred_home_margin=("pred_home_margin", "mean"),
            pred_home_win_probability=("pred_home_win_probability", "mean"))),
    ]
    reference = market[["game_id", "week", "actual_home_margin", "market_home_spread", "market_home_probability"]]
    rows = []
    for label, predictions in grouped:
        if set(predictions.game_id) != set(market.game_id):
            raise ValueError(f"Cohort mismatch for {label}")
        table = predictions.merge(reference, on="game_id", validate="one_to_one")
        for week in range(1, 8):
            seen = table.loc[table.week.le(week)]
            margin = seen.loc[seen.game_id.isin(spread_ids)]
            probability = seen.loc[seen.game_id.isin(probability_ids)]
            truth = probability.actual_home_margin.gt(0).to_numpy(float)
            rows.append({"through_week": week, "series": label,
                         "margin_games": len(margin), "brier_games": len(probability),
                         "margin_mae": float(np.mean(np.abs(margin.pred_home_margin - margin.actual_home_margin))),
                         "brier": float(np.mean((probability.pred_home_win_probability.to_numpy() - truth) ** 2)),
                         "status": "retrospective_scored"})
    for week in range(1, 8):
        seen = market.loc[market.week.le(week)]
        margin = seen.loc[seen.game_id.isin(spread_ids)]
        probability = seen.loc[seen.game_id.isin(probability_ids)]
        rows.append({"through_week": week, "series": "Archived market",
                     "margin_games": len(margin), "brier_games": len(probability),
                     "margin_mae": float(np.mean(np.abs(-margin.market_home_spread - margin.actual_home_margin))),
                     "brier": float(np.mean((probability.market_home_probability.to_numpy()
                                            - probability.actual_home_margin.gt(0).to_numpy(float)) ** 2)),
                     "status": "retrospective_market_quote_times_unverified"})
        for model_id in (*sorted(MODELS), "F20"):
            seen_f20 = f20.loc[f20.cfbd_week.le(week) & f20.game_id.isin(probability_ids)]
            if week == 7:
                brier = np.nan
                status = "unavailable_no_cfbd_week_07_archive_games"
            else:
                truth = seen_f20.actual_home_margin.gt(0).to_numpy(float)
                brier = float(np.mean((seen_f20[f"probability_{model_id}"].to_numpy() - truth) ** 2))
                status = "retrospective_replay_assumed_pregame_archive"
            rows.append({"through_week": week, "series": f"F20/{model_id}" if model_id != "F20" else "F20",
                         "margin_games": 0, "brier_games": len(seen_f20),
                         "margin_mae": np.nan, "brier": brier, "status": status})
    result = pd.DataFrame(rows)
    if len(result) != (20 + 120 + 2 + 1 + 7) * 7:
        raise ValueError("Cumulative scorecard has an unexpected number of series")
    return result.sort_values(["through_week", "series"]).reset_index(drop=True)


def draw(scores: pd.DataFrame, *, metric: str, path: Path, paired: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#F7F4ED")
    ax.set_facecolor("#FFFCF7")
    tiers = [f"F{i}" for i in range(17)] + ["F17-market", "F18", "F19"]
    colors = plt.cm.viridis(np.linspace(.1, .85, len(tiers)))
    for tier, color in zip(tiers, colors):
        row = scores.loc[scores.series.eq(tier) & (scores.through_week.le(6) if paired else True)]
        ax.plot(row.through_week, row[metric], color=color, alpha=.58, linewidth=1.2)
    for label, color, style in (("Frozen F0–F6", "#D13D7C", "-"),
                                ("Expanded F0–F19", "#24344B", "-"),
                                ("Archived market", "#AB7B22", "--")):
        row = scores.loc[scores.series.eq(label) & (scores.through_week.le(6) if paired else True)]
        ax.plot(row.through_week, row[metric], color=color, linestyle=style,
                linewidth=2.8, marker="o", label=label)
    if metric == "brier":
        for model_id, color in zip(sorted(MODELS), plt.cm.plasma(np.linspace(.2, .8, len(MODELS)))):
            row = scores.loc[scores.series.eq(f"F20/{model_id}") & scores.through_week.le(6)]
            ax.plot(row.through_week, row.brier, color=color, linestyle=":", linewidth=1.2,
                    alpha=.75, label=f"F20/{model_id}")
        row = scores.loc[scores.series.eq("F20") & scores.through_week.le(6)]
        ax.plot(row.through_week, row.brier, color="#D42121", linewidth=3,
                marker="D", label="F20 consensus")
    ax.set_xticks(range(1, 7 if paired else 8))
    ax.set_xlabel("Through week (2026)")
    ax.set_ylabel("Cumulative Brier score" if metric == "brier" else "Cumulative margin MAE (points)")
    ax.set_title("F0–F20 paired Brier · Weeks 1–6" if paired else
                 "F0–F20 scientific cumulative view · Weeks 1–7",
                 loc="left", color="#24344B", fontsize=15)
    ax.grid(alpha=.18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    note = ("Same 304 games with F20 replay and market probability, paired by CFBD game ID. No CFBD Week 7 archive games.\n"
            "Archive timing assumed pregame; the six F20 models were fitted after these games. Scores are retrospective."
            if paired else
            "F0–F19: 120 margin cells through Week 7. F20 has no margin prediction, so margin MAE is inapplicable.\n"
            "Market spread and model margin curves use the same games; historical market quote times remain unverified."
            if metric == "margin_mae" else
            "F0–F19: 120 cells through Week 7. F20: 6 model replay + consensus through CFBD Week 6; no Week 7 archive games.\n"
            "F20 uses a smaller matched cohort; use the paired companion chart to compare. Archive timing assumed pregame; model fit later.")
    fig.text(.08, .035, note,
             fontsize=8, color="#495C77")
    fig.subplots_adjust(left=.08, right=.78, top=.9, bottom=.17)
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--f20-archive", type=Path, default=F20_ARCHIVE)
    parser.add_argument("--f20-replay", type=Path, default=F20_REPLAY)
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Existing chart manifest is immutable; select a new output")
    models, market = load_matched(args.source)
    f20_audit = audit_f20(args.f20_archive)
    f20 = pd.read_parquet(args.f20_replay / "f20_replay.parquet")
    replay_receipt = json.loads((args.f20_replay / "receipt.json").read_text())
    if digest(args.f20_replay / "f20_replay.parquet") != replay_receipt["predictions_sha256"]:
        raise ValueError("F20 replay hash mismatch")
    if f20.game_id.duplicated().any() or not set(f20.cfbd_week).issubset(set(range(1, 7))):
        raise ValueError("Unexpected F20 replay coverage")
    paired = f20.merge(market[["game_id", "week", "actual_home_margin"]], on="game_id",
                       how="inner", validate="one_to_one", suffixes=("_f20", "_market"))
    if not paired.cfbd_week.eq(paired.week).all() or not np.allclose(
            paired.actual_home_margin_f20, paired.actual_home_margin_market):
        raise ValueError("F20 replay game/week/outcome mismatch")
    scores = cumulative_scores(models, market, f20)
    args.output.mkdir(parents=True, exist_ok=True)
    score_path = args.output / "cumulative_scores.csv"
    scores.to_csv(score_path, index=False, float_format="%.9f")
    paired_market = market.loc[market.game_id.isin(paired.game_id) & market.market_home_probability.notna()]
    paired_ids = set(paired_market.game_id)
    paired_models = models.loc[models.game_id.isin(paired_ids)]
    paired_f20 = f20.loc[f20.game_id.isin(paired_ids)]
    paired_scores = cumulative_scores(paired_models, paired_market, paired_f20)
    paired_path = args.output / "cumulative_scores_f20_paired_cohort.csv"
    paired_scores.to_csv(paired_path, index=False, float_format="%.9f")
    figures = {}
    for metric in ("brier", "margin_mae"):
        path = args.output / f"f0_f20_cumulative_{metric}.png"
        draw(scores, metric=metric, path=path)
        figures[metric] = {"path": str(path), "sha256": digest(path)}
    paired_figure = args.output / "f0_f20_paired_brier.png"
    draw(paired_scores, metric="brier", path=paired_figure, paired=True)
    figures["paired_brier"] = {"path": str(paired_figure), "sha256": digest(paired_figure)}
    manifest = {"scope": "2026 Weeks 1–7 retrospective common cohort",
                "source": {name: {"path": str(args.source / name), "sha256": digest(args.source / name)}
                           for name in ("common_120_model_game_results.parquet", "common_cohort_vegas_games.csv")},
                "common_games": int(market.game_id.nunique()),
                "margin_games": int(market.market_home_spread.notna().sum()),
                "brier_games": int(market.market_home_probability.notna().sum()),
                "f20": f20_audit,
                "f20_replay": {"path": str(args.f20_replay / "f20_replay.parquet"),
                               "sha256": digest(args.f20_replay / "f20_replay.parquet"),
                               "matched_common_games": len(paired),
                               "paired_brier_games": len(paired_ids),
                               "cfbd_week_counts": replay_receipt["cfbd_week_counts"],
                               "timing_assumption": replay_receipt["source_pregame_timing_assumption"]},
                "scores": {"path": str(score_path), "sha256": digest(score_path)},
                "paired_scores": {"path": str(paired_path), "sha256": digest(paired_path)},
                "figures": figures,
                "historical_market_quote_times_verified": False,
                "prospective_f20_week_01_07_score_available": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output / "README.md").write_text(
        "# F0–F20 scientific cumulative view, 2026 Weeks 1–7\n\n"
        "The main charts show 120 archived F0–F19 model cells through CFBD Week 7, "
        "and all six F20 probability models plus equal-weight F20 consensus through CFBD Week 6. "
        "The tracker archive's Week 7 maps to CFBD Week 6; no CFBD Week 7 F20 inputs exist. "
        "F20 is a retrospective replay using weekly projections assumed pregame by the owner; "
        "the six models were fitted after these games, so these are not prospective scores. "
        "The main F20 cohort is smaller; use `f0_f20_paired_brier.png` and "
        "`cumulative_scores_f20_paired_cohort.csv` "
        "for matched Brier comparisons. F20 has no margin MAE. Historical F19 and market "
        "quote times are unverified. See `manifest.json` for hashes and coverage.\n"
    )
    print(json.dumps({"output": str(args.output), "common_games": manifest["common_games"],
                      "margin_games": manifest["margin_games"], "brier_games": manifest["brier_games"],
                      "f20": f20_audit}, indent=2))


if __name__ == "__main__":
    main()
