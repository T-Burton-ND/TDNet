#!/usr/bin/env python3
"""Score and chart the 120-cell Week 6–7 retrospective scientific backfill."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "publication/2026"
OUT = PUB / "week_07/post_game/retrospective_replay/scientific/full_f0_f19"
CHART = PUB / "week_07/post_game/retrospective_replay/cumulative/scientific_f0_f19"
HISTORY = PUB / "week_05/post_game/scientific/current_season_f0_f19/scientific_2026_model_game_results.parquet"
GAMES = ROOT / "data/raw/cfbd/v2/games/2026.parquet"
ROSTER = ("M1", "M2", "M3", "M4", "M5", "M10")
STAGES = tuple(f"F{i}" for i in range(17)) + ("F17-market", "F18", "F19")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backfill_models() -> tuple[pd.DataFrame, list[Path]]:
    sources = []
    parts = []
    for week in (6, 7):
        path = (PUB / "week_06/pre_game/scientific/full_f0_f8/scientific_all_game_predictions.csv"
                if week == 6 else PUB / "week_07/post_game/retrospective_replay/scientific/full_f0_f8/scientific_model_game_results.csv")
        frame = pd.read_csv(path)
        if week == 7:
            frame = frame.loc[~frame.fingerprint.eq("F7")].copy()
        frame = frame[["game_id", "week", "home_team", "away_team", "fingerprint", "model_name",
                       "pred_home_margin", "pred_home_win_probability"]].copy()
        frame["model_id"] = frame.model_name.str.rsplit("_", n=1).str[-1]
        frame["provenance_class"] = ("frozen_pregame" if week == 6 else "retrospective_replay_after_games")
        parts.append(frame)
        sources.append(path)
    path = OUT / "f7_expost_predictions.parquet"
    parts.append(pd.read_parquet(path))
    sources.append(path)
    for stage in ("F09", "F10", "F11", "F12_corrected", "F13", "F14", "F15", "F16", "F17_market"):
        for model in ROSTER:
            path = OUT / "f09_f17" / f"{stage}_{model}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing retrospective scientific cell {path}")
            parts.append(pd.read_parquet(path))
            sources.append(path)
    path = OUT / "f18_f19_retrospective_predictions.parquet"
    parts.append(pd.read_parquet(path))
    sources.append(path)
    fields = ["game_id", "week", "home_team", "away_team", "fingerprint", "model_id",
              "model_name", "pred_home_margin", "pred_home_win_probability", "provenance_class"]
    models = pd.concat([frame[fields] for frame in parts], ignore_index=True)
    ex_post = models.week.eq(7) & models.fingerprint.isin(["F7", "F17-market", "F19"])
    models.loc[ex_post, "provenance_class"] = "retrospective_ex_post_market_after_games"
    models.loc[models.week.eq(7) & models.fingerprint.eq("F8"), "provenance_class"] = "retrospective_market_imputed_after_games"
    market = pd.read_parquet(OUT / "f17_f19_market_snapshot.parquet", columns=["target_game_id", "market_provenance"])
    fallback = set(market.loc[market.market_provenance.eq("week06_ex_post_market_fallback"), "target_game_id"])
    models.loc[models.week.eq(6) & models.game_id.isin(fallback)
               & models.fingerprint.isin(["F17-market", "F19"]), "provenance_class"] = "retrospective_ex_post_market_after_games"
    if len(models) != 117 * 120 or models.duplicated(["game_id", "model_name"]).any():
        raise ValueError("Week 6–7 F0–F19 backfill is not exactly 120 × 117")
    if set(models.fingerprint) != set(STAGES) or set(models.model_id) != set(ROSTER):
        raise ValueError("F0–F19 model roster differs from expected 120 cells")
    if models.groupby("game_id").model_name.nunique().value_counts().to_dict() != {120: 117}:
        raise ValueError("A game is missing a model vote")
    if not np.isfinite(models.pred_home_margin).all() or not models.pred_home_win_probability.between(0, 1).all():
        raise ValueError("A prediction margin or probability is invalid")
    score = pd.read_parquet(GAMES, columns=["id", "week", "home_points", "away_points"])
    score = score.loc[score.id.isin(models.game_id.unique())].rename(columns={"id": "game_id"})
    if len(score) != 117 or score[["home_points", "away_points"]].isna().any().any():
        raise ValueError("Week 6–7 official outcomes are incomplete")
    models = models.merge(score[["game_id", "home_points", "away_points"]], on="game_id", validate="many_to_one")
    models["actual_home_margin"] = models.home_points - models.away_points
    models["margin_error"] = (models.pred_home_margin - models.actual_home_margin).abs()
    models["winner_correct"] = models.pred_home_win_probability.ge(.5).eq(models.actual_home_margin.gt(0))
    models["brier_error"] = (models.pred_home_win_probability - models.actual_home_margin.gt(0).astype(float)) ** 2
    return models.sort_values(["week", "game_id", "fingerprint", "model_id"]), [*sources, GAMES]


def historical_models() -> pd.DataFrame:
    source = pd.read_parquet(HISTORY)
    common = source.groupby("game_id").forecast_status.apply(lambda s: s.eq("available").sum())
    common_ids = set(common[common.eq(120)].index)
    source = source.loc[source.game_id.isin(common_ids)].copy()
    if len(source) != 263 * 120 or source.duplicated(["game_id", "model_name"]).any():
        raise ValueError("Original common 120-cell cohort changed")
    frame = source[["game_id", "week", "home_team", "away_team", "fingerprint", "model_id", "model_name",
                    "pred_margin", "home_win_probability", "actual_home_margin", "absolute_margin_error",
                    "winner_correct", "brier_error"]].rename(columns={
                        "pred_margin": "pred_home_margin", "home_win_probability": "pred_home_win_probability",
                        "absolute_margin_error": "margin_error"})
    frame["provenance_class"] = "published_week05_exploratory_research"
    return frame


def aggregate(frame: pd.DataFrame, name: str) -> dict:
    brier = pd.to_numeric(frame.brier_error, errors="coerce")
    return {"model_name": name, "games": frame.game_id.nunique(), "brier_games": int(brier.notna().sum()),
            "margin_mae": float(frame.margin_error.mean()), "su_wins": int(frame.winner_correct.sum()),
            "su_accuracy": float(frame.winner_correct.mean()), "brier_score": float(brier.mean())}


def metrics(models: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    consensus = models.groupby(["week", "game_id"], as_index=False).agg(
        pred_home_margin=("pred_home_margin", "mean"),
        pred_home_win_probability=("pred_home_win_probability", "mean"),
        actual_home_margin=("actual_home_margin", "first"),
        model_votes=("model_name", "nunique"),
    )
    if not consensus.model_votes.eq(120).all() or len(consensus) != 380:
        raise ValueError("Full consensus lacks 120 equal votes on every common game")
    consensus["margin_error"] = (consensus.pred_home_margin - consensus.actual_home_margin).abs()
    consensus["winner_correct"] = consensus.pred_home_win_probability.ge(.5).eq(consensus.actual_home_margin.gt(0))
    consensus["brier_error"] = (consensus.pred_home_win_probability - consensus.actual_home_margin.gt(0).astype(float)) ** 2
    historical = pd.read_parquet(HISTORY)
    common_ids = set(models.loc[models.week.le(5), "game_id"])
    market = historical.loc[historical.game_id.isin(common_ids)].drop_duplicates("game_id")
    market = market[["week", "game_id", "actual_home_margin", "market_home_spread", "market_home_probability"]].copy()
    frozen = pd.read_parquet(PUB / "week_06/pre_game/frozen_bundle/public/predictions.parquet",
                             columns=["game_id", "market_spread_close", "market_win_probability"]).drop_duplicates("game_id")
    week6 = consensus.loc[consensus.week.eq(6), ["game_id", "actual_home_margin"]].merge(frozen, on="game_id", validate="one_to_one")
    week6["week"] = 6
    week6 = week6.rename(columns={"market_spread_close": "market_home_spread",
                                  "market_win_probability": "market_home_probability"})
    market = pd.concat([market, week6[market.columns]], ignore_index=True)
    market["margin_error"] = (-market.market_home_spread - market.actual_home_margin).abs()
    market["winner_correct"] = (-market.market_home_spread).gt(0).eq(market.actual_home_margin.gt(0)).astype("boolean")
    market.loc[market.market_home_spread.isna(), "winner_correct"] = pd.NA
    market["brier_error"] = (market.market_home_probability - market.actual_home_margin.gt(0).astype(float)) ** 2
    rows = []
    for week in range(1, 8):
        through = models.loc[models.week.le(week)]
        for name, frame in through.groupby("model_name", sort=True):
            rows.append({"through_week": week, "series_type": "model", **aggregate(frame, name)})
        rows.append({"through_week": week, "series_type": "consensus",
                     **aggregate(consensus.loc[consensus.week.le(week)], "F0–F19 equal consensus")})
        if week <= 6:
            valid = market.loc[market.week.le(week) & market.market_home_spread.notna()].copy()
            brier = market.loc[market.week.le(week), "brier_error"]
            row = aggregate(valid, "Vegas · frozen through Week 6")
            row["brier_games"] = int(brier.notna().sum())
            row["brier_score"] = float(brier.mean())
            rows.append({"through_week": week, "series_type": "vegas", **row})
    performance = pd.DataFrame(rows)
    performance["contains_retrospective_backfill"] = performance.through_week.ge(6) & ~performance.series_type.eq("vegas")
    performance["contains_ex_post_market_cells"] = performance.through_week.eq(7) & (
        performance.model_name.str.contains("F7_|F17-market_|F19_") | performance.series_type.eq("consensus"))
    return consensus, market, performance


def trajectory(performance: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="#F7F4ED")
    for axis, (column, title, percent) in zip(axes, (
        ("margin_mae", "Margin MAE", False), ("su_accuracy", "Straight-up accuracy", True),
        ("brier_score", "Brier score", False))):
        for _, group in performance.loc[performance.series_type.eq("model")].groupby("model_name"):
            group = group.sort_values("through_week")
            axis.plot(group.through_week, group[column], color="#71849A", alpha=.14, lw=.8)
            axis.plot(group.loc[group.through_week.ge(5), "through_week"],
                      group.loc[group.through_week.ge(5), column], color="#71849A", alpha=.28, lw=.8, ls=":")
        consensus = performance.loc[performance.series_type.eq("consensus")].sort_values("through_week")
        axis.plot(consensus.loc[consensus.through_week.le(5), "through_week"],
                  consensus.loc[consensus.through_week.le(5), column], color="#D62F76", lw=3, marker="o",
                  label="F0–F19 consensus · original Weeks 1–5")
        axis.plot(consensus.loc[consensus.through_week.ge(5), "through_week"],
                  consensus.loc[consensus.through_week.ge(5), column], color="#D62F76", lw=3, ls=":", marker="o",
                  label="Weeks 6–7 retrospective")
        vegas = performance.loc[performance.series_type.eq("vegas")].sort_values("through_week")
        axis.plot(vegas.through_week, vegas[column], color="#AE7F1C", lw=2.6, ls="--", marker="s",
                  label="Vegas · archived through Week 6")
        axis.axvspan(5.5, 7.3, color="#DEE0E2", alpha=.45)
        axis.set_title(title, fontsize=15, loc="left", weight="bold")
        axis.set_xticks(range(1, 8))
        axis.set_xlim(.8, 7.3)
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
        if percent:
            axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("2026 F0–F19 scientific · cumulative common-cohort performance", fontsize=21, weight="bold")
    fig.text(.5, .02, "263 common Weeks 1–5 games + 58 Week 6 + 59 Week 7; all 120 model cells. Weeks 6–7 reconstructed after games. Week 7 F7/F17/F19 use ex-post market lines; Vegas stops at Week 6.",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def scorecard(performance: pd.DataFrame, path: Path) -> None:
    table = performance.loc[performance.through_week.eq(7)].copy()
    table = pd.concat([table, performance.loc[performance.through_week.eq(6) & performance.series_type.eq("vegas")]], ignore_index=True)
    table["sort"] = table.series_type.map({"consensus": 0, "vegas": 1, "model": 2})
    table = table.sort_values(["sort", "margin_mae", "model_name"]).reset_index(drop=True)
    ranks = table.series_type.eq("model").cumsum().astype(str)
    ranks.loc[~table.series_type.eq("model")] = "—"
    labels = pd.DataFrame({
        "Rank": ranks,
        "Model": table.model_name.str.replace("scientific_", "", regex=False),
        "Games": table.games.astype(str), "Brier n": table.brier_games.astype(str),
        "MAE": table.margin_mae.map(lambda x: f"{x:.2f}"),
        "SU": table.su_wins.astype(str) + "/" + table.games.astype(str),
        "SU %": table.su_accuracy.map(lambda x: f"{x:.1%}"),
        "Brier": table.brier_score.map(lambda x: f"{x:.3f}"),
    })
    fig, ax = plt.subplots(figsize=(16, max(10, len(labels) * .38 + 2.5)), facecolor="#F7F4ED")
    ax.axis("off")
    artist = ax.table(cellText=labels.values, colLabels=labels.columns, cellLoc="left", colLoc="left",
                      bbox=[0, .035, 1, .91], colWidths=[.055, .39, .07, .07, .08, .1, .1, .1])
    artist.auto_set_font_size(False)
    artist.set_fontsize(9)
    for (row, _), cell in artist.get_celld().items():
        cell.set_edgecolor("#D4D7DB")
        cell.set_facecolor("#22324A" if row == 0 else "#FDE2F0" if row == 1 else "#EEF2F5" if row == 2 else "#FFFFFF" if row % 2 else "#EEF2F5")
        if row == 0:
            cell.set_text_props(color="white", weight="bold")
    ax.set_title("2026 Scientific F0–F19 roster · cumulative through Week 7", fontsize=19, weight="bold", pad=16)
    fig.text(.5, .008, "Model/consensus: 380 common games including retrospective Weeks 6–7. Vegas stops at Week 6. Brier n shows valid probabilities. ATS is omitted; Week 7 F7/F17/F19 market inputs were retrieved after games.",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHART.mkdir(parents=True, exist_ok=True)
    recent, sources = backfill_models()
    old = historical_models()
    fields = ["game_id", "week", "home_team", "away_team", "fingerprint", "model_id", "model_name",
              "pred_home_margin", "pred_home_win_probability", "actual_home_margin", "margin_error",
              "winner_correct", "brier_error", "provenance_class"]
    models = pd.concat([old[fields], recent[fields]], ignore_index=True)
    if len(models) != 380 * 120 or models.groupby("model_name").game_id.nunique().nunique() != 1:
        raise ValueError("Cumulative common 120-cell panel is incomplete")
    consensus, vegas, performance = metrics(models)
    paths = {
        "backfilled_model_games": OUT / "week06_07_model_game_results.parquet",
        "backfilled_consensus": OUT / "week06_07_consensus_game_results.csv",
        "cumulative_model_games": CHART / "common_120_model_game_results.parquet",
        "cumulative_consensus_games": CHART / "common_120_consensus_game_results.csv",
        "vegas_games": CHART / "common_cohort_vegas_games.csv",
        "cumulative_csv": CHART / "cumulative_performance.csv",
        "scorecard_csv": CHART / "cumulative_model_scorecard.csv",
        "performance_png": CHART / "cumulative_performance.png",
        "scorecard_png": CHART / "cumulative_model_scorecard.png",
        "coverage_png": CHART / "cumulative_available_performance.png",
    }
    recent.to_parquet(paths["backfilled_model_games"], index=False, compression="zstd")
    consensus.loc[consensus.week.ge(6)].to_csv(paths["backfilled_consensus"], index=False)
    models.to_parquet(paths["cumulative_model_games"], index=False, compression="zstd")
    consensus.to_csv(paths["cumulative_consensus_games"], index=False)
    vegas.to_csv(paths["vegas_games"], index=False)
    performance.to_csv(paths["cumulative_csv"], index=False)
    score = pd.concat([performance.loc[performance.through_week.eq(7)],
                       performance.loc[performance.through_week.eq(6) & performance.series_type.eq("vegas")]])
    score.to_csv(paths["scorecard_csv"], index=False)
    trajectory(performance, paths["performance_png"])
    shutil.copyfile(paths["performance_png"], paths["coverage_png"])
    scorecard(performance, paths["scorecard_png"])
    inputs = [HISTORY, PUB / "week_06/pre_game/frozen_bundle/public/predictions.parquet", *sources,
              OUT / "f17_f19_market_snapshot.parquet", OUT / "market_source_receipt.json",
              OUT / "f7_expost_receipt.json", OUT / "f18_f19_retrospective_receipt.json",
              ROOT / "data/what_if_2026_fingerprints/week06_07_retrospective_states/manifest.json"]
    manifest = {
        "scope": "2026_F0_F19_common_120_retrospective_week06_week07", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_cells": 120, "common_historical_games_weeks_1_5": 263, "week06_games": 58, "week07_games": 59,
        "model_games_per_cell": 380, "week06_07_issued_before_kickoff": False,
        "week07_market_quote_timing_verified": False,
        "week07_ex_post_market_cell_fingerprints": ["F7", "F17-market", "F19"],
        "week06_ex_post_market_fallback_game_ids": json.loads(
            (OUT / "market_source_receipt.json").read_text())["week06_ex_post_fallback_game_ids"],
        "week07_f8_market_feature_policy": "previously published no-market replay; market-dependent features imputed",
        "source_field_level_availability_verified": False,
        "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "outputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths.values()},
    }
    (OUT / "backfill_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT / "README.md").write_text(
        "# 2026 Weeks 6–7 F0–F19 retrospective backfill\n\n"
        "All 120 scientific model cells have 58 Week 6 and 59 Week 7 scored games. "
        "These model outputs were reconstructed October 21 after the target games. "
        "F0–F8 Week 6 rows were originally frozen before kickoff; F0–F6/F8 Week 7 rows came from the earlier retrospective replay. "
        "F9–F17 used through-2025 historical fits and score-free as-of feature states rebuilt from the later source cache. "
        "F18/F19 used frozen through-2025 bundles. F7 Week 7 was rerun from frozen checkpoints.\n\n"
        "Week 7 F7, F17-market and F19 used market quotes retrieved after the games, without verified quote timestamps. "
        "One Week 6 F17/F19 game also needed an after-game line because the frozen market was missing. "
        "The previously published Week 7 F8 replay had its market fields missing and imputed. "
        "Provider field-level historical availability and later corrections are not ruled out. "
        "Thus Weeks 6–7 are exploratory retrospective diagnostics, never timely or leakage-free forecasts. "
        "The cumulative chart uses the 263-game common F0–F19 Weeks 1–5 cohort, then adds these 117 games. "
        "The Vegas line uses archived market summaries only and stops at Week 6; its game count can be smaller.\n"
    )
    print({"model_cells": 120, "common_games": 380, "week06": 58, "week07": 59}, flush=True)


if __name__ == "__main__":
    main()
