#!/usr/bin/env python3
"""Rebuild the 2026 scientific season curves with the four missing Week 1 games.

This isolated what-if keeps the frozen week bundles untouched. The four added
games are forecast from Week 0 fingerprint rows and scored after the fact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pregame_market_snapshot_2026 import build_pregame_market_state

from gridiron_ml.cli.publication.build_2026_scientific_all_fingerprint_curves import (
    _full_scorecard,
    _plot,
)

PUB = ROOT / "publication/2026/week_05/post_game/scientific/current_season_f0_f17_market"
SOURCE_DIR = ROOT / "publication/2026/week_05/post_game/scientific/full_f0_f8"
BACKFILL = ROOT / "data/what_if_2026_fingerprints/f0_f8_week1_backfill/run_leak_checked/tables/all_game_model_predictions.parquet"
BACKFILL_RECEIPT = ROOT / "data/what_if_2026_fingerprints/f0_f8_week1_backfill/run_leak_checked/receipt.json"
GAMES = ROOT / "data/raw/cfbd/v2/games/2026.parquet"


def game_key(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("Int64").astype(str)


def _base_predictions() -> pd.DataFrame:
    frames = []
    for week in range(6):
        path = SOURCE_DIR / f"../../../../week_{week:02d}/post_game/scientific/full_f0_f8/scientific_model_game_results.csv"
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frames.append(frame)
    old = pd.concat(frames, ignore_index=True)
    old = old.rename(columns={"market_win_probability": "market_win_probability"})
    old["__game_key"] = game_key(old.game_id)
    old["__is_backfill"] = False
    f09_path = ROOT / "data/what_if_2026_fingerprints/f09_predictions/predictions.parquet"
    f09 = pd.read_parquet(f09_path)
    family = {"M1": "linear", "M2": "spline", "M3": "tree", "M4": "boosted", "M5": "neural", "M10": "knn"}
    f09["model_family"] = f09.model_name.str.rsplit("_", n=1).str[-1].map(family)
    f09["__game_key"] = game_key(f09.game_id)
    f09["__is_backfill"] = False
    f10 = pd.read_parquet(ROOT / "data/what_if_2026_fingerprints/f10_predictions/predictions.parquet")
    f10["model_family"] = f10.model_name.str.rsplit("_", n=1).str[-1].map(family)
    f10["__game_key"] = game_key(f10.game_id)
    f10["__is_backfill"] = False
    f11 = pd.read_parquet(ROOT / "data/what_if_2026_fingerprints/f11_predictions/predictions.parquet")
    f11["model_family"] = f11.model_name.str.rsplit("_", n=1).str[-1].map(family)
    f11["__game_key"] = game_key(f11.game_id)
    f11["__is_backfill"] = False
    f12 = pd.read_parquet(ROOT / "data/what_if_2026_fingerprints/f12_predictions/predictions.parquet")
    f12["model_family"] = f12.model_name.str.rsplit("_", n=1).str[-1].map(family)
    f12["__game_key"] = game_key(f12.game_id)
    f12["__is_backfill"] = False
    nextgen = []
    for generation in range(13, 17):
        frame = pd.read_parquet(ROOT / f"data/what_if_2026_fingerprints/f{generation}_predictions/predictions.parquet")
        frame["model_family"] = frame.model_name.str.rsplit("_", n=1).str[-1].map(family)
        frame["__game_key"] = game_key(frame.game_id)
        frame["__is_backfill"] = False
        nextgen.append(frame)
    f17 = pd.read_parquet(ROOT / "data/what_if_2026_fingerprints/f17_market_predictions/predictions.parquet")
    f17["model_family"] = f17.model_name.str.rsplit("_", n=1).str[-1].map(family)
    f17["__game_key"] = game_key(f17.game_id)
    f17["__is_backfill"] = False
    return pd.concat([old, f09, f10, f11, f12, *nextgen, f17], ignore_index=True, sort=False)


def _new_predictions() -> pd.DataFrame:
    frame = pd.read_parquet(BACKFILL)
    frame["__game_key"] = game_key(frame.game_id)
    return pd.DataFrame(
        {
            "game_id": pd.to_numeric(frame.game_id, errors="coerce").astype("Int64"),
            "__game_key": frame["__game_key"],
            "season": 2026,
            "week": 1,
            "home_team": frame.home_team,
            "away_team": frame.away_team,
            "model_name": frame.model_name,
            "model_family": frame.model_family,
            "fingerprint": frame.fingerprint,
            "pred_home_margin": pd.to_numeric(frame.pred_home_margin, errors="coerce"),
            "pred_home_win_probability": pd.to_numeric(frame.pred_home_win_probability, errors="coerce"),
            "pred_winner": frame.pred_winner,
            "market_spread_close": pd.to_numeric(frame.market_spread_close_schedule, errors="coerce"),
            "market_win_probability": pd.to_numeric(frame.market_win_probability_schedule, errors="coerce"),
            "forecast_source_type": frame.forecast_source_type,
            "feature_snapshot_timestamp_utc": frame.feature_snapshot_timestamp_utc,
            "market_quote_timestamp_available": frame.market_quote_timestamp_available,
            "market_snapshot_captured_pre_kickoff": frame.market_snapshot_captured_pre_kickoff,
            "__is_backfill": True,
        }
    )


def _completed_games() -> pd.DataFrame:
    games = pd.read_parquet(GAMES)
    games = games.loc[
        games.season_type.astype(str).str.lower().eq("regular")
        & games.completed.fillna(False).astype(bool)
        & games.home_classification.astype(str).str.lower().eq("fbs")
        & games.away_classification.astype(str).str.lower().eq("fbs")
    ].copy()
    games["__game_key"] = game_key(games.id)
    return games.rename(
        columns={"id": "source_game_id", "home_points": "home_points", "away_points": "away_points"}
    )[["__game_key", "source_game_id", "home_points", "away_points", "start_date"]]


def _score_model_rows(predictions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame = frame.merge(
        results.rename(columns={"home_points": "source_home_points", "away_points": "source_away_points"}),
        on="__game_key", how="left", validate="many_to_one",
    )
    frame["home_points"] = pd.to_numeric(frame.get("home_points"), errors="coerce").fillna(frame.source_home_points)
    frame["away_points"] = pd.to_numeric(frame.get("away_points"), errors="coerce").fillna(frame.source_away_points)
    if frame[["home_points", "away_points"]].isna().any().any():
        raise ValueError("One or more model rows are missing completed scores.")
    frame["actual_home_margin"] = frame.home_points - frame.away_points
    frame["actual_home_win"] = frame.actual_home_margin.gt(0).astype(float)
    frame["model_absolute_margin_error"] = (
        frame.pred_home_margin - frame.actual_home_margin
    ).abs()
    frame["model_winner_correct"] = (
        frame.pred_home_win_probability.ge(0.5) == frame.actual_home_win.eq(1)
    )
    frame["model_brier"] = (frame.pred_home_win_probability - frame.actual_home_win) ** 2
    spread = pd.to_numeric(frame.market_spread_close, errors="coerce")
    cover_margin = frame.actual_home_margin + spread
    frame["actual_cover_team"] = np.select(
        [cover_margin.gt(0), cover_margin.lt(0), cover_margin.eq(0)],
        [frame.home_team, frame.away_team, "Push"], default="No line",
    )
    model_cover_margin = frame.pred_home_margin + spread
    model_pick = np.where(model_cover_margin.gt(0), frame.home_team, frame.away_team)
    calculated_ats = np.select(
        [spread.isna(), frame.actual_cover_team.eq("Push"), pd.Series(model_pick, index=frame.index).eq(frame.actual_cover_team)],
        ["No line", "Push", "Win"], default="Loss",
    )
    if "model_ats_result" in frame:
        preserve_existing = ~frame["__is_backfill"] & frame["model_ats_result"].notna()
        frame["model_ats_result"] = frame["model_ats_result"].where(preserve_existing, calculated_ats)
    else:
        frame["model_ats_result"] = calculated_ats
    frame["actual_favorite_home"] = spread.lt(0)
    frame["actual_favorite_won"] = (
        (frame.actual_favorite_home & frame.actual_home_margin.gt(0))
        | (~frame.actual_favorite_home & frame.actual_home_margin.lt(0))
    )
    frame["actual_upset"] = spread.notna() & spread.ne(0) & ~frame.actual_favorite_won
    picked_home = frame.pred_home_win_probability.ge(0.5)
    picked_favorite = (frame.actual_favorite_home & picked_home) | (~frame.actual_favorite_home & ~picked_home)
    frame["model_recalled_upset"] = frame.actual_upset & ~picked_favorite
    return frame


def _metric_row(frame: pd.DataFrame, *, scope: str, week: int, name: str,
                family: str, fingerprint: str, kind: str) -> dict:
    if kind == "model":
        total = len(frame)
        valid_margin = frame.model_absolute_margin_error.notna()
        valid_probability = frame.model_brier.notna()
        ats = frame.model_ats_result
        wins, losses, pushes = int(ats.eq("Win").sum()), int(ats.eq("Loss").sum()), int(ats.eq("Push").sum())
        upset = frame.actual_upset
        return {
            "scope": scope, "through_week": week, "series_type": kind,
            "model_name": name, "model_family": family, "fingerprint": fingerprint,
            "games": int(frame.game_id.nunique()), "margin_games": int(valid_margin.sum()),
            "margin_mae": float(frame.loc[valid_margin, "model_absolute_margin_error"].mean()),
            "su_wins": int(frame.model_winner_correct.sum()),
            "su_losses": int(total - frame.model_winner_correct.sum()),
            "su_accuracy": float(frame.model_winner_correct.mean()),
            "brier_games": int(valid_probability.sum()),
            "brier_score": float(frame.loc[valid_probability, "model_brier"].mean()),
            "ats_wins": wins, "ats_losses": losses, "ats_pushes": pushes,
            "ats_accuracy": wins / (wins + losses) if wins + losses else np.nan,
            "upset_games": int(upset.sum()),
            "upset_recall": float(frame.loc[upset, "model_recalled_upset"].mean()) if upset.any() else np.nan,
        }
    raise ValueError(kind)


def _build_trajectory(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for through_week in range(1, 6):
        current = scored.loc[scored.week.le(through_week)].copy()
        for model_name, frame in current.groupby("model_name", sort=True):
            rows.append(_metric_row(
                frame, scope="cumulative", week=through_week, name=model_name,
                family=str(frame.model_family.iloc[0]), fingerprint=str(frame.fingerprint.iloc[0]), kind="model",
            ))

        # Full F0–F8 equal-weight consensus, one forecast per game.
        cons = current.groupby("__game_key", as_index=False).agg(
            game_id=("game_id", "first"), week=("week", "first"),
            home_team=("home_team", "first"), away_team=("away_team", "first"),
            home_points=("home_points", "first"), away_points=("away_points", "first"),
            actual_home_margin=("actual_home_margin", "first"),
            market_spread_close=("market_spread_close", "first"),
            market_win_probability=("market_win_probability", "first"),
            vegas_home_probability=("vegas_home_probability", "first"),
            pred_home_margin=("pred_home_margin", "mean"),
            pred_home_win_probability=("pred_home_win_probability", "mean"),
            actual_favorite_home=("actual_favorite_home", "first"),
            actual_favorite_won=("actual_favorite_won", "first"),
            actual_upset=("actual_upset", "first"),
            actual_cover_team=("actual_cover_team", "first"),
        )
        cons["actual_home_win"] = cons.actual_home_margin.gt(0).astype(float)
        cons["model_absolute_margin_error"] = (cons.pred_home_margin - cons.actual_home_margin).abs()
        cons["model_winner_correct"] = cons.pred_home_win_probability.ge(0.5).eq(cons.actual_home_win.eq(1))
        cons["model_brier"] = (cons.pred_home_win_probability - cons.actual_home_win) ** 2
        cons["model_recalled_upset"] = (
            cons.actual_upset & (cons.pred_home_win_probability.ge(0.5) != cons.actual_favorite_home)
        )
        cons_cover = cons.pred_home_margin + cons.market_spread_close
        cons_pick = np.where(cons_cover.gt(0), cons.home_team, cons.away_team)
        cons["model_ats_result"] = np.select(
            [cons.market_spread_close.isna(), cons.actual_cover_team.eq("Push"), pd.Series(cons_pick, index=cons.index).eq(cons.actual_cover_team)],
            ["No line", "Push", "Win"], default="Loss",
        )
        last_stage = max(17 if str(stage) == "F17-market" else int(str(stage)[1:])
                         for stage in current.fingerprint.unique())
        last_label = "F17-market" if last_stage == 17 else f"F{last_stage}"
        consensus_row = _metric_row(
            cons, scope="cumulative", week=through_week,
            name=f"Full F0–{last_label} scientific what-if consensus",
            family="consensus", fingerprint=last_label, kind="model",
        )
        consensus_row["series_type"] = "consensus"
        rows.append(consensus_row)

        # Vegas reference: spread as margin forecast, market probability as home win probability.
        unique = current.drop_duplicates("__game_key").copy()
        vegas_spread = pd.to_numeric(unique.market_spread_close, errors="coerce")
        valid_spread = vegas_spread.notna()
        vegas_probability = pd.to_numeric(unique.vegas_home_probability, errors="coerce")
        valid_prob = vegas_probability.between(0, 1)
        favorite_home = vegas_spread.lt(0)
        favorite_correct = (favorite_home & unique.actual_home_margin.gt(0)) | (~favorite_home & unique.actual_home_margin.lt(0))
        vegas_rows = {
            "scope": "cumulative", "through_week": through_week, "series_type": "vegas",
            "model_name": "Vegas closing-line baseline", "model_family": "market", "fingerprint": "evaluation-only",
            "games": int(unique.game_id.nunique()), "margin_games": int(valid_spread.sum()),
            "margin_mae": float(((-vegas_spread[valid_spread]) - unique.loc[valid_spread, "actual_home_margin"]).abs().mean()),
            "su_wins": int(favorite_correct[valid_spread].sum()),
            "su_losses": int(valid_spread.sum() - favorite_correct[valid_spread].sum()),
            "su_accuracy": float(favorite_correct[valid_spread].mean()),
            "brier_games": int(valid_prob.sum()),
            "brier_score": float(((vegas_probability[valid_prob] - unique.loc[valid_prob, "actual_home_margin"].gt(0).astype(float)) ** 2).mean()),
            "ats_wins": np.nan, "ats_losses": np.nan, "ats_pushes": np.nan,
            "ats_accuracy": 0.5 if valid_spread.any() else np.nan,
            "upset_games": int(unique.actual_upset.sum()), "upset_recall": 0.0 if unique.actual_upset.any() else np.nan,
        }
        rows.append(vegas_rows)
    return pd.DataFrame(rows)


def main() -> None:
    original = _base_predictions()
    additional = _new_predictions()
    if additional["__game_key"].nunique() != 4 or additional.model_name.nunique() != 54:
        raise ValueError("Expected four Week 1 games and all 54 F0–F8 models.")
    combined = pd.concat([original, additional], ignore_index=True, sort=False)
    if combined.duplicated(["__game_key", "model_name"]).any():
        raise ValueError("Duplicate model/game forecasts after the Week 1 backfill.")
    actual = _completed_games()
    target_schedule = pd.read_parquet(GAMES, columns=["id", "week", "start_date"])
    target_schedule = target_schedule.loc[
        target_schedule.id.astype(int).isin(set(combined.game_id.astype(int)))
    ].drop_duplicates("id")
    if len(target_schedule) != combined.game_id.nunique():
        raise ValueError("The target schedule does not cover every scored game exactly once")
    schedule_week = target_schedule.set_index("id").week
    # Some opening-slate bundles labeled Aug. 29 games Week 0 while CFBD's
    # schedule uses Week 1. Put every fingerprint's forecast on the canonical
    # target schedule week before drawing cumulative curves.
    combined["week"] = combined.game_id.map(schedule_week).astype(int)
    target_market_meta = target_schedule.rename(
        columns={"id": "target_game_id", "start_date": "target_start_utc"}
    )
    if target_market_meta.target_start_utc.isna().any():
        raise ValueError("A scored game lacks a schedule kickoff for market cutoff verification")
    market_state, market_report = build_pregame_market_state(target_market_meta)
    market_lookup = market_state.set_index("target_game_id")
    market_audit = target_market_meta.merge(market_state, on="target_game_id", validate="one_to_one")
    market_audit.to_csv(PUB / "scientific_2026_pregame_market_snapshots.csv", index=False)
    (PUB / "scientific_2026_pregame_market_snapshot_receipt.json").write_text(
        json.dumps(market_report, indent=2, sort_keys=True) + "\n"
    )
    for column in ("market_spread_close", "market_over_under", "market_spread_open", "market_win_probability"):
        combined[column] = pd.to_numeric(combined.game_id.map(market_lookup[column]), errors="coerce")
    combined["market_snapshot_captured_pre_kickoff"] = combined.game_id.map(
        market_lookup.market_snapshot_captured_pre_kickoff
    ).fillna(False)
    combined["market_snapshot_timestamp_utc"] = combined.game_id.map(market_lookup.snapshot_timestamp_utc)
    combined["market_snapshot_source"] = combined.game_id.map(market_lookup.snapshot_source)
    # Existing ATS outcomes may have used another line sidecar; recompute all
    # of them against the same verified pregame market snapshot.
    combined = combined.drop(columns=["model_ats_result"], errors="ignore")
    scored = _score_model_rows(combined, actual)
    backfill_for_publication = additional.drop(columns=["__game_key", "__is_backfill"], errors="ignore").copy()
    forecast_columns = [
        "game_id", "season", "week", "home_team", "away_team", "model_name",
        "model_family", "fingerprint", "pred_home_margin", "pred_home_win_probability",
        "pred_winner", "market_spread_close", "market_spread_open", "market_over_under",
        "market_win_probability", "forecast_source_type", "forecast_archive_timestamp_utc",
        "feature_snapshot_timestamp_utc", "market_quote_timestamp_available",
        "market_snapshot_captured_pre_kickoff", "checkpoint_sha256",
    ]
    backfill_for_publication = backfill_for_publication.reindex(columns=forecast_columns)
    backfill_for_publication.to_csv(PUB / "scientific_2026_week1_missing_forecasts.csv", index=False, float_format="%.8f")
    (PUB / "scientific_2026_week1_reconstruction_receipt.json").write_text(
        BACKFILL_RECEIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    expected_models = 18 * 6
    if scored.game_id.nunique() != 271 or scored.model_name.nunique() != expected_models or len(scored) != 271 * expected_models:
        raise ValueError(f"Unexpected combined cohort: {len(scored)} rows, {scored.game_id.nunique()} games, {scored.model_name.nunique()} models")
    vegas_probabilities = market_state.set_index("target_game_id").market_win_probability
    scored["vegas_home_probability"] = pd.to_numeric(
        scored.game_id.map(vegas_probabilities), errors="coerce"
    )
    scored.to_parquet(ROOT / "data/what_if_2026_fingerprints/2026_f0_f17_market_scored_model_games.parquet", index=False)
    trajectory = _build_trajectory(scored)
    trajectory_path = PUB / "scientific_2026_cumulative_trajectory.csv"
    trajectory.to_csv(trajectory_path, index=False)
    latest = trajectory.loc[trajectory.through_week.eq(5)].copy()
    scorecard = _full_scorecard(latest, season=2026, through_week=5)
    scorecard["upset_games"] = scorecard.model_name.map(latest.set_index("model_name").upset_games)
    scorecard["upset_recall"] = scorecard.model_name.map(latest.set_index("model_name").upset_recall)
    scorecard.to_csv(PUB / "scientific_2026_current_season_scorecard.csv", index=False)
    _plot(trajectory, PUB / "scientific_2026_full_f0_f17_market_cumulative_performance.png",
          season=2026, through_week=5, reconstructed_games=4)
    coverage = {
        "season": 2026, "through_week": 5, "completed_fbs_fbs_games": 271,
        "frozen_published_game_predictions": 267, "week_1_games_reconstructed": 4,
        "scored_model_fingerprint_cells": 108,
        "reconstructed_pregame_model_fingerprint_cells": 12,
        "partial_pregame_market_model_fingerprint_cells": 6,
        "unverified_quote_time_model_fingerprint_cells": 0,
        "pregame_market_games_with_snapshot": market_report["games_with_pregame_market_snapshot"],
        "pregame_market_games_without_snapshot": market_report["games_without_pregame_market_snapshot"],
        "expected_model_fingerprint_cells": 108,
        "scored_fingerprints": [*(f"F{i}" for i in range(17)), "F17-market"],
        "cutoff_checked_fingerprints": [*(f"F{i}" for i in range(17)), "F17-market"],
        "reconstructed_pregame_fingerprints": ["F7", "F8"],
        "partial_pregame_market_fingerprints": ["F17-market"],
        "exploratory_unverified_fingerprints": [], "not_scored_fingerprints": [],
        "backfill_feature_cutoff": "Week 0 rows for Week 1 targets; no 2026 game outcomes in forecast features",
        "market_line_provenance": "Where archived, target market summaries were captured before kickoff and are used consistently for model features, ATS scoring, and Vegas probability reference. Eight Aug. 29 openers have no archived pregame market snapshot and remain missing; provider quote timestamps and raw provider-level detail were not retained. Unavailable quote-level F17 features remain missing for fitted imputation. No refreshed postgame line cache is used.",
        "new_predictions": str(BACKFILL.relative_to(ROOT)),
        "week1_backfill_receipt": "data/what_if_2026_fingerprints/f0_f8_week1_backfill/run_leak_checked/receipt.json",
        "week1_reconstruction_predictions": "scientific_2026_week1_missing_forecasts.csv",
        "week1_reconstruction_receipt": "scientific_2026_week1_reconstruction_receipt.json",
        "pregame_market_snapshots": "scientific_2026_pregame_market_snapshots.csv",
        "pregame_market_snapshot_receipt_file": "scientific_2026_pregame_market_snapshot_receipt.json",
        "F9_predictions": "data/what_if_2026_fingerprints/f09_predictions/predictions.parquet",
        "F10_predictions": "data/what_if_2026_fingerprints/f10_predictions/predictions.parquet",
        "F11_predictions": "data/what_if_2026_fingerprints/f11_predictions/predictions.parquet",
        "F12_predictions": "data/what_if_2026_fingerprints/f12_predictions/predictions.parquet",
        "F13_predictions": "data/what_if_2026_fingerprints/f13_predictions/predictions.parquet",
        "F14_predictions": "data/what_if_2026_fingerprints/f14_predictions/predictions.parquet",
        "F15_predictions": "data/what_if_2026_fingerprints/f15_predictions/predictions.parquet",
        "F16_predictions": "data/what_if_2026_fingerprints/f16_predictions/predictions.parquet",
        "F17_market_predictions": "data/what_if_2026_fingerprints/f17_market_predictions/predictions.parquet",
        "F9_feature_cutoff": "prior regular-game play/drive sources with 48-hour reporting lag; same-game sources excluded",
        "F9_training": "2013-2025, residual calibration from held-out 2024-2025 predictions generated by fits through 2023; M3 seed 1701 only",
        "F10_training": "2013-2025, residual calibration from held-out 2024-2025 predictions generated by fits through 2023; M3 seed 1701 only",
        "F11_training": "2013-2025, prior-staff history uses the previous-season assignment only; residual calibration from held-out 2024-2025 predictions through 2023",
        "F12_training": "2013-2025, corrected F09 + F10 + F11 + F12 A; player sources require a 48-hour reporting lag",
        "F13_F16_training": "2013-2025, same scientific architectures and historical 2024-2025 calibration; current play/actor states use a 48-hour reporting lag; market inputs excluded",
        "F17_market_training": "2013-2025 with historical 2024-2025 calibration; 2026 target market summaries come from archived snapshots captured before kickoff; unavailable raw quote fields remain missing",
        "F17_market_status": "market-assisted what-if; 263 target market snapshots are captured pregame, eight opening-slate games lack archived pregame markets, and provider quote times and quote-level summaries are not archived",
        "pregame_market_snapshot_coverage": market_report,
        "combined_scored_rows": "data/what_if_2026_fingerprints/2026_f0_f17_market_scored_model_games.parquet",
        "trajectory": trajectory_path.name,
        "scorecard": "scientific_2026_current_season_scorecard.csv",
        "figure": "scientific_2026_full_f0_f17_market_cumulative_performance.png",
    }
    (PUB / "coverage_manifest.json").write_text(json.dumps(coverage, indent=2) + "\n")
    (PUB / "README.md").write_text(
        "# 2026 scientific fingerprint performance\n\n"
        "Season-to-date retrospective what-if for F0–F17-market, through Week 5 (271 completed FBS-vs-FBS games). "
        "The original published results covered 267 games. Four missing Week 1 games are reconstructed without target-game outcomes: F0–F6 reuse the original forecasts frozen on August 18; F7–F8 are inferred from the archived September 2 Week 0 fingerprint snapshot with market features included in the model inputs. The input snapshot predates each target kickoff. The 216 reconstructed model/game forecasts and their receipt are included beside this scorecard. "
        "This is isolated from the frozen weekly publication bundles. The F17-market branch remains an exploratory what-if: archived market snapshots cover 263 targets and were captured before kickoff, but eight Aug. 29 openers lack archived pregame market inputs, and provider quote timestamps and quote-level data are unavailable. Those market fields remain missing and are handled by fitted imputation. The complete per-game market snapshot audit and receipt sit beside the scorecard.\n\n"
        "The PNG shows cumulative Brier score, straight-up accuracy, and ATS accuracy. Fingerprint generation maps red to blue; line style maps architecture. Thick pink is the equal-weight F0–F17-market what-if consensus and dashed brass is the Vegas baseline. The adjacent CSV contains model/fingerprint metrics including margin MAE and upset recall. All 108 model/fingerprint cells are scored: 90 standard, 12 reconstructed F7/F8, and six F17-market partial-snapshot cells. Every available F17 target market snapshot was captured before kickoff; provider quote timestamps and quote-level details are not retained.\n\n"
        "F9 corrected-A uses prior-game play/drive statistics with a strict 48-hour availability lag. F10-A adds observed usage, recruiting, experience, and team continuity. F11 adds prior-season staff history without using a 2026 coach assignment. F12 corrected-A adds lagged special-teams execution and player-game PPA. F13 adds context residuals, F14 sequence transitions, F15 actor-role concentration, and F16 short-versus-long trends. F9–F16 features exclude their target-game outcomes. F7/F8 Week 1 inputs are traceable to the September 2 pregame snapshot. F17-market uses archived market summaries captured before kickoff; raw provider quote times and quote-level fields were not retained, and missing market features remain missing for fitted imputation. Models train through 2025 and use held-out 2024–25 residual calibration. No 2024–25 retrospective scores are inserted. Additional CFBD player/PBP sources are preserved under `data/what_if_2026_fingerprints/`.\n"
    )
    print(json.dumps({"scorecard_rows": len(scorecard), "trajectory_rows": len(trajectory),
                      "games": scored.game_id.nunique(), "models": scored.model_name.nunique(),
                      "scorecard": str(PUB / "scientific_2026_current_season_scorecard.csv"),
                      "figure": str(PUB / "scientific_2026_full_f0_f17_market_cumulative_performance.png")}, indent=2))


if __name__ == "__main__":
    main()
