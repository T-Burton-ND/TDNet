#!/usr/bin/env python3
"""Grade every available 2026 scientific cell without filling missing F20 weeks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "publication/2026/week_07/post_game/retrospective_replay/cumulative/scientific_f0_f19"
F20_SOURCE = ROOT / "data/private/f20/replay_2026_week01_07_v3_margin/f20_replay.parquet"
OUTPUT = ROOT / "publication/2026/research/scientific_f0_f20_posthoc_scorecards_week01_07"
MODELS = ("M1", "M2", "M3", "M4", "M5", "M10")
FINGERPRINTS = tuple(f"F{i}" for i in range(17)) + ("F17-market", "F18", "F19")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = pd.read_parquet(SOURCE / "common_120_model_game_results.parquet")
    market = pd.read_csv(SOURCE / "common_cohort_vegas_games.csv")
    f20 = pd.read_parquet(F20_SOURCE)
    expected = {(fingerprint, architecture) for fingerprint in FINGERPRINTS for architecture in MODELS}
    if set(zip(model.fingerprint, model.model_id)) != expected:
        raise ValueError("F0–F19 roster differs from the frozen 120 cells")
    if len(model) != 120 * 380 or model.duplicated(["game_id", "fingerprint", "model_id"]).any():
        raise ValueError("F0–F19 is missing a model-game result")
    if model.groupby("game_id").size().ne(120).any() or model.game_id.nunique() != 380:
        raise ValueError("F0–F19 common game cohort is incomplete")
    if market.game_id.duplicated().any() or set(market.game_id) != set(model.game_id):
        raise ValueError("Market and F0–F19 game IDs differ")
    if not np.allclose(model.actual_home_margin, model.game_id.map(market.set_index("game_id").actual_home_margin)):
        raise ValueError("Market and model final scores differ")
    if len(f20) != 320 or f20.game_id.duplicated().any() or set(f20.cfbd_week) != set(range(1, 7)):
        raise ValueError("F20 replay is not the archived 320-game Weeks 1–6 source")
    if f20.actual_home_margin.isna().any():
        raise ValueError("F20 has an unscored game")
    overlap = f20.merge(market, on="game_id", how="inner", validate="one_to_one")
    if not np.allclose(overlap.actual_home_margin_x, overlap.actual_home_margin_y):
        raise ValueError("F20 and archived final margins differ")
    if not overlap.cfbd_week.eq(overlap.week).all():
        raise ValueError("F20 and scientific week numbers differ")
    if not model.pred_home_win_probability.between(0, 1).all() or not np.isfinite(model.pred_home_margin).all():
        raise ValueError("F0–F19 has an invalid prediction")
    for architecture in MODELS:
        if not f20[f"probability_{architecture}"].between(0, 1).all():
            raise ValueError(f"F20/{architecture} has an invalid probability")
        if not np.isfinite(f20[f"margin_{architecture}"]).all():
            raise ValueError(f"F20/{architecture} has an invalid margin")
    return model, market, f20


def grade_model(model: pd.DataFrame, market: pd.DataFrame, *, roster: str,
                status: str) -> pd.DataFrame:
    frame = model.merge(market[["game_id", "market_home_spread"]], on="game_id",
                        how="left", validate="many_to_one")
    actual_home = frame.actual_home_margin.gt(0)
    if not frame.winner_correct.eq(frame.pred_home_win_probability.ge(.5).eq(actual_home)).all():
        raise ValueError("Saved straight-up flags do not match the project probability rule")
    if not np.allclose(frame.margin_error, (frame.pred_home_margin - frame.actual_home_margin).abs()):
        raise ValueError("Saved margin errors do not match predictions")
    if not np.allclose(frame.brier_error, (frame.pred_home_win_probability - actual_home.astype(float)) ** 2):
        raise ValueError("Saved Brier errors do not match predictions")
    frame["su_correct"] = frame.winner_correct.astype(bool)
    frame["log_loss"] = -(actual_home * np.log(frame.pred_home_win_probability.clip(1e-6, 1 - 1e-6))
                          + ~actual_home * np.log1p(-frame.pred_home_win_probability.clip(1e-6, 1 - 1e-6)))
    frame["ats_result"] = "no_line"
    has_line = frame.market_home_spread.notna()
    predicted_cover = frame.pred_home_margin + frame.market_home_spread
    actual_cover = frame.actual_home_margin + frame.market_home_spread
    frame.loc[has_line & predicted_cover.eq(0), "ats_result"] = "no_pick"
    pick = has_line & predicted_cover.ne(0)
    frame.loc[pick & actual_cover.eq(0), "ats_result"] = "P"
    resolved = pick & actual_cover.ne(0)
    frame.loc[resolved & predicted_cover.gt(0).eq(actual_cover.gt(0)), "ats_result"] = "W"
    frame.loc[resolved & predicted_cover.gt(0).ne(actual_cover.gt(0)), "ats_result"] = "L"
    frame = grade_upsets(frame)
    frame["roster"] = roster
    frame["status"] = status
    return frame


def grade_upsets(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.market_home_spread.notna() & frame.market_home_spread.ne(0)
    favorite_home = frame.market_home_spread.lt(0)
    predicted_home = frame.pred_home_win_probability.ge(.5)
    actual_home = frame.actual_home_margin.gt(0)
    frame["upset_eligible"] = eligible
    frame["actual_upset"] = eligible & actual_home.ne(favorite_home)
    frame["upset_call"] = eligible & predicted_home.ne(favorite_home)
    frame["upset_call_correct"] = frame.upset_call & predicted_home.eq(actual_home)
    return frame


def grade_f20(f20: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for architecture in MODELS:
        part = f20[["game_id", "cfbd_week", "actual_home_margin", f"probability_{architecture}",
                    f"margin_{architecture}"]].rename(
            columns={"cfbd_week": "week", f"probability_{architecture}": "pred_home_win_probability",
                     f"margin_{architecture}": "pred_home_margin"})
        part = part.copy()
        part["fingerprint"] = "F20"
        part["model_id"] = architecture
        actual_home = part.actual_home_margin.gt(0)
        part["winner_correct"] = part.pred_home_win_probability.ge(.5).eq(actual_home)
        part["margin_error"] = (part.pred_home_margin - part.actual_home_margin).abs()
        part["brier_error"] = (part.pred_home_win_probability - actual_home.astype(float)) ** 2
        part = grade_model(part, market, roster="F20",
                           status="retrospective_assumed_pregame_week01_06")
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def summarize(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    rows = []
    groups = frame.groupby(["fingerprint", "model_id"], sort=False)
    for (fingerprint, architecture), group in groups:
        ats = group.ats_result.value_counts()
        wins, losses = int(ats.get("W", 0)), int(ats.get("L", 0))
        margin = group.margin_error.dropna()
        upset_calls = int(group.upset_call.sum())
        actual_upsets = int(group.actual_upset.sum())
        correct_upset_calls = int(group.upset_call_correct.sum())
        rows.append({
            "fingerprint": fingerprint, "model_id": architecture, "model": f"{fingerprint}/{architecture}",
            "period": period, "through_week": int(group.week.max()), "games": len(group),
            "su_wins": int(group.su_correct.sum()), "su_losses": int((~group.su_correct).sum()),
            "su_accuracy": float(group.su_correct.mean()),
            "margin_games": len(margin), "margin_mae": float(margin.mean()) if len(margin) else np.nan,
            "brier_games": int(group.brier_error.notna().sum()), "brier": float(group.brier_error.mean()),
            "log_loss": float(group.log_loss.mean()),
            "ats_line_games": int(group.market_home_spread.notna().sum()),
            "ats_wins": wins, "ats_losses": losses, "ats_pushes": int(ats.get("P", 0)),
            "ats_no_pick": int(ats.get("no_pick", 0)), "ats_no_line": int(ats.get("no_line", 0)),
            "ats_accuracy": wins / (wins + losses) if wins + losses else np.nan,
            "upset_eligible_games": int(group.upset_eligible.sum()),
            "actual_upsets": actual_upsets, "upset_calls": upset_calls,
            "correct_upset_calls": correct_upset_calls,
            "upset_recall": correct_upset_calls / actual_upsets if actual_upsets else np.nan,
            "upset_call_accuracy": correct_upset_calls / upset_calls if upset_calls else np.nan,
            "roster": group.roster.iloc[0], "status": group.status.iloc[0],
        })
    return pd.DataFrame(rows).sort_values(["fingerprint", "model_id"]).reset_index(drop=True)


def main() -> None:
    model, market, f20 = load()
    scored = pd.concat([grade_model(model, market, roster="F0–F19", status="retrospective_week01_07"),
                        grade_f20(f20, market)], ignore_index=True)
    weekly = pd.concat([
        summarize(group, "week").assign(week=int(week))
        for week, group in scored.groupby("week", sort=True)
    ], ignore_index=True)
    cumulative = pd.concat([
        summarize(scored.loc[scored.week.le(week)], "cumulative").assign(week=int(week))
        for week in range(1, 8)
    ], ignore_index=True)
    latest = cumulative.loc[(cumulative.fingerprint.ne("F20") & cumulative.week.eq(7))
                            | (cumulative.fingerprint.eq("F20") & cumulative.week.eq(6))].copy()
    # Repeating a Week 6 F20 value under a Week 7 label would imply source coverage that does not exist.
    cumulative = cumulative.loc[~(cumulative.fingerprint.eq("F20") & cumulative.week.eq(7))].copy()
    paired_ids = set(f20.game_id) & set(market.loc[market.market_home_probability.notna(), "game_id"])
    paired_ids &= set(model.game_id)
    if len(paired_ids) != 304:
        raise ValueError(f"Expected 304 F20/shared-market games, got {len(paired_ids)}")
    paired_scored = scored.loc[scored.game_id.isin(paired_ids)]
    paired = summarize(paired_scored, "F20_paired_304_games")
    paired_weekly = pd.concat([
        summarize(group, "F20_paired_week").assign(week=int(week))
        for week, group in paired_scored.groupby("week", sort=True)
    ], ignore_index=True)
    if (len(latest) != 126 or len(paired) != 126 or len(weekly) != 876
            or len(cumulative) != 876 or len(paired_weekly) != 756):
        raise ValueError("Scientific scorecard roster or week coverage is incomplete")
    if latest.loc[latest.fingerprint.eq("F20"), ["margin_mae", "ats_accuracy"]].isna().any().any():
        raise ValueError("F20 margin and ATS grades are missing")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = {"weekly_scorecard.csv": weekly, "cumulative_scorecard.csv": cumulative,
             "latest_scorecard.csv": latest, "f20_paired_304_game_scorecard.csv": paired,
             "f20_paired_weekly_scorecard.csv": paired_weekly}
    for name, data in files.items():
        data.to_csv(OUTPUT / name, index=False, float_format="%.12g")
    manifest = {
        "scope": "2026 retrospective F0–F20 scientific M1–M5 and M10 scorecards",
        "f0_f19_games_per_cell": 380, "f20_games_per_cell": 320,
        "f20_paired_game_count": 304, "latest_f0_f19_week": 7, "latest_f20_week": 6,
        "f20_margin_and_ats": "trained_margin_regressors_and_available_market_spreads",
        "week_7_market_quote_timing_verified": False,
        "f20_original_expert_publication_timestamps_verified": False,
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in (
            SOURCE / "common_120_model_game_results.parquet", SOURCE / "common_cohort_vegas_games.csv", F20_SOURCE)},
        "outputs": {name: sha256(OUTPUT / name) for name in files},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print({"cells": len(latest), "weekly_rows": len(weekly), "cumulative_rows": len(cumulative),
           "paired_games": len(paired_ids)})


if __name__ == "__main__":
    main()
