#!/usr/bin/env python3
"""Reconstruct the four missing 2026 Week 1 scientific forecasts leak-free.

F0–F6 are restored from the original August 18 pregame prediction archive.
F7–F8 are re-inferred from the September 2 frozen Week 0 fingerprint rows;
market features are explicitly included in the feature matrix, matching each
checkpoint's training feature contract. The October 6 refreshed market cache
is never read.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.experiments.opponent_adjusted import (
    StaticFrameFingerprints,
)
from gridiron_ml.fingerprints.features import FeatureSpec
from gridiron_ml.models import load_model_checkpoint
from gridiron_ml.td_run.matchups import MatchupBuilder

DATA = ROOT / "data"
INPUTS = DATA / "publication/2026/weekly_operations/week_01"
SCHEDULE_PATH = DATA / "what_if_2026_fingerprints/f0_f8_week1_backfill/missing_game_schedule.parquet"
FINGERPRINT_PATH = INPUTS / "fingerprint_ladder_v3/canonical_fingerprint.parquet"
INVENTORY_PATH = INPUTS / "scientific_runtime_inventory.csv"
ORIGINAL_PREDICTIONS = DATA / "publication/2026/legacy_preseason_layout/scientific_market_free/week_01_predictions/tables/all_game_model_predictions.csv"
ORIGINAL_MANIFEST = DATA / "publication/2026/legacy_preseason_layout/scientific_market_free/week_01_predictions/metadata/report_manifest.json"
PRIOR_BACKFILL = DATA / "what_if_2026_fingerprints/f0_f8_week1_backfill/run_all54/tables/all_game_model_predictions.parquet"
OUTPUT_DIR = DATA / "what_if_2026_fingerprints/f0_f8_week1_backfill/run_leak_checked"
OUTPUT_PATH = OUTPUT_DIR / "tables/all_game_model_predictions.parquet"
MARKET_COLUMNS = [
    "market_over_under",
    "market_spread_close",
    "market_spread_open",
    "market_win_probability",
]
TARGET_IDS = {401858423, 401858204, 401856776, 401858424}
MODEL_IDS = ("M1", "M2", "M3", "M4", "M5", "M10")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    schedule = pd.read_parquet(SCHEDULE_PATH)
    schedule = schedule.loc[schedule.id.astype(int).isin(TARGET_IDS)].copy()
    if set(schedule.id.astype(int)) != TARGET_IDS or schedule.id.duplicated().any():
        raise ValueError("The four-game Week 1 target schedule is incomplete or duplicated")
    if not schedule.home_classification.astype(str).str.lower().eq("fbs").all() or not schedule.away_classification.astype(str).str.lower().eq("fbs").all():
        raise ValueError("All reconstruction targets must be FBS-vs-FBS")

    # Only static schedule identity and kickoff are used from this postgame file.
    # Its score and postgame fields are deliberately not passed to model inference.
    schedule = schedule[["id", "season", "week", "start_date", "home_team", "away_team", "neutral_site", "conference_game", "season_type"]].copy()
    schedule["id"] = schedule.id.astype(int)
    schedule["game_start_time_utc"] = pd.to_datetime(schedule.start_date, utc=True, errors="raise")

    fingerprints = pd.read_parquet(FINGERPRINT_PATH)
    target_rows = fingerprints.loc[
        pd.to_numeric(fingerprints.keys_season, errors="coerce").eq(2026)
        & pd.to_numeric(fingerprints.keys_week, errors="coerce").eq(0)
        & pd.to_numeric(fingerprints.next_week, errors="coerce").eq(1)
        & pd.to_numeric(fingerprints.next_game_id, errors="coerce").isin(TARGET_IDS)
    ].copy()
    if len(target_rows) != 8 or target_rows.next_game_id.nunique() != 4:
        raise ValueError("Expected exactly two archived Week 0 team states per target game")
    if target_rows.duplicated(["next_game_id", "keys_team"]).any():
        raise ValueError("Duplicate team state for a target game")
    if target_rows[["y_margin_this_week", "y_next_margin"]].notna().any().any():
        raise ValueError("A target-period score label appears in a reconstruction feature row")
    if not target_rows["fp_build_timestamp"].notna().all():
        raise ValueError("A target fingerprint lacks its saved build timestamp")
    target_rows["fp_build_timestamp"] = pd.to_datetime(target_rows.fp_build_timestamp, utc=True, errors="raise")
    kickoff = schedule.set_index("id").game_start_time_utc
    target_rows["target_kickoff_utc"] = pd.to_numeric(target_rows.next_game_id).astype(int).map(kickoff)
    if not (target_rows.fp_build_timestamp < target_rows.target_kickoff_utc).all():
        raise ValueError("A target fingerprint snapshot was built after kickoff")
    for row in schedule.itertuples(index=False):
        pair = target_rows.loc[target_rows.next_game_id.eq(float(row.id))]
        actual_pair = set(pair.keys_team.astype(str))
        if actual_pair != {str(row.home_team), str(row.away_team)}:
            raise ValueError(f"Fingerprint team pair disagrees with schedule for {row.id}")
        home = pair.loc[pair.next_game_is_home.astype(bool)]
        away = pair.loc[~pair.next_game_is_home.astype(bool)]
        if len(home) != 1 or len(away) != 1:
            raise ValueError(f"Home/away identity is ambiguous for game {row.id}")
        if home[MARKET_COLUMNS].isna().any().any():
            raise ValueError(f"Pregame market inputs are incomplete for game {row.id}")
        if not np.allclose(home.market_spread_close.to_numpy(float), -away.market_spread_close.to_numpy(float)):
            raise ValueError(f"Home/away spread signs disagree for game {row.id}")
        if not np.allclose(home.market_win_probability.to_numpy(float), 1.0 - away.market_win_probability.to_numpy(float)):
            raise ValueError(f"Home/away win probabilities disagree for game {row.id}")

    input_spec = FeatureSpec(include_market=True, allow_market_features_for_training=True)
    X, _, meta, _ = StaticFrameFingerprints(target_rows).split_frame(target_rows, input_spec)
    full_inventory = pd.read_csv(INVENTORY_PATH)
    inventory = full_inventory.loc[full_inventory.fingerprint.isin(["F7", "F8"])].copy()
    if len(inventory) != 12 or inventory.duplicated(["fingerprint", "model_level"]).any():
        raise ValueError("F7/F8 checkpoint inventory must contain six models per generation")

    inferred: list[dict] = []
    builder = MatchupBuilder(representation="unit_matchup", safe_math=True)
    for generation in ("F7", "F8"):
        feature_block = X[MARKET_COLUMNS] if generation == "F7" else X
        matchup_X, matchup_meta, _ = builder.matchups(feature_block, meta)
        if len(matchup_X) != 4 or matchup_meta.next_game_id.astype(int).nunique() != 4:
            raise ValueError(f"{generation} inference did not produce four unique games")
        for row in inventory.loc[inventory.fingerprint.eq(generation)].itertuples(index=False):
            checkpoint = Path(str(row.checkpoint_path))
            if sha256(checkpoint) != row.checkpoint_sha256:
                raise ValueError(f"Checkpoint hash mismatch for {row.final_model_name}")
            model = load_model_checkpoint(checkpoint)
            expected = list(model.feature_names_)
            if list(matchup_X.columns) != expected:
                raise ValueError(
                    f"{generation}/{row.model_level} input contract mismatch: "
                    f"{len(matchup_X.columns)} built columns vs {len(expected)} checkpoint columns"
                )
            prediction = model.predict(matchup_X)
            if len(prediction) != 4 or not np.isfinite(prediction[["pred_margin", "pred_proba_home_win"]].to_numpy(float)).all():
                raise ValueError(f"Non-finite or incomplete predictions for {row.final_model_name}")
            for i, game in matchup_meta.reset_index(drop=True).iterrows():
                inferred.append({
                    "game_id": int(game.next_game_id),
                    "model_name": row.final_model_name,
                    "fingerprint": generation,
                    "pred_home_margin": float(prediction.pred_margin.iloc[i]),
                    "pred_home_win_probability": float(prediction.pred_proba_home_win.iloc[i]),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": row.checkpoint_sha256,
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "forecast_source_type": "frozen_checkpoint_on_archived_pregame_fingerprint",
                    "feature_snapshot_timestamp_utc": target_rows.fp_build_timestamp.min().isoformat(),
                    "market_quote_timestamp_available": False,
                    "market_snapshot_captured_pre_kickoff": True,
                })

    inferred_df = pd.DataFrame(inferred)
    original = pd.read_csv(ORIGINAL_PREDICTIONS)
    original = original.loc[original.game_id.astype(int).isin(TARGET_IDS)].copy()
    original = original.loc[original.fingerprint.isin([f"F{i}" for i in range(7)])].copy()
    if len(original) != 168 or original.duplicated(["game_id", "model_name"]).any():
        raise ValueError("Archived original forecasts must provide 24 rows for each of F0–F6")
    checkpoint_hashes = full_inventory.set_index("final_model_name").checkpoint_sha256.astype(str)
    original_hashes = original.model_name.map(checkpoint_hashes)
    if original_hashes.isna().any() or not original.checkpoint_sha256.astype(str).eq(original_hashes).all():
        raise ValueError("Archived F0–F6 forecast checkpoints differ from the Week 1 scientific roster")

    prior = pd.read_parquet(PRIOR_BACKFILL)
    rows = []
    for base in prior.to_dict("records"):
        game_id = int(base["game_id"])
        generation = str(base["fingerprint"])
        model_name = str(base["model_name"])
        if generation in {f"F{i}" for i in range(7)}:
            source = original.loc[original.game_id.astype(int).eq(game_id) & original.model_name.eq(model_name)]
            if len(source) != 1:
                raise ValueError(f"Missing archived pregame prediction for {game_id} {model_name}")
            source = source.iloc[0]
            base.update({
                "pred_home_margin": float(source.pred_home_margin),
                "pred_home_win_probability": float(source.pred_home_win_probability),
                "pred_winner": str(source.pred_winner),
                "confidence": float(source.confidence),
                "checkpoint_path": str(source.checkpoint_path),
                "checkpoint_sha256": str(source.checkpoint_sha256),
                "created_at_utc": str(source.created_at_utc),
                "forecast_source_type": "archived_original_pregame_prediction",
                "forecast_archive_timestamp_utc": str(source.created_at_utc),
                "feature_snapshot_timestamp_utc": None,
                "market_quote_timestamp_available": False,
                "market_snapshot_captured_pre_kickoff": True,
            })
        else:
            source = inferred_df.loc[
                inferred_df.game_id.eq(game_id)
                & inferred_df.fingerprint.eq(generation)
                & inferred_df.model_name.eq(model_name)
            ]
            if len(source) != 1:
                raise ValueError(f"Missing rebuilt prediction for {game_id} {model_name}")
            source = source.iloc[0]
            base.update(source.to_dict())
            base["pred_winner"] = base["home_team"] if base["pred_home_win_probability"] >= 0.5 else base["away_team"]
            base["confidence"] = abs(base["pred_home_win_probability"] - 0.5) * 2.0

        # Use the saved pregame market snapshot for evaluation and ATS; never
        # substitute the refreshed postgame CFBD line cache.
        game_rows = target_rows.loc[pd.to_numeric(target_rows.next_game_id).eq(game_id)]
        home = game_rows.loc[game_rows.next_game_is_home.astype(bool)].iloc[0]
        away = game_rows.loc[~game_rows.next_game_is_home.astype(bool)].iloc[0]
        base["market_spread_close"] = float(home.market_spread_close)
        base["market_spread_open"] = float(home.market_spread_open)
        base["market_over_under"] = float(home.market_over_under)
        base["market_win_probability"] = float(home.market_win_probability)
        base["market_spread_close_schedule"] = float(home.market_spread_close)
        base["market_spread_open_schedule"] = float(home.market_spread_open)
        base["market_over_under_schedule"] = float(home.market_over_under)
        base["market_win_probability_schedule"] = float(home.market_win_probability)
        rows.append(base)

    result = pd.DataFrame(rows).sort_values(["game_id", "fingerprint", "model_name"]).reset_index(drop=True)
    if len(result) != 216 or result.duplicated(["game_id", "model_name"]).any():
        raise ValueError("Rebuilt output must contain exactly 54 model forecasts for each target game")
    if result.groupby("fingerprint").size().to_dict() != {f"F{i}": 24 for i in range(9)}:
        raise ValueError("Rebuilt output does not have complete F0–F8 coverage")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False, compression="zstd")
    receipt = {
        "status": "success_leak_checked_week1_reconstruction",
        "season": 2026,
        "week": 1,
        "games": 4,
        "models": 54,
        "forecast_rows": len(result),
        "forecast_sources": {
            "F0-F6": "archived original Week 1 forecasts created 2026-08-18 before target kickoffs",
            "F7-F8": "frozen through-2025 checkpoints run on the 2026-09-02 Week 0 fingerprint snapshot",
        },
        "feature_snapshot_timestamp_utc": target_rows.fp_build_timestamp.min().isoformat(),
        "latest_feature_snapshot_timestamp_utc": target_rows.fp_build_timestamp.max().isoformat(),
        "target_games_kickoff_utc": schedule.set_index("id").game_start_time_utc.astype(str).to_dict(),
        "market_quote_timestamp_available": False,
        "market_snapshot_capture_precedes_every_target_kickoff": True,
        "postgame_refreshed_market_cache_used": False,
        "target_score_columns_used_in_feature_build": False,
        "target_score_labels_present_in_feature_rows": False,
        "f7_f8_market_features_included": True,
        "f0_f6_original_forecasts_reused": True,
        "inputs": {
            "fingerprints_path": str(FINGERPRINT_PATH.relative_to(ROOT)),
            "fingerprints_sha256": sha256(FINGERPRINT_PATH),
            "inventory_path": str(INVENTORY_PATH.relative_to(ROOT)),
            "inventory_sha256": sha256(INVENTORY_PATH),
            "original_predictions_path": str(ORIGINAL_PREDICTIONS.relative_to(ROOT)),
            "original_predictions_sha256": sha256(ORIGINAL_PREDICTIONS),
            "original_manifest_path": str(ORIGINAL_MANIFEST.relative_to(ROOT)),
            "original_manifest_sha256": sha256(ORIGINAL_MANIFEST),
        },
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT_PATH),
        "prediction_counts_by_fingerprint": result.groupby("fingerprint").size().to_dict(),
    }
    (OUTPUT_DIR / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
