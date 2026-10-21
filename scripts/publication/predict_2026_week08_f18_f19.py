#!/usr/bin/env python3
"""Apply frozen F18/F19 models to verified score-free Week 8 inputs."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import constraint_free_2026_matrix as matrix_builder  # noqa: E402
import constraint_free_freeze as freeze_builder  # noqa: E402
from constraint_free_predict_2026 import predict  # noqa: E402
from gridiron_ml.experiments.constraint_free import SNAPSHOT_MARKET_COLUMNS  # noqa: E402

STATES = ROOT / "data/what_if_2026_fingerprints/week08_expanded_states"
OUT = ROOT / "publication/2026/week_08/pre_game/scientific/full_f0_f19"
FROZEN_MODELS = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/") / (
    "broad_search/final_fit/freeze/freeze_manifest.json"
)
MARGIN = ROOT / "publication/2026/week_08/pre_game/frozen_bundle/public/predictions.parquet"
LADDER = ROOT / "data/publication/2026/weekly_operations/week_08/fingerprint_ladder_v3/canonical_fingerprint.parquet"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path: Path) -> dict:
    return {"path": str(path), "sha256": digest(path)}


def target_schedule() -> pd.DataFrame:
    target = pd.read_parquet(STATES / "target_schedule_without_scores.parquet")
    selected = target.rename(columns={"id": "target_game_id", "start_date": "target_start_utc"})
    selected["target_start_utc"] = pd.to_datetime(selected.target_start_utc, utc=True)
    selected = selected.sort_values("target_game_id").reset_index(drop=True)
    if len(selected) != 53 or not selected.week.eq(8).all() or selected.target_game_id.duplicated().any():
        raise ValueError("Invalid Week 8 score-free expanded target schedule")
    if pd.Timestamp.now(tz="UTC") >= selected.target_start_utc.min():
        raise ValueError("Week 8 first remaining kickoff has passed; new forecasts cannot be issued")
    return selected[["target_game_id", "week", "target_start_utc", "home_team", "away_team"]]


def market_snapshot(games: pd.DataFrame) -> Path:
    columns = ["game_id", "created_at_utc", "game_start_time_utc", "market_spread_close",
               "market_spread_open", "market_over_under", "market_win_probability"]
    frozen = pd.read_parquet(MARGIN, columns=columns)
    for column in columns[1:]:
        if frozen.groupby("game_id")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Frozen market field differs among models: {column}")
    frozen = frozen.drop_duplicates("game_id").rename(columns={"game_id": "target_game_id"})
    market = games[["target_game_id", "target_start_utc"]].merge(
        frozen, on="target_game_id", validate="one_to_one")
    if len(market) != 53 or market.market_spread_close.isna().any():
        raise ValueError("Week 8 frozen market spread coverage incomplete")
    captured = pd.to_datetime(market.created_at_utc, utc=True)
    if not (captured < market.target_start_utc).all():
        raise ValueError("Market snapshot was not frozen before target kickoff")
    result = pd.DataFrame({"target_game_id": market.target_game_id})
    result["market_home_spread"] = market.market_spread_close.astype(float)
    result["market_total"] = market.market_over_under.astype(float)
    result["market_open_spread"] = market.market_spread_open.astype(float)
    result["market_spread_move"] = result.market_home_spread - result.market_open_spread
    result["market_home_implied_no_vig"] = market.market_win_probability.astype(float)
    for column in ("market_open_total", "market_total_move", "market_home_moneyline",
                   "market_away_moneyline", "market_spread_quote_sd", "market_total_quote_sd",
                   "market_quote_count"):
        result[column] = np.nan
    result["snapshot_timestamp_utc"] = captured
    result["market_snapshot_captured_pre_kickoff"] = True
    result["market_quote_timestamp_available"] = False
    if result[list(SNAPSHOT_MARKET_COLUMNS)].isna().any().any():
        raise ValueError("Week 8 market summary has missing F19 feature")
    path = OUT / "f17_f19_market_snapshot.parquet"
    result.to_parquet(path, index=False, compression="zstd")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    states_manifest = json.loads((STATES / "manifest.json").read_text())
    if states_manifest["target_games"] != 53 or not states_manifest["no_target_outcome_columns_read"]:
        raise ValueError("Week 8 expanded source state is not score-free")
    games = target_schedule()
    market_path = market_snapshot(games)
    matrix_builder.LADDER = LADDER
    matrix_builder.STATE_PATHS = {
        "F09": STATES / "f09_target_state.parquet",
        "F10": STATES / "f10_target_state.parquet",
        "F11": STATES / "f11_target_state.parquet",
        "F12": STATES / "f12_target_state.parquet",
        "research": STATES / "research_target_state.parquet",
    }
    matrix_builder.target_schedule = target_schedule
    f18, meta, report = matrix_builder.build()
    if len(f18) != 53 or not f18.target_game_id.equals(meta.target_game_id):
        raise ValueError("Week 8 F18 matrix does not match target schedule")
    market = pd.read_parquet(market_path)
    f19 = f18.merge(market[["target_game_id", *SNAPSHOT_MARKET_COLUMNS]],
                    on="target_game_id", validate="one_to_one")
    historical, _, records, _ = matrix_builder.load_broad_historical("F19")
    if list(f19.columns[1:]) != [record.name for record in records] or f19.shape[1] - 1 != historical.shape[1]:
        raise ValueError("Week 8 F19 columns differ from frozen historical contract")
    f18_path, f19_path, schedule_path = (OUT / "f18_inputs.parquet", OUT / "f19_inputs.parquet",
                                          OUT / "target_schedule_without_scores.parquet")
    f18.to_parquet(f18_path, index=False, compression="zstd")
    f19.to_parquet(f19_path, index=False, compression="zstd")
    meta.to_parquet(schedule_path, index=False, compression="zstd")
    report["no_2026_outcome_columns_read"] = True
    report["f18_inputs_sha256"] = digest(f18_path)
    report["f19_inputs_sha256"] = digest(f19_path)
    report["feature_source_build_manifest_sha256"] = digest(STATES / "manifest.json")
    report["market_snapshot_sha256"] = digest(market_path)
    report_path = OUT / "input_receipt.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    freeze_builder.source_frames = matrix_builder.source_frames
    availability_path = OUT / "feature_source_availability.parquet"
    availability = freeze_builder._documented_source_availability(meta, availability_path)
    frozen = json.loads(FROZEN_MODELS.read_text())
    frozen["target_schedule_without_scores"] = reference(schedule_path)
    frozen["target_input_receipt"] = reference(report_path)
    frozen["target_input_matrices"] = {"F18": reference(f18_path), "F19": reference(f19_path)}
    frozen["feature_source_availability"] = availability
    frozen["market_policy"] = {
        "market_source": str(market_path), "market_source_sha256": digest(market_path),
        "eligible_games": 53, "unavailable_games": 0, "unavailable_game_ids": [],
        "forecast_cutoff": "target_start_utc exclusive",
        "actual_market_as_of": "frozen Week 8 margin prediction created_at_utc",
        "historical_quote_level_timestamp_verified": False,
        "prospective_quote_level_timestamp_available": False,
    }
    frozen["week08_application_created_at_utc"] = datetime.now(timezone.utc).isoformat()
    frozen["week08_original_model_freeze"] = reference(FROZEN_MODELS)
    freeze_path = OUT / "week08_application_manifest.json"
    freeze_path.write_text(json.dumps(frozen, indent=2, sort_keys=True, default=str) + "\n")
    receipt = predict(freeze_path=freeze_path, output=OUT / "f18_f19_frozen_predictions")
    print(json.dumps({"games": len(meta), "f18_features": f18.shape[1] - 1,
                      "f19_features": f19.shape[1] - 1,
                      "prediction_rows": receipt["rows"], "tier_coverage": receipt["tier_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
