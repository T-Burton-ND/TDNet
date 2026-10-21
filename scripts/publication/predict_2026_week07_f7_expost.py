#!/usr/bin/env python3
"""Apply frozen F7 models to Week 7 market inputs retrieved after the games."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_ml.models import load_model_checkpoint

ROOT = Path(__file__).resolve().parents[2]
MARKET = ROOT / "publication/2026/week_07/post_game/retrospective_replay/scientific/full_f0_f19/f17_f19_market_snapshot.parquet"
TARGETS = ROOT / "data/what_if_2026_fingerprints/week06_07_retrospective_states/target_schedule_without_scores.parquet"
MODEL_ROOT = Path("/groups/bsavoie2/tburton2/TDNet/publication_artifacts/scientific_roster_refits/f0_f8_margin_through_2025_v1/cells/F7")
OUT = MARKET.parent / "f7_expost_predictions.parquet"
ROSTER = ("M1", "M2", "M3", "M4", "M5", "M10")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    market = pd.read_parquet(MARKET)
    market = market.loc[market.market_provenance.eq("ex_post_oct21_cache_unverified_quote_time")].copy()
    games = pd.read_parquet(TARGETS, columns=["id", "week", "home_team", "away_team"])
    games = games.loc[games.week.eq(7)].rename(columns={"id": "target_game_id"})
    games = games.merge(market, on="target_game_id", validate="one_to_one")
    if len(games) != 59:
        raise ValueError("Week 7 market cohort incomplete")
    features = pd.DataFrame(index=games.index)
    for side in ("home", "away"):
        sign = 1 if side == "home" else -1
        features[f"{side}_market_over_under"] = games.market_total
        features[f"{side}_market_spread_close"] = sign * games.market_home_spread
        features[f"{side}_market_spread_open"] = sign * games.market_open_spread
        features[f"{side}_market_win_probability"] = (
            games.market_home_implied_no_vig if side == "home" else 1 - games.market_home_implied_no_vig)
    for name in ("market_over_under", "market_spread_close", "market_spread_open", "market_win_probability"):
        features[f"net_{name}"] = features[f"home_{name}"] - features[f"away_{name}"]
    rows, hashes = [], {}
    for model_id in ROSTER:
        path = MODEL_ROOT / model_id / "checkpoint.pkl"
        model = load_model_checkpoint(path)
        if list(features.columns) != list(model.feature_names_):
            raise ValueError(f"F7 feature contract changed: {model_id}")
        pred = model.predict(features)
        if len(pred) != 59 or not np.isfinite(pred.pred_margin).all():
            raise ValueError(f"Invalid Week 7 F7 reconstruction: {model_id}")
        item = games[["target_game_id", "week", "home_team", "away_team"]].rename(columns={"target_game_id": "game_id"}).copy()
        item["fingerprint"] = "F7"
        item["model_id"] = model_id
        item["model_name"] = f"scientific_F7_{model_id}"
        item["pred_home_margin"] = pred.pred_margin.to_numpy(float)
        item["pred_home_win_probability"] = pred.pred_proba_home_win.to_numpy(float)
        item["provenance_class"] = "retrospective_ex_post_market_after_games"
        rows.append(item)
        hashes[str(path)] = digest(path)
    result = pd.concat(rows, ignore_index=True)
    if len(result) != 354 or result.duplicated(["game_id", "model_id"]).any():
        raise ValueError("Incomplete Week 7 F7 roster")
    result.to_parquet(OUT, index=False, compression="zstd")
    (OUT.parent / "f7_expost_receipt.json").write_text(json.dumps({
        "models": len(ROSTER), "games": 59, "checkpoint_sha256": hashes,
        "market_sha256": digest(MARKET), "predictions_sha256": digest(OUT),
        "market_inputs_retrieved_after_games": True,
        "pregame_forecast_claim_allowed": False,
    }, indent=2, sort_keys=True) + "\n")
    print({"models": len(ROSTER), "games": 59})


if __name__ == "__main__":
    main()
