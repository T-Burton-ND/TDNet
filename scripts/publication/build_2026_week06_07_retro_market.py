#!/usr/bin/env python3
"""Bind Week 6 archived and Week 7 ex-post market summaries for a replay.

Week 7 quotes are from an October 21 cache, after the target games. They are
explicitly unsuitable as a pregame market benchmark or leakage-free forecast.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATES = ROOT / "data/what_if_2026_fingerprints/week06_07_retrospective_states"
OUT = ROOT / "publication/2026/week_07/post_game/retrospective_replay/scientific/full_f0_f19"
FROZEN = ROOT / "publication/2026/week_06/pre_game/frozen_bundle/public/predictions.parquet"
POSTGAME = ROOT / "data/raw/cfbd/v2/lines/2026.parquet"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implied(odds: float) -> float:
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = pd.read_parquet(STATES / "target_schedule_without_scores.parquet", columns=["id", "week"])
    if targets.groupby("week").size().to_dict() != {6: 58, 7: 59}:
        raise ValueError("Retrospective target cohort changed")
    frozen = pd.read_parquet(FROZEN, columns=["game_id", "created_at_utc", "market_spread_close",
                                                "market_spread_open", "market_over_under", "market_win_probability"])
    frozen = frozen.drop_duplicates("game_id")
    if len(frozen) != 58 or set(frozen.game_id) != set(targets.loc[targets.week.eq(6), "id"]):
        raise ValueError("Week 6 frozen market cohort changed")
    rows = []
    for row in frozen.itertuples(index=False):
        rows.append({
            "target_game_id": int(row.game_id), "market_home_spread": row.market_spread_close,
            "market_total": row.market_over_under, "market_open_spread": row.market_spread_open,
            "market_open_total": np.nan, "market_spread_move": row.market_spread_close - row.market_spread_open,
            "market_total_move": np.nan, "market_home_moneyline": np.nan,
            "market_away_moneyline": np.nan, "market_home_implied_no_vig": row.market_win_probability,
            "market_spread_quote_sd": np.nan, "market_total_quote_sd": np.nan,
            "market_quote_count": np.nan, "market_provenance": "frozen_pregame_summary",
            "snapshot_timestamp_utc": row.created_at_utc,
        })
    raw = pd.read_parquet(POSTGAME, columns=["id", "week", "lines"])
    raw = raw.loc[raw.id.isin(targets.id)].copy()
    if len(raw) != 117:
        raise ValueError("Ex-post Week 7 market cache lacks target games")
    for item in raw.itertuples(index=False):
        if item.week != 7:
            continue
        quotes = list(item.lines)
        def number(quote: dict, name: str) -> float:
            value = quote.get(name)
            return float(value) if value is not None else np.nan
        def median(name: str) -> float:
            values = np.array([number(q, name) for q in quotes], float)
            return float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else np.nan
        home, away = median("homeMoneyline"), median("awayMoneyline")
        ph, pa = implied(home), implied(away)
        prob = ph / (ph + pa) if np.isfinite(ph + pa) and ph + pa > 0 else np.nan
        spread, opening = median("spread"), median("spreadOpen")
        total, open_total = median("overUnder"), median("overUnderOpen")
        spread_quotes = np.array([number(q, "spread") for q in quotes], float)
        total_quotes = np.array([number(q, "overUnder") for q in quotes], float)
        rows.append({
            "target_game_id": int(item.id), "market_home_spread": spread,
            "market_total": total, "market_open_spread": opening,
            "market_open_total": open_total, "market_spread_move": spread - opening,
            "market_total_move": total - open_total, "market_home_moneyline": home,
            "market_away_moneyline": away, "market_home_implied_no_vig": prob,
            "market_spread_quote_sd": float(np.nanstd(spread_quotes)) if np.isfinite(spread_quotes).any() else np.nan,
            "market_total_quote_sd": float(np.nanstd(total_quotes)) if np.isfinite(total_quotes).any() else np.nan,
            "market_quote_count": len(quotes), "market_provenance": "ex_post_oct21_cache_unverified_quote_time",
            "snapshot_timestamp_utc": pd.NaT,
        })
    market = pd.DataFrame(rows).sort_values("target_game_id")
    if len(market) != 117 or market.target_game_id.duplicated().any():
        raise ValueError("Week 6–7 market rows are incomplete")
    # One Week 6 frozen game lacked a line. Supply it only as an explicitly
    # after-game research input so the full F17/F19 roster can be inspected.
    missing6 = market.loc[market.market_provenance.eq("frozen_pregame_summary")
                          & market.market_home_spread.isna(), "target_game_id"]
    for game_id in missing6:
        quotes = list(raw.loc[raw.id.eq(game_id), "lines"].iloc[0])
        if not quotes:
            raise ValueError(f"No retrospective line for Week 6 game {game_id}")
        q = quotes[0]
        loc = market.target_game_id.eq(game_id)
        spread, opening = float(q["spread"]), float(q["spreadOpen"])
        ph, pa = implied(float(q["homeMoneyline"])), implied(float(q["awayMoneyline"]))
        market.loc[loc, "market_home_spread"] = spread
        market.loc[loc, "market_open_spread"] = opening
        market.loc[loc, "market_total"] = float(q["overUnder"])
        market.loc[loc, "market_spread_move"] = spread - opening
        market.loc[loc, "market_home_implied_no_vig"] = ph / (ph + pa)
        market.loc[loc, "market_provenance"] = "week06_ex_post_market_fallback"
    path = OUT / "f17_f19_market_snapshot.parquet"
    market.to_parquet(path, index=False, compression="zstd")
    (OUT / "market_source_receipt.json").write_text(json.dumps({
        "week06_frozen_summary_sha256": digest(FROZEN),
        "week07_ex_post_cache_sha256": digest(POSTGAME),
        "week06_ex_post_fallback_game_ids": missing6.astype(int).tolist(),
        "week07_quote_timing_verified": False,
        "week07_pregame_market_claim_allowed": False,
        "week07_market_features_use_postgame_cache": True,
        "market_output_sha256": digest(path),
    }, indent=2, sort_keys=True) + "\n")
    print(market.groupby("market_provenance").size().to_dict(), flush=True)


if __name__ == "__main__":
    main()
