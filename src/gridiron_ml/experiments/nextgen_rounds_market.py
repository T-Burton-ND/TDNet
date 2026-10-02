"""Retrospective market inputs for the separate F17 research comparator.

The archived CFBD line records have no quote timestamp. These values can be
used to measure a historical market-assisted model, but they cannot certify a
point-in-time operational forecast. F06–F16 never import this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MARKET_COLUMNS = (
    "market_home_spread", "market_total", "market_open_spread",
    "market_open_total", "market_spread_move", "market_total_move",
    "market_home_moneyline", "market_away_moneyline",
    "market_home_implied_no_vig", "market_spread_quote_sd",
    "market_total_quote_sd", "market_quote_count",
)


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _median(values):
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else np.nan


def _american_probability(odds):
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (100.0 - odds)


def summarize_quotes(quotes) -> dict[str, float]:
    """Select consensus when available and summarize all numeric quotes.

    Provider order has no statistical meaning. An opener or moneyline is
    missing unless the chosen provider supplies it; mixed-provider synthetic
    line movement and synthetic no-vig pairs are forbidden.
    """
    rows = [dict(q) for q in (quotes if quotes is not None else [])]
    rows.sort(key=lambda q: (str(q.get("provider", "")), json.dumps(q, sort_keys=True, default=str)))
    spreads = [_number(q.get("spread")) for q in rows]
    totals = [_number(q.get("overUnder")) for q in rows]
    consensus = [q for q in rows if str(q.get("provider", "")).lower() == "consensus"
                 and np.isfinite(_number(q.get("spread")))]
    chosen = consensus[0] if consensus else None
    # Prefer a provider with a complete opener/current pair for movement.
    movement = next((q for q in rows if np.isfinite(_number(q.get("spread")))
                     and np.isfinite(_number(q.get("spreadOpen")))), None)
    total_movement = next((q for q in rows if np.isfinite(_number(q.get("overUnder")))
                           and np.isfinite(_number(q.get("overUnderOpen")))), None)
    money = next((q for q in rows if np.isfinite(_number(q.get("homeMoneyline")))
                  and np.isfinite(_number(q.get("awayMoneyline")))), None)
    home_odds = _number(money.get("homeMoneyline")) if money else np.nan
    away_odds = _number(money.get("awayMoneyline")) if money else np.nan
    ph, pa = _american_probability(home_odds), _american_probability(away_odds)
    spread = _number(chosen.get("spread")) if chosen else _median(spreads)
    total = _number(chosen.get("overUnder")) if chosen else np.nan
    if not np.isfinite(total):
        total = _median(totals)
    open_spread = _number(movement.get("spreadOpen")) if movement else np.nan
    open_total = _number(total_movement.get("overUnderOpen")) if total_movement else np.nan
    finite_spreads = [x for x in spreads if np.isfinite(x)]
    finite_totals = [x for x in totals if np.isfinite(x)]
    return {
        "market_home_spread": spread,
        "market_total": total,
        "market_open_spread": open_spread,
        "market_open_total": open_total,
        "market_spread_move": (_number(movement.get("spread")) - open_spread
                               if movement else np.nan),
        "market_total_move": (_number(total_movement.get("overUnder")) - open_total
                              if total_movement else np.nan),
        "market_home_moneyline": home_odds,
        "market_away_moneyline": away_odds,
        "market_home_implied_no_vig": ph / (ph + pa) if np.isfinite(ph + pa) and ph + pa > 0 else np.nan,
        "market_spread_quote_sd": float(np.std(finite_spreads)) if len(finite_spreads) >= 2 else np.nan,
        "market_total_quote_sd": float(np.std(finite_totals)) if len(finite_totals) >= 2 else np.nan,
        "market_quote_count": float(len(rows)) if rows else np.nan,
    }


def load_market_archive(ledger_snapshot: Path, eligible_game_ids: set[int]):
    """Return one market row per eligible game and source-provenance report."""
    snapshot = Path(ledger_snapshot)
    records = json.loads(snapshot.read_text())
    lines = [r for r in records if r.get("endpoint") == "/lines"
             and r.get("status") in {"success_complete", "skipped_existing_complete"}
             and r.get("year") is not None and 2013 <= int(r["year"]) <= 2025]
    sources, rows = [], []
    for item in sorted(lines, key=lambda r: int(r["year"])):
        path = Path(item["cache_path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item.get("sha256"):
            raise ValueError(f"Market cache hash mismatch: {path}")
        frame = pd.read_parquet(path)
        frame = frame.loc[frame.id.isin(eligible_game_ids) & frame.season_type.eq("regular")]
        for game in frame.itertuples(index=False):
            rows.append({"target_game_id": int(game.id), "season": int(game.season),
                         **summarize_quotes(game.lines)})
        sources.append({"year": int(item["year"]), "request_id": item["request_id"],
                        "sha256": digest, "eligible_games": len(frame)})
    result = pd.DataFrame(rows)
    if result.empty or result.target_game_id.duplicated().any():
        raise ValueError("Missing or duplicate market archive identity")
    if set(result.target_game_id) - eligible_game_ids:
        raise ValueError("Market record outside eligible game scope")
    coverage = result.groupby("season")[list(MARKET_COLUMNS)].count().to_dict(orient="index")
    report = {"scope": "retrospective_F17_only", "quote_timestamp_available": False,
              "operational_point_in_time_claim_allowed": False,
              "ledger_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              "source_files": sources, "eligible_games": len(eligible_game_ids),
              "market_games": len(result), "coverage_by_season": coverage}
    return result.sort_values("target_game_id").reset_index(drop=True), report
