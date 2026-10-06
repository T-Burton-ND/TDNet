"""Recover only pre-kickoff 2026 market summaries for the scientific what-if.

The source archives retain weekly market summaries, not raw provider quotes.
Quote-level fields are deliberately left missing instead of being filled from
the refreshed CFBD line cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MARKET_FEATURES = (
    "market_home_spread", "market_total", "market_open_spread",
    "market_open_total", "market_spread_move", "market_total_move",
    "market_home_moneyline", "market_away_moneyline",
    "market_home_implied_no_vig", "market_spread_quote_sd",
    "market_total_quote_sd", "market_quote_count",
)
SUMMARY_COLUMNS = (
    "market_spread_close", "market_over_under", "market_spread_open",
    "market_win_probability",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pregame_market_state(target_meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build the F17 market block from archived snapshots captured pre-kickoff."""
    required = {"target_game_id", "week", "target_start_utc"}
    if required - set(target_meta):
        raise ValueError(f"Target metadata missing {sorted(required - set(target_meta))}")
    targets = target_meta[["target_game_id", "week", "target_start_utc"]].copy()
    targets["target_game_id"] = pd.to_numeric(targets.target_game_id, errors="raise").astype(int)
    targets["week"] = pd.to_numeric(targets.week, errors="raise").astype(int)
    if targets.target_game_id.duplicated().any():
        raise ValueError("Target market request contains duplicate games")

    rows: list[dict] = []
    sources: dict[str, dict] = {}
    for week in range(1, 5):
        source = ROOT / f"data/publication/2026/weekly_operations/week_{week:02d}/fingerprint_ladder_v3/canonical_fingerprint.parquet"
        frame = pd.read_parquet(source, columns=[
            "keys_season", "next_week", "next_game_id", "next_game_is_home",
            "market_over_under", "market_spread_close", "market_spread_open",
            "market_win_probability", "fp_build_timestamp",
        ])
        wanted = set(targets.loc[targets.week.eq(week), "target_game_id"])
        frame = frame.loc[
            pd.to_numeric(frame.keys_season, errors="coerce").eq(2026)
            & pd.to_numeric(frame.next_week, errors="coerce").eq(week)
            & frame.next_game_is_home.fillna(False).astype(bool)
            & pd.to_numeric(frame.next_game_id, errors="coerce").isin(wanted)
        ].copy()
        if frame.next_game_id.duplicated().any():
            raise ValueError(f"Week {week} archived market snapshot has duplicate target games")
        for item in frame.itertuples(index=False):
            rows.append({
                "target_game_id": int(item.next_game_id), "week": week,
                "market_spread_close": item.market_spread_close,
                "market_over_under": item.market_over_under,
                "market_spread_open": item.market_spread_open,
                "market_win_probability": item.market_win_probability,
                "snapshot_timestamp_utc": item.fp_build_timestamp,
                "snapshot_source": str(source.relative_to(ROOT)),
            })
        sources[str(source.relative_to(ROOT))] = {"sha256": _sha256(source), "week": week}

    # Four Week 1 games are absent from the canonical fingerprint file. Their
    # captured market summaries are preserved in the frozen Week 1 package.
    week1_fallback = ROOT / "publication/2026/week_01/pre_game/scientific/scientific_all_game_predictions.csv"
    week1_manifest = ROOT / "publication/2026/week_01/pre_game/frozen_bundle/public/prediction_manifest.json"
    fallback_ids = set(targets.loc[targets.week.eq(1), "target_game_id"]) - {
        int(row["target_game_id"]) for row in rows if row["week"] == 1
    }
    if fallback_ids:
        frozen = pd.read_csv(week1_fallback)
        frozen = frozen.loc[pd.to_numeric(frozen.game_id, errors="coerce").isin(fallback_ids)].copy()
        if frozen.empty or set(frozen.game_id.astype(int)) != fallback_ids:
            raise ValueError("Frozen Week 1 package does not cover missing market snapshot IDs")
        timestamp = json.loads(week1_manifest.read_text())["created_at_utc"]
        for game_id, group in frozen.groupby("game_id", sort=True):
            values = group.market_spread_close.dropna().unique()
            if len(values) != 1:
                raise ValueError(f"Week 1 fallback spread is missing or conflicting for {game_id}")
            rows.append({
                "target_game_id": int(game_id), "week": 1,
                "market_spread_close": float(values[0]),
                "market_over_under": np.nan, "market_spread_open": np.nan,
                "market_win_probability": np.nan,
                "snapshot_timestamp_utc": timestamp,
                "snapshot_source": str(week1_fallback.relative_to(ROOT)),
            })
        sources[str(week1_fallback.relative_to(ROOT))] = {
            "sha256": _sha256(week1_fallback),
            "manifest": str(week1_manifest.relative_to(ROOT)),
            "manifest_sha256": _sha256(week1_manifest),
            "capture_timestamp_utc": timestamp,
            "games": len(fallback_ids),
        }

    week5_source = ROOT / "data/publication/2026/weekly_operations/week_05/scientific/full_f0_f8/scientific_all_game_predictions.csv"
    week5_manifest = ROOT / "data/publication/2026/weekly_operations/week_05/scientific/full_f0_f8/manifest.json"
    week5 = pd.read_csv(week5_source)
    week5 = week5.loc[pd.to_numeric(week5.week, errors="coerce").eq(5)].copy()
    week5 = week5.drop_duplicates("game_id")
    wanted5 = set(targets.loc[targets.week.eq(5), "target_game_id"])
    if set(week5.game_id.astype(int)) != wanted5:
        raise ValueError("Frozen Week 5 scientific package does not cover its 2026 target cohort")
    capture5 = pd.Timestamp(json.loads(week5_manifest.read_text())["created_at_eastern"]).tz_convert("UTC").isoformat()
    for item in week5.itertuples(index=False):
        rows.append({
            "target_game_id": int(item.game_id), "week": 5,
            "market_spread_close": item.market_spread_close,
            "market_over_under": item.market_over_under,
            "market_spread_open": item.market_spread_open,
            "market_win_probability": item.market_win_probability,
            "snapshot_timestamp_utc": capture5,
            "snapshot_source": str(week5_source.relative_to(ROOT)),
        })
    sources[str(week5_source.relative_to(ROOT))] = {
        "sha256": _sha256(week5_source), "manifest": str(week5_manifest.relative_to(ROOT)),
        "manifest_sha256": _sha256(week5_manifest), "capture_timestamp_utc": capture5,
        "games": len(wanted5),
    }

    summary = pd.DataFrame(rows)
    if summary.target_game_id.duplicated().any() or set(summary.target_game_id) != set(targets.target_game_id):
        raise ValueError("Pregame market snapshots do not cover each target game exactly once")
    summary = targets.merge(summary, on=["target_game_id", "week"], validate="one_to_one")
    summary["target_start_utc"] = pd.to_datetime(summary.target_start_utc, utc=True, errors="raise")
    summary["snapshot_timestamp_utc"] = pd.to_datetime(summary.snapshot_timestamp_utc, utc=True, errors="coerce")
    late = summary.snapshot_timestamp_utc.ge(summary.target_start_utc)
    if late.any():
        # Week 1 contains eight Aug 29 openers, but the archived Week 1 state
        # was built Sep 2. Keep those market inputs missing; never use this
        # post-kickoff snapshot for them.
        summary.loc[late, list(SUMMARY_COLUMNS)] = np.nan
        summary.loc[late, "snapshot_timestamp_utc"] = pd.NaT
        summary.loc[late, "snapshot_source"] = "no_pregame_market_snapshot_archived"
    observed = summary[list(SUMMARY_COLUMNS)].notna().any(axis=1)
    if not (summary.loc[observed, "snapshot_timestamp_utc"] < summary.loc[observed, "target_start_utc"]).all():
        raise ValueError("At least one observed market feature lacks a verified pre-kickoff snapshot")

    result = pd.DataFrame({"target_game_id": summary.target_game_id})
    result["market_home_spread"] = pd.to_numeric(summary.market_spread_close, errors="coerce")
    result["market_total"] = pd.to_numeric(summary.market_over_under, errors="coerce")
    result["market_open_spread"] = pd.to_numeric(summary.market_spread_open, errors="coerce")
    # The archives never retained quote-level totals, moneylines, dispersion,
    # or provider counts, so those fields stay missing for fitted imputation.
    result["market_open_total"] = np.nan
    result["market_spread_move"] = result.market_home_spread - result.market_open_spread
    result["market_total_move"] = np.nan
    result["market_home_moneyline"] = np.nan
    result["market_away_moneyline"] = np.nan
    result["market_home_implied_no_vig"] = pd.to_numeric(summary.market_win_probability, errors="coerce")
    result["market_spread_quote_sd"] = np.nan
    result["market_total_quote_sd"] = np.nan
    result["market_quote_count"] = np.nan
    result["market_spread_close"] = result.market_home_spread
    result["market_over_under"] = result.market_total
    result["market_spread_open"] = result.market_open_spread
    result["market_win_probability"] = result.market_home_implied_no_vig
    result["snapshot_timestamp_utc"] = summary.snapshot_timestamp_utc
    result["snapshot_source"] = summary.snapshot_source
    result["market_snapshot_captured_pre_kickoff"] = summary.snapshot_timestamp_utc.notna().to_numpy()
    result["market_quote_timestamp_available"] = False
    for column in MARKET_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    coverage = {column: int(result[column].notna().sum()) for column in MARKET_FEATURES}
    report = {
        "source_type": "archived_pregame_market_summaries",
        "target_games": len(result),
        "all_available_snapshots_pre_kickoff": True,
        "full_cohort_market_snapshot_complete": bool(summary.snapshot_timestamp_utc.notna().all()),
        "games_with_pregame_market_snapshot": int(summary.snapshot_timestamp_utc.notna().sum()),
        "games_without_pregame_market_snapshot": int(summary.snapshot_timestamp_utc.isna().sum()),
        "games_without_pregame_market_snapshot_ids": summary.loc[
            summary.snapshot_timestamp_utc.isna(), "target_game_id"
        ].astype(int).tolist(),
        "market_quote_timestamps_available": False,
        "market_features": list(MARKET_FEATURES),
        "feature_nonmissing_counts": coverage,
        "quote_level_features_unavailable": [
            "market_open_total", "market_total_move", "market_home_moneyline",
            "market_away_moneyline", "market_spread_quote_sd",
            "market_total_quote_sd", "market_quote_count",
        ],
        "missing_market_data_is_left_missing": True,
        "sources": sources,
        "snapshot_timestamp_min_utc": summary.snapshot_timestamp_utc.min().isoformat(),
        "snapshot_timestamp_max_utc": summary.snapshot_timestamp_utc.max().isoformat(),
        "target_kickoff_min_utc": summary.target_start_utc.min().isoformat(),
        "target_kickoff_max_utc": summary.target_start_utc.max().isoformat(),
    }
    return result.sort_values("target_game_id").reset_index(drop=True), report
