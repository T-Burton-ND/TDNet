#!/usr/bin/env python3
"""Score the immutable F18/F19 forecast export after its freeze and prediction.

This is the first F18/F19 program that reads 2026 outcomes. It refuses to run
without a hash-verified freeze and score-free prediction receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from constraint_free_predict_2026 import DEFAULT_OUTPUT as PREDICTION_DIR, FREEZE
from constraint_free_search import digest, write_json
from gridiron_ml.experiments.constraint_free import ROSTER

REPO = Path(__file__).resolve().parents[1]
GAMES = REPO / "data/raw/cfbd/v2/games/2026.parquet"
MARKET = REPO / "data/what_if_2026_fingerprints/f17_market_features/f17_market_target_state_2026.parquet"
PRIOR = REPO / "data/what_if_2026_fingerprints/2026_f0_f17_market_scored_model_games.parquet"
DEFAULT_OUTPUT = PREDICTION_DIR / "evaluation"


def _metrics(frame: pd.DataFrame, name: str, tier: str, scope: str) -> dict:
    data = frame.loc[frame.pred_margin.notna()].copy()
    if data.target_game_id.duplicated().any():
        raise ValueError(f"Duplicate scored forecast: {tier} {name} {scope}")
    actual = data.actual_margin.to_numpy(float)
    pred = data.pred_margin.to_numpy(float)
    prob = data.home_win_probability.to_numpy(float)
    valid_prob = np.isfinite(prob)
    home_win = actual > 0
    winner = prob >= 0.5
    spread = data.market_home_spread.to_numpy(float)
    valid_line = np.isfinite(spread) & (spread != 0) & (actual != 0)
    valid_market = valid_line & valid_prob
    favorite_home = -spread > 0
    upset = valid_market & (favorite_home != home_win)
    chalk = valid_market & ~upset
    cover = actual + spread
    predicted_cover = pred + spread
    ats = valid_line & (cover != 0) & (predicted_cover != 0)
    return {
        "tier": tier, "model_id": name, "scope": scope,
        "games": len(data), "mae": float(np.abs(actual - pred).mean()),
        "rmse": float(np.sqrt(np.square(actual - pred).mean())),
        "brier_games": int(valid_prob.sum()),
        "brier": float(np.square(prob[valid_prob] - home_win[valid_prob]).mean()) if valid_prob.any() else np.nan,
        "winner_games": int(valid_prob.sum()),
        "winner_accuracy": float((winner[valid_prob] == home_win[valid_prob]).mean()) if valid_prob.any() else np.nan,
        "upset_games": int(upset.sum()),
        "upset_recall": float((winner[upset] != favorite_home[upset]).mean()) if upset.any() else np.nan,
        "chalk_games": int(chalk.sum()),
        "chalk_accuracy": float((winner[chalk] == favorite_home[chalk]).mean()) if chalk.any() else np.nan,
        "ats_games": int(ats.sum()),
        "ats_accuracy": float((np.sign(predicted_cover[ats]) == np.sign(cover[ats])).mean()) if ats.any() else np.nan,
    }


def _bootstrap_paired(a: pd.DataFrame, b: pd.DataFrame, *, seed: int = 1819,
                      repeats: int = 4000) -> dict:
    joined = a[["target_game_id", "week", "actual_margin", "pred_margin"]].merge(
        b[["target_game_id", "pred_margin"]], on="target_game_id", suffixes=("_a", "_b"),
        validate="one_to_one")
    if len(joined) == 0 or joined.actual_margin.isna().any():
        raise ValueError("Paired comparison has no shared scored games")
    delta = np.abs(joined.actual_margin - joined.pred_margin_a).to_numpy() - np.abs(
        joined.actual_margin - joined.pred_margin_b).to_numpy()
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(joined.week.to_numpy() == week) for week in sorted(joined.week.unique())]
    estimates = np.empty(repeats)
    for repeat in range(repeats):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        estimates[repeat] = delta[indices].mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"games": len(joined), "mae_a_minus_b": float(delta.mean()),
            "ci95_low": float(low), "ci95_high": float(high),
            "bootstrap": "paired week-stratified game bootstrap", "repeats": repeats,
            "seed": seed, "positive_means_b_better": True}


def _prior_models(target_ids: set[int], outcomes: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    old = pd.read_parquet(PRIOR, columns=["game_id", "fingerprint", "model_name",
                                         "pred_home_margin", "pred_home_win_probability",
                                         "actual_home_margin"])
    family = {}
    for label, tier in (("F16", "F16"), ("F17-market", "F17-market")):
        subset = old.loc[old.fingerprint.eq(label)].copy()
        subset["model_id"] = subset.model_name.str.rsplit("_", n=1).str[-1]
        if len(subset) != 271 * 6 or set(subset.game_id.astype(int)) != target_ids:
            raise ValueError(f"Prior {label} archive differs from target cohort")
        for architecture in ROSTER:
            rows = subset.loc[subset.model_id.eq(architecture)].rename(columns={
                "game_id": "target_game_id", "pred_home_margin": "pred_margin",
                "pred_home_win_probability": "home_win_probability"})
            rows = rows[["target_game_id", "pred_margin", "home_win_probability",
                         "actual_home_margin"]].merge(outcomes[["target_game_id", "week", "actual_margin",
                                                             "market_home_spread"]],
                                                      on="target_game_id", validate="one_to_one")
            if not np.allclose(rows.actual_home_margin, rows.actual_margin):
                raise ValueError(f"Prior {label} outcomes differ from current score source")
            family[(tier, architecture)] = rows.drop(columns="actual_home_margin")
        consensus = subset.groupby("game_id", as_index=False).agg(
            pred_margin=("pred_home_margin", "mean"),
            home_win_probability=("pred_home_win_probability", "mean"))
        consensus = consensus.rename(columns={"game_id": "target_game_id"}).merge(
            outcomes[["target_game_id", "week", "actual_margin", "market_home_spread"]],
            on="target_game_id", validate="one_to_one")
        family[(tier, "equal")] = consensus
    return family


def score(predictions_dir: Path = PREDICTION_DIR, output: Path = DEFAULT_OUTPUT) -> dict:
    receipt_path = predictions_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    freeze_path = Path(receipt["freeze_manifest"])
    if freeze_path != FREEZE or digest(freeze_path) != receipt["freeze_manifest_sha256"]:
        raise ValueError("Prospective freeze identity changed")
    frozen = json.loads(freeze_path.read_text())
    if digest(MARKET) != frozen["market_policy"]["market_source_sha256"]:
        raise ValueError("Archived market source changed after forecast freeze")
    if not receipt.get("no_2026_outcomes_read"):
        raise ValueError("Prediction receipt did not quarantine 2026 outcomes")
    prediction_path = Path(receipt["predictions"])
    if digest(prediction_path) != receipt["predictions_sha256"]:
        raise ValueError("Prospective forecast changed before scoring")
    if (output / "score_receipt.json").exists():
        raise FileExistsError("2026 score receipt is immutable")
    predictions = pd.read_parquet(prediction_path)
    if predictions.duplicated(["tier", "model_id", "target_game_id"]).any():
        raise ValueError("Duplicate F18/F19 forecast")
    target_ids = set(predictions.loc[predictions.tier.eq("F18"), "target_game_id"].astype(int))
    games = pd.read_parquet(GAMES, columns=["id", "week", "home_points", "away_points"])
    games = games.loc[games.id.isin(target_ids)].rename(columns={"id": "target_game_id"})
    if len(games) != 271 or games.target_game_id.duplicated().any():
        raise ValueError("Outcome source differs from frozen game cohort")
    games["actual_margin"] = games.home_points - games.away_points
    if games.actual_margin.isna().any():
        raise ValueError("Incomplete outcomes in frozen target cohort")
    market = pd.read_parquet(MARKET, columns=["target_game_id", "market_home_spread",
                                             "market_home_implied_no_vig",
                                             "market_snapshot_captured_pre_kickoff"])
    if market.target_game_id.duplicated().any():
        raise ValueError("Duplicate market game")
    outcomes = games[["target_game_id", "week", "actual_margin"]].merge(
        market, on="target_game_id", validate="one_to_one")
    scored = predictions.merge(outcomes, on="target_game_id", suffixes=("", "_outcome"),
                               validate="many_to_one")
    if not scored.week.eq(scored.week_outcome).all():
        raise ValueError("Forecast week differs from outcome schedule")
    scored = scored.drop(columns="week_outcome")
    scored.loc[scored.status.ne("available"), ["pred_margin", "home_win_probability"]] = np.nan
    if scored.loc[scored.tier.eq("F18") & scored.model_id.eq("equal"), "pred_margin"].notna().sum() != 271:
        raise ValueError("F18 coverage changed")
    if scored.loc[scored.tier.eq("F19") & scored.model_id.eq("equal"), "pred_margin"].notna().sum() != 263:
        raise ValueError("F19 coverage changed")
    prior = _prior_models(target_ids, outcomes)
    market_baseline = outcomes[["target_game_id", "week", "actual_margin", "market_home_spread"]].copy()
    market_baseline["pred_margin"] = -market_baseline.market_home_spread
    market_baseline["home_win_probability"] = outcomes.market_home_implied_no_vig
    market_baseline.loc[~outcomes.market_snapshot_captured_pre_kickoff.fillna(False),
                        ["pred_margin", "home_win_probability"]] = np.nan
    lookup = {(tier, model): group.copy() for (tier, model), group in scored.groupby(["tier", "model_id"])}
    lookup.update(prior)
    lookup[("market", "baseline")] = market_baseline
    rows = []
    for (tier, model), frame in lookup.items():
        rows.append(_metrics(frame, model, tier, "available"))
        if tier in ("F18", "F19", "F16", "F17-market", "market"):
            common = frame.loc[frame.target_game_id.isin(set(market_baseline.loc[
                market_baseline.pred_margin.notna(), "target_game_id"]))]
            rows.append(_metrics(common, model, tier, "common_market_263"))
    metrics = pd.DataFrame(rows).sort_values(["scope", "tier", "model_id"])
    pairs = []
    for model in (*ROSTER, "equal"):
        for tier_a, tier_b in (("F16", "F18"), ("F18", "F19"),
                               ("F17-market", "F19"), ("market", "F19"),
                               ("market", "F18")):
            left = lookup[(tier_a, "baseline" if tier_a == "market" else model)]
            right = lookup[(tier_b, model)]
            left = left.loc[left.pred_margin.notna()]
            right = right.loc[right.pred_margin.notna()]
            estimate = _bootstrap_paired(left, right)
            pairs.append({"model_id": model, "a": tier_a, "b": tier_b, **estimate})
    comparisons = pd.DataFrame(pairs)
    output.mkdir(parents=True, exist_ok=True)
    scored_path = output / "scored_forecasts.parquet"
    metrics_path = output / "metrics.csv"
    pairs_path = output / "paired_comparisons.csv"
    scored.to_parquet(scored_path, index=False, compression="zstd")
    metrics.to_csv(metrics_path, index=False, float_format="%.8f")
    comparisons.to_csv(pairs_path, index=False, float_format="%.8f")
    score_receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_receipt": str(receipt_path), "prediction_receipt_sha256": digest(receipt_path),
        "freeze_manifest_sha256": receipt["freeze_manifest_sha256"],
        "2026_outcomes_first_read_after_prediction_receipt": True,
        "broader_project_2026_outcomes_previously_analyzed": True,
        "interpretation": "post-freeze archived-pregame replay; not a fully blinded prospective experiment for the broader project",
        "source_sha256": {str(path): digest(path) for path in (GAMES, MARKET, PRIOR)},
        "outputs": {str(path): digest(path) for path in (scored_path, metrics_path, pairs_path)},
        "metric_rows": len(metrics), "paired_rows": len(comparisons),
    }
    write_json(output / "score_receipt.json", score_receipt)
    return score_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(score(args.predictions_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
