#!/usr/bin/env python3
"""Supplement frozen 2026 scores with paired errors and a decomposition figure.

This is post-freeze analysis only. It never selects or refits a model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constraint_free_score_2026 import DEFAULT_OUTPUT as SCORE, MARKET, PRIOR
from constraint_free_search import digest, write_json

SCORECARD = Path("publication/2026/week_05/post_game/scientific/current_season_f0_f17_market/scientific_2026_current_season_scorecard.csv")
OUTPUT = SCORE / "postfreeze_analysis"


def paired(a: pd.DataFrame, b: pd.DataFrame, *, seed: int = 1819,
           repeats: int = 4000) -> list[dict]:
    joined = a[["target_game_id", "week", "actual_margin", "pred_margin",
                "home_win_probability"]].merge(
        b[["target_game_id", "pred_margin", "home_win_probability"]],
        on="target_game_id", suffixes=("_a", "_b"), validate="one_to_one")
    if joined.target_game_id.duplicated().any() or not joined.actual_margin.notna().all():
        raise ValueError("Invalid paired evaluation cohort")
    actual = joined.actual_margin.to_numpy(float)
    pa = joined.pred_margin_a.to_numpy(float)
    pb = joined.pred_margin_b.to_numpy(float)
    qa = joined.home_win_probability_a.to_numpy(float)
    qb = joined.home_win_probability_b.to_numpy(float)
    truth = (actual > 0).astype(float)
    errors = {
        "mae": (np.abs(actual - pa), np.abs(actual - pb),
                np.isfinite(pa) & np.isfinite(pb)),
        "brier": ((qa - truth) ** 2, (qb - truth) ** 2,
                  np.isfinite(qa) & np.isfinite(qb)),
        "winner_error": ((qa >= .5) != truth, (qb >= .5) != truth,
                         np.isfinite(qa) & np.isfinite(qb)),
    }
    rows = []
    for number, (metric, (left, right, valid)) in enumerate(errors.items()):
        if not valid.any():
            raise ValueError(f"No paired {metric} games")
        delta = np.asarray(left[valid], float) - np.asarray(right[valid], float)
        weeks = joined.week.to_numpy()[valid]
        strata = [np.flatnonzero(weeks == week) for week in sorted(set(weeks))]
        rng = np.random.default_rng(seed + number)
        draws = np.empty(repeats)
        for repeat in range(repeats):
            indices = np.concatenate([rng.choice(group, len(group), replace=True)
                                      for group in strata])
            draws[repeat] = delta[indices].mean()
        low, high = np.quantile(draws, [.025, .975])
        rows.append({"metric": metric, "games": int(valid.sum()),
                     "mean_a": float(np.mean(np.asarray(left[valid], float))),
                     "mean_b": float(np.mean(np.asarray(right[valid], float))),
                     "a_minus_b": float(delta.mean()), "ci95_low": float(low),
                     "ci95_high": float(high), "positive_favors_b": True,
                     "bootstrap": "paired week-stratified game bootstrap",
                     "repeats": repeats, "seed": seed + number})
    return rows


def build(output: Path = OUTPUT) -> dict:
    if (output / "manifest.json").exists():
        raise FileExistsError("Post-freeze analysis manifest is immutable")
    score_receipt_path = SCORE / "score_receipt.json"
    score_receipt = json.loads(score_receipt_path.read_text())
    scored_path = SCORE / "scored_forecasts.parquet"
    if digest(scored_path) != score_receipt["outputs"][str(scored_path)]:
        raise ValueError("Scored forecasts changed after receipt")
    scored = pd.read_parquet(scored_path)
    forecasts = {(tier, model): frame for (tier, model), frame
                 in scored.groupby(["tier", "model_id"])}
    canonical = forecasts[("F18", "equal")]
    games = canonical[["target_game_id", "week", "actual_margin"]]
    if len(games) != 271:
        raise ValueError("Frozen 2026 cohort changed")
    old = pd.read_parquet(PRIOR)
    for tier in ("F16", "F17-market"):
        subset = old.loc[old.fingerprint.eq(tier)]
        if len(subset) != 271 * 6:
            raise ValueError(f"Prior {tier} roster has changed")
        mean = subset.groupby("game_id", as_index=False).agg(
            pred_margin=("pred_home_margin", "mean"),
            home_win_probability=("pred_home_win_probability", "mean"))
        forecasts[(tier, "equal")] = mean.rename(columns={"game_id": "target_game_id"}).merge(
            games, on="target_game_id", validate="one_to_one")
    scorecard = pd.read_csv(SCORECARD)
    prior_roster = scorecard.loc[scorecard.series_type.eq("model") &
                                 scorecard.fingerprint.ne("F17-market")]
    observed_prior = prior_roster.sort_values(["margin_mae", "fingerprint", "model_name"]).iloc[0]
    best_label = str(observed_prior.model_name)
    best_prior = old.loc[old.model_name.eq(best_label) &
                         old.fingerprint.eq(observed_prior.fingerprint)]
    if len(best_prior) != 271:
        raise ValueError("Observed prior-best archive has changed")
    forecasts[("observed_prior_best", best_label)] = best_prior.rename(columns={
        "game_id": "target_game_id", "pred_home_margin": "pred_margin",
        "pred_home_win_probability": "home_win_probability"})[[
            "target_game_id", "pred_margin", "home_win_probability"]].merge(
                games, on="target_game_id", validate="one_to_one")
    market = pd.read_parquet(MARKET, columns=[
        "target_game_id", "market_home_spread", "market_home_implied_no_vig",
        "market_snapshot_captured_pre_kickoff"])
    market = games.merge(market, on="target_game_id", validate="one_to_one")
    market["pred_margin"] = -market.market_home_spread
    market["home_win_probability"] = market.market_home_implied_no_vig
    market.loc[~market.market_snapshot_captured_pre_kickoff.fillna(False),
               ["pred_margin", "home_win_probability"]] = np.nan
    forecasts[("market", "baseline")] = market
    eligible = set(market.loc[market.pred_margin.notna(), "target_game_id"])
    comparisons = [
        ("F16", "equal", "F18", "equal", "all_271"),
        ("F16", "equal", "F18", "equal", "common_market_263"),
        ("F18", "equal", "F19", "equal", "common_market_263"),
        ("F17-market", "equal", "F19", "equal", "common_market_263"),
        ("market", "baseline", "F19", "equal", "common_market_263"),
        ("market", "baseline", "F19", "M4", "common_market_263"),
        ("F18", "equal", "market", "baseline", "common_market_263"),
        ("observed_prior_best", best_label, "F18", "equal", "all_271"),
        ("observed_prior_best", best_label, "F18", "M1", "all_271"),
        ("observed_prior_best", best_label, "F19", "equal", "common_market_263"),
    ]
    rows = []
    for index, (ta, ma, tb, mb, scope) in enumerate(comparisons):
        a, b = forecasts[(ta, ma)], forecasts[(tb, mb)]
        if scope == "common_market_263":
            a = a.loc[a.target_game_id.isin(eligible)]
            b = b.loc[b.target_game_id.isin(eligible)]
        for row in paired(a, b, seed=1819 + 7 * index):
            rows.append({"a_tier": ta, "a_model": ma, "b_tier": tb,
                         "b_model": mb, "scope": scope, **row})
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table_path = output / "paired_metrics.csv"
    table.to_csv(table_path, index=False, float_format="%.8f")
    metrics = pd.read_csv(SCORE / "metrics.csv")
    common = metrics.loc[metrics.scope.eq("common_market_263")]
    labels = ["F16", "F18", "F19", "Market"]
    identities = [("F16", "equal"), ("F18", "equal"),
                  ("F19", "equal"), ("market", "baseline")]
    values = [float(common.loc[common.tier.eq(tier) &
                               common.model_id.eq(model), "mae"].iloc[0])
              for tier, model in identities]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#F8F4EA")
    ax.set_facecolor("#FFFCF5")
    ax.plot(range(3), values[:3], color="#356B82", lw=2, marker="o", markersize=9)
    ax.axhline(values[3], color="#A37F3D", ls="--", label="Archived market spread")
    for i, value in enumerate(values[:3]):
        ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=11)
    d1 = table.loc[table.a_tier.eq("F16") & table.b_tier.eq("F18") &
                   table.scope.eq("common_market_263") & table.metric.eq("mae")].iloc[0]
    d2 = table.loc[table.a_tier.eq("F18") & table.b_tier.eq("F19") &
                   table.scope.eq("common_market_263") & table.metric.eq("mae")].iloc[0]
    ax.text(.5, max(values[:2]) + .2, f"F16 → F18: {d1.a_minus_b:+.3f}  [{d1.ci95_low:+.3f}, {d1.ci95_high:+.3f}]",
            ha="center", fontsize=10)
    ax.text(1.5, max(values[1:3]) + .38,
            f"F18 → F19: {d2.a_minus_b:+.3f}  [{d2.ci95_low:+.3f}, {d2.ci95_high:+.3f}]",
            ha="center", fontsize=10)
    ax.set_xticks(range(3), labels[:3])
    ax.set_ylabel("2026 margin MAE · 263 common games · lower is better")
    ax.set_title("Representation and market contributions after matched optimization")
    ax.set_ylim(bottom=0, top=max(values) + 1.3)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.text(.5, .02, "Paired week-stratified 95% intervals; F18/F19 are performance tiers. The market line is the 263-game spread baseline.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, 1])
    figure = output / "optimization_decomposition.png"
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_receipt": str(score_receipt_path), "score_receipt_sha256": digest(score_receipt_path),
        "paired_table": str(table_path), "paired_table_sha256": digest(table_path),
        "figure": str(figure), "figure_sha256": digest(figure),
        "observed_prior_best": {"fingerprint": str(observed_prior.fingerprint),
                                "model_name": best_label,
                                "2026_selected_post_hoc": True,
                                "warning": "2026 selection makes this an exploratory comparator, not an unbiased preselected winner"},
        "all_forecasts_frozen_before_this_analysis": True,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
