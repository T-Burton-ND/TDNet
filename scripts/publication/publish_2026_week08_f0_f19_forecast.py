#!/usr/bin/env python3
"""Publish the complete score-free F0–F19 Week 8 research forecast roster."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "publication/2026/week_08/pre_game"
OUT = BASE / "scientific/full_f0_f19"
F09 = OUT / "f09_f17"
F18 = OUT / "f18_f19_frozen_predictions"
STATES = ROOT / "data/what_if_2026_fingerprints/week08_expanded_states"
ROSTER = ("M1", "M2", "M3", "M4", "M5", "M10")
STAGES = tuple(f"F{i}" for i in range(17)) + ("F17-market", "F18", "F19")
FAMILIES = {"M1": "linear", "M2": "spline", "M3": "tree", "M4": "boosted", "M5": "neural", "M10": "knn"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_rows() -> tuple[pd.DataFrame, list[Path]]:
    base_path = BASE / "scientific/full_f0_f8/scientific_all_game_predictions.csv"
    original = pd.read_csv(base_path)
    early = original[["game_id", "season", "week", "home_team", "away_team", "fingerprint",
                      "model_name", "model_family", "pred_home_margin",
                      "pred_home_win_probability"]].copy()
    early["model_id"] = early.model_name.str.rsplit("_", n=1).str[-1]
    paths = [base_path]
    middle = []
    for stage in ("F09", "F10", "F11", "F12_corrected", "F13", "F14", "F15", "F16", "F17_market"):
        for model in ROSTER:
            path = F09 / f"{stage}_{model}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Missing Week 8 expanded cell: {path}")
            middle.append(pd.read_parquet(path))
            paths.append(path)
    middle = pd.concat(middle, ignore_index=True)
    path18 = F18 / "predictions.parquet"
    frozen = pd.read_parquet(path18)
    frozen = frozen.loc[frozen.model_id.isin(ROSTER)].copy()
    if not frozen.status.eq("available").all():
        raise ValueError("F18/F19 frozen forecast lacks full Week 8 market coverage")
    later = frozen.rename(columns={"target_game_id": "game_id", "tier": "fingerprint",
                                   "pred_margin": "pred_home_margin",
                                   "home_win_probability": "pred_home_win_probability"})
    later["season"] = 2026
    later["model_name"] = "scientific_" + later.fingerprint + "_" + later.model_id
    later["model_family"] = later.model_id.map(FAMILIES)
    fields = ["game_id", "season", "week", "home_team", "away_team", "fingerprint",
              "model_id", "model_name", "model_family", "pred_home_margin",
              "pred_home_win_probability"]
    models = pd.concat([early[fields], middle[fields], later[fields]], ignore_index=True)
    if len(models) != 120 * 53 or models.duplicated(["game_id", "fingerprint", "model_id"]).any():
        raise ValueError("Expanded Week 8 model roster is not 120 × 53")
    if set(models.fingerprint) != set(STAGES) or set(models.model_id) != set(ROSTER):
        raise ValueError("Expanded Week 8 fingerprint/model roster changed")
    if not models.groupby(["fingerprint", "model_id"]).game_id.nunique().eq(53).all():
        raise ValueError("A Week 8 scientific model cell is missing a game")
    if not np.isfinite(models.pred_home_margin).all() or not models.pred_home_win_probability.between(0, 1).all():
        raise ValueError("Week 8 expanded forecast has an invalid margin/probability")
    paths.append(path18)
    return models.sort_values(["game_id", "fingerprint", "model_id"]).reset_index(drop=True), paths


def consensus(models: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    margin_path = BASE / "frozen_bundle/public/predictions.parquet"
    raw = pd.read_parquet(margin_path, columns=["game_id", "game_start_time_utc", "created_at_utc",
                                                     "market_spread_close"])
    market = raw.drop_duplicates("game_id")
    if len(market) != 53 or market.market_spread_close.isna().any():
        raise ValueError("Week 8 market cohort incomplete")
    totals = models.groupby("game_id").agg(
        consensus_pred_home_margin=("pred_home_margin", "mean"),
        consensus_pred_home_win_probability=("pred_home_win_probability", "mean"),
        model_votes=("model_name", "nunique"),
    ).reset_index()
    by_fingerprint = models.groupby(["game_id", "fingerprint"]).agg(
        pred_home_margin=("pred_home_margin", "mean"),
        pred_home_win_probability=("pred_home_win_probability", "mean"),
        model_votes=("model_name", "nunique"),
    ).reset_index()
    if not totals.model_votes.eq(120).all() or not by_fingerprint.model_votes.eq(6).all():
        raise ValueError("Consensus uses fewer than the required model votes")
    teams = models.drop_duplicates("game_id")[["game_id", "home_team", "away_team"]]
    totals = totals.merge(teams, on="game_id", validate="one_to_one")
    totals = totals.merge(market, on="game_id", validate="one_to_one")
    totals["consensus_straight_up_pick"] = np.where(
        totals.consensus_pred_home_win_probability.ge(.5), totals.home_team, totals.away_team)
    totals["consensus_predicted_winner_margin"] = totals.consensus_pred_home_margin.abs()
    totals["consensus_model_edge_vs_spread_points"] = (
        totals.consensus_pred_home_margin + totals.market_spread_close)
    totals["consensus_against_spread_pick"] = np.where(
        totals.consensus_model_edge_vs_spread_points.gt(0), totals.home_team,
        np.where(totals.consensus_model_edge_vs_spread_points.lt(0), totals.away_team, "push"))
    totals["consensus_pick_confidence"] = totals.consensus_pred_home_win_probability.clip(lower=.5).where(
        totals.consensus_pred_home_win_probability.ge(.5),
        1 - totals.consensus_pred_home_win_probability)
    totals["season"] = 2026
    totals["week"] = 8
    totals["phase"] = "pre_game"
    totals["roster"] = "expanded_scientific_f0_f19_120"
    return totals.sort_values(["game_start_time_utc", "game_id"]).reset_index(drop=True), by_fingerprint


def figure(table: pd.DataFrame, path: Path) -> None:
    colors = {"paper": "#F7F4ED", "ink": "#24344B", "pink": "#D13D7C", "gold": "#AB7B22"}
    fig, axes = plt.subplots(1, 2, figsize=(24, 18), facecolor=colors["paper"])
    for axis, chunk in zip(axes, np.array_split(table, 2)):
        axis.axis("off")
        rows = []
        for game in chunk.itertuples(index=False):
            rows.append([
                pd.Timestamp(game.game_start_time_utc).strftime("%a %I:%M %p UTC"),
                f"{game.away_team} @ {game.home_team}",
                f"{game.consensus_straight_up_pick} {game.consensus_predicted_winner_margin:.1f}",
                f"{game.consensus_pick_confidence:.0%}",
                str(game.consensus_against_spread_pick),
                f"{abs(game.consensus_model_edge_vs_spread_points):.1f}",
            ])
        artist = axis.table(cellText=rows, colLabels=["Kickoff", "Matchup", "Projected winner", "Win p", "ATS", "Edge"],
                            cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1],
                            colWidths=[.16, .33, .23, .075, .15, .055])
        artist.auto_set_font_size(False)
        artist.set_fontsize(8)
        for (row, _), cell in artist.get_celld().items():
            cell.set_edgecolor("#D2D8DE")
            cell.set_facecolor(colors["ink"] if row == 0 else "#FFFFFF" if row % 2 else "#EDF1F5")
            if row == 0:
                cell.set_text_props(color="white", weight="bold")
    fig.suptitle("2026 Week 8 · expanded F0–F19 scientific consensus", fontsize=22,
                 color=colors["ink"], weight="bold", y=.982)
    fig.text(.5, .018, "120 equal model votes per game · 53 remaining games · frozen before first kickoff · ATS edge is projected home margin plus home spread",
             ha="center", color=colors["ink"], fontsize=10)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Existing expanded Week 8 publication is immutable")
    models, inputs = model_rows()
    table, fingerprints = consensus(models)
    first_start = pd.to_datetime(table.game_start_time_utc, utc=True).min()
    if pd.Timestamp.now(tz="UTC") >= first_start:
        raise ValueError("First remaining Week 8 kickoff passed before publication")
    f18_receipt = json.loads((F18 / "receipt.json").read_text())
    if f18_receipt["tier_coverage"] != {"F18": 53, "F19": 53} or not f18_receipt["no_2026_outcomes_read"]:
        raise ValueError("Frozen F18/F19 prediction receipt failed")
    state_receipt = json.loads((STATES / "manifest.json").read_text())
    if not state_receipt["no_target_outcome_columns_read"]:
        raise ValueError("Week 8 state build read a target outcome")
    availability = pd.read_parquet(OUT / "feature_source_availability.parquet")
    available = pd.to_datetime(availability.latest_documented_source_utc, utc=True)
    market = pd.read_parquet(OUT / "f17_f19_market_snapshot.parquet")
    market_time = pd.to_datetime(market.snapshot_timestamp_utc, utc=True)
    freeze_time = pd.Timestamp.now(tz="UTC")
    if (len(availability) != 53 or len(market) != 53
            or not (available < freeze_time).all() or not (market_time < freeze_time).all()
            or freeze_time >= first_start):
        raise ValueError("Expanded Week 8 inputs or publication are not pregame")
    proof_path = OUT / "cutoff_proof.json"
    proof_path.write_text(json.dumps({
        "status": "pass_structural_cutoff", "season": 2026, "week": 8,
        "target_games": 53, "model_cells": 120,
        "forecast_freeze_utc": freeze_time.isoformat(),
        "first_remaining_kickoff_utc": first_start.isoformat(),
        "latest_documented_source_utc": available.max().isoformat(),
        "latest_market_snapshot_utc": market_time.max().isoformat(),
        "source_reporting_lag_hours": 48,
        "no_target_outcome_columns_read": True,
        "no_target_game_in_own_features": True,
        "historical_training_through_season": 2025,
        "provider_field_level_availability_verified": False,
        "historical_market_quote_timing_verified": False,
        "source_availability_sha256": digest(OUT / "feature_source_availability.parquet"),
        "market_snapshot_sha256": digest(OUT / "f17_f19_market_snapshot.parquet"),
        "states_manifest_sha256": digest(STATES / "manifest.json"),
    }, indent=2, sort_keys=True) + "\n")
    saved = {
        "models_csv": OUT / "scientific_all_game_predictions.csv",
        "consensus_csv": OUT / "scientific_consensus_game_predictions.csv",
        "fingerprint_csv": OUT / "scientific_fingerprint_consensus_game_predictions.csv",
        "consensus_png": OUT / "scientific_all_game_predictions.png",
        "cutoff_proof": proof_path,
    }
    models.to_csv(saved["models_csv"], index=False, float_format="%.6f")
    table.to_csv(saved["consensus_csv"], index=False, float_format="%.6f")
    fingerprints.to_csv(saved["fingerprint_csv"], index=False, float_format="%.6f")
    figure(table, saved["consensus_png"])
    readme = OUT / "README.md"
    readme.write_text(
        "# Week 8 F0–F19 scientific research forecasts\n\n"
        "These 53 games were still unplayed at the October 21 forecast freeze. "
        "The 120-cell roster has six architectures per fingerprint: F0–F16, F17-market, F18, and F19. "
        "The game consensus is the equal mean of all 120 predicted margins and win probabilities. "
        "F0–F8 came from the separately frozen full scientific Week 8 bundle. F9–F17 were refit "
        "on saved historical designs ending in 2025 and calibrated with saved historical residuals. "
        "F18/F19 used their frozen through-2025 model bundles and calibration. No Week 8 target "
        "scores were read in feature construction or model fitting.\n\n"
        "Week 6–7 play, player, and drive data were acquired October 21. Every derived source "
        "game used a 48-hour lag and was checked to precede its target kickoff. The source "
        "cache's field-level historical publication times and possible later corrections are "
        "not proven; the structural proof cannot guarantee their original Week 7 availability. "
        "Market summaries were copied from the pregame Week 8 frozen margin bundle. Individual "
        "provider quote timestamps are unavailable, and historical F17/F19 quote timing is "
        "unverified. Earlier 2026 outcomes had informed broader roster research, so this "
        "expanded panel is exploratory even though these 53 game forecasts precede kickoff.\n"
    )
    saved["readme"] = readme
    if pd.Timestamp.now(tz="UTC") >= first_start:
        raise ValueError("First remaining Week 8 kickoff passed during publication build")
    manifest = {
        "season": 2026, "week": 8, "phase": "pre_game", "scope": "research_full_f0_f19",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_remaining_kickoff_utc": first_start.isoformat(),
        "games": 53, "fingerprints": list(STAGES), "model_ids": list(ROSTER),
        "model_cells": 120, "model_game_rows": len(models), "consensus_votes_per_game": 120,
        "no_target_outcomes_read": True, "historical_training_through_season": 2025,
        "confirmatory_protocol_status": "exploratory_after_prior_2026_research",
        "market_snapshot_all_games_pre_kickoff": True,
        "historical_quote_level_timing_verified": False,
        "source_field_level_availability_verified": False,
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "supporting_receipts": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (STATES / "manifest.json", F18 / "receipt.json",
                         OUT / "week08_application_manifest.json", OUT / "f17_f19_market_snapshot.parquet")
        },
        "outputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
                    for name, path in saved.items()},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    status_path = BASE / "expanded_scientific_status.json"
    status_path.write_text(json.dumps({
        "status": "frozen_complete_research", "generated_at_utc": manifest["created_at_utc"],
        "season": 2026, "week": 8, "current_week_games": 53,
        "trained_model_cells": 120, "timely_predictions_issued": True,
        "first_remaining_kickoff_utc": first_start.isoformat(),
        "forecast_manifest": str((OUT / "manifest.json").relative_to(ROOT)),
        "forecast_manifest_sha256": digest(OUT / "manifest.json"),
        "prior_blocker": "Week 1–5-only expanded feature states; resolved by October 21 Week 6–7 context acquisition and score-free Week 8 rebuild",
        "limits": [
            "Provider field-level historical availability and later corrections are unverified.",
            "Market quote-level timing is unavailable; market-bearing tiers remain research outputs.",
        ],
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"games": 53, "model_cells": 120, "rows": len(models),
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
