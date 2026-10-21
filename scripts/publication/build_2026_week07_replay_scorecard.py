#!/usr/bin/env python3
"""Grade the explicitly retrospective Week 7 catch-up against frozen prior weeks."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/publication/2026/catchup_2026_10_21"
OUTPUT = ROOT / "publication/2026/week_07/post_game/retrospective_replay"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    proof_path = INPUT / "week_07_replay_structural_proof.json"
    proof = json.loads(proof_path.read_text())
    if proof.get("status") != "retrospective_structural_pass" or proof.get("timely_forecast_issued") is not False:
        raise ValueError("Week 7 replay must be labeled retrospective")
    source = INPUT / "week_07_replay_operational/tables/all_game_model_predictions.parquet"
    consensus_path = INPUT / "week_07_replay_operational/tables/all_games.csv"
    results_path = INPUT / "week_07_replay_results.csv"
    model = pd.read_parquet(source)
    consensus = pd.read_csv(consensus_path)
    results = pd.read_csv(results_path)
    if len(model) != 59 * 33 or model.duplicated(["game_id", "model_name"]).any():
        raise ValueError("Week 7 model replay is incomplete")
    if len(consensus) != 59 or set(consensus.game_id) != set(results.game_id):
        raise ValueError("Week 7 consensus/results game IDs differ")
    if results.game_id.duplicated().any() or results[["home_points", "away_points"]].isna().any().any():
        raise ValueError("Week 7 results are incomplete")
    model = model.merge(results, on="game_id", validate="many_to_one")
    model["actual_home_margin"] = model.home_points - model.away_points
    model["actual_home_win"] = model.actual_home_margin.gt(0)
    model["winner_correct"] = model.pred_winner.eq(
        model.home_team.where(model.actual_home_win, model.away_team)
    )
    model["absolute_margin_error"] = (model.pred_home_margin - model.actual_home_margin).abs()
    model["provenance_class"] = "retrospective_replay_after_games"
    model["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
    consensus = consensus.merge(results, on="game_id", validate="one_to_one")
    consensus["actual_home_margin"] = consensus.home_points - consensus.away_points
    consensus["winner_correct"] = consensus.pred_winner.eq(
        consensus.home_team.where(consensus.actual_home_margin.gt(0), consensus.away_team)
    )
    consensus["absolute_margin_error"] = (
        consensus.pred_home_margin - consensus.actual_home_margin
    ).abs()
    consensus["brier_error"] = (
        consensus.pred_home_win_probability - consensus.actual_home_margin.gt(0).astype(float)
    ) ** 2
    consensus["provenance_class"] = "retrospective_replay_after_games"
    historical = []
    input_hashes = {}
    for week in range(7):
        path = ROOT / f"publication/2026/week_{week:02d}/post_game/scoring/scored_predictions.parquet"
        frame = pd.read_parquet(path)
        frame["provenance_class"] = "frozen_pregame"
        historical.append(frame)
        input_hashes[str(path.relative_to(ROOT))] = digest(path)
    all_models = pd.concat([*historical, model], ignore_index=True)
    rows = []
    for (name, week), group in all_models.groupby(["model_name", "week"], sort=True):
        probability = pd.to_numeric(group.pred_home_win_probability, errors="coerce")
        actual = group.actual_home_win.astype(float)
        rows.append({
            "model_name": name, "week": int(week), "games": group.game_id.nunique(),
            "winner_accuracy": float(group.winner_correct.mean()),
            "margin_mae": float(group.absolute_margin_error.mean()),
            "brier_score": float(((probability - actual) ** 2).mean()),
            "provenance_class": group.provenance_class.iloc[0],
        })
    weekly = pd.DataFrame(rows).sort_values(["week", "model_name"])
    cumulative_rows = []
    for (name, week), _ in weekly.groupby(["model_name", "week"], sort=True):
        group = all_models.loc[all_models.model_name.eq(name) & all_models.week.le(week)]
        probability = pd.to_numeric(group.pred_home_win_probability, errors="coerce")
        cumulative_rows.append({
            "model_name": name, "through_week": int(week), "games": group.game_id.nunique(),
            "winner_accuracy": float(group.winner_correct.mean()),
            "margin_mae": float(group.absolute_margin_error.mean()),
            "brier_score": float(((probability - group.actual_home_win.astype(float)) ** 2).mean()),
            "contains_retrospective_replay": bool(week >= 7),
        })
    cumulative = pd.DataFrame(cumulative_rows).sort_values(["through_week", "model_name"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = {
        "model_games": OUTPUT / "week07_replay_model_games.parquet",
        "consensus_games": OUTPUT / "week07_replay_consensus_games.csv",
        "weekly": OUTPUT / "operational_weekly_metrics.csv",
        "cumulative": OUTPUT / "operational_cumulative_metrics.csv",
    }
    model.to_parquet(files["model_games"], index=False)
    consensus.to_csv(files["consensus_games"], index=False)
    weekly.to_csv(files["weekly"], index=False)
    cumulative.to_csv(files["cumulative"], index=False)
    input_hashes.update({str(path.relative_to(ROOT)): digest(path) for path in
                         (source, consensus_path, results_path, proof_path)})
    readme = OUTPUT / "README.md"
    readme.write_text(
        "# 2026 Week 7 retrospective replay\n\n"
        "These 59 FBS versus FBS games were inferred on October 21 after completion. "
        "A score-free Week 7 schedule and reconstructed Week 6 feature state were used, "
        "but the source cache itself was collected after the games. The structural cutoff "
        "does not prove historical provider field availability or exclude later corrections. "
        "These are not timely pregame forecasts. Three FBS versus FCS games were outside "
        "the frozen roster's supported matchup scope. No historical market line was used.\n\n"
        "Weeks 0–6 in the cumulative table use their actual frozen pregame predictions. "
        "The Week 7 endpoint includes this retrospective replay and is marked accordingly.\n"
    )
    manifest = {
        "schema": "tdnet-2026-week07-retrospective-replay-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_issued_before_kickoff": False,
        "score_free_schedule": True,
        "historical_provider_field_timing_verified": False,
        "games": 59, "models": 33, "excluded_fbs_fcs_games": 3,
        "inputs": input_hashes,
        "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
                  for name, path in files.items()},
        "readme_sha256": digest(readme),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"games": 59, "models": 33, "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
