#!/usr/bin/env python3
"""Refit frozen historical F09–F17 recipes and forecast the 53 Week 8 games.

Only 2010–2025 labels enter model fitting and calibration. Week 8 inputs come
from verified score-free state files and the frozen margin-roster market summary.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/publication")]
from nextgen_rounds_train import _record, load_stage_matrix  # noqa: E402
from nextgen_rounds_scientific_train import make_model  # noqa: E402
from gridiron_ml.experiments.nextgen_rounds_market import MARKET_COLUMNS  # noqa: E402
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import (  # noqa: E402
    build_estimator, residual_probability, source_to_matchup,
)

STAGES = ("F09", "F10", "F11", "F12_corrected", "F13", "F14", "F15", "F16", "F17_market")
MODELS = ("M1", "M2", "M3", "M4", "M5", "M10")
FAMILIES = {"M1": "linear", "M2": "spline", "M3": "tree", "M4": "boosted", "M5": "neural", "M10": "knn"}
DATA = ROOT / "data/what_if_2026_fingerprints/week08_expanded_states"
ROUND_DATA = ROOT / "data/nextgen_rounds_2026"
OUT = ROOT / "publication/2026/week_08/pre_game/scientific/full_f0_f19/f09_f17"
LADDER = ROOT / "data/publication/2026/weekly_operations/week_08/fingerprint_ladder_v3/canonical_fingerprint.parquet"
MARKET = OUT.parent / "f17_f19_market_snapshot.parquet"
MANIFEST = Path("/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/fingerprints/F12_F_a/feature_manifest.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities() -> pd.DataFrame:
    games = pd.read_parquet(DATA / "target_schedule_without_scores.parquet",
                            columns=["id", "season", "week", "start_date", "home_team", "away_team"])
    games = games.rename(columns={"id": "target_game_id", "start_date": "target_start_utc"})
    games["target_start_utc"] = pd.to_datetime(games.target_start_utc, utc=True)
    games = games.sort_values("target_game_id").reset_index(drop=True)
    if len(games) != 53 or not games.week.eq(8).all():
        raise ValueError("Week 8 score-free target cohort changed")
    if pd.Timestamp.now(tz="UTC") >= games.target_start_utc.min():
        raise ValueError("Week 8 first remaining kickoff has passed; new forecasts cannot be issued")
    return games


def team_sources(games: pd.DataFrame) -> pd.DataFrame:
    base = pd.concat([
        games[["target_game_id", "home_team"]].rename(columns={"home_team": "team"}).assign(side="home"),
        games[["target_game_id", "away_team"]].rename(columns={"away_team": "team"}).assign(side="away"),
    ], ignore_index=True)
    ladder = pd.read_parquet(LADDER)
    ladder = ladder.loc[ladder.keys_season.eq(2026) & ladder.keys_week.eq(7)].copy()
    if ladder.keys_team.duplicated().any():
        raise ValueError("Duplicate Week 7 ladder team")
    joined = base.merge(ladder, left_on="team", right_on="keys_team", validate="many_to_one")
    if len(joined) != 106:
        raise ValueError("Week 8 prior ladder is incomplete")
    for stage in ("f09", "f10", "f11", "f12", "research"):
        state = pd.read_parquet(DATA / f"{stage}_target_state.parquet")
        names = [column for column in state if column not in joined.columns and column not in
                 ("target_game_id", "team", "side", "target_start_utc")]
        joined = joined.merge(state[["target_game_id", "team", *names]],
                              on=["target_game_id", "team"], validate="one_to_one")
        if len(joined) != 106:
            raise ValueError(f"Incomplete Week 8 {stage} team state merge")
    return joined


def target_matrix(games: pd.DataFrame, team: pd.DataFrame, stage: str,
                  evidence: dict) -> pd.DataFrame:
    names = list(evidence["source_features"])
    if set(names) - set(team.columns):
        raise ValueError(f"{stage} missing source features: {sorted(set(names) - set(team.columns))}")
    records_by_name = {record["name"]: record for record in json.loads(MANIFEST.read_text())}
    records = [records_by_name[name] if name in records_by_name else _record(name) for name in names]
    home = games[["target_game_id"]].merge(team.loc[team.side.eq("home"), ["target_game_id", *names]],
                                            on="target_game_id", validate="one_to_one")
    away = games[["target_game_id"]].merge(team.loc[team.side.eq("away"), ["target_game_id", *names]],
                                            on="target_game_id", validate="one_to_one")
    if len(home) != 53 or len(away) != 53:
        raise ValueError(f"{stage} missing matchup side")
    values = np.column_stack((home[names].to_numpy(float), away[names].to_numpy(float)))
    matchup = source_to_matchup(values, records)
    if stage == "F17_market":
        market = games[["target_game_id"]].merge(pd.read_parquet(MARKET), on="target_game_id", validate="one_to_one")
        if len(market) != 53 or not market.market_snapshot_captured_pre_kickoff.all():
            raise ValueError("F17 current-week market snapshot incomplete")
        matchup = np.column_stack((matchup, market[list(MARKET_COLUMNS)].to_numpy(float)))
        columns = [f"matchup__{name}" for name in names] + list(MARKET_COLUMNS)
    else:
        columns = [f"matchup__{name}" for name in names]
    if matchup.shape[1] != len(columns):
        raise ValueError(f"{stage} target matchup feature count changed")
    return pd.DataFrame(matchup, columns=columns)


def fit_stage(games: pd.DataFrame, team: pd.DataFrame, stage: str,
              model_ids: tuple[str, ...]) -> None:
    x_hist, historical_meta, _, evidence = load_stage_matrix(ROUND_DATA, stage)
    x_target = target_matrix(games, team, stage, evidence)
    if historical_meta.season.max() != 2025 or historical_meta.season.min() < 2010:
        raise ValueError("Historical fit cohort is not confined to 2010–2025")
    if x_hist.shape[1] != x_target.shape[1] or len(x_hist) != len(historical_meta):
        raise ValueError(f"{stage} historical/target shape mismatch")
    X = pd.DataFrame(x_hist, columns=x_target.columns)
    y = historical_meta.next_game_margin.to_numpy(float)
    cfg = json.loads((ROOT / "configs/experiments/nextgen_rounds_scientific_models_v1.json").read_text())
    points = json.loads((ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json").read_text())
    display = {"F09": "F9", "F12_corrected": "F12", "F17_market": "F17-market"}.get(stage, stage)
    for architecture in model_ids:
        target = OUT / f"{stage}_{architecture}.parquet"
        if target.exists():
            prior = pd.read_parquet(target)
            if len(prior) == 53 and set(prior.game_id) == set(games.target_game_id):
                print(f"{stage}/{architecture} already complete", flush=True)
                continue
            raise ValueError(f"Existing Week 8 prediction is incomplete: {target}")
        print(f"Fitting {stage}/{architecture} on through-2025 data", flush=True)
        specs = ([('setpoint', point) for point in points[architecture]]
                 if architecture in ("M2", "M4")
                 else [('seed', seed) for seed in ([1701] if architecture == "M3" else cfg["seeds"])])
        predictions = []
        for kind, spec in specs:
            if kind == "setpoint":
                saved = pd.read_parquet(ROUND_DATA / "experiments" / stage / architecture
                                        / spec["id"] / "predictions.parquet")
                model = build_estimator(architecture, spec)
                model.fit(X, y)
                pred = np.asarray(model.predict(x_target), float)
            else:
                seed = int(spec)
                saved = pd.read_parquet(ROUND_DATA / "scientific_model_runs/experiments"
                                        / stage / architecture / f"seed_{seed}" / "predictions.parquet")
                model, _ = make_model(architecture, seed, stage)
                model.train(X, y)
                pred = np.asarray(model.predict_margin(x_target), float).reshape(-1)
            residuals = (saved.actual_margin - saved.predicted_margin).to_numpy(float)
            predictions.append((pred, residual_probability(pred, residuals)))
            del model
            gc.collect()
        margin = np.mean([part[0] for part in predictions], axis=0)
        probability = np.mean([part[1] for part in predictions], axis=0)
        if margin.shape != (53,) or not np.isfinite(margin).all() or not np.isfinite(probability).all():
            raise ValueError(f"Invalid Week 8 forecast: {stage}/{architecture}")
        result = games[["target_game_id", "season", "week", "home_team", "away_team"]].rename(
            columns={"target_game_id": "game_id"}).copy()
        result["fingerprint"] = display
        result["model_id"] = architecture
        result["model_name"] = f"scientific_{display}_{architecture}"
        result["model_family"] = FAMILIES[architecture]
        result["pred_home_margin"] = margin
        result["pred_home_win_probability"] = probability
        result.to_parquet(target, index=False, compression="zstd")
        print(f"Saved {stage}/{architecture}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    games = identities()
    team = team_sources(games)
    stages = (args.stage,) if args.stage else STAGES
    models = (args.model,) if args.model else MODELS
    for stage in stages:
        fit_stage(games, team, stage, models)
    outputs = {path.stem: digest(path) for path in OUT.glob("*.parquet")}
    receipt_name = (f"run_receipt_{args.stage}_{args.model}.json"
                    if args.stage and args.model else "run_receipt.json")
    (OUT / receipt_name).write_text(json.dumps({
        "target_games": 53, "requested_stages": stages, "requested_models": models,
        "saved_model_cells": len(outputs), "training_through_season": 2025,
        "calibration_labels": "saved 2024–2025 held-out residuals",
        "target_outcomes_read": False,
        "week08_states_manifest_sha256": digest(DATA / "manifest.json"),
        "market_snapshot_sha256": digest(MARKET),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "outputs": outputs,
    }, indent=2, sort_keys=True) + "\n")
    print(f"Saved {len(outputs)} Week 8 F09–F17 model cells", flush=True)


if __name__ == "__main__":
    main()
