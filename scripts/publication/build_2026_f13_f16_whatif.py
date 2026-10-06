#!/usr/bin/env python3
"""Build cutoff-safe 2026 F13–F16 scientific what-if forecasts.

Current-season play context, sequence transitions, and role observations are
combined with the verified 2010–2025 archive. Feature availability is delayed
48 hours after each source game; the forecast target game is excluded.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/publication")]

from build_2026_f09_whatif import DATA, ROUND_DATA, read_schedule
from build_2026_f12_whatif import FAMILY
from nextgen_rounds_train import load_stage_matrix
from nextgen_rounds_train import _record as research_record
from nextgen_rounds_scientific_train import make_model
from gridiron_ml.experiments.nextgen_rounds_features import (
    actor_game_statistics, context_plays, context_totals, past_only_context_baselines,
    play_actor_lookup, residual_game_statistics, sequence_game_statistics,
    trailing_research_state,
)
from gridiron_ml.experiments.nextgen_microstructure import play_flags
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import (
    build_estimator, residual_probability, source_to_matchup,
)
from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file

ARCHIVE = Path("/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen")
RESEARCH = DATA / "f13_f16_features"
MODEL_IDS = ("M1", "M2", "M3", "M4", "M5", "M10")
FAMILIES = {"M1": "linear", "M2": "spline", "M3": "tree", "M4": "boosted", "M5": "neural", "M10": "knn"}


def normalize_plays(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={
        "gameId": "game_id", "driveId": "drive_id", "driveNumber": "drive_number",
        "playNumber": "play_number", "offenseScore": "offense_score",
        "defenseScore": "defense_score", "yardsToGoal": "yards_to_goal",
        "yardsGained": "yards_gained", "playType": "play_type",
    })


def current_research_stats(schedule: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    games = schedule.loc[
        schedule.season.eq(2026)
        & schedule.season_type.astype(str).str.lower().eq("regular")
        & schedule.completed.fillna(False).astype(bool)
        & schedule.home_classification.astype(str).str.lower().eq("fbs")
        & schedule.away_classification.astype(str).str.lower().eq("fbs")
    ].copy()
    allowed = set(games.id.astype(int))
    baselines = past_only_context_baselines(
        pd.read_parquet(ROUND_DATA / "context_training_totals.parquet"), years=range(2010, 2027)
    )[2026]
    residual_parts, sequence_parts, lookup_by_game = [], [], {}
    seq_audit = Counter()
    for path in sorted((DATA / "raw_cache/plays").glob("*.parquet")):
        raw = normalize_plays(pd.read_parquet(path))
        raw = raw.loc[raw.game_id.isin(allowed)].copy()
        if raw.empty:
            continue
        flagged = play_flags(raw)
        residual_parts.append(residual_game_statistics(flagged, baselines))
        sequence, report = sequence_game_statistics(flagged)
        sequence_parts.append(sequence)
        seq_audit.update(report)
        lookup = play_actor_lookup(flagged)
        for game, frame in lookup.groupby("game_id"):
            lookup_by_game[int(game)] = frame.copy()

    stats_parts = []
    residual = pd.concat([p for p in residual_parts if not p.empty], ignore_index=True)
    sequence = pd.concat([p for p in sequence_parts if not p.empty], ignore_index=True)
    plays_stats_dir = DATA / "raw_cache/plays_stats"
    actor_parts = []
    actor_audit = Counter()
    actor_game_ids = set()
    for path in sorted(plays_stats_dir.glob("gameId_*.parquet")):
        actors = pd.read_parquet(path).rename(columns={
            "gameId": "game_id", "playId": "play_id", "athleteId": "athlete_id",
            "statType": "stat_type",
        })
        actors = actors.loc[pd.to_numeric(actors.game_id, errors="coerce").isin(allowed)].copy()
        if actors.empty:
            continue
        actors["game_id"] = pd.to_numeric(actors.game_id, errors="raise").astype(int)
        actors["play_id"] = actors.play_id.astype(str)
        actors["athlete_id"] = actors.athlete_id.astype(str)
        for game, block in actors.groupby("game_id"):
            lookup = lookup_by_game.get(int(game))
            if lookup is None:
                continue
            lookup = lookup.copy()
            lookup["play_id"] = lookup.play_id.astype(str)
            result, audit = actor_game_statistics(block, lookup)
            actor_audit.update(audit)
            if not result.empty:
                actor_parts.append(result)
            actor_game_ids.add(int(game))
    if actor_parts:
        actors = pd.concat(actor_parts, ignore_index=True)
    else:
        actors = pd.DataFrame(columns=["game_id", "team"])
    for part in (residual, sequence, actors):
        if not part.empty and part.duplicated(["game_id", "team"]).any():
            raise ValueError("Current research component has duplicate game/team rows")
    current = residual.merge(sequence, on=["game_id", "team"], how="outer", validate="one_to_one")
    current = current.merge(actors, on=["game_id", "team"], how="outer", validate="one_to_one")
    historical = pd.read_parquet(ROUND_DATA / "research_game_statistics.parquet")
    combined = pd.concat([historical, current], ignore_index=True, sort=False)
    if combined.duplicated(["game_id", "team"]).any():
        raise ValueError("Historical and 2026 source game/team rows overlap")
    targets = games[["id", "season", "week", "start_date", "home_team", "away_team"]].copy()
    targets = targets.rename(columns={"id": "target_game_id", "start_date": "target_start_utc"})
    targets["target_start_utc"] = pd.to_datetime(targets.target_start_utc, utc=True)
    team_targets = pd.concat([
        targets[["target_game_id", "target_start_utc", "home_team"]].rename(columns={"home_team": "team"}),
        targets[["target_game_id", "target_start_utc", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True)
    schedule_rows = schedule.loc[schedule.id.isin(set(combined.game_id.astype(int)))].copy()
    schedule_rows["start_date"] = pd.to_datetime(schedule_rows.start_date, utc=True)
    state = trailing_research_state(combined, schedule_rows, team_targets)
    if state.target_game_id.nunique() != len(games) or len(state) != len(games) * 2:
        raise ValueError("F13–F16 2026 target state is not complete")
    if not (pd.to_datetime(state.feature_available_utc, utc=True)
            < state.target_game_id.map(games.set_index("id").start_date).pipe(pd.to_datetime, utc=True)).all():
        raise ValueError("A F13–F16 source state is not pre-target")
    if state.latest_source_game_id.eq(state.target_game_id).any():
        raise ValueError("A target game contributed to its F13–F16 state")
    state_path = RESEARCH / "f13_f16_target_state_2026.parquet"
    RESEARCH.mkdir(parents=True, exist_ok=True)
    state.to_parquet(state_path, index=False, compression="zstd")
    receipt = {
        "status": "features_ready", "target_games": int(len(games)),
        "target_team_rows": len(state), "source_game_rows_2026": int(current.game_id.nunique()),
        "source_games_with_plays_stats": len(actor_game_ids),
        "reporting_lag_hours": 48, "target_game_excluded": True,
        "market_features_used": False, "sequence_audit": dict(seq_audit),
        "actor_audit": dict(actor_audit), "historical_stats_sha256": sha256_file(ROUND_DATA / "research_game_statistics.parquet"),
        "state_sha256": sha256_file(state_path),
    }
    (RESEARCH / "features_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(receipt, indent=2, default=str), flush=True)
    return state


def target_state(state: pd.DataFrame) -> pd.DataFrame:
    state = state.rename(columns={"latest_source_game_utc": "latest_source_game_utc_f13"})
    states = [pd.read_parquet(DATA / "f09_predictions/f09_target_state.parquet"),
              pd.read_parquet(DATA / "f10_features/f10_target_state.parquet"),
              pd.read_parquet(DATA / "f11_predictions/f11_target_state.parquet"),
              pd.read_parquet(DATA / "f12_features/f12_target_state.parquet")]
    manifest_path = ARCHIVE / "fingerprints/F12_F_a/feature_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected_states = [states[0]]
    for generation, frame in zip(("F10", "F11", "F12"), states[1:]):
        names = [record["name"] for record in manifest if record["generation"] == generation]
        if generation == "F11":
            names = [name for name in frame if name.startswith("prior_staff_")]
        selected_states.append(frame[["target_game_id", "team", *names]])
    combined = selected_states[0]
    for frame in selected_states[1:]:
        combined = combined.merge(frame, on=["target_game_id", "team"], validate="one_to_one")
    names = [c for c in state if c not in {
        "target_game_id", "team", "latest_source_game_id", "latest_source_game_utc_f13",
        "feature_available_utc", "source_game_count",
    }]
    return combined.merge(state[["target_game_id", "team", *names]],
                          on=["target_game_id", "team"], validate="one_to_one")


def extended_target_matrix(state: pd.DataFrame, evidence: dict):
    games = pd.read_parquet(ROOT / "data/raw/cfbd/v2/games/2026.parquet")
    games = games.loc[games.season_type.astype(str).str.lower().eq("regular")
                      & games.completed.fillna(False).astype(bool)
                      & games.home_classification.astype(str).str.lower().eq("fbs")
                      & games.away_classification.astype(str).str.lower().eq("fbs")].copy()
    games["kickoff"] = pd.to_datetime(games.start_date, utc=True)
    games["week"] = pd.to_numeric(games.week, errors="raise").astype(int)
    ladder = pd.read_parquet(ROOT / "data/publication/2026/weekly_operations/week_06/fingerprint_ladder_v3/canonical_fingerprint.parquet")
    week_key = ladder.loc[ladder.keys_season.eq(2026)]
    names = evidence["source_features"]
    extra = [name for name in names if name not in ladder.columns]
    if set(extra) - set(state.columns):
        raise ValueError(f"Missing current-season dynamic states: {sorted(set(extra)-set(state.columns))}")
    frames = []
    for game in games.itertuples(index=False):
        row = {"target_game_id": int(game.id), "season": 2026, "week": int(game.week),
               "target_start_utc": game.kickoff, "home_team": game.home_team,
               "away_team": game.away_team, "next_game_margin": float(game.home_points-game.away_points)}
        for side, team in (("home", game.home_team), ("away", game.away_team)):
            base = week_key.loc[week_key.keys_week.eq(int(game.week)-1)
                                & week_key.keys_team.eq(team)]
            if len(base) != 1:
                raise ValueError(f"Expected one F0–F8 row for {team}, week {game.week}; found {len(base)}")
            dynamic = state.loc[state.target_game_id.eq(int(game.id)) & state.team.eq(team)]
            if len(dynamic) != 1:
                raise ValueError(f"Expected one dynamic state for {team}, game {game.id}")
            for name in names:
                row[f"{side}__{name}"] = base.iloc[0][name] if name in base.columns else dynamic.iloc[0][name]
        frames.append(row)
    raw = pd.DataFrame(frames)
    parent_manifest = json.loads((ARCHIVE / "fingerprints/F12_F_a/feature_manifest.json").read_text())
    record_by_name = {r["name"]: r for r in parent_manifest}
    records = [record_by_name.get(name, research_record(name)) for name in names]
    source = np.column_stack([raw[f"home__{name}"].to_numpy(float) for name in names]
                             + [raw[f"away__{name}"].to_numpy(float) for name in names])
    x = source_to_matchup(source, records)
    meta = raw[["target_game_id", "season", "week", "target_start_utc", "home_team", "away_team", "next_game_margin"]]
    lines_path = next((path for path in (DATA / "2026_f0_f12_scored_model_games.parquet",
                                         DATA / "2026_f0_f10_scored_model_games.parquet",
                                         DATA / "2026_f0_f8_scored_model_games.parquet") if path.exists()), None)
    if lines_path is None:
        raise FileNotFoundError("No current-season evaluation line sidecar")
    lines = pd.read_parquet(lines_path).drop_duplicates("game_id").set_index("game_id").market_spread_close
    spread = meta.target_game_id.map(lines).to_numpy(float)
    return x, meta, spread


def train_generation(generation: str, state: pd.DataFrame) -> dict:
    stage = generation
    x_hist, meta, _, evidence = load_stage_matrix(ROUND_DATA, stage)
    x_target, target_meta, spread = extended_target_matrix(state, evidence)
    columns = [f"matchup__{name}" for name in evidence["source_features"]]
    X = pd.DataFrame(x_hist, columns=columns)
    T = pd.DataFrame(x_target, columns=columns)
    y = meta.next_game_margin.to_numpy(float)
    cfg = json.loads((ROOT / "configs/experiments/nextgen_rounds_scientific_models_v1.json").read_text())
    points = json.loads((ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json").read_text())
    out = DATA / f"{generation.lower()}_predictions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "predictions.parquet"
    rows = pd.read_parquet(path).to_dict("records") if path.exists() else []
    done = {str(row["model_name"]).rsplit("_", 1)[-1] for row in rows}
    for row in rows:
        row["model_family"] = FAMILIES.get(str(row["model_name"]).rsplit("_", 1)[-1], row["model_family"])
    for arch in MODEL_IDS:
        if arch in done:
            continue
        print(f"{generation} refit {arch}", flush=True)
        specs = ([('setpoint', point) for point in points[arch]] if arch in ("M2", "M4")
                 else [('seed', seed) for seed in ([1701] if arch == "M3" else cfg["seeds"])])
        predictions = []
        for kind, spec in specs:
            if kind == "setpoint":
                saved = pd.read_parquet(ROUND_DATA / "experiments" / generation / arch / spec["id"] / "predictions.parquet")
                residuals = (saved.actual_margin - saved.predicted_margin).to_numpy(float)
                model = build_estimator(arch, spec)
                model.fit(X, y)
                pred = np.asarray(model.predict(T), dtype=float)
            else:
                seed = int(spec)
                saved = pd.read_parquet(ROUND_DATA / "scientific_model_runs" / "experiments" / generation / arch / f"seed_{seed}" / "predictions.parquet")
                residuals = (saved.actual_margin - saved.predicted_margin).to_numpy(float)
                model, _ = make_model(arch, seed, generation)
                model.train(X, y)
                pred = np.asarray(model.predict_margin(T), dtype=float).reshape(-1)
            predictions.append((pred, residual_probability(pred, residuals)))
        pred = np.mean([part[0] for part in predictions], axis=0)
        prob = np.mean([part[1] for part in predictions], axis=0)
        for index, game in target_meta.reset_index(drop=True).iterrows():
            rows.append({"game_id": int(game.target_game_id), "season": 2026, "week": int(game.week),
                         "home_team": game.home_team, "away_team": game.away_team,
                         "model_name": f"scientific_{generation}_{arch}", "model_family": FAMILIES[arch],
                         "fingerprint": generation, "pred_home_margin": float(pred[index]),
                         "pred_home_win_probability": float(prob[index]),
                         "market_spread_close": float(spread[index]) if np.isfinite(spread[index]) else np.nan,
                         "market_win_probability": np.nan})
        pd.DataFrame(rows).to_parquet(path, index=False, compression="zstd")
        done.add(arch)
    receipt = {"status": "success" if len(done) == len(MODEL_IDS) else "partial", "generation": generation,
               "models": sorted(done), "games": int(target_meta.target_game_id.nunique()),
               "features": len(evidence["source_features"]), "training_through_season": 2025,
               "calibration_years": [2024, 2025], "market_features_used": False,
               "M3_replicates": 1, "other_seed_replicates": 3, "M2_M4_setpoints": 10,
               "target_matrix_sha256": sha256_file(path)}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    state_path = RESEARCH / "f13_f16_target_state_2026.parquet"
    state = pd.read_parquet(state_path) if state_path.exists() else current_research_stats(read_schedule())
    combined = target_state(state)
    all_receipts = []
    for generation in ("F13", "F14", "F15", "F16"):
        all_receipts.append(train_generation(generation, combined))
    print(json.dumps(all_receipts, indent=2), flush=True)


if __name__ == "__main__":
    main()
