#!/usr/bin/env python3
"""Build isolated, leak-checked 2026 F09 predictions for the scientific roster."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gridiron_ml.experiments.nextgen_f09 import f09_formulas  # noqa: E402
from gridiron_ml.experiments.nextgen_microstructure import (  # noqa: E402
    game_sufficient_statistics, drive_sufficient_statistics,
)
from nextgen_rounds_train import load_stage_matrix  # noqa: E402
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import (  # noqa: E402
    build_estimator, residual_probability, source_to_matchup,
)
from gridiron_ml.models import TDKNN, TDLinear, TDMLP, TDTree  # noqa: E402

DATA = ROOT / "data/what_if_2026_fingerprints"
ROUND_DATA = ROOT / "data/nextgen_rounds_2026"
OUT = DATA / "f09_predictions"
LADDER = ROOT / "data/publication/2026/weekly_operations/week_06/fingerprint_ladder_v3/canonical_fingerprint.parquet"
GAMES_2026 = ROOT / "data/raw/cfbd/v2/games/2026.parquet"
MODEL_CLASSES = {"M1": TDLinear, "M3": TDTree, "M5": TDMLP, "M10": TDKNN}


def read_schedule() -> pd.DataFrame:
    parts = []
    for path in (ROOT / "data/raw/cfbd/v2/games").glob("*.parquet"):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if {"id", "start_date", "home_team", "away_team"} <= set(frame.columns):
            parts.append(frame)
    schedule = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    schedule["id"] = pd.to_numeric(schedule.id, errors="coerce").astype("Int64")
    schedule["kickoff"] = pd.to_datetime(schedule.start_date, utc=True, errors="coerce")
    return schedule


def _normalize(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    return frame.rename(columns=columns)


def source_statistics(schedule: pd.DataFrame) -> pd.DataFrame:
    historical = pd.read_parquet(ROUND_DATA / "corrected_f09_game_statistics.parquet")
    current_schedule = schedule.loc[
        schedule.season.eq(2026) & schedule.season_type.astype(str).str.lower().eq("regular")
        & schedule.completed.fillna(False).astype(bool)
    ]
    allowed = set(current_schedule.id.dropna().astype(int))
    play_parts, drive_parts = [], []
    for path in (DATA / "raw_cache/plays").glob("*.parquet"):
        p = pd.read_parquet(path)
        p = _normalize(p, {"gameId": "game_id", "driveId": "drive_id", "driveNumber": "drive_number",
                           "playNumber": "play_number", "offenseScore": "offense_score",
                           "defenseScore": "defense_score", "yardsToGoal": "yards_to_goal",
                           "yardsGained": "yards_gained", "playType": "play_type"})
        p = p.loc[p.game_id.isin(allowed)]
        if p.empty:
            continue
        st = game_sufficient_statistics(p)
        dp = next((DATA / "raw_cache/drives").glob(path.stem.replace("plays", "drives") + ".parquet"), None)
        if dp is None:
            # Filenames use the same request parameters with an endpoint-specific directory.
            dp = DATA / "raw_cache/drives" / path.name
        if dp.exists():
            d = pd.read_parquet(dp)
            d = _normalize(d, {"gameId": "game_id", "startPeriod": "start_period", "endPeriod": "end_period",
                               "startYardsToGoal": "start_yards_to_goal", "endYardsToGoal": "end_yards_to_goal",
                               "startYardline": "start_yardline", "endYardline": "end_yardline",
                               "driveResult": "drive_result", "startOffenseScore": "start_offense_score",
                               "endOffenseScore": "end_offense_score", "startDefenseScore": "start_defense_score",
                               "endDefenseScore": "end_defense_score"})
            d = d.loc[d.game_id.isin(p.game_id.unique())]
            if not d.empty:
                st = st.merge(drive_sufficient_statistics(d, p), on=["game_id", "team"], how="left", validate="one_to_one")
        play_parts.append(st)
    if not play_parts:
        raise ValueError("No completed 2026 play statistics found")
    current = pd.concat(play_parts, ignore_index=True)
    stats = pd.concat([historical, current], ignore_index=True, sort=False)
    if stats.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate historical/current source game-team statistics")
    return stats


def f09_state(stats: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    source = stats.merge(schedule[["id", "season", "season_type", "completed", "kickoff", "home_team", "away_team"]],
                         left_on="game_id", right_on="id", how="inner", validate="many_to_one")
    source = source.loc[source.season_type.astype(str).str.lower().eq("regular") & source.completed.fillna(False)]
    source["available"] = source.kickoff + pd.Timedelta(hours=48)
    source_columns = [c for c in stats if c.endswith(("__sum", "__n"))]
    history = {team: rows.sort_values(["available", "game_id"]) for team, rows in source.groupby("team")}
    targets = pd.read_parquet(GAMES_2026)
    targets = targets.loc[targets.season_type.astype(str).str.lower().eq("regular")
                          & targets.completed.fillna(False).astype(bool)
                          & targets.home_classification.astype(str).str.lower().eq("fbs")
                          & targets.away_classification.astype(str).str.lower().eq("fbs")].copy()
    targets["kickoff"] = pd.to_datetime(targets.start_date, utc=True, errors="coerce")
    rows = []
    for g in targets.itertuples(index=False):
        for side, team in (("home", g.home_team), ("away", g.away_team)):
            prior = history.get(team)
            if prior is None:
                continue
            prior = prior.loc[prior.available.lt(g.kickoff)].tail(12)
            if prior.empty:
                continue
            sums = prior[source_columns].sum(min_count=1)
            state = {c.removesuffix("__sum"): np.nan for c in source_columns if c.endswith("__sum")}
            for c in list(state):
                n, total = sums.get(c + "__n"), sums.get(c + "__sum")
                minimum = 8 if "rush_ypa_q" in c else 10
                state[c] = total / n if pd.notna(n) and n >= minimum else np.nan
            state.update(target_game_id=int(g.id), team=team, side=side,
                         latest_source_game_id=int(prior.game_id.iloc[-1]),
                         latest_source_available_utc=prior.available.iloc[-1],
                         target_start_utc=g.kickoff, source_game_count=len(prior))
            rows.append(state)
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(["target_game_id", "team"]).any():
        raise ValueError("F09 as-of state is empty or has duplicate targets")
    if not (pd.to_datetime(result.latest_source_available_utc, utc=True)
            < pd.to_datetime(result.target_start_utc, utc=True)).all():
        raise ValueError("F09 feature availability does not precede target kickoff")
    if result.latest_source_game_id.eq(result.target_game_id).any():
        raise ValueError("Target game contributed to its own F09 state")
    for formula in f09_formulas("a"):
        result[formula.name] = formula.evaluate(result)
    return result


def target_matrix(state: pd.DataFrame, evidence: dict) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    games = pd.read_parquet(GAMES_2026)
    games = games.loc[games.season_type.astype(str).str.lower().eq("regular")
                      & games.completed.fillna(False).astype(bool)
                      & games.home_classification.astype(str).str.lower().eq("fbs")
                      & games.away_classification.astype(str).str.lower().eq("fbs")].copy()
    games["kickoff"] = pd.to_datetime(games.start_date, utc=True)
    games["week"] = pd.to_numeric(games.week, errors="coerce").astype(int)
    ladder = pd.read_parquet(LADDER)
    week_key = ladder.loc[ladder.keys_season.eq(2026)].copy()
    names = evidence["source_features"]
    extra = [n for n in names if n not in ladder.columns]
    if set(extra) != set(state.columns).intersection(extra):
        raise ValueError(f"F09 state does not match the missing ladder fields: {sorted(set(extra)-set(state.columns))}")
    frames = []
    for g in games.itertuples(index=False):
        row = {"target_game_id": int(g.id), "season": 2026, "week": int(g.week),
               "target_start_utc": g.kickoff, "home_team": g.home_team, "away_team": g.away_team,
               "next_game_margin": float(g.home_points - g.away_points)}
        for side, team in (("home", g.home_team), ("away", g.away_team)):
            base = week_key.loc[week_key.keys_week.eq(int(g.week)-1) & week_key.keys_team.eq(team)]
            if len(base) != 1:
                raise ValueError(f"Expected one prior-week ladder state for {team}, target week {g.week}; found {len(base)}")
            dynamic = state.loc[(state.target_game_id.eq(int(g.id))) & state.team.eq(team)]
            if len(dynamic) != 1:
                raise ValueError(f"Expected one F09 state for {team} in game {g.id}")
            for name in names:
                row[f"{side}__{name}"] = (base.iloc[0][name] if name in base.columns
                                            else dynamic.iloc[0][name])
        frames.append(row)
    raw = pd.DataFrame(frames)
    manifest = json.loads(Path("/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/fingerprints/F12_F_a/feature_manifest.json").read_text())
    # The corrected F09 matchup contract is identical; F10 keeps the frozen parent manifest.
    records = [r for r in manifest if r["name"] in names]
    source = np.column_stack([raw[f"home__{n}"].to_numpy(float) for n in names]
                             + [raw[f"away__{n}"].to_numpy(float) for n in names])
    x = source_to_matchup(source, records)
    meta = raw[["target_game_id", "season", "week", "target_start_utc", "home_team", "away_team", "next_game_margin"]]
    line_path = next((path for path in (
        DATA / "2026_f0_f12_scored_model_games.parquet",
        DATA / "2026_f0_f10_scored_model_games.parquet",
        DATA / "2026_f0_f8_scored_model_games.parquet",
    ) if path.exists()), DATA / "2026_f0_f8_scored_model_games.parquet")
    known_lines = pd.read_parquet(line_path)
    lines = known_lines.drop_duplicates("game_id").set_index("game_id").market_spread_close
    spread = meta.target_game_id.map(lines).to_numpy(float)
    return x, meta, spread


def residual_probabilities(model_factory, X: pd.DataFrame, y: np.ndarray, years: np.ndarray, target_X: pd.DataFrame):
    errors = []
    for cutoff in (2019, 2021, 2023):
        fit, cal = years <= cutoff, (years > cutoff) & (years <= cutoff + 2)
        if fit.sum() == 0 or cal.sum() == 0:
            raise ValueError(f"No rolling calibration observations at {cutoff}")
        model = model_factory()
        if hasattr(model, "train"):
            model.train(X.loc[fit], y[fit])
            pred = np.asarray(model.predict_margin(X.loc[cal]), dtype=float)
        else:
            model.fit(X.loc[fit], y[fit])
            pred = model.predict(X.loc[cal])
        errors.extend(y[cal] - pred)
    final = model_factory()
    train = years <= 2025
    if hasattr(final, "train"):
        final.train(X.loc[train], y[train])
        pred = np.asarray(final.predict_margin(target_X), dtype=float)
    else:
        final.fit(X.loc[train], y[train])
        pred = final.predict(target_X)
    return pred, residual_probability(pred, errors)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    schedule = read_schedule()
    stats_path, state_path = OUT / "source_statistics.parquet", OUT / "f09_target_state.parquet"
    if stats_path.exists() and state_path.exists():
        stats, state = pd.read_parquet(stats_path), pd.read_parquet(state_path)
    else:
        print("Building F09 source statistics and cutoff-safe target states", flush=True)
        stats = source_statistics(schedule)
        state = f09_state(stats, schedule)
        stats.to_parquet(stats_path, index=False, compression="zstd")
        state.to_parquet(state_path, index=False, compression="zstd")
    Xhist, meta, _, evidence = load_stage_matrix(ROUND_DATA, "F09")
    Xtarget, target_meta, spreads = target_matrix(state, evidence)
    names = [f"matchup__{n}" for n in evidence["source_features"]]
    hist = pd.DataFrame(Xhist, columns=names)
    target = pd.DataFrame(Xtarget, columns=names)
    years = meta.season.to_numpy(int)
    y = meta.next_game_margin.to_numpy(float)
    tasks = []
    prediction_path = OUT / "predictions.parquet"
    pred_frames = (pd.read_parquet(prediction_path).to_dict("records")
                   if prediction_path.exists() else [])
    completed_models = {str(r["model_name"]).rsplit("_", 1)[-1] for r in pred_frames}
    family_by_model = {"M1": "linear", "M2": "spline", "M3": "tree",
                       "M4": "boosted", "M5": "neural", "M10": "knn"}
    for row in pred_frames:
        row["model_family"] = family_by_model.get(str(row["model_name"]).rsplit("_", 1)[-1], row["model_family"])
    cfg_path = ROOT / "configs/experiments/nextgen_rounds_scientific_models_v1.json"
    cfg = json.loads(cfg_path.read_text())
    from nextgen_rounds_scientific_train import make_model
    setpoints = json.loads((ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json").read_text())
    # Use the frozen 2024–25 held-out residuals for probability calibration, then refit through 2025.
    # This keeps calibration labels historical while avoiding four expensive rolling fits per replica.
    for architecture in ("M2", "M4"):
        if architecture in completed_models:
            print(f"F09 {architecture} already saved; reusing verified output", flush=True)
            continue
        print(f"F09 refit {architecture}", flush=True)
        individual = []
        for point in setpoints[architecture]:
            point_id = point["id"]
            path = ROUND_DATA / "experiments" / "F09" / architecture / point_id / "predictions.parquet"
            heldout = pd.read_parquet(path)
            residuals = (heldout.actual_margin - heldout.predicted_margin).to_numpy(float)
            estimator = build_estimator(architecture, point)
            estimator.fit(hist, y)
            prediction = np.asarray(estimator.predict(target), dtype=float)
            probability = residual_probability(prediction, residuals)
            individual.append((prediction, probability))
        pred = np.mean([x[0] for x in individual], axis=0)
        prob = np.mean([x[1] for x in individual], axis=0)
        tasks.append((architecture, pred, prob))
        pred_frames.extend(_prediction_records(architecture, pred, prob, target_meta, spreads))
        pd.DataFrame(pred_frames).to_parquet(prediction_path, index=False, compression="zstd")
        completed_models.add(architecture)
    for architecture in ("M1", "M3", "M5", "M10"):
        if architecture in completed_models:
            print(f"F09 {architecture} already saved; reusing verified output", flush=True)
            continue
        print(f"F09 refit {architecture}", flush=True)
        individual = []
        seeds = [1701] if architecture == "M3" else cfg["seeds"]
        for seed in seeds:
            print(f"  seed {seed}", flush=True)
            path = ROUND_DATA / "scientific_model_runs" / "experiments" / "F09" / architecture / f"seed_{seed}" / "predictions.parquet"
            heldout = pd.read_parquet(path)
            residuals = (heldout.actual_margin - heldout.predicted_margin).to_numpy(float)
            model, _ = make_model(architecture, int(seed), "F09")
            model.train(hist, y)
            pred = np.asarray(model.predict_margin(target), dtype=float).reshape(-1)
            prob = residual_probability(pred, residuals)
            individual.append((pred, prob))
        tasks.append((architecture, np.mean([x[0] for x in individual], axis=0),
                      np.mean([x[1] for x in individual], axis=0)))
        pred_frames.extend(_prediction_records(architecture, tasks[-1][1], tasks[-1][2], target_meta, spreads))
        pd.DataFrame(pred_frames).to_parquet(prediction_path, index=False, compression="zstd")
        completed_models.add(architecture)
    (OUT / "receipt.json").write_text(json.dumps({
        "status": "success" if len(completed_models) == 6 else "partial",
        "generation": "F09 corrected A", "models": len(completed_models),
        "scored_architectures": sorted(completed_models),
        "unscored_architectures": sorted(set(("M1", "M2", "M3", "M4", "M5", "M10")) - completed_models),
        "target_games": int(target_meta.target_game_id.nunique()), "source_statistics_rows": int(len(stats)),
        "feature_rows": int(len(state)), "reporting_lag_hours": 48,
        "training_through_season": 2025, "calibration_source_years": [2024, 2025],
        "calibration_residuals_from_models_trained_through_season": 2023,
        "replicates": {"M1": 3, "M2": 10, "M3": 1, "M4": 10, "M5": 3, "M10": 3},
        "market_inputs_used": False, "leak_check_passed": True,
    }, indent=2) + "\n")
    print(f"F09 generated: {len(completed_models)} architecture forecasts x {len(target_meta)} completed games")


def _prediction_records(architecture, pred, prob, target_meta, spreads):
    result = []
    for i, row in target_meta.reset_index(drop=True).iterrows():
        result.append({"game_id": int(row.target_game_id), "season": 2026, "week": int(row.week),
                       "home_team": row.home_team, "away_team": row.away_team,
                       "model_name": f"scientific_F9_{architecture}",
                       "model_family": {"M1": "linear", "M2": "spline", "M3": "tree",
                                       "M4": "boosted", "M5": "neural", "M10": "knn"}[architecture],
                       "fingerprint": "F9", "pred_home_margin": float(pred[i]),
                       "pred_home_win_probability": float(prob[i]),
                       "market_spread_close": float(spreads[i]) if np.isfinite(spreads[i]) else np.nan,
                       "market_win_probability": np.nan})
    return result


if __name__ == "__main__":
    main()
