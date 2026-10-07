#!/usr/bin/env python3
"""Build F18/F19 target inputs from archived pregame states, without scores.

This prepares a candidate input matrix; it does not select or score a model.
Only declared F12 A/B/C formulas and the saved F13–F16 state are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from constraint_free_search import digest
from constraint_free_broad_universe import load_broad_historical
from nextgen_rounds_train import _record, stage_sources
from gridiron_ml.experiments.constraint_free import SNAPSHOT_MARKET_COLUMNS
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import source_to_matchup

ARCHIVE = Path("/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen")
CURRENT = ROOT / "data/what_if_2026_fingerprints"
LADDER = ROOT / "data/publication/2026/weekly_operations/week_06/fingerprint_ladder_v3/canonical_fingerprint.parquet"
GAMES = ROOT / "data/raw/cfbd/v2/games/2026.parquet"
MARKET = CURRENT / "f17_market_features/f17_market_target_state_2026.parquet"
DEFAULT_OUTPUT = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/broad_target_inputs")
KEY = ["target_game_id", "team"]
STATE_PATHS = {
    "F09": CURRENT / "f09_predictions/f09_target_state.parquet",
    "F10": CURRENT / "f10_features/f10_target_state.parquet",
    "F11": CURRENT / "f11_predictions/f11_target_state.parquet",
    "F12": CURRENT / "f12_features/f12_target_state.parquet",
    "research": CURRENT / "f13_f16_features/f13_f16_target_state_2026.parquet",
}


def projection_digest(frame: pd.DataFrame) -> str:
    """Bind only the explicitly selected columns, never a scored source file."""
    header = json.dumps([(name, str(frame[name].dtype)) for name in frame.columns],
                        sort_keys=True).encode()
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(np.uint64).tobytes()
    return hashlib.sha256(header + values).hexdigest()


def target_schedule() -> pd.DataFrame:
    # Scores and any scored 2026 files are absent from this read list.
    columns = ["id", "season", "week", "start_date", "home_team", "away_team",
               "season_type", "completed", "home_classification", "away_classification"]
    games = pd.read_parquet(GAMES, columns=columns)
    games = games.loc[
        games.season.eq(2026)
        & games.season_type.astype(str).str.lower().eq("regular")
        & games.completed.fillna(False).astype(bool)
        & games.home_classification.astype(str).str.lower().eq("fbs")
        & games.away_classification.astype(str).str.lower().eq("fbs")
    ].copy()
    games["target_start_utc"] = pd.to_datetime(games.start_date, utc=True)
    games = games.rename(columns={"id": "target_game_id"}).sort_values("target_game_id")
    if len(games) != 271 or games.target_game_id.duplicated().any():
        raise ValueError("2026 target schedule differs from frozen 271-game cohort")
    return games[["target_game_id", "week", "target_start_utc", "home_team", "away_team"]].reset_index(drop=True)


def source_frames(games: pd.DataFrame, manifests: dict[str, list[dict]]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rows = []
    for game in games.itertuples(index=False):
        for side, team in (("home", game.home_team), ("away", game.away_team)):
            rows.append({"target_game_id": int(game.target_game_id), "team": team,
                         "side": side, "week": int(game.week),
                         "target_start_utc": game.target_start_utc})
    teams = pd.DataFrame(rows)
    index = pd.MultiIndex.from_frame(teams[KEY])
    if index.has_duplicates:
        raise ValueError("Duplicate target team")
    a_names = {record["name"] for record in manifests["a"]}
    needed = {generation: set() for generation in ("F06", "F09", "F10", "F11", "F12")}
    for record in manifests["a"]:
        needed[record["generation"]].add(record["name"])
    for design in "bc":
        for record in manifests[design]:
            if record["name"] in a_names:
                continue
            if record["operation"] == "identity":
                needed[record["generation"]].add(record["name"])
            else:
                needed[record["generation"]].update(record["source_inputs"])
    ladder_columns = ["keys_season", "keys_week", "keys_team", "keys_game_date",
                      "keys_game_id", *sorted(needed["F06"])]
    ladder = pd.read_parquet(LADDER, columns=ladder_columns)
    ladder = ladder.loc[ladder.keys_season.eq(2026)].copy()
    ladder["target_week"] = ladder.keys_week.astype(int) + 1
    prior = teams.merge(ladder, left_on=["team", "week"],
                        right_on=["keys_team", "target_week"], how="left", validate="many_to_one")
    if len(prior) != len(teams) or prior.keys_team.isna().any():
        raise ValueError("Missing prior-week canonical ladder state")
    source_date = pd.to_datetime(prior.keys_game_date, utc=True, errors="coerce")
    if (source_date.notna() & ~(source_date < prior.target_start_utc)).any():
        raise ValueError("Prior-week ladder state is dated after target kickoff")
    if prior.keys_game_id.eq(prior.target_game_id).any():
        raise ValueError("Ladder contains the target game")
    frames = {"F06": prior.set_index(KEY).reindex(index)}
    cutoff_columns = {
        "F09": ["latest_source_available_utc"],
        "F10": ["latest_source_available_utc", "recruitment_available_utc"],
        "F11": ["staff_available_utc"],
        "F12": ["latest_source_available_utc_st", "latest_source_available_utc_ppa"],
        "research": ["feature_available_utc"],
    }
    target_start = pd.to_datetime(teams.target_start_utc, utc=True).to_numpy()
    for generation, path in STATE_PATHS.items():
        source_names = (set(stage_sources("F16")) if generation == "research"
                        else needed[generation])
        schema = set(pq.read_schema(path).names)
        identities = [name for name in ("latest_source_game_id", "latest_source_game_id_st",
                                        "latest_source_game_id_ppa") if name in schema]
        columns = [*KEY, *sorted(source_names | set(cutoff_columns[generation]) | set(identities))]
        if set(columns) - schema:
            raise ValueError(f"{generation} missing declared source columns: {sorted(set(columns) - schema)}")
        frame = pd.read_parquet(path, columns=columns)
        if frame.duplicated(KEY).any() or len(frame) != len(teams):
            raise ValueError(f"{generation} target state has duplicate or missing team rows")
        if set(map(tuple, frame[KEY].itertuples(index=False, name=None))) != set(index):
            raise ValueError(f"{generation} target identities differ")
        frame = frame.set_index(KEY).reindex(index)
        for name in cutoff_columns[generation]:
            available = pd.to_datetime(frame[name], utc=True, errors="coerce").to_numpy()
            nonmissing = ~pd.isna(available)
            if (available[nonmissing] >= target_start[nonmissing]).any():
                raise ValueError(f"{generation} {name} is not pregame")
        for name in ("latest_source_game_id", "latest_source_game_id_st",
                     "latest_source_game_id_ppa"):
            if name in frame and frame[name].eq(teams.target_game_id.to_numpy()).any():
                raise ValueError(f"{generation} contains its target game")
        frames[generation] = frame
    return frames, teams


def formula(record: dict, source: pd.DataFrame) -> np.ndarray:
    name, operation = record["name"], record["operation"]
    if operation == "identity":
        if name not in source:
            raise ValueError(f"Missing declared identity output: {name}")
        return source[name].to_numpy(float)
    inputs = record["source_inputs"]
    if set(inputs) - set(source):
        raise ValueError(f"Missing declared inputs for {name}: {sorted(set(inputs) - set(source))}")
    data = source[inputs].to_numpy(float)
    complete = np.isfinite(data).all(axis=1)
    result = np.full(len(source), np.nan)
    if operation == "mean":
        result[complete] = data[complete].mean(axis=1)
    elif operation == "product" and len(inputs) == 2:
        result[complete] = data[complete, 0] * data[complete, 1]
    elif operation == "difference" and len(inputs) == 2:
        result[complete] = data[complete, 0] - data[complete, 1]
    elif operation == "ratio" and len(inputs) == 2:
        valid = complete & (data[:, 1] > 0)
        result[valid] = data[valid, 0] / data[valid, 1]
    else:
        raise ValueError(f"Unimplemented declared formula: {name} {operation}")
    return result


def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    games = target_schedule()
    manifests = {}
    sources = {"selected_target_schedule": projection_digest(games)}
    for design in "abc":
        path = ARCHIVE / "fingerprints" / f"F12_F_{design}" / "feature_manifest.json"
        manifests[design] = json.loads(path.read_text())
        sources[str(path)] = digest(path)
    frames, teams = source_frames(games, manifests)
    sources["selected_prior_week_ladder"] = projection_digest(frames["F06"])
    for generation, path in STATE_PATHS.items():
        sources[f"selected_{generation}_state:{path}"] = projection_digest(frames[generation])
    a = manifests["a"]
    a_names = {record["name"] for record in a}
    extra = {}
    for design in "bc":
        for record in manifests[design]:
            if record["name"] not in a_names:
                if record["name"] in extra:
                    prior = extra[record["name"]]
                    if (prior["operation"], prior["source_inputs"], prior["matchup_formula"]) != (
                        record["operation"], record["source_inputs"], record["matchup_formula"]
                    ):
                        raise ValueError(f"F12 B/C formula disagreement: {record['name']}")
                extra[record["name"]] = record
    if len(extra) != 98:
        raise ValueError(f"Expected 98 added feature formulas, found {len(extra)}")
    values = {}
    for record in [*a, *extra.values()]:
        generation = record["generation"]
        source = frames[generation]
        name = record["name"]
        values[name] = (source[name].to_numpy(float) if name in a_names
                        else formula(record, source))
    if len(values) != 437:
        raise ValueError("F12 A/B/C team-state union changed")
    names = [record["name"] for record in a] + sorted(extra)
    for record in [*a, *(extra[name] for name in sorted(extra))]:
        if record.get("market_derived") or record.get("target_derived"):
            raise ValueError(f"Inadmissible F18 feature: {record['name']}")
    team_matrix = np.column_stack([values[name] for name in names])
    home = team_matrix[teams.side.eq("home").to_numpy()]
    away = team_matrix[teams.side.eq("away").to_numpy()]
    if np.isinf(home).any() or np.isinf(away).any():
        raise ValueError("Infinite F12 feature value")
    all_matchup = source_to_matchup(
        np.column_stack((home, away)),
        [*a, *(extra[name] for name in sorted(extra))],
    )
    research_names = list(stage_sources("F16"))
    research = frames["research"]
    if set(research_names) - set(research):
        raise ValueError("F13–F16 target state has missing columns")
    research_team = research[research_names].to_numpy(float)
    research_matchup = source_to_matchup(
        np.column_stack((research_team[teams.side.eq("home").to_numpy()],
                         research_team[teams.side.eq("away").to_numpy()])),
        [_record(name) for name in research_names],
    )
    a_count = len(a)
    f18 = np.column_stack((all_matchup[:, :a_count], research_matchup,
                           all_matchup[:, a_count:]))
    historical, _, records, evidence = load_broad_historical("F18")
    names_f18 = [record.name for record in records]
    if f18.shape[1] != historical.shape[1] or names_f18 != [
        *(f"matchup__{name}" for name in [record["name"] for record in a]),
        *(f"matchup__{name}" for name in research_names),
        *(f"matchup__{name}" for name in sorted(extra)),
    ]:
        raise ValueError("2026 matrix differs from broad historical column contract")
    matrix = pd.DataFrame(f18, columns=names_f18)
    matrix.insert(0, "target_game_id", games.target_game_id.astype(int))
    meta = games.copy()
    report = {"games": len(games), "features_f18": f18.shape[1],
              "extra_bc_features": len(extra), "no_2026_outcome_columns_read": True,
              "extra_feature_names": evidence["extra_source_features"],
              "source_sha256": sources,
              "missing_values_per_feature": {name: int(matrix[name].isna().sum())
                                             for name in names_f18},
              "market_not_included": True}
    return matrix, meta, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    matrix, meta, report = build()
    market_columns = ["target_game_id", *SNAPSHOT_MARKET_COLUMNS,
                      "snapshot_timestamp_utc", "market_snapshot_captured_pre_kickoff",
                      "market_quote_timestamp_available"]
    market = pd.read_parquet(MARKET, columns=market_columns)
    if market.target_game_id.duplicated().any() or set(market.target_game_id) != set(meta.target_game_id):
        raise ValueError("F19 market snapshot cohort differs from F18")
    market = meta[["target_game_id", "target_start_utc"]].merge(
        market, on="target_game_id", validate="one_to_one")
    captured = market.market_snapshot_captured_pre_kickoff.fillna(False).astype(bool)
    timestamp = pd.to_datetime(market.snapshot_timestamp_utc, utc=True, errors="coerce")
    if (captured & ~(timestamp < market.target_start_utc)).any():
        raise ValueError("F19 market snapshot is not pregame")
    if market.market_quote_timestamp_available.fillna(False).astype(bool).any():
        raise ValueError("Historical F19 quote-time limitation was silently changed")
    if market.loc[~captured, list(SNAPSHOT_MARKET_COLUMNS)].notna().any().any():
        raise ValueError("F19 has market data without a pregame snapshot")
    if int(captured.sum()) != 263:
        raise ValueError("F19 pregame market snapshot coverage changed")
    f19 = matrix.merge(market[["target_game_id", *SNAPSHOT_MARKET_COLUMNS]],
                       on="target_game_id", validate="one_to_one")
    _, _, records19, _ = load_broad_historical("F19")
    if list(f19.columns[1:]) != [record.name for record in records19]:
        raise ValueError("2026 F19 inputs differ from broad historical column contract")
    args.output.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(args.output / "f18_inputs.parquet", index=False, compression="zstd")
    f19.to_parquet(args.output / "f19_inputs.parquet", index=False, compression="zstd")
    meta.to_parquet(args.output / "target_schedule_without_scores.parquet", index=False,
                    compression="zstd")
    report["f18_inputs_sha256"] = digest(args.output / "f18_inputs.parquet")
    report["f19_inputs_sha256"] = digest(args.output / "f19_inputs.parquet")
    report["features_f19"] = f19.shape[1] - 1
    report["f19_pregame_market_games"] = int(captured.sum())
    report["historical_market_quote_timing"] = "unverified"
    report["source_sha256"][f"selected_market_state:{MARKET}"] = projection_digest(market)
    (args.output / "receipt.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("games", "features_f18", "features_f19",
                                             "extra_bc_features", "f19_pregame_market_games",
                                             "no_2026_outcome_columns_read", "f18_inputs_sha256",
                                             "f19_inputs_sha256")},
                     indent=2))


if __name__ == "__main__":
    main()
