#!/usr/bin/env python3
"""Materialize one historical scientific tier on the shared 2013-2025 game cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from gridiron_ml.experiments.publication import filter_frame_for_feature_config
from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints
from gridiron_ml.fingerprints.features import FeatureSpec
from gridiron_ml.td_run.matchups import MatchupBuilder
from nextgen_rounds_train import load_stage_matrix
from constraint_free_broad_universe import load_broad_historical

OLD_SOURCE = Path("/groups/bsavoie2/tburton2/TDNet/publication_artifacts/fingerprint_ladder_v3/canonical_fingerprint.parquet")
NEW_SOURCE = Path("/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/fingerprints/F12_F_a/values.parquet")
NEW_ROOT = ROOT / "data/nextgen_rounds_2026"
DEFAULT_OUTPUT = Path("/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1/data")
STAGES = {"F9": "F09", "F10": "F10", "F11": "F11",
          "F12": "F12_corrected", **{f"F{i}": f"F{i}" for i in range(13, 17)},
          "F17": "F17_market"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def build_cohort(output: Path) -> dict:
    source = pd.read_parquet(NEW_SOURCE, columns=["target_game_id", "season", "week", "team", "is_home",
                                                  "next_game_margin", "target_start_utc", "feature_available_utc",
                                                  "latest_source_game_id"])
    source = source.loc[source.season.between(2013, 2025)].copy()
    if source.duplicated(["target_game_id", "team"]).any():
        raise ValueError("Duplicate target/team rows in new source")
    home = source.loc[source.is_home].set_index("target_game_id").sort_index()
    away = source.loc[~source.is_home].set_index("target_game_id").sort_index()
    if not home.index.equals(away.index) or len(home) != 7358:
        raise ValueError("New source home/away cohort changed")
    if not np.allclose(home.next_game_margin, -away.next_game_margin):
        raise ValueError("New source reciprocal labels differ")
    for side in (home, away):
        if not (pd.to_datetime(side.feature_available_utc, utc=True) <
                pd.to_datetime(side.target_start_utc, utc=True)).all():
            raise ValueError("New source feature availability after kickoff")
        if side.latest_source_game_id.eq(side.index.to_series()).any():
            raise ValueError("New source target contributed to its own features")
    cohort = pd.DataFrame({
        "target_game_id": home.index.astype("int64"),
        "season": home.season.to_numpy(int), "week": home.week.to_numpy(int),
        "target_start_utc": home.target_start_utc.to_numpy(),
        "home_team": home.team.to_numpy(str), "away_team": away.team.to_numpy(str),
        "actual_margin": home.next_game_margin.to_numpy(float),
    })
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cohort.parquet"
    cohort.to_parquet(path, index=False, compression="zstd")
    receipt = {"status": "materialized", "games": len(cohort),
               "games_by_season": {str(k): int(v) for k, v in cohort.season.value_counts().sort_index().items()},
               "source": str(NEW_SOURCE), "source_sha256": sha256(NEW_SOURCE),
               "cohort_sha256": sha256(path), "latest_test_season": 2025,
               "new_source_availability_before_kickoff": True,
               "new_source_no_target_as_source": True}
    write_json(output / "cohort_receipt.json", receipt)
    return receipt


def load_old(tier: str, cohort: pd.DataFrame):
    source = pd.read_parquet(OLD_SOURCE)
    source = source.loc[source.keys_season.between(2013, 2025)].copy()
    market = tier in {"F7", "F8"}
    filtered, contract = filter_frame_for_feature_config(
        source, feature_config=tier,
        registry_path=ROOT / "configs/features/feature_registry.yaml",
        ladders_path=ROOT / "configs/features/feature_ladders.yaml", strict_registry=True)
    spec = FeatureSpec(include_market=market, allow_market_features_for_training=market)
    X, y, metadata, market_sidecar = StaticFrameFingerprints(filtered).training_block(
        range(2013, 2026), feature_spec=spec)
    builder = MatchupBuilder(representation="unit_matchup")
    values, paired, _ = builder.matchups(X, metadata, market_df=market_sidecar, y=y)
    ids = paired.next_game_id.astype("int64")
    if ids.duplicated().any():
        raise ValueError(f"{tier} duplicate old target game")
    values.insert(0, "target_game_id", ids.to_numpy())
    old_labels = paired.set_index(ids).y.astype(float)
    target = cohort.set_index("target_game_id")
    common = target.index.intersection(ids)
    mismatches = old_labels.reindex(common).ne(target.loc[common, "actual_margin"])
    if len(common) != len(target):
        raise ValueError(f"{tier} lacks {len(target)-len(common)} common games")
    old_source = source.loc[source.next_game_id.notna() & source.next_game_is_home.notna()].copy()
    old_source["target_game_id"] = old_source.next_game_id.astype("int64")
    old_source = old_source.loc[old_source.target_game_id.isin(target.index)]
    if old_source.duplicated(["target_game_id", "keys_team"]).any():
        raise ValueError("Old source has duplicate team/target rows")
    kickoff = old_source.target_game_id.map(target.target_start_utc)
    old_date = pd.to_datetime(old_source.keys_game_date, utc=True)
    kickoff_date = pd.to_datetime(kickoff, utc=True)
    missing_date = old_date.isna()
    if (old_source.loc[missing_date, "keys_week"] != 0).any():
        raise ValueError("Old source has missing dates outside synthetic preseason week zero")
    if (old_date.dt.date >= kickoff_date.dt.date).fillna(False).any():
        raise ValueError("Old source game date is not before target kickoff date")
    if old_source.keys_game_id.eq(old_source.target_game_id).any():
        raise ValueError("Old target game contributed to its own features")
    return values, {"source": str(OLD_SOURCE), "source_sha256": sha256(OLD_SOURCE),
                    "selected_source_features": contract["selected_features"],
                    "prior_target_disagreement_count": int(mismatches.sum()),
                    "prior_target_disagreement_game_ids": common[mismatches.to_numpy()].astype(int).tolist(),
                    "source_date_before_target_date": True,
                    "missing_source_date_only_in_preseason_week_zero": True,
                    "source_game_not_target_game": True,
                    "pregame_market_quote_timing": "unverified" if market else "not_applicable"}


def load_new(tier: str):
    if tier in STAGES:
        X, metadata, _, evidence = load_stage_matrix(NEW_ROOT, STAGES[tier])
        names = [f"matchup__{name}" for name in evidence["source_features"]] + list(evidence["market_features"])
    else:
        X, metadata, records, evidence = load_broad_historical(tier)
        names = [r.name for r in records]
    if X.shape[1] != len(names):
        raise ValueError(f"{tier} matrix feature names do not match its width")
    frame = pd.DataFrame(X, columns=names)
    frame.insert(0, "target_game_id", metadata.target_game_id.astype("int64").to_numpy())
    if frame.target_game_id.duplicated().any():
        raise ValueError(f"{tier} duplicate target game")
    return frame, {"source": str(NEW_ROOT), "source_evidence": evidence,
                   "pregame_market_quote_timing": "unverified" if tier in {"F17", "F19"} else "not_applicable"}


def materialize(tier: str, output: Path) -> dict:
    cohort_path = output / "cohort.parquet"
    cohort_receipt = json.loads((output / "cohort_receipt.json").read_text())
    if sha256(cohort_path) != cohort_receipt["cohort_sha256"]:
        raise ValueError("Frozen common cohort changed")
    cohort = pd.read_parquet(cohort_path)
    source, evidence = load_old(tier, cohort) if tier in {f"F{i}" for i in range(9)} else load_new(tier)
    target = cohort.set_index("target_game_id")
    source = source.set_index("target_game_id").reindex(target.index)
    if source.isna().all(axis=1).any():
        raise ValueError(f"{tier} missing games on common cohort")
    if source.columns.duplicated().any():
        raise ValueError(f"{tier} duplicate feature names")
    frame = pd.concat([cohort.reset_index(drop=True), source.reset_index(drop=True)], axis=1)
    folder = output / tier
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "features.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    receipt = {"status": "materialized", "tier": tier, "games": len(frame),
               "feature_count": len(source.columns), "feature_names": source.columns.tolist(),
               "matrix_sha256": sha256(path), "cohort_sha256": cohort_receipt["cohort_sha256"],
               "evidence": evidence, "latest_season": int(frame.season.max()),
               "checkpoint_saved": False}
    write_json(folder / "receipt.json", receipt)
    return {k: receipt[k] for k in ("status", "tier", "games", "feature_count", "matrix_sha256")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["cohort", *[f"F{i}" for i in range(20)]], required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_cohort(args.output) if args.tier == "cohort" else materialize(args.tier, args.output)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
