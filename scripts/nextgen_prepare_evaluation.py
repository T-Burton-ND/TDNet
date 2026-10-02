#!/usr/bin/env python3
"""Create a market-only evaluation sidecar and a fail-closed baseline audit."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.experiments.nextgen_f09 import verified_endpoint_records
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, load_authoritative_schedule, sha256_file


def main():
    config = json.loads((ROOT / "configs/experiments/nextgen_fingerprints_v1.json").read_text())
    root = Path(config["artifact_root"])
    inventory = json.loads((ROOT / "configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json").read_text())
    schedule, schedule_hash = load_authoritative_schedule(root, inventory)
    by_id = schedule.set_index("id")
    rows, sources = [], []
    for record in verified_endpoint_records(root, "/lines"):
        d = pd.read_parquet(record["cache_path"])
        sources.append({"request_id": record["request_id"], "sha256": record["sha256"]})
        for row in d.itertuples(index=False):
            if int(row.id) not in by_id.index:
                continue
            game = by_id.loc[int(row.id)]
            if row.home_team != game.home_team or row.away_team != game.away_team:
                raise ValueError("Market home/away identity disagrees with fresh schedule")
            lines = [x for x in row.lines if isinstance(x, dict) and x.get("spread") is not None]
            consensus = [float(x["spread"]) for x in lines if str(x.get("provider")).lower() == "consensus"]
            quotes = consensus or [float(x["spread"]) for x in lines]
            value = float(np.median(quotes)) if quotes else np.nan
            rows.append({"target_game_id": int(row.id), "season": int(game.season),
                         "home_spread": value, "provider_quotes": len(quotes),
                         "selection": "consensus" if consensus else "median_available_providers"})
    market = pd.DataFrame(rows)
    if market.empty or market.target_game_id.duplicated().any() or market.season.gt(2025).any():
        raise ValueError("Invalid market sidecar scope or identity")
    path = root / "canonical/evaluation_market_sidecar.parquet"
    market.to_parquet(path, index=False, compression="zstd")
    atomic_json(path.with_suffix(".provenance.json"), {
        "data_sha256": sha256_file(path), "schedule_sha256": schedule_hash, "sources": sources,
        "role": "evaluation_only_never_features", "home_spread_convention": "negative_home_favorite",
        "selection": "consensus_if_available_else_median_provider_spread",
        "coverage_by_season": market.groupby("season").home_spread.count().to_dict()})

    baseline = root / "canonical/f06_aligned.parquet"
    frame = pd.read_parquet(baseline)
    provenance = json.loads(baseline.with_suffix(".provenance.json").read_text())
    from gridiron_ml.experiments.nextgen_source_policy import baseline_features, assert_excluded_absent
    original = json.loads((ROOT / config['source_f6_manifest']).read_text())['feature_names']
    expected = baseline_features(original, config)
    assert_excluded_absent(frame)
    if provenance.get('feature_names') != expected or provenance.get('baseline_source_policy') != config['baseline_source_policy']:
        raise ValueError('Baseline exclusion policy or retained feature schema mismatch')
    static_columns = [c for c in frame if c.startswith(("coach_", "roster_"))]
    checks = {
        "baseline_hash_matches": sha256_file(baseline) == provenance["data_sha256"],
        "schedule_hash_matches": schedule_hash == provenance["schedule_sha256"],
        "design_years_only": frame.season.between(2010, 2025).all(),
        "no_duplicate_keys": not frame.duplicated(["target_game_id", "team"]).any(),
        "paired_targets": frame.groupby("target_game_id").size().eq(2).all(),
        "availability_before_target": frame.feature_available_utc.lt(frame.target_start_utc).all(),
        "static_roster_coach_frozen": not frame.groupby(["season", "team"])[static_columns].nunique(dropna=False).gt(1).any().any(),
    }
    checks = {k: bool(v) for k, v in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"Baseline audit failed: {checks}")
    report = {
        "training_authorized": True,
        "excluded_features": config["baseline_source_policy"]["excluded_features"],
        "baseline_source_policy": config["baseline_source_policy"],
        "f06_aligned_sha256": provenance["data_sha256"], "schedule_sha256": schedule_hash,
        "verified_checks": checks, "target_games": int(frame.target_game_id.nunique()),
        "availability_assumption": "48 hours after latest league-wide kickoff in source week; reconstructed reporting lag, not archived API publication timestamp",
        "known_baseline_coverage_limitation": "Historical weekly F6 may omit one of multiple games in a team-week; retained baseline values preserved after authorized coach SP exclusions and not claimed equivalent to a fresh all-game reconstruction",

    }
    audit_path = root / "results/source_semantics_audit.json"
    if audit_path.exists() and json.loads(audit_path.read_text()).get("training_authorized"):
        raise ValueError("Refusing to overwrite an already resolved source-semantics audit")
    atomic_json(audit_path, report)
    print(json.dumps({"market_games": len(market), "audit": report}, indent=2))


if __name__ == "__main__":
    main()
