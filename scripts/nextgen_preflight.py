#!/usr/bin/env python3
"""One-command nextgen preflight; no bulk acquisition or model experiment.

Use --live-plays once to spend a minimal number of budgeted CFBD requests.
Subsequent invocations verify the ledger cache and do not repeat that play call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.experiments.nextgen_contract import (  # noqa: E402
    all_fingerprint_ids, assert_design_operation_frame, assert_design_years, assert_pair_closed, assert_safe_inputs,
    assert_temporal_feature_rows,
    load_json, validate_setup,
)
from gridiron_ml.experiments.nextgen_temporal import next_game_rows  # noqa: E402
from gridiron_ml.models.td_spline import TDSpline  # noqa: E402
from gridiron_ml.models.td_tree import TDTree  # noqa: E402
from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import (  # noqa: E402
    CFBDClient, CallBudget, load_cfbd_key_file,
)
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (  # noqa: E402
    AcquisitionLedger, atomic_json, build_manifest, execute_request, materialize_plan, quota_allows,
    sha256_file, verify_cache,
)
from gridiron_ml.td_run.matchups.builder import MatchupBuilder  # noqa: E402


def expect_rejected(action) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Expected safety guard to reject invalid input")


def check_contract_and_models() -> dict:
    validate_setup(ROOT)
    ids = all_fingerprint_ids()
    assert len(ids) == 42
    assert all("F07" not in item and "F08" not in item for item in ids)
    expect_rejected(lambda: assert_safe_inputs(["market_spread"]))
    expect_rejected(lambda: assert_safe_inputs(["pregame_win_probability"]))
    expect_rejected(lambda: assert_design_years([2024, 2026]))
    records = load_json(ROOT / "configs/experiments/nextgen_seed_features_v1.json")
    names = [item["name"] for item in records]
    assert_pair_closed(names, records)
    expect_rejected(lambda: assert_pair_closed(names[:1], records))
    pair = pd.DataFrame({"offense_rush_ypa_q4_minus_q1": [0.3, -0.2],
                         "defense_rush_ypa_q4_minus_q1": [-0.1, 0.4]})
    built = MatchupBuilder(representation="diff").build(pair.iloc[:1], pair.iloc[1:])
    assert len(built) == 1
    setpoints = load_json(ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json")
    rng = np.random.default_rng(1701)
    X = pd.DataFrame(rng.normal(size=(160, 3)), columns=["a", "b", "c"])
    y = 3 * X.a - X.b + rng.normal(size=160)
    for point in setpoints["M2"]:
        config = {"model_type": "spline_ridge", "loss_function": "MAE",
                  "params": {"alpha": point["alpha"]},
                  "spline": {"n_knots": point["n_knots"], "degree": point["degree"]},
                  "seed": 1701}
        model = TDSpline(config)
        model._build_pipeline().fit(X, y)
    for point in setpoints["M4"]:
        params = {key: value for key, value in point.items() if key != "id"}
        model = TDTree({"model_type": "hist_gradient_boosted", "loss_function": "MAE", "params": params,
                        "seed": 1701})
        model._build_pipeline().fit(X, y)
    return {"lineage_variants": 42, "m2_setpoints_fitted": 10,
            "m4_setpoints_fitted": 10, "matchup_pair_smoke": "passed"}


def check_temporal() -> dict:
    fixture = pd.DataFrame([
        dict(id=1, season=2025, week=1, season_type="regular", completed=True,
             start_date="2025-08-30T12:00:00Z", home_team="A", away_team="FCS",
             home_classification="fbs", away_classification="fcs", home_points=21, away_points=7),
        dict(id=2, season=2025, week=2, season_type="regular", completed=True,
             start_date="2025-09-06T12:00:00Z", home_team="A", away_team="B",
             home_classification="fbs", away_classification="fbs", home_points=10, away_points=17),
        dict(id=3, season=2025, week=3, season_type="postseason", completed=True,
             start_date="2025-09-13T12:00:00Z", home_team="A", away_team="B",
             home_classification="fbs", away_classification="fbs", home_points=90, away_points=0),
    ])
    rows = next_game_rows(fixture)
    a = rows.loc[rows.team.eq("A")].iloc[0]
    assert int(a.source_game_id) == 1 and int(a.game_id) == 2
    assert a.prior_mean_margin == 14 and a.next_game_margin == -7
    assert 3 not in rows.game_id.tolist()
    feature_row = pd.DataFrame({"season": [2025], "season_type": ["regular"],
                                "source_game_id": [int(a.source_game_id)],
                                "target_game_id": [int(a.game_id)],
                                "feature_available_utc": ["2025-08-30T16:00:00Z"],
                                "target_start_utc": ["2025-09-06T12:00:00Z"],
                                "prior_mean_margin": [a.prior_mean_margin]})
    assert_temporal_feature_rows(feature_row, ["prior_mean_margin"])
    expect_rejected(lambda: assert_temporal_feature_rows(
        feature_row.assign(feature_available_utc="2025-09-07T00:00:00Z"), ["prior_mean_margin"]))
    expect_rejected(lambda: next_game_rows(pd.concat([
        fixture, fixture.iloc[:1].assign(season=2026, id=4)], ignore_index=True)))
    for operation in ("scaling", "correlation", "shap", "feature_discovery", "recommendations"):
        expect_rejected(lambda operation=operation: assert_design_operation_frame(
            pd.DataFrame({"season": [2025, 2026]}), operation))
    return {"next_game_alignment": "passed", "postseason_exclusion": "passed",
            "design_2026_exclusion": "passed_for_scaling_correlation_shap_discovery_recommendations"}


def storage_estimate(manifest: list[dict], root: Path) -> dict:
    """Conservative planning estimate; sample-based and explicitly provisional."""
    legacy_sizes = [p.stat().st_size for p in (ROOT / "data/raw/cfbd/v2").glob("*/*.parquet")]
    p95 = sorted(legacy_sizes)[int(0.95 * (len(legacy_sizes)-1))] if legacy_sizes else 2_000_000
    play_sample = [p for p in (root / "raw_cache/v1/plays").glob("*.parquet")]
    play_size = max((p.stat().st_size for p in play_sample), default=max(p95, 2_000_000))
    new = [item for item in manifest if item["status"] == "planned"]
    raw = sum(play_size * 3 if item["endpoint"] == "/plays" else
              p95 * 3 if item["partition"].find("-w") < 0 else
              min(play_size, p95 * 3) for item in new)
    legacy = sum(legacy_sizes)
    total = (raw + legacy) * 4  # canonical superset, manifests, transient matrices
    return {"method": "provisional_p95_legacy_and_sampled_plays_with_3x_raw_4x_total",
            "legacy_bytes": legacy, "plays_sample_bytes": play_size,
            "estimated_raw_new_bytes": raw, "estimated_total_bytes": total,
            "soft_limit_bytes": 100 * 1024**3,
            "under_soft_limit": total < 100 * 1024**3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--live-plays", action="store_true",
                        help="Check live quota and execute only the first 2025 /plays week")
    args = parser.parse_args()
    config = load_json(ROOT / "configs/experiments/nextgen_fingerprints_v1.json")
    root = args.artifact_root or Path(config["artifact_root"])
    checks = {**check_contract_and_models(), **check_temporal()}
    inventory = load_json(ROOT / "configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json")
    manifest, summary = build_manifest(inventory, ROOT, root)
    if summary["unresolved_partial_requests"] or summary["unresolved_review_requests"]:
        raise RuntimeError("Unresolved capped or anomalous responses require review before launch")
    budget_path = Path(config["cfbd_api_call_budget"]["ledger"])
    reserved = json.loads(budget_path.read_text())["reserved"] if budget_path.exists() else 0
    if reserved + summary["new_planned_calls"] > 20000:
        raise RuntimeError("Preferred 20,000-call target exceeded")
    if not storage_estimate(manifest, root)["under_soft_limit"]:
        raise RuntimeError("100 GB storage soft limit exceeded")
    quota = None
    quota_snapshot_recorded_at_utc = None
    if args.live_plays:
        budget = CallBudget(budget_path, 24000)
        os.environ["CFBD_API_KEY"] = load_cfbd_key_file(ROOT / ".env")
        client = CFBDClient(call_budget=budget, max_retries=2)
        quota = client.get_json("/info", {}, max_retries=0)
        if isinstance(quota, list):
            quota = quota[0] if quota else {}
        quota_snapshot_recorded_at_utc = datetime.now(timezone.utc).isoformat()
        remaining = quota.get("remainingCalls")
        if not isinstance(remaining, int) or not quota_allows(
            remaining, summary["new_planned_calls"] * 3, budget.status()["reserved"]):
            raise RuntimeError("Live quota cannot cover plan and 6,000-call reserve")
        plays = next(item for item in manifest if item["endpoint"] == "/plays" and
                     item["year"] == 2025 and item["week"] == 1)
        ledger = AcquisitionLedger(root)
        first = execute_request(plays, ledger, client)
        if first["status"] not in ("success_complete", "skipped_existing_complete"):
            raise RuntimeError(f"/plays smoke failed: {first['status']} {first.get('error_summary')}")
        calls_before = client.api_calls
        second = execute_request(plays, ledger, client)
        if client.api_calls != calls_before or not verify_cache(Path(second["cache_path"]), second):
            raise RuntimeError("/plays cache/resume verification failed")
        checks["plays_smoke"] = {"request_id": first["request_id"],
                                  "rows": first["row_count"], "bytes": first["byte_size"],
                                  "resume_without_call": True}
        manifest, summary = build_manifest(inventory, ROOT, root)
    estimate = storage_estimate(manifest, root)
    checks["request_ledger_materialized"] = materialize_plan(manifest, AcquisitionLedger(root))
    results = root / "results/preflight"
    results.mkdir(parents=True, exist_ok=True)
    manifest_path = results / "cfbd_request_manifest_v1.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    old_summary_path = results / "cfbd_plan_summary_v1.json"
    if quota is None and old_summary_path.exists():
        old_summary = json.loads(old_summary_path.read_text())
        quota = old_summary.get("live_quota_snapshot")
        quota_snapshot_recorded_at_utc = (old_summary.get("quota_snapshot_recorded_at_utc")
                                          or old_summary.get("generated_at_utc"))
        if old_summary.get("checks", {}).get("plays_smoke"):
            checks["plays_smoke"] = old_summary["checks"]["plays_smoke"]
    if "plays_smoke" not in checks:
        verified_play = next((item for item in manifest if item["endpoint"] == "/plays"
                              and item["status"] == "skipped_existing_complete"
                              and item.get("reuse_source") == "verified_request_ledger"), None)
        if verified_play is not None:
            checks["plays_smoke"] = {"request_id": verified_play["request_id"],
                                      "rows": verified_play["row_count"],
                                      "bytes": verified_play["byte_size"],
                                      "resume_without_call": True}
    summary.update({"checks": checks, "storage_estimate": estimate,
                    "budget_reserved_attempts": (budget.status()["reserved"] if args.live_plays else reserved),
                    "live_quota_snapshot": quota, "manifest_path": str(manifest_path),
                    "quota_snapshot_recorded_at_utc": quota_snapshot_recorded_at_utc,
                    "manifest_sha256": sha256_file(manifest_path),
                    "2026_policy": "quarantined_cache_only_not_in_2010_2025_plan"})
    atomic_json(results / "cfbd_plan_summary_v1.json", summary)
    print(json.dumps({"new_planned_calls": summary["new_planned_calls"],
                      "reused_cached_partitions": summary["reused_cached_partitions"],
                      "checks": checks, "storage_estimate": estimate,
                      "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
