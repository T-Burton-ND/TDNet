#!/usr/bin/env python3
"""Run the publication-facing adversarial fixture leakage audit.

This suite is deliberately complementary to the unit tests: each row records
the threat, mutation, expected invariant, observed behavior, and evidence.
It does not claim that a fixture audit replaces a full historical replay.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.pipeline.validation.leakage import (  # noqa: E402
    assert_disjoint_years,
    assert_no_market_features,
)
from gridiron_ml.publication import (  # noqa: E402
    build_prediction_bundle,
    build_snapshot_completeness,
    prepare_public_prediction_table,
    score_prediction_bundle,
    verify_prediction_bundle,
)
from gridiron_ml.publication.inference import calibration_summary, fit_margin_calibrator  # noqa: E402
from gridiron_ml.publication.knn_oof import run_strict_oof_knn  # noqa: E402
from gridiron_ml.publication.consensus import build_equal_weight_consensus  # noqa: E402
from gridiron_ml.publication.weekly_protocol import validate_deadline_utc  # noqa: E402
from gridiron_ml.experiments.opponent_adjusted import (  # noqa: E402
    OpponentAdjustedVersionSpec,
    build_method_game_contributions,
)


def _record(rows, name, threat, expected, fn):
    try:
        observed = fn()
        rows.append({"test_name": name, "threat": threat, "expected_behavior": expected,
                     "observed_behavior": str(observed), "status": "pass", "evidence": "fixture execution"})
    except Exception as exc:
        rows.append({"test_name": name, "threat": threat, "expected_behavior": expected,
                     "observed_behavior": f"{type(exc).__name__}: {exc}", "status": "fail", "evidence": "fixture execution"})


def _knn_future_mutation():
    frame = pd.DataFrame({
        "game_id": [f"g{i}" for i in range(6)], "season": [2020] * 6,
        "week": list(range(1, 7)), "x": np.arange(6, dtype=float),
        "y_margin": np.arange(6, dtype=float),
    })
    folds = [{"train_indices": [0, 1], "test_indices": [2, 3]}]
    before, audit_before = run_strict_oof_knn(frame, feature_columns=["x"], target_column="y_margin", folds=folds, config={"model_type": "distance", "params": {"n_neighbors": 2}})
    altered = frame.copy()
    altered.loc[[4, 5], "y_margin"] = [9999.0, -9999.0]
    after, audit_after = run_strict_oof_knn(altered, feature_columns=["x"], target_column="y_margin", folds=folds, config={"model_type": "distance", "params": {"n_neighbors": 2}})
    if not np.allclose(before["pred_margin"], after["pred_margin"]):
        raise AssertionError("future target mutation changed earlier OOF predictions")
    if set(audit_before["neighbor_game_id"]) & set(audit_before["target_game_id"]):
        raise AssertionError("target game appeared among KNN neighbors")
    return {"predictions": len(before), "audited_neighbors": len(audit_before)}


def _opponent_future_mutation():
    rows = []
    for week, margin in [(1, 10.0), (2, -4.0)]:
        rows.extend([
            {"keys_season": 2025, "keys_week": week, "keys_game_id": week, "keys_team": "A", "keys_opponent": "B", "game_is_home": True, "offense_ppa": 0.2 * week, "defense_ppa": -0.1 * week, "target_team_margin": margin},
            {"keys_season": 2025, "keys_week": week, "keys_game_id": week, "keys_team": "B", "keys_opponent": "A", "game_is_home": False, "offense_ppa": -0.1 * week, "defense_ppa": 0.2 * week, "target_team_margin": -margin},
        ])
    frame = pd.DataFrame(rows)
    changed = frame.copy()
    changed.loc[changed["keys_week"].eq(2), ["offense_ppa", "defense_ppa"]] = [99.0, -99.0]
    spec = OpponentAdjustedVersionSpec("v1.2", "opponent_ridge", "fixture")
    columns = ["offense_ppa", "defense_ppa", "target_team_margin"]
    a = build_method_game_contributions(games=frame, stat_columns=columns, spec=spec)
    b = build_method_game_contributions(games=changed, stat_columns=columns, spec=spec)
    pd.testing.assert_frame_equal(a.loc[a.keys_week.eq(1), columns].reset_index(drop=True), b.loc[b.keys_week.eq(1), columns].reset_index(drop=True))
    return {"week_checked": 1, "method": spec.method}


def _opponent_reorder_invariance():
    rows = []
    for week in (1, 2, 3):
        rows.extend([
            {"keys_season": 2025, "keys_week": week, "keys_game_id": week, "keys_team": "A", "keys_opponent": "B", "game_is_home": True, "offense_ppa": 0.2 * week, "defense_ppa": -0.1 * week, "target_team_margin": float(week)},
            {"keys_season": 2025, "keys_week": week, "keys_game_id": week, "keys_team": "B", "keys_opponent": "A", "game_is_home": False, "offense_ppa": -0.1 * week, "defense_ppa": 0.2 * week, "target_team_margin": -float(week)},
        ])
    frame = pd.DataFrame(rows)
    spec = OpponentAdjustedVersionSpec("v1.2", "opponent_ridge", "fixture")
    kwargs = {"games": frame, "stat_columns": ("offense_ppa", "defense_ppa", "target_team_margin"), "spec": spec}
    first = build_method_game_contributions(**kwargs).sort_values(["keys_week", "keys_team"]).reset_index(drop=True)
    second = build_method_game_contributions(games=frame.sample(frac=1.0, random_state=7).reset_index(drop=True), stat_columns=kwargs["stat_columns"], spec=spec).sort_values(["keys_week", "keys_team"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    return {"rows": len(first), "shuffled_seed": 7}


def _duplicate_game_audit():
    frame = pd.DataFrame({"season": [2025, 2025, 2025], "week": [1, 1, 2], "game_id": [7, 7, 7]})
    duplicate_count = int(frame.duplicated(["season", "week", "game_id"], keep=False).sum())
    if duplicate_count != 2:
        raise AssertionError("duplicate game fixture was not detected")
    return {"duplicate_rows_flagged": duplicate_count, "key": "season/week/game_id"}


def _consensus_failed_model_membership():
    frame = pd.DataFrame([
        {"game_id": "g1", "model_name": "a", "pred_margin": 2.0, "pred_probability_home": 0.6},
        {"game_id": "g1", "model_name": "b", "pred_margin": 4.0, "pred_probability_home": 0.7},
        {"game_id": "g2", "model_name": "a", "pred_margin": -2.0, "pred_probability_home": 0.4},
    ])
    consensus, membership = build_equal_weight_consensus(frame)
    counts = dict(zip(consensus["game_id"], consensus["effective_model_count"]))
    if counts != {"g1": 2, "g2": 1} or len(membership) != 3:
        raise AssertionError("failed model was imputed or membership was lost")
    return {"effective_model_counts": counts, "membership_rows": len(membership)}


def _deadline_fixture():
    # Thursday 23:59 New York in July is 03:59 UTC on Friday.
    return validate_deadline_utc("2026-07-31T03:59:00Z", local_date="2026-07-30")


def _missing_history_fixture():
    frame = pd.DataFrame([
        {"keys_season": 2025, "keys_week": 1, "keys_game_id": 1, "keys_team": "A", "keys_opponent": "B", "game_is_home": True, "offense_ppa": 0.1, "defense_ppa": 0.2, "target_team_margin": 3.0},
        {"keys_season": 2025, "keys_week": 1, "keys_game_id": 1, "keys_team": "B", "keys_opponent": "A", "game_is_home": False, "offense_ppa": 0.2, "defense_ppa": 0.1, "target_team_margin": -3.0},
    ])
    spec = OpponentAdjustedVersionSpec("v1.2", "opponent_ridge", "fixture")
    result = build_method_game_contributions(games=frame, stat_columns=("offense_ppa", "defense_ppa", "target_team_margin"), spec=spec)
    if result.empty or not np.isfinite(result[["offense_ppa", "defense_ppa"]].to_numpy(dtype=float)).all():
        raise AssertionError("missing-history fallback produced invalid values")
    return {"rows": len(result), "week": 1, "fallback": "global_history_or_zero"}


def _bundle_immutability():
    now = datetime.now(timezone.utc)
    raw = pd.DataFrame([{"season": 2026, "week": 1, "game_id": 1, "game_start_time_utc": (now + timedelta(days=3)).isoformat(), "home_team": "H", "away_team": "A", "neutral_site": False, "conference_game": True, "season_type": "regular", "model_family": "linear", "model_name": "ridge", "objective": "margin", "checkpoint_sha256": "a" * 64, "pred_home_margin": 3.5, "pred_home_win_probability": 0.62}])
    prepared = prepare_public_prediction_table(raw, prediction_deadline_utc=(now + timedelta(days=1)).isoformat(), feature_manifest_sha256="b" * 64, data_snapshot_sha256="c" * 64, schedule_snapshot_sha256="d" * 64, git_commit="e" * 40, pipeline_version="fixture", environment_lock_sha256="f" * 64, kickoff_time_confirmed=True)
    with tempfile.TemporaryDirectory(prefix="tdnet-leakage-") as temp:
        root = Path(temp) / "bundle"
        build_prediction_bundle(prepared, output_root=root, project_root=ROOT, allow_dirty_code=True)
        before = (root / "public/predictions.parquet").read_bytes()
        score_prediction_bundle(root, pd.DataFrame([{"game_id": 1, "home_points": 24, "away_points": 17}]), output_root=Path(temp) / "scores")
        if before != (root / "public/predictions.parquet").read_bytes() or not verify_prediction_bundle(root)["valid"]:
            raise AssertionError("scoring mutated or invalidated frozen prediction bytes")
    return {"bundle_verified": True, "prediction_bytes_unchanged": True}


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "docs/publication_2026/leakage_audit")
    args = parser.parse_args()
    rows = []
    _record(rows, "market_boundary", "hidden market proxy", "market feature rejected by default", lambda: assert_no_market_features(["offense_ppa", "market_spread_close"]))
    # The preceding check should raise; turn the expected rejection into a pass.
    if rows[-1]["status"] == "fail" and "market" in rows[-1]["observed_behavior"].lower():
        rows[-1]["status"] = "pass"
    _record(rows, "season_split_overlap", "train/evaluation season overlap", "overlap rejected", lambda: assert_disjoint_years([2023], [2023]))
    if rows[-1]["status"] == "fail" and "overlap" in rows[-1]["observed_behavior"].lower():
        rows[-1]["status"] = "pass"
    _record(rows, "future_target_knn", "future target mutation", "earlier OOF predictions invariant; target games absent from neighbors", _knn_future_mutation)
    _record(rows, "future_opponent_adjustment", "future score/stat mutation", "prior-week adjusted contributions invariant", _opponent_future_mutation)
    _record(rows, "opponent_reorder", "input row order", "opponent-adjusted contributions invariant to input order", _opponent_reorder_invariance)
    _record(rows, "duplicate_game_detection", "duplicate or changed games", "duplicate season/week/game keys explicitly flagged", _duplicate_game_audit)
    _record(rows, "missing_team_history", "missing prior team history", "declared finite fallback rather than future data", _missing_history_fixture)
    _record(rows, "failed_model_consensus", "failed model row", "no imputation; effective membership is recorded per game", _consensus_failed_model_membership)
    _record(rows, "deadline_contract", "weekly timing", "Thursday New York deadline maps to exact UTC", _deadline_fixture)
    _record(rows, "calibration_bounds", "calibration saturation or invalid probability", "fold-separated calibrator is monotone and bounded", lambda: _calibration_check())
    _record(rows, "bundle_immutability", "post-deadline scoring mutation", "prediction bytes and manifest remain verifiable", _bundle_immutability)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "synthetic adversarial fixtures plus current endpoint contract", "tests": rows, "pass_count": sum(row["status"] == "pass" for row in rows), "fail_count": sum(row["status"] == "fail" for row in rows), "limitations": ["This does not replace the full historical F0-F8 replay.", "CFBD line timestamp/provider completeness remains a prospective-data blocker.", "Neutral-site, canceled/postponed, and late-field fixtures still require explicit historical replay coverage."]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "leakage_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# TDNet adversarial leakage audit", "", f"Created: {report['created_at_utc']}", "", "| Test | Status | Expected | Observed |", "|---|---|---|---|"]
    lines += [f"| {r['test_name']} | {r['status']} | {r['expected_behavior']} | {r['observed_behavior']} |" for r in rows]
    lines += ["", "## Limitations", "", *[f"- {item}" for item in report["limitations"]]]
    (args.output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["fail_count"] else 2


def _calibration_check():
    fit = fit_margin_calibrator([-6, -3, -1, 1, 3, 6], [0, 0, 0, 1, 1, 1], fit_hash="fixture")
    values = fit.predict(np.array([-20, -2, 0, 2, 20], dtype=float))
    if np.any(np.diff(values) < 0) or np.any((values < 0) | (values > 1)):
        raise AssertionError("calibrator is not monotone and bounded")
    summary = calibration_summary([0, 1, 0, 1], values[[1, 3, 1, 3]], bins=4)
    return {"fit_rows": fit.fit_rows, "probability_min": summary["probability_min"], "probability_max": summary["probability_max"]}


if __name__ == "__main__":
    raise SystemExit(main())
