from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pandas as pd

from gridiron_ml.publication import (
    build_preseason_freeze,
    build_prediction_bundle,
    prepare_public_prediction_table,
    score_prediction_bundle,
    verify_prediction_bundle,
    verify_preseason_freeze,
)


def _predictions():
    now = datetime.now(timezone.utc)
    return pd.DataFrame([{
        "season": 2026, "week": 1, "game_id": 100, "game_start_time_utc": (now + timedelta(days=3)).isoformat(),
        "home_team": "Home", "away_team": "Away", "neutral_site": False, "conference_game": True,
        "season_type": "regular", "created_at_utc": now.isoformat(), "model_family": "linear",
        "model_name": "ridge", "objective": "margin", "checkpoint_sha256": "a" * 64,
        "pred_home_margin": 3.5, "pred_home_win_probability": 0.62,
    }])


def test_prediction_bundle_is_verifiable_and_scored_without_mutation(tmp_path):
    now = datetime.now(timezone.utc)
    prepared = prepare_public_prediction_table(
        _predictions(), prediction_deadline_utc=(now + timedelta(days=1)).isoformat(),
        feature_manifest_sha256="b" * 64, data_snapshot_sha256="c" * 64,
        schedule_snapshot_sha256="d" * 64, git_commit="e" * 40,
        pipeline_version="test", environment_lock_sha256="f" * 64,
        kickoff_time_confirmed=True,
    )
    root = tmp_path / "bundle"
    build_prediction_bundle(prepared, output_root=root, project_root=Path(__file__).resolve().parents[1], allow_dirty_code=True)
    assert verify_prediction_bundle(root)["valid"]
    before = (root / "public/predictions.parquet").read_bytes()
    scored = score_prediction_bundle(root, pd.DataFrame([{"game_id": 100, "home_points": 24, "away_points": 17}]), output_root=tmp_path / "scores")
    assert scored["scorecard"].iloc[0]["margin_mae"] == 3.5
    assert (root / "public/predictions.parquet").read_bytes() == before


def test_bundle_verifier_detects_tampering(tmp_path):
    now = datetime.now(timezone.utc)
    prepared = prepare_public_prediction_table(
        _predictions(), prediction_deadline_utc=(now + timedelta(days=1)).isoformat(),
        feature_manifest_sha256="b" * 64, data_snapshot_sha256="c" * 64,
        schedule_snapshot_sha256="d" * 64, git_commit="e" * 40,
        pipeline_version="test", environment_lock_sha256="f" * 64, kickoff_time_confirmed=True,
    )
    root = tmp_path / "bundle"
    build_prediction_bundle(prepared, output_root=root, project_root=Path(__file__).resolve().parents[1], allow_dirty_code=True)
    with (root / "public/predictions.csv.gz").open("ab") as handle:
        handle.write(b"tamper")
    assert not verify_prediction_bundle(root)["valid"]


def test_preseason_freeze_build_and_verify(tmp_path):
    project = Path(__file__).resolve().parents[1]
    artifacts = {}
    for name in ["checkpoint", "preprocessing", "calibration", "historical_evaluation"]:
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = path
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame([{
        "model_id": "TDNET-2026-WIN-01", "model_family": "linear", "objective": "winner",
        "feature_config": "F1", "checkpoint_path": artifacts["checkpoint"],
        "preprocessing_path": artifacts["preprocessing"], "calibration_path": artifacts["calibration"],
        "historical_evaluation_path": artifacts["historical_evaluation"],
    }]).to_csv(inventory, index=False)
    selection = tmp_path / "selection.md"; selection.write_text("selected\n")
    environment = tmp_path / "environment.lock"; environment.write_text("python=test\n")
    data_manifest = tmp_path / "data.json"; data_manifest.write_text("{}\n")
    schedule = tmp_path / "schedule.csv"; schedule.write_text("game_id\n1\n")
    bundle = tmp_path / "freeze"
    build_preseason_freeze(
        project_root=project, bundle_root=bundle, inventory_path=inventory,
        selection_report_path=selection, feature_registry_path=project / "configs/features/feature_registry.yaml",
        feature_ladders_path=project / "configs/features/feature_ladders.yaml",
        split_paths=[project / "configs/splits/final_historical_holdout.yaml"],
        environment_lock_path=environment, data_snapshot_manifest_path=data_manifest,
        schedule_snapshot_path=schedule, allow_dirty=True,
    )
    assert verify_preseason_freeze(bundle)["valid"]
    (bundle / "model_selection_report.md").write_text("tampered\n")
    assert not verify_preseason_freeze(bundle)["valid"]
