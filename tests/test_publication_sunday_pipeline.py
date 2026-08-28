from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd

from gridiron_ml.publication import build_prediction_bundle, prepare_public_prediction_table


def test_sunday_pipeline_requires_certified_snapshot_and_writes_review_bundle(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc)
    predictions = pd.DataFrame([{
        "season": 2026,
        "week": 1,
        "game_id": 42,
        "game_start_time_utc": (now + timedelta(days=1)).isoformat(),
        "home_team": "Home",
        "away_team": "Away",
        "neutral_site": False,
        "conference_game": False,
        "season_type": "regular",
        "created_at_utc": now.isoformat(),
        "model_family": "linear",
        "model_name": "ridge",
        "objective": "margin",
        "checkpoint_sha256": "a" * 64,
        "pred_home_margin": 3.0,
        "pred_home_win_probability": 0.65,
    }])
    prepared = prepare_public_prediction_table(
        predictions,
        prediction_deadline_utc=(now + timedelta(hours=1)).isoformat(),
        feature_manifest_sha256="b" * 64,
        data_snapshot_sha256="c" * 64,
        schedule_snapshot_sha256="d" * 64,
        git_commit="e" * 40,
        pipeline_version="test",
        environment_lock_sha256="f" * 64,
        kickoff_time_confirmed=True,
    )
    bundle = tmp_path / "bundle"
    build_prediction_bundle(prepared, output_root=bundle, project_root=project, allow_dirty_code=True)
    results = tmp_path / "results.parquet"
    pd.DataFrame([{"game_id": 42, "home_points": 24, "away_points": 17}]).to_parquet(results, index=False)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"status": "pass", "certification": "weekly_snapshot_certified"}) + "\n")
    output = tmp_path / "publication/2026/week_01/post_game"
    env = {**os.environ, "PYTHONPATH": str(project / "src"), "MPLCONFIGDIR": str(tmp_path / "mpl")}
    completed = subprocess.run(
        [sys.executable, "src/gridiron_ml/cli/publication/run_sunday_publication_pipeline.py",
         "--bundle", str(bundle), "--results", str(results),
         "--snapshot-completeness", str(snapshot), "--output-root", str(output)],
        cwd=project, env=env, capture_output=True, text=True, check=True,
    )
    assert "sunday_review_bundle_ready" in completed.stdout
    assert (output / "sunday_publication_manifest.json").exists()
    assert (output / "x_post_package/manifest.json").exists()
    assert (bundle / "public/predictions.parquet").exists()
