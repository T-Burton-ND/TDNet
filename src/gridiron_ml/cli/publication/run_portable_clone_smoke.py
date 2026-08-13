#!/usr/bin/env python3
"""Exercise core publication operations from an arbitrary temporary clone."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile


ROOT = project_root()


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    completed = subprocess.run(command, cwd=cwd, env=env, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def smoke() -> dict:
    with tempfile.TemporaryDirectory(prefix="tdnet-portable-clone-") as temporary:
        clone = Path(temporary) / "clone"
        clone.mkdir()
        for relative in ("src", "configs", "scripts", "README.md", "pyproject.toml"):
            source = ROOT / relative
            destination = clone / relative
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        clean_env = os.environ.copy()
        run(["git", "init", "-q"], cwd=clone, env=clean_env)
        run(["git", "config", "user.email", "portable-smoke@example.invalid"], cwd=clone, env=clean_env)
        run(["git", "config", "user.name", "TDNet portable smoke"], cwd=clone, env=clean_env)
        run(["git", "add", "src", "configs", "scripts", "README.md", "pyproject.toml"], cwd=clone, env=clean_env)
        run(["git", "commit", "-qm", "portable smoke fixture"], cwd=clone, env=clean_env)
        smoke_code = r'''
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from gridiron_ml.models import build_model_from_config
from gridiron_ml.publication import build_prediction_bundle, prepare_public_prediction_table, score_prediction_bundle, verify_prediction_bundle
from gridiron_ml.publication.chart_contracts import validate_chart_domains

model = build_model_from_config({"family": "linear", "model_type": "ridge", "params": {"alpha": 1.0}})
X = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0]})
y = np.array([-2.0, -1.0, 1.0, 2.0])
model.train(X, y)
assert len(model.predict(X)) == 4
validate_chart_domains(chart_kind="probability", y_domain=(0, 1))
now = datetime.now(timezone.utc)
frame = pd.DataFrame([{
    "season": 2026, "week": 1, "game_id": 1,
    "game_start_time_utc": (now + timedelta(days=1)).isoformat(),
    "home_team": "Home", "away_team": "Away", "neutral_site": False,
    "conference_game": False, "season_type": "regular", "created_at_utc": now.isoformat(),
    "model_family": "linear", "model_name": "portable_ridge", "objective": "margin",
    "checkpoint_sha256": "a" * 64, "pred_home_margin": 3.0,
    "pred_home_win_probability": 0.65,
}])
prepared = prepare_public_prediction_table(frame, prediction_deadline_utc=(now + timedelta(hours=1)).isoformat(), feature_manifest_sha256="b" * 64, data_snapshot_sha256="c" * 64, schedule_snapshot_sha256="d" * 64, git_commit="e" * 40, pipeline_version="portable-smoke", environment_lock_sha256="f" * 64, kickoff_time_confirmed=True)
bundle = Path("portable_bundle")
build_prediction_bundle(prepared, output_root=bundle, project_root=Path.cwd())
assert verify_prediction_bundle(bundle)["valid"]
results = pd.DataFrame([{"game_id": 1, "home_points": 24, "away_points": 17}])
results.to_parquet("portable_results.parquet", index=False)
score_prediction_bundle(bundle, results, output_root=Path("portable_scores"))
assert (Path("portable_scores") / "scorecard.csv").exists()
print("portable_import_train_predict_score_figure=PASS")
'''
        env = os.environ.copy()
        env["PYTHONPATH"] = str(clone / "src")
        env["MPLCONFIGDIR"] = str(Path(temporary) / "mpl")
        run([sys.executable, "-c", smoke_code], cwd=clone, env=env)
        return {
            "status": "portable_clone_smoke_pass",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "clone_root": "temporary_clone_removed_after_test",
            "operations": ["imports", "fixture training", "prediction", "bundle verification", "bundle scoring", "chart-domain validation"],
            "git_commit_in_clone": run(["git", "rev-parse", "HEAD"], cwd=clone, env=env),
        }


def main() -> int:
    root = ROOT
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "docs/publication_2026/portable_clone_smoke.json")
    args = parser.parse_args()
    report = smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
