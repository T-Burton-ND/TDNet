#!/usr/bin/env python3
"""Apply frozen F20 to the 2026 tracker archive without reading outcomes as features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fit_f20_baselines import cohort

from gridiron_ml.f20.model import (
    FEATURE_COLUMNS,
    HISTORICAL_COLUMNS,
    historical_features,
)
from gridiron_ml.f20.scientific import MODEL_IDS, predict_margin_panel, predict_panel

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/private/f20/historical_2026"
MODEL = ROOT / "data/private/f20/models/f20_scientific_2017_2024_v2_margin"
OUTPUT = ROOT / "data/private/f20/replay_2026_week01_07_v3_margin"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(archive: Path, schedule_dir: Path, model_dir: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if manifest["canonical_name"] != "F20" or manifest["training_seasons"][-1] != 2024:
        raise ValueError("F20 frozen model contract changed")
    if tuple(manifest.get("model_ids", [])) != MODEL_IDS:
        raise ValueError("F20 scientific architecture roster changed")
    models = {}
    margin_models = {}
    for model_id in MODEL_IDS:
        model_path = model_dir / f"{model_id}.joblib"
        if digest(model_path) != manifest["model_sha256"][model_id]:
            raise ValueError(f"F20 {model_id} model hash changed")
        models[model_id] = joblib.load(model_path)
        margin_path = model_dir / f"{model_id}_margin.joblib"
        if digest(margin_path) != manifest["margin_model_sha256"][model_id]:
            raise ValueError(f"F20 {model_id} margin model hash changed")
        margin_models[model_id] = joblib.load(margin_path)
    frame, audit = cohort(archive, schedule_dir, range(2026, 2027))
    if not set(frame.week_y).issubset(set(range(1, 8))):
        raise ValueError("2026 archive contains an unexpected CFBD week")
    source_columns = list(dict.fromkeys(HISTORICAL_COLUMNS.values()))
    # The model sees only these original projection columns. Scores and the
    # tracker 'actual', 'phwin', and aggregate columns never enter features.
    X = historical_features(frame[source_columns])
    if list(X) != list(FEATURE_COLUMNS):
        raise ValueError("F20 source feature order changed")
    probabilities = predict_panel(models, X.loc[:, list(FEATURE_COLUMNS)])
    margins = predict_margin_panel(margin_models, X.loc[:, list(FEATURE_COLUMNS)])
    output = pd.DataFrame({
        "game_id": frame.id.astype(int),
        "cfbd_week": frame.week_y.astype(int),
        "tracker_week": frame.week_x.astype(int),
        "actual_home_margin": frame.hscore.astype(float) - frame.vscore.astype(float),
        "provenance": "retrospective_model_on_assumed_pregame_weekly_archive",
    })
    for model_id, probability in probabilities.items():
        output[f"probability_{model_id}"] = probability
        output[f"margin_{model_id}"] = margins[model_id]
    if (output.game_id.duplicated().any() or not output.probability_F20.between(0, 1).all()
            or not np.isfinite(output[[f"margin_{model_id}" for model_id in (*MODEL_IDS, "F20")]]).all().all()):
        raise ValueError("F20 replay has duplicate games or invalid predictions")
    return output, {"archive_audit": audit["2026"],
                    "model_sha256": manifest["model_sha256"],
                    "margin_model_sha256": manifest["margin_model_sha256"],
                    "model_manifest_sha256": digest(model_dir / "manifest.json"),
                    "archive_week_is_not_cfbd_week": bool(output.tracker_week.ne(output.cfbd_week).any()),
                    "original_weekly_publication_times_verified": False,
                    "source_pregame_timing_assumption": "owner_accepted_for_retrospective_chart",
                    "model_was_frozen_before_2026_games": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--schedule-dir", type=Path, default=ROOT / "data/raw/cfbd/v2/games")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("F20 replay already exists; choose a new output")
    predictions, audit = replay(args.archive, args.schedule_dir, args.model)
    args.output.mkdir(parents=True)
    path = args.output / "f20_replay.parquet"
    predictions.to_parquet(path, index=False)
    receipt = {**audit, "rows": len(predictions),
               "cfbd_week_counts": {str(k): int(v) for k, v in predictions.cfbd_week.value_counts().sort_index().items()},
               "predictions_sha256": digest(path)}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "rows": len(predictions),
                      "cfbd_week_counts": receipt["cfbd_week_counts"]}, indent=2))


if __name__ == "__main__":
    main()
