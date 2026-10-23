#!/usr/bin/env python3
"""Issue append-only F20-A and market-control probabilities from frozen inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.f20.model import BASE_COLUMNS, FEATURE_COLUMNS, captured_features


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cutoff(kickoff: datetime) -> datetime:
    if kickoff.tzinfo is None:
        raise ValueError("Kickoff has no timezone")
    eastern = kickoff.astimezone(ZoneInfo("America/New_York"))
    thursday = eastern.date() - timedelta(days=(eastern.weekday() - 3) % 7)
    return datetime.combine(thursday, time(23, 59), ZoneInfo("America/New_York")).astimezone(UTC)


def write_once(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"Immutable forecast collision: {path}")
        return False
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o400)
    return True


def issue(archive: Path, receipt_path: Path, model_dir: Path, *, now: datetime,
          revision_id: str = "initial", week: int | None = None,
          official_enabled: bool = True) -> dict:
    if now.tzinfo is None:
        raise ValueError("Issuance time must be timezone-aware")
    now = now.astimezone(UTC)
    if not revision_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in revision_id):
        raise ValueError("Invalid revision ID")
    receipt = json.loads(receipt_path.read_text())
    raw = archive / receipt["raw_filename"]
    if sha(raw) != receipt["raw_sha256"] or not receipt["parser_version"].startswith("f20-tracker-html-v"):
        raise ValueError("Capture hash or parser version mismatch")
    collected = datetime.fromisoformat(receipt["collected_at_utc"])
    updated = receipt.get("site_updated_at_utc")
    if collected > now or (updated and datetime.fromisoformat(updated) > collected):
        raise ValueError("Expert source timing after issuance")
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["training_seasons"][-1] > 2024 or 2026 not in manifest["excluded_seasons"]:
        raise ValueError("Model training boundary changed")
    fitted = manifest.get("fit_at_utc")
    if fitted and datetime.fromisoformat(fitted) > now:
        raise ValueError("Model was fitted after issuance")
    models = {}
    for arm in ("market_only", "F20-A"):
        path = model_dir / f"{arm}.joblib"
        if sha(path) != manifest["model_sha256"][arm]:
            raise ValueError(f"Frozen {arm} model hash changed")
        models[arm] = joblib.load(path)
    output = {"issued": 0, "existing": 0, "official": 0, "late_revision": 0,
              "precutoff_revision": 0,
              "early_game_pregame": 0, "already_started": 0,
              "kickoff_unverified": 0, "missing_expert": 0, "games": []}
    for observation in receipt["observations"]:
        if week is not None and observation["week"] != week:
            continue
        kickoff = datetime.fromisoformat(observation["kickoff_utc"])
        if observation["status"] == "kickoff_unverified":
            output["kickoff_unverified"] += 1
            continue
        if observation["status"] != "prospective_pregame" or kickoff <= now:
            output["already_started"] += 1
            continue
        X = captured_features(observation)
        available = int(X.iloc[0].expert_count)
        if available == 0:
            output["missing_expert"] += 1
        probabilities = {}
        for arm, columns in {"market_only": BASE_COLUMNS, "F20-A": FEATURE_COLUMNS}.items():
            value = float(models[arm].predict_proba(X.loc[:, list(columns)])[0, 1])
            if not 0 <= value <= 1:
                raise ValueError("Invalid model probability")
            probabilities[arm] = value
        game_id = observation["game_id"]
        deadline = cutoff(kickoff)
        official_eligible = official_enabled and deadline - timedelta(hours=4) <= now <= deadline < kickoff
        if official_eligible:
            schedule_time = receipt.get("schedule_modified_at_utc")
            if not schedule_time:
                raise ValueError("Official forecast lacks schedule freshness evidence")
            schedule_time = datetime.fromisoformat(schedule_time)
            if schedule_time > now or now - schedule_time > timedelta(days=5):
                raise ValueError("Official forecast schedule is stale or future-dated")
        issuance_class = ("eligible_weekly" if official_eligible else
                          "early_game_pregame" if kickoff <= deadline else
                          "precutoff_revision" if now < deadline else "late_pregame_revision")
        version_seed = f"{game_id}:{sha(receipt_path)}:{sha(manifest_path)}:{revision_id}"
        forecast_id = hashlib.sha256(version_seed.encode()).hexdigest()
        path = archive / "forecasts" / str(game_id) / f"{forecast_id}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if (previous["forecast_id"] != forecast_id
                    or previous["raw_snapshot_sha256"] != receipt["raw_sha256"]
                    or previous["receipt_sha256"] != sha(receipt_path)
                    or previous["model_manifest_sha256"] != sha(manifest_path)):
                raise ValueError("Existing forecast identity changed")
            if previous["issuance_class"] == "eligible_weekly":
                official_path = path.parent / "official.json"
                if not official_path.exists():
                    pointer = {"forecast_id": forecast_id, "record_sha256": sha(path),
                               "record_file": str(path), "weekly_cutoff_utc": deadline.isoformat()}
                    if write_once(official_path,
                                  (json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode()):
                        output["official"] += 1
            output["existing"] += 1
            output["games"].append(game_id)
            continue
        record = {"schema_version": 1, "forecast_id": forecast_id,
                  "season": 2026, "week": observation["week"], "game_id": game_id,
                  "home_team": observation["home_team"], "away_team": observation["away_team"],
                  "kickoff_utc": kickoff.isoformat(), "issued_at_utc": now.isoformat(),
                  "weekly_cutoff_utc": deadline.isoformat(),
                  "issuance_class": issuance_class,
                  "source_id": receipt["source_id"], "source_url": receipt["source_url"],
                  "source_collected_at_utc": receipt["collected_at_utc"],
                  "source_site_updated_at_utc": updated,
                  "raw_snapshot_sha256": receipt["raw_sha256"],
                  "receipt_sha256": sha(receipt_path), "receipt_file": str(receipt_path),
                  "input_values": observation["values"],
                  "feature_vector": {name: (None if pd.isna(value) else float(value))
                                     for name, value in X.iloc[0].items()},
                  "fingerprint_version": receipt["parser_version"],
                  "model_manifest_sha256": sha(manifest_path),
                  "model_manifest_file": str(manifest_path),
                  "model_sha256": manifest["model_sha256"],
                  "model_training_through": 2024,
                  "revision_id": revision_id,
                  "model_historical_timing": manifest["historical_source_timing"],
                  "expert_count": available, "home_win_probability": probabilities}
        payload = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        created = write_once(path, payload)
        output["issued" if created else "existing"] += 1
        if official_eligible:
            pointer = {"forecast_id": forecast_id, "record_sha256": sha(path),
                       "record_file": str(path), "weekly_cutoff_utc": deadline.isoformat()}
            official_path = path.parent / "official.json"
            if not official_path.exists() and write_once(
                    official_path, (json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode()):
                output["official"] += 1
        elif issuance_class == "early_game_pregame":
            output["early_game_pregame"] += 1
        elif issuance_class == "precutoff_revision":
            output["precutoff_revision"] += 1
        else:
            output["late_revision"] += 1
        output["games"].append(game_id)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/private/f20/prospective")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "data/private/f20/models/baseline_2017_2024_v1")
    parser.add_argument("--revision-id", default="initial")
    parser.add_argument("--week", type=int)
    args = parser.parse_args()
    print(json.dumps(issue(args.archive, args.receipt, args.models, now=datetime.now(UTC),
                           revision_id=args.revision_id, week=args.week), indent=2))


if __name__ == "__main__":
    main()
