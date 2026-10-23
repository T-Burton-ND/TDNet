#!/usr/bin/env python3
"""Issue append-only F20 scientific probabilities and margins from frozen inputs."""
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
from gridiron_ml.f20.model import FEATURE_COLUMNS, captured_features
from gridiron_ml.f20.scientific import MODEL_IDS, predict_margin_panel, predict_panel


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
          official_enabled: bool = False, weekly_cutoff_utc: datetime | None = None,
          target_game_ids: set[int] | None = None) -> dict:
    if now.tzinfo is None:
        raise ValueError("Issuance time must be timezone-aware")
    now = now.astimezone(UTC)
    if not revision_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in revision_id):
        raise ValueError("Invalid revision ID")
    if official_enabled and (target_game_ids is None or weekly_cutoff_utc is None):
        raise ValueError("Official F20 requires a frozen target slate and owner-selected cutoff")
    receipt = json.loads(receipt_path.read_text())
    raw = archive / receipt["raw_filename"]
    if sha(raw) != receipt["raw_sha256"] or not receipt["parser_version"].startswith("f20-tracker-html-v"):
        raise ValueError("Capture hash or parser version mismatch")
    collected = datetime.fromisoformat(receipt["collected_at_utc"])
    updated = receipt.get("site_updated_at_utc")
    if collected > now or (updated and datetime.fromisoformat(updated) > collected):
        raise ValueError("Expert source timing after issuance")
    if target_game_ids is not None:
        if not official_enabled or weekly_cutoff_utc is None or not target_game_ids:
            raise ValueError("Canonical issuance requires a cutoff and nonempty target slate")
        if weekly_cutoff_utc.tzinfo is None:
            raise ValueError("Canonical F20 cutoff must be timezone-aware")
        weekly_cutoff_utc = weekly_cutoff_utc.astimezone(UTC)
        if collected > weekly_cutoff_utc or now > weekly_cutoff_utc:
            raise ValueError("Canonical F20 capture or issuance is after the frozen weekly cutoff")
        eligible = [item for item in receipt["observations"]
                    if item["game_id"] in target_game_ids and (week is None or item["week"] == week)]
        if len(eligible) != len(target_game_ids) or {item["game_id"] for item in eligible} != target_game_ids:
            raise ValueError("Canonical F20 capture does not cover the exact frozen target slate")
        if any(item["status"] != "prospective_pregame"
               or datetime.fromisoformat(item["kickoff_utc"]) <= weekly_cutoff_utc
               for item in eligible):
            raise ValueError("Canonical F20 target has unverified or pre-cutoff kickoff")
        schedule_time = receipt.get("schedule_modified_at_utc")
        if not schedule_time:
            raise ValueError("Canonical F20 capture lacks schedule freshness evidence")
        schedule_time = datetime.fromisoformat(schedule_time)
        if schedule_time > now or now - schedule_time > timedelta(days=5):
            raise ValueError("Canonical F20 schedule is stale or future-dated")
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("canonical_name") != "F20" or
            manifest["training_seasons"] != list(range(2017, 2025)) or
            2026 not in manifest["excluded_seasons"] or
            tuple(manifest.get("model_ids", [])) != MODEL_IDS or
            set(manifest.get("margin_model_sha256", {})) != set(MODEL_IDS) or
            manifest.get("feature_columns") != list(FEATURE_COLUMNS)):
        raise ValueError("Model training boundary changed")
    fitted = manifest.get("fit_at_utc")
    if fitted and datetime.fromisoformat(fitted) > now:
        raise ValueError("Model was fitted after issuance")
    models = {}
    margin_models = {}
    for arm in MODEL_IDS:
        path = model_dir / f"{arm}.joblib"
        if sha(path) != manifest["model_sha256"][arm]:
            raise ValueError(f"Frozen {arm} model hash changed")
        models[arm] = joblib.load(path)
        margin_path = model_dir / f"{arm}_margin.joblib"
        if sha(margin_path) != manifest["margin_model_sha256"][arm]:
            raise ValueError(f"Frozen {arm} margin model hash changed")
        margin_models[arm] = joblib.load(margin_path)
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
        probabilities = {arm: float(values[0]) for arm, values in
                         predict_panel(models, X.loc[:, list(FEATURE_COLUMNS)]).items()}
        margins = {arm: float(values[0]) for arm, values in
                   predict_margin_panel(margin_models, X.loc[:, list(FEATURE_COLUMNS)]).items()}
        game_id = observation["game_id"]
        deadline = weekly_cutoff_utc if target_game_ids is not None else cutoff(kickoff)
        official_eligible = (official_enabled and
                             (game_id in target_game_ids if target_game_ids is not None else
                              deadline - timedelta(hours=4) <= now <= deadline < kickoff))
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
                               "record_file": str(path), "weekly_cutoff_utc": deadline.isoformat(),
                               "canonical_variant": "F20"}
                    if write_once(official_path,
                                  (json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode()):
                        output["official"] += 1
            output["existing"] += 1
            output["games"].append(game_id)
            continue
        record = {"schema_version": 2, "forecast_id": forecast_id,
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
                  "margin_model_sha256": manifest["margin_model_sha256"],
                  "model_training_through": 2024,
                  "canonical_variant": "F20" if official_eligible else None,
                  "revision_id": revision_id,
                  "model_historical_timing": manifest["historical_source_timing"],
                  "expert_count": available, "home_win_probability": probabilities,
                  "pred_home_margin": margins}
        payload = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        created = write_once(path, payload)
        output["issued" if created else "existing"] += 1
        if official_eligible:
            pointer = {"forecast_id": forecast_id, "record_sha256": sha(path),
                       "record_file": str(path), "weekly_cutoff_utc": deadline.isoformat(),
                       "canonical_variant": "F20"}
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
    parser.add_argument("--models", type=Path, default=ROOT / "data/private/f20/models/f20_scientific_2017_2024_v2_margin")
    parser.add_argument("--revision-id", default="initial")
    parser.add_argument("--week", type=int)
    args = parser.parse_args()
    print(json.dumps(issue(args.archive, args.receipt, args.models, now=datetime.now(UTC),
                           revision_id=args.revision_id, week=args.week), indent=2))


if __name__ == "__main__":
    main()
