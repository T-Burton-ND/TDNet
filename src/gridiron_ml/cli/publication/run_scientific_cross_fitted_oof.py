#!/usr/bin/env python3
"""Generate one deterministic cross-fitted OOF archive for the frozen matrix.

This deliberately retrains temporary historical folds only.  It never opens
the frozen checkpoint for writing and records the exact frozen checkpoint hash
to which the resulting calibrator will be bound.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from gridiron_ml.cli._paths import project_root

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints  # noqa: E402
from gridiron_ml.experiments.publication import build_tuned_model_config, filter_frame_for_feature_config  # noqa: E402
from gridiron_ml.models import build_model_from_config, validate_model_contract  # noqa: E402
from gridiron_ml.td_run.evaluator import TDEval  # noqa: E402
from gridiron_ml.td_run.matchups import MatchupBuilder  # noqa: E402


LEVELS = {"M1", "M2", "M3", "M4", "M5", "M10"}
FINGERPRINTS = {f"F{i}" for i in range(9)}
MARKET_BEARING_FINGERPRINTS = {"F7", "F8"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        try:
            return ROOT / path.resolve().relative_to(ROOT)
        except ValueError:
            return path
    return ROOT / path


def as_bool(value: object) -> bool:
    """Parse manifest booleans without treating the string ``False`` as true."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-inventory", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--first-season", type=int, default=2011)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.first_season < 2011 or args.last_season > 2025 or args.first_season > args.last_season:
        raise ValueError("OOF seasons must be an ordered subset of 2011-2025")

    manifest = pd.read_csv(args.manifest)
    selected = manifest.loc[manifest["task_id"].astype(int).eq(args.task_id)]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one manifest row for task {args.task_id}")
    row = selected.iloc[0].to_dict()
    tier, level = str(row["feature_config"]), str(row["model_level"])
    if tier not in FINGERPRINTS or level not in LEVELS:
        raise ValueError(f"invalid scientific cell {tier}/{level}")
    inventory = pd.read_csv(args.frozen_inventory)
    frozen = inventory.loc[
        inventory["fingerprint"].astype(str).eq(tier)
        & inventory["model_id"].astype(str).str.endswith(f"_{level}")
    ]
    if len(frozen) != 1:
        raise ValueError(f"frozen inventory lacks unique {tier}/{level} entry")
    frozen_row = frozen.iloc[0]
    if str(frozen_row["model_family"]) != str(row["model_family"]):
        raise ValueError(f"model-family mismatch for {tier}/{level}")
    allow_market = as_bool(row.get("market_bearing", tier in MARKET_BEARING_FINGERPRINTS))
    if allow_market != (tier in MARKET_BEARING_FINGERPRINTS):
        raise ValueError(f"market-bearing status is inconsistent for {tier}")
    out = args.output_root / tier / level
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    status = {
        "status": "running",
        "task_id": int(args.task_id),
        "fingerprint_id": tier,
        "model_level": level,
        "model_family": str(row["model_family"]),
        "model_role": str(frozen_row["model_id"]),
        "frozen_checkpoint_path": str(frozen_row["checkpoint_path"]),
        "frozen_checkpoint_sha256": str(frozen_row["checkpoint_sha256"]),
        "oof_seasons_requested": list(range(args.first_season, args.last_season + 1)),
        "prospective_2026_used_for_fit": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    try:
        source = resolve_repo_path(row["source_fingerprint_path"])
        model_config = resolve_repo_path(row["model_config"])
        registry = resolve_repo_path(row["feature_registry"])
        ladders = resolve_repo_path(row["feature_ladders"])
        declared_hashes = {
            source: str(row["source_sha256"]),
            model_config: str(row["model_config_sha256"]),
            registry: str(row["feature_registry_sha256"]),
            ladders: str(row["feature_ladders_sha256"]),
        }
        for path in declared_hashes:
            if not path.exists():
                raise FileNotFoundError(path)
            if sha256(path) != declared_hashes[path]:
                raise ValueError(f"input hash changed after refit manifest creation: {path}")
        raw = pd.read_parquet(source)
        season_column = str(row["source_season_column"])
        if season_column not in raw:
            raise ValueError(f"source fingerprint lacks declared season column {season_column}")
        source_seasons = pd.to_numeric(raw[season_column], errors="coerce")
        raw = raw.loc[source_seasons.le(args.last_season)].copy()
        if raw.empty or pd.to_numeric(raw[season_column], errors="coerce").max() > args.last_season:
            raise ValueError("failed to exclude post-calibration seasons from source frame")
        frame, metadata = filter_frame_for_feature_config(
            raw, feature_config=tier, registry_path=registry, ladders_path=ladders, strict_registry=True
        )
        selected_features = list(metadata.get("selected_features", []))
        if not allow_market and any("market" in str(value).lower() or "vegas" in str(value).lower() for value in selected_features):
            raise ValueError(f"market feature selected for confirmatory {tier}/{level}")
        predictions, failures = [], []
        for season in range(args.first_season, args.last_season + 1):
            try:
                config = build_tuned_model_config(
                    base_config_path=model_config, params=json.loads(str(row.get("params_json") or "{}"))
                )
                config["seed"] = int(row["seed"])
                config["loss_function"] = "MAE"
                if str(row["model_family"]) == "tree":
                    config.setdefault("params", {})["n_jobs"] = max(1, int(args.workers))
                if allow_market:
                    config["allow_market_features_for_training"] = True
                model = build_model_from_config({"family": str(row["model_family"]), **config})
                validate_model_contract(model)
                evaluator = TDEval(
                    config={
                        "model": {"family": str(row["model_family"]), "allow_market_features_for_training": allow_market},
                        "feature_spec": {"include_market": allow_market, "allow_market_features_for_training": allow_market},
                    },
                    fingerprints=StaticFrameFingerprints(frame),
                    matchup_builder=MatchupBuilder(representation="unit_matchup", safe_math=True),
                    model=model,
                )
                evaluator.train(train_years=list(range(2010, season)), val_years=[])
                predicted, _ = evaluator.evaluate(years=[season], label=f"scientific_oof_{tier}_{level}_{season}")
                if "next_game_id" in predicted:
                    predicted["keys_game_id"] = predicted["next_game_id"]
                if "next_week" in predicted:
                    predicted["keys_week"] = predicted["next_week"]
                if "actual_margin" not in predicted and "y" in predicted:
                    predicted["actual_margin"] = predicted["y"]
                predicted["season"] = season
                predicted["fingerprint_id"] = tier
                predicted["model_role"] = str(frozen_row["model_id"])
                keep = ["keys_game_id", "keys_week", "season", "pred_margin", "actual_margin", "fingerprint_id", "model_role"]
                fold = predicted[[column for column in keep if column in predicted]].copy()
                if fold.empty:
                    raise ValueError("evaluation produced no games")
                predictions.append(fold)
                del predicted, evaluator, model
                gc.collect()
            except Exception as exc:  # preserve fold-level evidence for audit
                failures.append({"season": season, "error": repr(exc)})
        if failures:
            raise RuntimeError(f"one or more OOF seasons failed: {failures}")
        if not predictions:
            raise RuntimeError("no OOF seasons completed")
        result = pd.concat(predictions, ignore_index=True).drop_duplicates(["keys_game_id", "season"])
        required = {"keys_game_id", "season", "pred_margin", "actual_margin", "fingerprint_id", "model_role"}
        missing = sorted(required - set(result.columns))
        if missing:
            raise ValueError(f"OOF result missing required columns: {missing}")
        expected_seasons = list(range(args.first_season, args.last_season + 1))
        actual_seasons = sorted(result["season"].unique().astype(int).tolist())
        if actual_seasons != expected_seasons:
            raise ValueError(f"OOF season coverage mismatch: expected {expected_seasons}, got {actual_seasons}")
        numeric = result[["pred_margin", "actual_margin"]].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise ValueError("OOF result contains missing or non-numeric margins")
        if not result["fingerprint_id"].astype(str).eq(tier).all():
            raise ValueError("OOF result fingerprint binding mismatch")
        if not result["model_role"].astype(str).eq(str(frozen_row["model_id"])).all():
            raise ValueError("OOF result model binding mismatch")
        result.to_parquet(out / "oof_predictions.parquet", index=False)
        result.to_csv(out / "oof_predictions.csv", index=False)
        status.update({
            "status": "success",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha256(source),
            "selected_features_sha256": hashlib.sha256(json.dumps(selected_features, separators=(",", ":")).encode()).hexdigest(),
            "prediction_rows": int(len(result)),
            "successful_seasons": actual_seasons,
            "failed_seasons": [],
            "oof_predictions_sha256": sha256(out / "oof_predictions.parquet"),
        })
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(status, indent=2))
        return 0
    except Exception as exc:
        status.update({"status": "failed", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "error": repr(exc)})
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
