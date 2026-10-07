#!/usr/bin/env python3
"""Fit selected F18/F19 models and write true historical rolling OOF margins.

The training program has no 2026 file path. Prospective prediction is a later
step gated by a separate immutable freeze manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

import cloudpickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from constraint_free_search import CONFIG, digest, make_architecture, representation, write_json
from constraint_free_broad_universe import load_broad_historical
from constraint_free_advanced_search import adjusted_model
from gridiron_ml.experiments.constraint_free_advanced import GenomeRepresentation
from gridiron_ml.experiments.constraint_free_extra_reducers import extra_representation

SEARCH_ROOT = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/broad_search")
DEFAULT_OUTPUT = SEARCH_ROOT / "final_fit"
SELECTION = SEARCH_ROOT / "finalists.json"
YEARS = (2022, 2023, 2024, 2025)
BOUND = (
    ROOT / "scripts/constraint_free_search.py",
    ROOT / "scripts/constraint_free_broad_universe.py",
    ROOT / "scripts/constraint_free_advanced_search.py",
    ROOT / "src/gridiron_ml/experiments/constraint_free.py",
    ROOT / "src/gridiron_ml/experiments/constraint_free_advanced.py",
    ROOT / "src/gridiron_ml/experiments/constraint_free_extra_reducers.py",
    ROOT / "configs/experiments/constraint_free_v1.json",
    ROOT / "configs/experiments/constraint_free_advanced_v1.json",
    ROOT / "configs/experiments/constraint_free_extra_v1.json",
    Path(__file__),
)


def bindings() -> dict:
    return {str(path): digest(path) for path in BOUND}


def _representation(winner: dict, records, n_features: int):
    track = winner["track"]
    if track in ("broad_stage1", "broad_stage2"):
        seed = int(json.loads(CONFIG.read_text())["seed"])
        return representation(winner["representation"], records, seed=seed)
    if track == "extra_stage2":
        return extra_representation(winner["representation"], n_features, seed=1729)
    if track in ("ga", "bo"):
        return GenomeRepresentation(winner["genome"], records, winner["tier"])
    raise ValueError(f"Unsupported finalist track: {track}")


def _architecture(winner: dict):
    if winner["track"] in ("ga", "bo"):
        model, backend, _ = adjusted_model(winner["architecture"],
                                            winner["genome"]["hp_index"])
        return model, backend
    return make_architecture(winner["architecture"], json.loads(CONFIG.read_text()))


def _baseline(winner: dict, X: np.ndarray, y: np.ndarray, records,
              train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    genome = winner.get("genome")
    if not genome or genome["target_mode"] != "residual":
        return np.zeros(int(train.sum())), np.zeros(int(test.sum()))
    if winner["tier"] == "F18":
        value = float(y[train].mean())
        return np.full(int(train.sum()), value), np.full(int(test.sum()), value)
    spread = [i for i, record in enumerate(records) if record.name == "market_home_spread"]
    if len(spread) != 1:
        raise ValueError("F19 market residual requires one home spread")
    baseline_train, baseline_test = -X[train, spread[0]], -X[test, spread[0]]
    if not np.isfinite(baseline_train).all() or not np.isfinite(baseline_test).all():
        raise ValueError("F19 residual baseline contains missing historical spreads")
    return baseline_train, baseline_test


def _fit(model, backend: str, fit: np.ndarray, y: np.ndarray) -> None:
    if backend == "sklearn":
        model.fit(fit, y)
    else:
        frame = pd.DataFrame(fit, columns=[f"repr_{i:04d}" for i in range(fit.shape[1])])
        model.train(frame, y)


def _predict(model, backend: str, future: np.ndarray) -> np.ndarray:
    if backend == "sklearn":
        return np.asarray(model.predict(future), dtype=float).reshape(-1)
    frame = pd.DataFrame(future, columns=[f"repr_{i:04d}" for i in range(future.shape[1])])
    return np.asarray(model.predict_margin(frame), dtype=float).reshape(-1)


def fit_split(winner: dict, X: np.ndarray, y: np.ndarray, records,
              train: np.ndarray, test: np.ndarray):
    transformer = _representation(winner, records, X.shape[1])
    fit = transformer.fit_transform(X[train], y[train])
    future = transformer.transform(X[test])
    if not np.isfinite(fit).all() or not np.isfinite(future).all():
        raise ValueError("Finalist preprocessing produced nonfinite values")
    baseline_train, baseline_test = _baseline(winner, X, y, records, train, test)
    model, backend = _architecture(winner)
    _fit(model, backend, fit, y[train] - baseline_train)
    prediction = baseline_test + _predict(model, backend, future)
    if prediction.shape != (int(test.sum()),) or not np.isfinite(prediction).all():
        raise ValueError("Finalist model produced invalid predictions")
    return transformer, model, backend, prediction, baseline_train


def _selected_sources(transformer, records) -> list[str]:
    names = [record.name for record in records]
    if hasattr(transformer, "columns_") and hasattr(transformer, "kept_"):
        return [names[int(transformer.columns_[i])] for i in transformer.kept_]
    if hasattr(transformer, "named_steps") and "selector" in transformer.named_steps:
        mask = transformer.named_steps["selector"].get_support()
        return [name for name, chosen in zip(names, mask, strict=True) if chosen]
    if hasattr(transformer, "named_steps") and "select" in transformer.named_steps:
        mask = transformer.named_steps["select"].get_support()
        return [name for name, chosen in zip(names, mask, strict=True) if chosen]
    return names


def plan(output: Path = DEFAULT_OUTPUT, selection: Path = SELECTION) -> dict:
    chosen = json.loads(selection.read_text())
    if len(chosen["winners"]) != 12 or tuple(chosen["years"]) != YEARS:
        raise ValueError("Finalist selection has incomplete paired roster")
    identities = [(row["tier"], row["architecture"]) for row in chosen["winners"]]
    if len(set(identities)) != 12:
        raise ValueError("Finalists duplicate an architecture")
    for tier in ("F18", "F19"):
        X, meta, records, _ = load_broad_historical(tier)
        if len(meta) != 7358 or X.shape[1] != (467 if tier == "F18" else 472):
            raise ValueError("Historical finalist input universe changed")
    for winner in chosen["winners"]:
        if digest(Path(winner["receipt"])) != winner["receipt_sha256"]:
            raise ValueError(f"Finalist source receipt changed: {winner['receipt']}")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=ROOT, text=True).strip(),
        "selection": str(selection), "selection_sha256": digest(selection),
        "bindings": bindings(), "years": list(YEARS),
        "tasks": chosen["winners"], "task_concurrency_max": 10,
        "training_seasons": [2013, 2025],
        "no_2026_features_or_outcomes_read": True,
    }
    path = output / "manifest.json"
    if path.exists():
        raise FileExistsError(f"Immutable final fit manifest exists: {path}")
    write_json(path, manifest)
    return {"manifest": str(path), "sha256": digest(path), "tasks": 12}


def run(manifest_path: Path, task_id: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["bindings"] != bindings() or manifest["selection_sha256"] != digest(Path(manifest["selection"])):
        raise ValueError("Final fit source or finalist selection changed")
    if task_id < 1 or task_id > len(manifest["tasks"]):
        raise IndexError("Final fit task outside manifest")
    winner = manifest["tasks"][task_id - 1]
    if digest(Path(winner["receipt"])) != winner["receipt_sha256"]:
        raise ValueError("Finalist source receipt changed after planning")
    cell = manifest_path.parent / f"{winner['tier']}_{winner['architecture']}"
    result_path = cell / "result.json"
    if result_path.exists():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "success" and prior.get("manifest_sha256") == digest(manifest_path):
            for name, expected in prior["artifacts"].items():
                if digest(cell / name) != expected:
                    raise ValueError(f"Saved final fit artifact changed: {cell / name}")
            return prior
        raise FileExistsError(f"Nonreusable final fit result: {result_path}")
    cell.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "tier": winner["tier"],
              "architecture": winner["architecture"], "track": winner["track"],
              "manifest_sha256": digest(manifest_path),
              "selection_receipt_sha256": winner["receipt_sha256"],
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        X, meta, records, evidence = load_broad_historical(winner["tier"])
        years = meta.season.to_numpy(int)
        y = meta.next_game_margin.to_numpy(float)
        oof_rows = []
        for year in YEARS:
            train, test = years < year, years == year
            _, _, _, prediction, _ = fit_split(winner, X, y, records, train, test)
            oof_rows.append(pd.DataFrame({
                "target_game_id": meta.loc[test, "target_game_id"].to_numpy(int),
                "season": year, "actual_margin": y[test], "pred_margin": prediction,
            }))
        oof = pd.concat(oof_rows, ignore_index=True).sort_values("target_game_id")
        if len(oof) != sum(np.isin(years, YEARS)) or oof.target_game_id.duplicated().any():
            raise ValueError("OOF game coverage changed")
        # Fit a full-data representation and model without reading prospective rows.
        transformer = _representation(winner, records, X.shape[1])
        fit = transformer.fit_transform(X, y)
        if not np.isfinite(fit).all():
            raise ValueError("Full-data preprocessing produced nonfinite values")
        if winner.get("genome") and winner["genome"]["target_mode"] == "residual":
            if winner["tier"] == "F18":
                baseline = np.full(len(y), float(y.mean()))
            else:
                spread = next(i for i, record in enumerate(records)
                              if record.name == "market_home_spread")
                baseline = -X[:, spread]
                if not np.isfinite(baseline).all():
                    raise ValueError("Full F19 historical spread is missing")
        else:
            baseline = np.zeros(len(y))
        model, backend = _architecture(winner)
        _fit(model, backend, fit, y - baseline)
        check = _predict(model, backend, transformer.transform(X[:3]))
        if check.shape != (3,) or not np.isfinite(check).all():
            raise ValueError("Saved model has invalid roundtrip probe")
        bundle = {
            "tier": winner["tier"], "architecture": winner["architecture"],
            "track": winner["track"], "representation": winner["representation"],
            "genome": winner["genome"], "records": records,
            "source_feature_names": [record.name for record in records],
            "source_evidence": evidence,
            "transformer": transformer, "model": model, "backend": backend,
            "residual_baseline_mean": (float(y.mean()) if winner.get("genome")
                                       and winner["genome"]["target_mode"] == "residual"
                                       and winner["tier"] == "F18" else None),
            "historical_training_games": len(y),
        }
        model_path = cell / "fitted_bundle.pkl"
        with model_path.open("wb") as handle:
            cloudpickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        preprocessing_path = cell / "preprocessing.pkl"
        with preprocessing_path.open("wb") as handle:
            cloudpickle.dump(transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with model_path.open("rb") as handle:
            loaded = pickle.load(handle)
        replay = _predict(loaded["model"], loaded["backend"],
                          loaded["transformer"].transform(X[:3]))
        if not np.allclose(check, replay):
            raise ValueError("Saved model roundtrip changed predictions")
        oof_path = cell / "oof.parquet"
        oof.to_parquet(oof_path, index=False)
        result.update(status="success", artifacts={
            model_path.name: digest(model_path),
            preprocessing_path.name: digest(preprocessing_path),
            oof_path.name: digest(oof_path)},
            model_fits=len(YEARS) + 1, oof_games=len(oof),
            oof_mae=float(np.abs(oof.actual_margin - oof.pred_margin).mean()),
            representation_features=int(fit.shape[1]),
            source_features=int(X.shape[1]),
            selected_source_features=_selected_sources(transformer, records),
            historical_market_quote_timing=("unverified" if winner["tier"] == "F19" else "none"))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=15))
    result["runtime_seconds"] = time.monotonic() - started
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    if args.action == "plan":
        result = plan(args.output, args.selection)
    else:
        if args.manifest is None or args.task_id is None:
            parser.error("run requires --manifest and --task-id")
        result = run(args.manifest, args.task_id)
    print(json.dumps(result, indent=2, default=str), flush=True)
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
