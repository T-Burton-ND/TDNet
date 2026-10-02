# Versioned parallel-full runner: preserves the original live F06 runner unchanged.
"""Shared checked M2/M4 screening, future-outcome metrics and source SHAP.

One task fits one frozen setpoint on 2010–2023 and reports 2024/2025
separately. Probability calibration uses only rolling-origin training residuals.
No model checkpoint is retained.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import inspect
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .nextgen_contract import assert_design_operation_frame, parse_fingerprint_id, validate_feature_manifest
from .nextgen_artifacts import NextgenModelBoundary
from gridiron_ml.models.td_spline import TDSpline
from gridiron_ml.models.td_tree import TDTree
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]
SEED = 1701


def execution_binding(root: Path, provenance: dict, architecture: str, point: dict):
    """Bind resumable results to the computation, evaluation data and inputs."""
    paths = [Path(__file__), Path(inspect.getfile(TDSpline)), Path(inspect.getfile(TDTree)),
             ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json",
             root / "results/source_semantics_audit.json",
             ROOT / "configs/experiments/nextgen_parallel_authorization.json"]
    market = root / "canonical/evaluation_market_sidecar.parquet"
    return {
        "scheduling_policy": "parallel_full_v1",
        "fingerprint_provenance": provenance,
        "architecture": architecture, "setpoint": point, "seed": SEED,
        "files": {str(p): sha256_file(p) for p in paths},
        "market_sha256": sha256_file(market) if market.exists() else None,
        "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "shap")},
    }


def verify_reusable_result(prior: dict, binding: dict, out: Path):
    if prior.get("execution_binding") != binding:
        raise ValueError("Refusing to reuse a result from different inputs or computation")
    hashes = prior.get("output_sha256", {})
    for name in ("predictions.parquet", "source_shap.parquet"):
        path = out / name
        if not path.exists() or hashes.get(name) != sha256_file(path):
            raise ValueError(f"Result output missing or changed: {name}")


def build_estimator(architecture: str, setpoint: dict):
    params = {k: v for k, v in setpoint.items() if k != "id"}
    if architecture == "M2":
        return TDSpline({"model_type": "spline_ridge", "seed": SEED, "loss_function": "MAE",
                         "params": {"alpha": params["alpha"]},
                         "spline": {"n_knots": params["n_knots"], "degree": params["degree"]}})._build_pipeline()
    if architecture == "M4":
        # The same histogram estimator and frozen parameters, with native NaNs.
        return TDTree({"model_type": "hist_gradient_boosted", "seed": SEED, "loss_function": "MAE",
                       "params": params})._build_estimator()
    raise ValueError("Only M2/M4 are authorized for screening")


def source_to_matchup(values: np.ndarray, records: list[dict]) -> np.ndarray:
    """Reciprocal offense/defense sums and same-measure home/away differences.

    Input coordinates remain home/away team-week values for end-to-end SHAP.
    Each fingerprint uses the same representation across architectures.
    """
    names = [r["name"] for r in records]
    index = {n: i for i, n in enumerate(names)}
    counterpart = [index[r["matchup_counterpart"]] for r in records]
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2*len(names):
        raise ValueError("Wrong source-coordinate dimension")
    signs = np.array([-1.0 if r["matchup_counterpart"] == r["name"] else 1.0 for r in records])
    for r, sign in zip(records, signs):
        expected = f"home.[{r['name']}]{'-' if sign < 0 else '+'}away.[{r['matchup_counterpart']}]"
        if r["matchup_formula"] != expected:
            raise ValueError("Runner does not implement the declared matchup equation")
    return x[:, :len(names)]+signs*x[:, len(names):][:, counterpart]


def checked_fingerprint(root: Path, fingerprint: str):
    generation, _, _ = parse_fingerprint_id(fingerprint)
    dest = root / "fingerprints" / fingerprint
    provenance = json.loads((dest / "provenance.json").read_text())
    manifest_path, data_path = dest / "feature_manifest.json", dest / "values.parquet"
    if sha256_file(manifest_path) != provenance["manifest_sha256"] or sha256_file(data_path) != provenance["data_sha256"]:
        raise ValueError("Fingerprint data/manifest hash mismatch")
    records = json.loads(manifest_path.read_text())
    schema = json.loads((ROOT / "configs/experiments/nextgen_feature_manifest_schema_v1.json").read_text())
    validate_feature_manifest(records, schema)
    audit_path = root / "results/source_semantics_audit.json"
    if not audit_path.exists():
        raise ValueError("Source-semantics audit is required before training")
    audit = json.loads(audit_path.read_text())
    if not audit.get("training_authorized") or audit.get("schedule_sha256") != provenance["schedule_sha256"]:
        raise ValueError("Source-semantics audit has not authorized this schedule")
    baseline_path = root / "canonical/f06_aligned.parquet"
    if audit.get("f06_aligned_sha256") != sha256_file(baseline_path):
        raise ValueError("Baseline source audit is stale")
    canonical_values = []
    for family in provenance.get("canonical_families", []):
        boundary = NextgenModelBoundary.from_canonical(root, family["generation"], family["family"], Path(family["manifest_path"]))
        canonical_values.append(boundary.for_fit())
    if generation != "F06" and not provenance.get("canonical_families"):
        raise ValueError("New-generation fingerprint lacks checked canonical ancestry")
    frame = pd.read_parquet(data_path)
    from .nextgen_source_policy import assert_excluded_absent, EXCLUDED_COACH_SP
    assert_excluded_absent(frame, records)
    if set(audit.get('excluded_features', [])) != EXCLUDED_COACH_SP:
        raise ValueError('Baseline exclusion decision is not bound to source audit')
    from .nextgen_assembly import assert_family_values
    for values in canonical_values:
        assert_family_values(frame, values)
    assert_design_operation_frame(frame, "model_evaluation")
    names = [r["name"] for r in records]
    if frame.duplicated(["target_game_id", "team"]).any():
        raise ValueError("Duplicate target-team rows")
    home = frame.loc[frame.is_home].set_index("target_game_id").sort_index()
    away = frame.loc[~frame.is_home].set_index("target_game_id").sort_index()
    if not home.index.equals(away.index) or not home.season.equals(away.season):
        raise ValueError("Fingerprint does not contain paired scheduled targets")
    if not np.allclose(home.next_game_margin, -away.next_game_margin):
        raise ValueError("Reciprocal outcomes disagree")
    raw = np.column_stack([home[names].to_numpy(float), away[names].to_numpy(float)])
    meta = home[["season", "week", "target_start_utc", "next_game_margin", "next_game_win"]].reset_index()
    return raw, meta, records, provenance


def future_metrics(actual, prediction, probability, spread=None) -> dict:
    y, p, probability = map(lambda a: np.asarray(a, dtype=float), (actual, prediction, probability))
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(probability)
    y, p, probability = y[valid], p[valid], probability[valid]
    if not len(y):
        raise ValueError("No finite evaluation observations")
    correct = (p > 0) == (y > 0)
    result = {"n_games": len(y), "mae": float(np.abs(y-p).mean()),
              "rmse": float(np.sqrt(np.square(y-p).mean())),
              "brier": float(np.square(probability-(y > 0)).mean()),
              "winner_accuracy": float(correct.mean()), "ats_accuracy": None,
              "chalk_accuracy": None, "upset_accuracy": None, "n_market_games": 0}
    if spread is not None:
        s = np.asarray(spread, dtype=float)[valid]
        market = np.isfinite(s)
        result["n_market_games"] = int(market.sum())
        nonpush = market & (np.abs(y+s) > 1e-8) & (np.abs(p+s) > 1e-8)
        if nonpush.any():
            result["ats_accuracy"] = float(((p[nonpush]+s[nonpush] > 0) == (y[nonpush]+s[nonpush] > 0)).mean())
        # Consistent with TDNet season_vs_vegas: conditioned on actual upset/chalk.
        meaningful = market & (s != 0) & (y != 0)
        chalk = meaningful & ((y > 0) == (s < 0))
        upset = meaningful & ~chalk
        for label, mask in (("chalk_accuracy", chalk), ("upset_accuracy", upset)):
            if mask.any():
                result[label] = float(correct[mask].mean())
    return result


def residual_probability(prediction, residuals):
    r = np.sort(np.asarray(residuals, dtype=float))
    r = r[np.isfinite(r)]
    if len(r) < 100:
        raise ValueError("Insufficient training-only calibration residuals")
    # Smoothed empirical CDF avoids exact 0/1 from a finite calibration sample.
    return (len(r)-np.searchsorted(r, -np.asarray(prediction), side="right")+0.5)/(len(r)+1)


def market_sidecar(root: Path, game_ids):
    path = root / "canonical/evaluation_market_sidecar.parquet"
    if not path.exists():
        return np.full(len(game_ids), np.nan)
    d = pd.read_parquet(path)
    if d.target_game_id.duplicated().any():
        raise ValueError("Duplicate evaluation sidecar game")
    return d.set_index("target_game_id").home_spread.reindex(game_ids).to_numpy(float)


def permutation_source_shap(estimator, records, background, explain, *, seed=SEED):
    """Full empirical-background permutation SHAP, matching the existing study.

    Uses all selected background rows for each explained row, with an
    antithetic forward/reverse permutation traversal. Source home/away
    coordinates are aggregated only after explaining the complete predictor.
    """
    import shap
    n = len(records)
    def predict(values):
        return estimator.predict(source_to_matchup(values, records))
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(predict, masker, algorithm="permutation", seed=seed)
    explanation = explainer(explain, max_evals=2*explain.shape[1]+1, batch_size=512, silent=True)
    values = np.asarray(explanation.values, dtype=float)
    importance = np.abs(values[:, :n]).mean(axis=0)+np.abs(values[:, n:]).mean(axis=0)
    total = importance.sum()
    table = pd.DataFrame({"source_feature": [r["name"] for r in records],
                          "mean_abs_shap": importance,
                          "normalized_importance": importance/total if total > 0 else np.zeros(n)})
    residual = predict(explain)-np.asarray(explanation.base_values)-values.sum(axis=1)
    return table, {"method": "end_to_end_source_coordinate_permutation", "background_games": len(background),
                   "explanation_games": len(explain), "seed": seed,
                   "max_absolute_additivity_error": float(np.abs(residual).max())}


def generation_barrier(root: Path, fingerprint: str):
    """User-authorized parallel full screening; reductions retain dependencies."""
    generation, lineage, design = parse_fingerprint_id(fingerprint)
    if lineage != 'F':
        raise ValueError('Parallel runner currently supports independent full fingerprints only')
    policy = json.loads((ROOT / 'configs/experiments/nextgen_parallel_authorization.json').read_text())
    if policy.get('independent_full_training_authorized') is not True:
        raise ValueError('Parallel full screening is not authorized')


def run_task(root: Path, fingerprint: str, architecture: str, setpoint_id: str):
    generation, lineage, design = parse_fingerprint_id(fingerprint)
    generation_barrier(root, fingerprint)
    points = json.loads((ROOT / "configs/experiments/nextgen_screening_setpoints_v1.json").read_text())
    point = next(p for p in points[architecture] if p["id"] == setpoint_id)
    raw, meta, records, provenance = checked_fingerprint(root, fingerprint)
    run_id = f"{fingerprint}__{architecture}__{setpoint_id}"
    out = root / "experiments" / generation / run_id
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    binding = execution_binding(root, provenance, architecture, point)
    prior = None
    if result_path.exists():
        prior = json.loads(result_path.read_text())
        if prior.get("status") == "success":
            verify_reusable_result(prior, binding, out)
            return prior
    attempt_number = int(prior.get("attempt_number", 1))+1 if prior else 1
    if attempt_number > 4:
        raise ValueError("Initial attempt plus three screening retries exhausted")
    if prior:
        atomic_json(out / "attempts" / f"attempt_{attempt_number-1}.json", prior)
    start = time.monotonic()
    result = {"row_type": "run", "run_id": run_id, "fingerprint_id": fingerprint,
              "execution_binding": binding,
              "attempt_number": attempt_number,
              "generation": generation, "lineage": lineage, "design": design, "model": architecture,
              "hyperparameter_setpoint": setpoint_id, "seed": SEED, "feature_count": len(records),
              "feature_manifest_sha256": provenance["manifest_sha256"], "data_sha256": provenance["data_sha256"],
              "max_design_year_used": int(meta.season.max()), "prospective_boundary_year": 2026,
              "status": "incomplete", "started_at_utc": datetime.now(timezone.utc).isoformat()}
    acquisition_manifest = root / "results/preflight/cfbd_request_manifest_v1.jsonl"
    result.update(
        acquisition_manifest_sha256=sha256_file(acquisition_manifest),
        acquisition_summary_ref=str(root / "results/preflight/cfbd_plan_summary_v1.json"),
        missingness_summary_json=json.dumps({r["name"]: float(np.isnan(raw[:, [i, i+len(records)]]).mean())
                                            for i, r in enumerate(records)}, sort_keys=True),
        coverage_summary_json=json.dumps({str(year): int(count) for year, count in meta.season.value_counts().items()}, sort_keys=True),
    )
    atomic_json(out / "progress.json", {**result, "phase": "training"})
    try:
        x = source_to_matchup(raw, records)
        y, years = meta.next_game_margin.to_numpy(float), meta.season.to_numpy(int)
        residuals = []
        # Neither 2024 nor 2025 influences fitting or probability calibration.
        for cutoff in (2019, 2021):
            fit, calibrate = years <= cutoff, (years > cutoff) & (years <= cutoff+2)
            m = build_estimator(architecture, point)
            m.fit(x[fit], y[fit])
            residuals.extend(y[calibrate]-m.predict(x[calibrate]))
        estimator = build_estimator(architecture, point)
        train = years <= 2023
        estimator.fit(x[train], y[train])
        predictions = estimator.predict(x)
        probabilities = residual_probability(predictions, residuals)
        spreads = market_sidecar(root, meta.target_game_id)
        for year in (2024, 2025):
            mask = years == year
            metrics = future_metrics(y[mask], predictions[mask], probabilities[mask], spreads[mask])
            result[f"development_{year}_metrics_json"] = json.dumps(metrics)
            for metric in ("mae", "rmse", "brier", "winner_accuracy", "ats_accuracy", "chalk_accuracy", "upset_accuracy"):
                result[f"{metric}_{year}"] = metrics[metric]
        result["train_metrics_json"] = json.dumps(future_metrics(y[train], predictions[train], probabilities[train]))
        result["calibration_residual_count"] = len(residuals)
        pd.DataFrame({"target_game_id": meta.target_game_id, "season": years,
                      "actual_margin": y, "predicted_margin": predictions,
                      "home_win_probability": probabilities}).loc[years >= 2024].to_parquet(out / "predictions.parquet", index=False)
        atomic_json(out / "progress.json", {**result, "phase": "shap", "elapsed_seconds": time.monotonic()-start})
        rng = np.random.default_rng(SEED)
        bg_idx = rng.choice(np.flatnonzero(train), size=min(256, int(train.sum())), replace=False)
        # Equal year representation, held constant across models and setpoints.
        explain_idx = np.concatenate([rng.choice(np.flatnonzero(years == year), size=min(256, int((years == year).sum())), replace=False)
                                      for year in (2024, 2025)])
        importance, shap_report = permutation_source_shap(estimator, records, raw[bg_idx], raw[explain_idx])
        importance.to_parquet(out / "source_shap.parquet", index=False)
        result.update(status="success", shap_summary_ref=str(out / "source_shap.parquet"),
                      output_sha256={name: sha256_file(out / name) for name in ("predictions.parquet", "source_shap.parquet")},
                      shap_report=shap_report, failure_reason=None)
    except Exception as exc:
        result.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=8))
    result["runtime_seconds"] = time.monotonic()-start
    atomic_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--model", choices=["M2", "M4"], required=True)
    parser.add_argument("--setpoint", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/experiments/nextgen_fingerprints_v1.json").read_text())
    result = run_task(Path(config["artifact_root"]), args.fingerprint, args.model, args.setpoint)
    print(json.dumps(result, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
