#!/usr/bin/env python
"""Audit finalized fingerprint-search models for leakage and robustness signals."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


REPO_ROOT = project_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gridiron_ml.experiments.hyperparameter_search import safe_label


LEAKY_NAME_PATTERNS = [
    r"\by\b",
    r"target",
    r"margin",
    r"points_for",
    r"points_against",
    r"postseason",
    r"final",
    r"bowl",
    r"playoff",
    r"result",
    r"winner",
    r"win_pct",
    r"win_percentage",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("data/experiments/fingerprint_hyperparameter_search"),
    )
    parser.add_argument("--objectives", nargs="*", default=["winner", "balanced", "margin"])
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/fingerprint_hyperparameter_search/audits"))
    parser.add_argument("--confidence-bins", nargs="*", type=float, default=[0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--spread-bins", nargs="*", type=float, default=[0, 3, 7, 14, 21, 1000])
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.project_root.resolve()
    search_root = resolve_path(root, args.search_root)
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_weekly = []
    all_confidence = []
    all_spread = []
    all_leak_flags = []
    all_duplicates = []
    all_errors = []
    manifests = []

    for objective in args.objectives:
        final_root = search_root / str(objective) / "final_artifacts"
        if not final_root.exists():
            continue
        for model_dir in sorted(path for path in final_root.glob("*/*") if path.is_dir()):
            context = model_context(objective, model_dir)
            manifests.append(context)
            pred_path = model_dir / "artifacts" / "predictions" / "predictions.csv"
            source_path = model_dir / "source_search_row.csv"
            if source_path.exists():
                all_leak_flags.extend(feature_leakage_flags(objective, model_dir, source_path))
            if not pred_path.exists():
                summaries.append({**context, "status": "missing_predictions"})
                continue
            try:
                pred = pd.read_csv(pred_path)
            except pd.errors.EmptyDataError:
                summaries.append({**context, "status": "empty_predictions"})
                continue
            metrics = prediction_metrics(pred)
            summaries.append({**context, "status": "audited", **metrics})
            all_weekly.append(with_context(context, weekly_metrics(pred)))
            all_confidence.append(with_context(context, confidence_metrics(pred, args.confidence_bins)))
            all_spread.append(with_context(context, spread_metrics(pred, args.spread_bins)))
            all_duplicates.append(with_context(context, duplicate_checks(pred)))
            errors = error_table(pred)
            if not errors.empty:
                all_errors.append(with_context(context, errors.head(50)))

    write_table(output_dir / "model_audit_summary.csv", pd.DataFrame(summaries))
    write_table(output_dir / "weekly_metrics.csv", concat_or_empty(all_weekly))
    write_table(output_dir / "confidence_bins.csv", concat_or_empty(all_confidence))
    write_table(output_dir / "spread_bins.csv", concat_or_empty(all_spread))
    write_table(output_dir / "feature_leakage_flags.csv", pd.DataFrame(all_leak_flags))
    write_table(output_dir / "duplicate_checks.csv", concat_or_empty(all_duplicates))
    write_table(output_dir / "high_confidence_errors.csv", concat_or_empty(all_errors))
    write_table(output_dir / "artifact_manifest.csv", pd.DataFrame(manifests))
    (output_dir / "audit_readme.md").write_text(audit_readme())
    print(f"Audit outputs: {output_dir}")


def model_context(objective: str, model_dir: Path) -> dict:
    return {
        "objective": objective,
        "family": model_dir.parent.name,
        "model": model_dir.name,
        "model_dir": str(model_dir),
    }


def prediction_metrics(pred: pd.DataFrame) -> dict:
    frame = normalized_predictions(pred)
    if frame.empty:
        return {"n_rows": 0}
    y = frame["actual_home_win"]
    pick = frame["pred_pick_home"]
    correct = pick == y
    market_pick = frame["market_pick_home"]
    home_pick = pd.Series(True, index=frame.index)
    return {
        "n_rows": int(len(frame)),
        "winner_accuracy": float(correct.mean()),
        "balanced_accuracy": balanced_accuracy(y, pick),
        "mcc": matthews_corrcoef(y, pick),
        "brier_score": brier_score(frame),
        "log_loss": log_loss(frame),
        "home_team_accuracy": float((home_pick == y).mean()),
        "market_accuracy": float((market_pick == y).mean()) if market_pick.notna().any() else np.nan,
        "majority_class_accuracy": majority_accuracy(y),
        "favorite_correct": favorite_correct(frame, correct, favorite_won=True),
        "upset_recall": favorite_correct(frame, correct, favorite_won=False),
        "high_confidence_error_count": int((~correct & (frame["confidence"] >= 0.8)).sum()),
        "prediction_rate_home": float(pick.mean()),
        "actual_home_win_rate": float(y.mean()),
        "market_pick_available_rate": float(market_pick.notna().mean()),
    }


def normalized_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    required = {"y", "pred_margin"}
    if not required.issubset(pred.columns):
        return pd.DataFrame()
    frame = pred.copy()
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame["pred_margin"] = pd.to_numeric(frame["pred_margin"], errors="coerce")
    frame = frame.loc[frame["y"].notna() & frame["pred_margin"].notna()].copy()
    if frame.empty:
        return frame
    frame["actual_home_win"] = frame["y"] > 0
    frame["pred_pick_home"] = frame["pred_margin"] > 0
    if "pred_proba_home_win" in frame.columns:
        frame["pred_proba_home_win"] = pd.to_numeric(frame["pred_proba_home_win"], errors="coerce").clip(1e-6, 1 - 1e-6)
    else:
        frame["pred_proba_home_win"] = 1.0 / (1.0 + np.exp(-frame["pred_margin"] / 14.0))
    frame["confidence"] = np.maximum(frame["pred_proba_home_win"], 1.0 - frame["pred_proba_home_win"])
    frame["market_pick_home"] = market_pick_home(frame)
    frame["abs_market_spread"] = market_abs_spread(frame)
    return frame


def market_pick_home(frame: pd.DataFrame) -> pd.Series:
    if "market_win_probability" in frame.columns:
        prob = pd.to_numeric(frame["market_win_probability"], errors="coerce")
        if prob.notna().any():
            return prob > 0.5
    if "market_spread_close" in frame.columns:
        spread = pd.to_numeric(frame["market_spread_close"], errors="coerce")
        return spread < 0
    return pd.Series(np.nan, index=frame.index)


def market_abs_spread(frame: pd.DataFrame) -> pd.Series:
    if "market_spread_close" not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame["market_spread_close"], errors="coerce").abs()


def weekly_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    frame = normalized_predictions(pred)
    if frame.empty or "next_week" not in frame.columns:
        return pd.DataFrame()
    frame["next_week"] = pd.to_numeric(frame["next_week"], errors="coerce")
    rows = []
    for week, group in frame.groupby("next_week", dropna=True):
        rows.append({"week": int(week), **prediction_metrics(group)})
    return pd.DataFrame(rows)


def confidence_metrics(pred: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    frame = normalized_predictions(pred)
    if frame.empty:
        return pd.DataFrame()
    return binned_metrics(frame, "confidence", bins)


def spread_metrics(pred: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    frame = normalized_predictions(pred)
    frame = frame.loc[frame["abs_market_spread"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    return binned_metrics(frame, "abs_market_spread", bins)


def binned_metrics(frame: pd.DataFrame, column: str, bins: list[float]) -> pd.DataFrame:
    out = []
    cats = pd.cut(frame[column], bins=bins, include_lowest=True, right=True)
    for bucket, group in frame.groupby(cats, observed=True):
        metrics = prediction_metrics(group)
        out.append({"bucket": str(bucket), "bucket_column": column, **metrics})
    return pd.DataFrame(out)


def duplicate_checks(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["next_game_id", "keys_game_id"]:
        if col in pred.columns:
            values = pred[col].dropna().astype(str)
            rows.append(
                {
                    "check": f"duplicate_{col}",
                    "rows": int(len(values)),
                    "unique_values": int(values.nunique()),
                    "duplicate_rows": int(values.duplicated().sum()),
                }
            )
    if {"keys_team_home", "keys_team_away", "next_week"}.issubset(pred.columns):
        key = (
            pred["next_week"].astype(str)
            + "::"
            + pred[["keys_team_home", "keys_team_away"]].astype(str).apply(lambda row: "::".join(sorted(row)), axis=1)
        )
        rows.append(
            {
                "check": "mirrored_home_away_week",
                "rows": int(len(key)),
                "unique_values": int(key.nunique()),
                "duplicate_rows": int(key.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)


def error_table(pred: pd.DataFrame) -> pd.DataFrame:
    frame = normalized_predictions(pred)
    if frame.empty:
        return pd.DataFrame()
    frame["correct"] = frame["pred_pick_home"] == frame["actual_home_win"]
    frame["absolute_error"] = (frame["pred_margin"] - frame["y"]).abs()
    keep = [
        "keys_season",
        "next_week",
        "next_game_id",
        "keys_team_home",
        "keys_team_away",
        "y",
        "pred_margin",
        "pred_proba_home_win",
        "confidence",
        "market_spread_close",
        "market_win_probability",
        "absolute_error",
    ]
    keep = [col for col in keep if col in frame.columns]
    return frame.loc[~frame["correct"], keep].sort_values(["confidence", "absolute_error"], ascending=[False, False])


def feature_leakage_flags(objective: str, model_dir: Path, source_path: Path) -> list[dict]:
    try:
        row = pd.read_csv(source_path).iloc[0].to_dict()
    except (pd.errors.EmptyDataError, IndexError):
        return []
    features = parse_json(row.get("selected_features_json"), default=[])
    out = []
    for feature in features:
        reasons = [pattern for pattern in LEAKY_NAME_PATTERNS if re.search(pattern, str(feature), flags=re.IGNORECASE)]
        if reasons:
            out.append(
                {
                    "objective": objective,
                    "family": model_dir.parent.name,
                    "model": model_dir.name,
                    "feature": feature,
                    "matched_patterns": json.dumps(reasons),
                }
            )
    return out


def balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    pos = y_true
    neg = ~y_true
    tpr = (y_pred[pos] == y_true[pos]).mean() if pos.any() else np.nan
    tnr = (y_pred[neg] == y_true[neg]).mean() if neg.any() else np.nan
    return float(np.nanmean([tpr, tnr]))


def matthews_corrcoef(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float(((tp * tn) - (fp * fn)) / denom) if denom else np.nan


def brier_score(frame: pd.DataFrame) -> float:
    y = frame["actual_home_win"].astype(float)
    p = frame["pred_proba_home_win"].astype(float)
    return float(np.mean((p - y) ** 2))


def log_loss(frame: pd.DataFrame) -> float:
    y = frame["actual_home_win"].astype(float)
    p = frame["pred_proba_home_win"].clip(1e-6, 1 - 1e-6).astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def majority_accuracy(y: pd.Series) -> float:
    rate = float(y.astype(bool).mean())
    return max(rate, 1.0 - rate)


def favorite_correct(frame: pd.DataFrame, correct: pd.Series, *, favorite_won: bool) -> float:
    market_pick = frame["market_pick_home"]
    if market_pick.isna().all():
        return np.nan
    favored_won = market_pick == frame["actual_home_win"]
    mask = favored_won if favorite_won else ~favored_won
    return float(correct[mask].mean()) if mask.any() else np.nan


def parse_json(value, *, default):
    if value is None or pd.isna(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def with_context(context: dict, frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    for key, value in reversed(list(context.items())):
        out.insert(0, key, value)
    return out


def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def audit_readme() -> str:
    return """# Fingerprint Final Model Audit

This directory is generated by `src/gridiron_ml/cli/audit_fingerprint_final_models.py`.

Primary tables:
- `model_audit_summary.csv`: overall accuracy, calibration, market/home/majority baselines, MCC, and high-confidence error counts.
- `weekly_metrics.csv`: per-week winner metrics.
- `confidence_bins.csv`: performance by model confidence.
- `spread_bins.csv`: performance by closing-spread bucket.
- `feature_leakage_flags.csv`: selected feature names matching leakage-sensitive patterns; these are flags for review, not proof of leakage.
- `duplicate_checks.csv`: duplicate and mirrored-game checks from prediction rows.
- `high_confidence_errors.csv`: most confident wrong predictions.
"""


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
