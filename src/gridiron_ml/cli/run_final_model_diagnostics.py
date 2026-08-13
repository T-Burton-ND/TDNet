#!/usr/bin/env python
"""Build diagnostics for finalized fingerprint-search models.

The runner works from saved final artifacts so it can be submitted as a small
SGE array.  Each task handles one objective/family/model directory and writes
tables plus PNG diagnostics for prediction quality, confidence behavior,
feature reliance, and fingerprint sanity checks.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = project_root()
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.finalize_fingerprint_hyperparameter_search import load_selected_frame


OBJECTIVES = ("winner", "balanced", "margin")
CONFIDENCE_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
LEAKY_PATTERNS = (
    "next_game_id",
    "keys_game_id",
    "game_id",
    "target",
    "points_for",
    "points_against",
    "team_margin",
    "winner",
    "result",
    "final",
)
REQUESTED_OUTPUTS = (
    "01_confusion_matrix",
    "02_roc_curve",
    "03_precision_recall_curve",
    "04_calibration_curve",
    "05_confidence_histogram",
    "06_accuracy_vs_confidence",
    "07_coverage_vs_accuracy",
    "08_cumulative_gains_lift",
    "09_brier_decomposition",
    "10_probability_by_actual_outcome",
    "11_residual_error_distribution",
    "12_error_vs_confidence",
    "13_error_vs_game_closeness",
    "14_performance_by_subgroup",
    "15_temporal_performance",
    "16_learning_curve",
    "17_complexity_validation_curve",
    "18_hyperparameter_validation_curve",
    "19_cross_validation_fold_performance",
    "20_high_confidence_mistakes",
    "21_permutation_importance",
    "22_shap_summary",
    "23_grouped_feature_importance",
    "24_feature_importance_stability",
    "25_fingerprint_correlation_heatmap",
    "26_fingerprint_feature_clustering",
    "27_pca_explained_variance",
    "28_pca_projection",
    "29_umap_or_tsne_embedding",
    "30_fingerprint_distance_outcome_similarity",
    "31_nearest_neighbor_outcome_consistency",
    "32_within_between_team_distance",
    "33_fingerprint_drift_over_time",
    "34_class_separation_by_feature",
    "35_partial_dependence",
    "36_ice_plots",
    "37_two_dimensional_dependence",
    "38_feature_ablation_performance",
    "39_cumulative_feature_performance",
    "40_random_or_shuffled_baseline",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-manifest")
    add_common(build)
    build.add_argument("--include-gradient-boosted", action="store_true")

    run = sub.add_parser("run-job")
    add_common(run)
    run.add_argument("--job-index", type=int, default=None)
    run.add_argument("--sge-task-id", type=int, default=None)
    run.add_argument("--manifest", type=Path, default=None)
    run.add_argument("--force", action="store_true")
    run.add_argument("--fingerprint-sample", type=int, default=2500)

    merge = sub.add_parser("merge")
    add_common(merge)

    return parser.parse_args()


def add_common(parser):
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("data/experiments/fingerprint_hyperparameter_search"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/experiments/fingerprint_hyperparameter_search/final_model_diagnostics"),
    )
    parser.add_argument(
        "--source-fingerprint-root",
        type=Path,
        default=Path("data/experiments/opponent_adjusted_fingerprints"),
    )
    parser.add_argument("--objectives", nargs="*", default=list(OBJECTIVES))


def main():
    args = parse_args()
    root = args.project_root.resolve()
    search_root = resolve(root, args.search_root)
    output_root = resolve(root, args.output_root)
    source_root = resolve(root, args.source_fingerprint_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.command == "build-manifest":
        manifest = build_manifest(
            search_root=search_root,
            output_root=output_root,
            objectives=tuple(args.objectives),
            include_gradient_boosted=bool(args.include_gradient_boosted),
        )
        path = output_root / "job_manifest.csv"
        manifest.to_csv(path, index=False)
        print(f"Manifest: {path}")
        print(f"Jobs: {len(manifest)}")
        return

    if args.command == "run-job":
        manifest_path = args.manifest or output_root / "job_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        job_index = resolve_job_index(args.job_index, args.sge_task_id)
        if job_index < 0 or job_index >= len(manifest):
            raise IndexError(f"job_index {job_index} outside manifest size {len(manifest)}")
        row = manifest.iloc[job_index].to_dict()
        result_path = Path(row["diagnostic_dir"]) / "tables" / "overall_metrics.csv"
        if result_path.exists() and not args.force:
            print(f"SKIP existing {result_path}")
            return
        run_diagnostics(
            row=row,
            source_root=source_root,
            fingerprint_sample=int(args.fingerprint_sample),
        )
        print(f"Output: {row['diagnostic_dir']}")
        return

    if args.command == "merge":
        merged = merge_outputs(output_root)
        print(f"Merged diagnostics: {output_root / 'summary'}")
        for name, frame in merged.items():
            print(f"{name}: {len(frame)} rows")
        return


def build_manifest(*, search_root: Path, output_root: Path, objectives: tuple[str, ...], include_gradient_boosted: bool) -> pd.DataFrame:
    rows = []
    for objective in objectives:
        final_root = search_root / objective / "final_artifacts"
        selected_path = final_root / "selected_best_by_model.csv"
        if not selected_path.exists():
            continue
        selected = pd.read_csv(selected_path)
        for item in selected.to_dict("records"):
            family = str(item["family"])
            model = str(item["model"])
            if model == "gradient_boosted" and not include_gradient_boosted:
                continue
            model_dir = final_root / safe(family) / safe(model)
            pred_path = model_dir / "artifacts" / "predictions" / "predictions.csv"
            checkpoint_paths = list((model_dir / "checkpoints").glob("*.pkl"))
            if not pred_path.exists() or not checkpoint_paths:
                continue
            diag_dir = output_root / "runs" / objective / safe(family) / safe(model)
            rows.append(
                {
                    "job_index": len(rows),
                    "objective": objective,
                    "family": family,
                    "model": model,
                    "model_dir": str(model_dir),
                    "prediction_path": str(pred_path),
                    "source_search_row": str(model_dir / "source_search_row.csv"),
                    "diagnostic_dir": str(diag_dir),
                    "checkpoint_path": str(checkpoint_paths[0]),
                }
            )
    return pd.DataFrame(rows)


def run_diagnostics(*, row: dict, source_root: Path, fingerprint_sample: int) -> None:
    out = Path(row["diagnostic_dir"])
    tables = out / "tables"
    plots = out / "plots"
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    status = {name: "not_started" for name in REQUESTED_OUTPUTS}

    pred = pd.read_csv(row["prediction_path"])
    frame = normalize_predictions(pred)
    if frame.empty:
        pd.DataFrame([{"status": "empty_predictions"}]).to_csv(tables / "overall_metrics.csv", index=False)
        write_requested_status(tables, status)
        return

    overall = overall_metrics(frame, row)
    pd.DataFrame([overall]).to_csv(tables / "overall_metrics.csv", index=False)
    confidence_table(frame).to_csv(tables / "confidence_thresholds.csv", index=False)
    coverage_accuracy_table(frame).to_csv(tables / "coverage_accuracy.csv", index=False)
    performance_by_group(frame).to_csv(tables / "performance_by_group.csv", index=False)
    high_confidence_errors(frame).to_csv(tables / "high_confidence_errors.csv", index=False)
    memorization_checks(frame, row).to_csv(tables / "memorization_checks.csv", index=False)
    status["20_high_confidence_mistakes"] = "table"

    feature_rows = feature_tables(row, tables)
    status.update(fingerprint_diagnostics(row, source_root, tables, plots, fingerprint_sample))

    plot_confusion(frame, plots / "confusion_matrix.png")
    status["01_confusion_matrix"] = "figure"
    plot_roc(frame, plots / "roc_curve.png")
    status["02_roc_curve"] = "figure"
    plot_pr(frame, plots / "precision_recall_curve.png")
    status["03_precision_recall_curve"] = "figure"
    plot_calibration(frame, plots / "calibration_curve.png")
    status["04_calibration_curve"] = "figure"
    plot_confidence_hist(frame, plots / "confidence_histogram.png")
    status["05_confidence_histogram"] = "figure"
    plot_accuracy_vs_confidence(frame, plots / "accuracy_vs_confidence.png")
    status["06_accuracy_vs_confidence"] = "figure"
    plot_coverage_accuracy(frame, plots / "coverage_vs_accuracy.png")
    status["07_coverage_vs_accuracy"] = "figure"
    plot_cumulative_gains(frame, plots / "cumulative_gains_lift.png")
    status["08_cumulative_gains_lift"] = "figure"
    brier_decomposition(frame).to_csv(tables / "brier_decomposition.csv", index=False)
    plot_brier_decomposition(frame, plots / "brier_decomposition.png")
    status["09_brier_decomposition"] = "figure"
    plot_probability_by_outcome(frame, plots / "probability_by_actual_outcome.png")
    status["10_probability_by_actual_outcome"] = "figure"
    plot_residual_distribution(frame, plots / "residual_error_distribution.png")
    status["11_residual_error_distribution"] = "figure"
    plot_error_vs_confidence(frame, plots / "error_vs_confidence.png")
    status["12_error_vs_confidence"] = "figure"
    plot_error_vs_closeness(frame, plots / "error_vs_game_closeness.png")
    status["13_error_vs_game_closeness"] = "figure"
    plot_subgroup_grid(frame, plots / "performance_by_subgroup.png")
    status["14_performance_by_subgroup"] = "figure"
    plot_performance_bars(frame, plots / "performance_by_week.png", group_col="week")
    status["15_temporal_performance"] = "figure"
    if "season" in frame.columns:
        plot_performance_bars(frame, plots / "performance_by_season.png", group_col="season")
    plot_feature_importance(feature_rows, plots / "feature_importance.png")
    if feature_rows is not None and not feature_rows.empty:
        status["21_permutation_importance"] = "not_available_requires_model_reprediction"
        status["22_shap_summary"] = "not_available_run_shap_job"
        status["24_feature_importance_stability"] = "not_available_requires_folds_or_bootstraps"
        status["34_class_separation_by_feature"] = "figure"
        plot_class_separation(row, source_root, feature_rows, plots / "class_separation_by_feature.png")
    plot_feature_family_importance(feature_rows, plots / "feature_family_importance.png")
    if feature_rows is not None and not feature_rows.empty:
        status["23_grouped_feature_importance"] = "figure"
    status["16_learning_curve"] = "not_available_requires_refits_at_train_sizes"
    status["17_complexity_validation_curve"] = "not_available_requires_hyperparameter_sweep"
    status["18_hyperparameter_validation_curve"] = "not_available_requires_hyperparameter_sweep"
    status["19_cross_validation_fold_performance"] = "not_available_requires_fold_refits"
    status["35_partial_dependence"] = "not_available_requires_model_reprediction"
    status["36_ice_plots"] = "not_available_requires_model_reprediction"
    status["37_two_dimensional_dependence"] = "not_available_requires_model_reprediction"
    status["38_feature_ablation_performance"] = "not_available_use_ablation_array"
    status["39_cumulative_feature_performance"] = "not_available_requires_refits"
    status["40_random_or_shuffled_baseline"] = "not_available_requires_refits"
    write_requested_status(tables, status)


def normalize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    if not {"y", "pred_margin"}.issubset(pred.columns):
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
        prob = pd.to_numeric(frame["pred_proba_home_win"], errors="coerce")
    else:
        prob = 1.0 / (1.0 + np.exp(-frame["pred_margin"] / 14.0))
    frame["pred_proba_home_win"] = prob.clip(1e-6, 1 - 1e-6)
    frame["confidence"] = np.maximum(frame["pred_proba_home_win"], 1.0 - frame["pred_proba_home_win"])
    frame["correct"] = frame["actual_home_win"] == frame["pred_pick_home"]
    frame["abs_margin"] = frame["y"].abs()
    frame["prob_error"] = frame["actual_home_win"].astype(float) - frame["pred_proba_home_win"]
    frame["abs_prob_error"] = frame["prob_error"].abs()
    frame["week"] = first_numeric(frame, ["week", "keys_week", "next_week"])
    frame["season"] = first_numeric(frame, ["season", "keys_season"])
    frame["market_pick_home"] = market_pick_home(frame)
    frame["favorite_won"] = favorite_won(frame)
    return frame.reset_index(drop=True)


def overall_metrics(frame: pd.DataFrame, row: dict) -> dict:
    y = frame["actual_home_win"].astype(int)
    p = frame["pred_proba_home_win"]
    pick = frame["pred_pick_home"].astype(int)
    out = {
        "objective": row["objective"],
        "family": row["family"],
        "model": row["model"],
        "n_rows": int(len(frame)),
        "winner_accuracy": float(frame["correct"].mean()),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_confidence": float(frame["confidence"].mean()),
        "high_confidence_error_count": int((~frame["correct"] & (frame["confidence"] >= 0.8)).sum()),
        "prediction_rate_home": float(frame["pred_pick_home"].mean()),
        "actual_home_win_rate": float(frame["actual_home_win"].mean()),
    }
    if y.nunique() == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
    market = frame["market_pick_home"]
    if market.notna().any():
        out["market_accuracy"] = float((market.dropna().astype(bool) == frame.loc[market.notna(), "actual_home_win"]).mean())
    return out


def confidence_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in CONFIDENCE_THRESHOLDS:
        subset = frame.loc[frame["confidence"] >= threshold]
        rows.append(
            {
                "confidence_threshold": threshold,
                "coverage": float(len(subset) / max(len(frame), 1)),
                "n_rows": int(len(subset)),
                "winner_accuracy": float(subset["correct"].mean()) if len(subset) else np.nan,
                "error_rate": float((~subset["correct"]).mean()) if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def coverage_accuracy_table(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("confidence", ascending=False).reset_index(drop=True)
    rows = []
    for coverage in np.linspace(0.05, 1.0, 20):
        n = max(1, int(round(len(ordered) * coverage)))
        subset = ordered.head(n)
        rows.append({"coverage": float(coverage), "n_rows": int(n), "winner_accuracy": float(subset["correct"].mean())})
    return pd.DataFrame(rows)


def performance_by_group(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [("season", "season"), ("week", "week"), ("home_pick", "pred_pick_home"), ("favorite_won", "favorite_won")]
    for group_name, col in specs:
        if col not in frame.columns:
            continue
        for value, subset in frame.groupby(col, dropna=False):
            if len(subset) == 0:
                continue
            rows.append({"group": group_name, "value": value, "n_rows": len(subset), "winner_accuracy": subset["correct"].mean(), "mean_confidence": subset["confidence"].mean()})
    return pd.DataFrame(rows)


def high_confidence_errors(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["season", "week", "home_team", "away_team", "keys_team", "keys_opponent", "y", "pred_margin", "pred_proba_home_win", "confidence"] if c in frame.columns]
    errors = frame.loc[~frame["correct"]].sort_values("confidence", ascending=False)
    return errors.loc[:, cols].head(100)


def brier_decomposition(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    y = frame["actual_home_win"].astype(float)
    p = frame["pred_proba_home_win"].astype(float)
    base_rate = y.mean()
    bucket = pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = frame.assign(bucket=bucket).groupby("bucket", observed=False)
    rows = []
    reliability = 0.0
    resolution = 0.0
    for value, part in grouped:
        if part.empty:
            continue
        weight = len(part) / len(frame)
        pred_mean = part["pred_proba_home_win"].mean()
        actual_mean = part["actual_home_win"].mean()
        reliability += weight * (pred_mean - actual_mean) ** 2
        resolution += weight * (actual_mean - base_rate) ** 2
        rows.append({"bucket": str(value), "n_rows": len(part), "mean_prediction": pred_mean, "actual_rate": actual_mean})
    uncertainty = base_rate * (1.0 - base_rate)
    rows.append({"bucket": "__decomposition__", "n_rows": len(frame), "mean_prediction": np.nan, "actual_rate": base_rate, "reliability": reliability, "resolution": resolution, "uncertainty": uncertainty, "brier_recomposed": reliability - resolution + uncertainty})
    return pd.DataFrame(rows)


def memorization_checks(frame: pd.DataFrame, row: dict) -> pd.DataFrame:
    checks = []
    for col in ["next_game_id", "keys_game_id", "game_id"]:
        if col in frame.columns:
            non_null = frame[col].dropna()
            checks.append({"check": f"duplicate_{col}", "rows": len(non_null), "unique_values": non_null.nunique(), "duplicate_rows": int(non_null.duplicated().sum())})
    source = Path(row["source_search_row"])
    if source.exists():
        try:
            source_row = pd.read_csv(source).iloc[0].to_dict()
            features = json.loads(source_row.get("selected_features_json", "[]"))
        except Exception:
            features = []
        leaky = [f for f in features if any(pattern in str(f).lower() for pattern in LEAKY_PATTERNS)]
        checks.append({"check": "selected_feature_count", "rows": len(features), "unique_values": len(set(features)), "duplicate_rows": len(features) - len(set(features))})
        checks.append({"check": "leaky_name_feature_count", "rows": len(features), "unique_values": len(leaky), "duplicate_rows": 0})
    return pd.DataFrame(checks)


def feature_tables(row: dict, tables: Path) -> pd.DataFrame:
    model_dir = Path(row["model_dir"])
    importance_path = model_dir / "artifacts" / "feature_importance" / "feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
    else:
        importance = pd.DataFrame()
    source = Path(row["source_search_row"])
    features = []
    if source.exists():
        try:
            source_row = pd.read_csv(source).iloc[0].to_dict()
            features = json.loads(source_row.get("selected_features_json", "[]"))
        except Exception:
            features = []
    if importance.empty and features:
        importance = pd.DataFrame({"feature": features, "importance": np.nan})
    if not importance.empty:
        importance["feature_family"] = importance["feature"].map(feature_family)
        importance["leakage_name_flag"] = importance["feature"].map(lambda f: any(pattern in str(f).lower() for pattern in LEAKY_PATTERNS))
        importance.to_csv(tables / "feature_importance.csv", index=False)
        (
            importance.groupby("feature_family", as_index=False)
            .agg(feature_count=("feature", "count"), importance=("importance", "sum"), leakage_name_flags=("leakage_name_flag", "sum"))
            .sort_values("importance", ascending=False)
            .to_csv(tables / "feature_family_importance.csv", index=False)
        )
    return importance


def fingerprint_diagnostics(row: dict, source_root: Path, tables: Path, plots: Path, sample_size: int) -> None:
    status = {}
    source = Path(row["source_search_row"])
    if not source.exists():
        return
    try:
        source_row = pd.read_csv(source).iloc[0].to_dict()
        frame = load_selected_frame(row=source_row, source_root=source_root)
    except Exception as exc:
        pd.DataFrame([{"status": "failed", "reason": str(exc)}]).to_csv(tables / "fingerprint_status.csv", index=False)
        return status
    feature_cols = [c for c in frame.columns if c not in {"keys_season", "keys_week", "keys_team", "keys_opponent", "keys_game_id"} and pd.api.types.is_numeric_dtype(frame[c])]
    if not feature_cols:
        return status
    data = frame.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if len(data) > sample_size:
        data = data.sample(sample_size, random_state=42)
    corr = data.corr().fillna(0.0)
    corr.to_csv(tables / "fingerprint_correlation.csv")
    plot_heatmap(corr.iloc[:60, :60], plots / "fingerprint_correlation_heatmap.png", "Fingerprint Correlation")
    status["25_fingerprint_correlation_heatmap"] = "figure"
    plot_feature_clustering(corr, plots / "fingerprint_feature_clustering.png")
    status["26_fingerprint_feature_clustering"] = "figure"

    pca = PCA(n_components=min(10, data.shape[1], data.shape[0]))
    coords = pca.fit_transform(data)
    pd.DataFrame({"component": range(1, len(pca.explained_variance_ratio_) + 1), "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(tables / "pca_explained_variance.csv", index=False)
    plot_pca_variance(pca.explained_variance_ratio_, plots / "pca_explained_variance.png")
    status["27_pca_explained_variance"] = "figure"
    if coords.shape[1] >= 2:
        pca_df = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1]})
        pca_df.to_csv(tables / "pca_projection.csv", index=False)
        plot_scatter(pca_df, plots / "pca_projection.png", "pc1", "pc2", "PCA Projection")
        status["28_pca_projection"] = "figure"
        status["29_umap_or_tsne_embedding"] = "not_available_pca_generated_instead"

    if len(data) > 10:
        nn = NearestNeighbors(n_neighbors=min(6, len(data))).fit(data)
        distances, _ = nn.kneighbors(data)
        pd.DataFrame({"neighbor_rank": range(1, distances.shape[1]), "mean_distance": distances[:, 1:].mean(axis=0)}).to_csv(tables / "nearest_neighbor_distances.csv", index=False)
        plot_neighbor_distances(distances, plots / "nearest_neighbor_outcome_consistency.png")
        status["31_nearest_neighbor_outcome_consistency"] = "distance_proxy_figure"
        status["30_fingerprint_distance_outcome_similarity"] = "distance_proxy_table"
    if {"keys_team", "keys_week"}.issubset(frame.columns):
        drift = fingerprint_drift(frame, feature_cols)
        if not drift.empty:
            drift.to_csv(tables / "fingerprint_drift_over_time.csv", index=False)
            plot_drift(drift, plots / "fingerprint_drift_over_time.png")
            status["33_fingerprint_drift_over_time"] = "figure"
    status["32_within_between_team_distance"] = "not_available_requires_pairwise_team_sampling"
    return status


def plot_confusion(frame, path):
    mat = confusion_matrix(frame["actual_home_win"], frame["pred_pick_home"], labels=[False, True])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="Blues")
    for (i, j), value in np.ndenumerate(mat):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set_xticks([0, 1], ["Pred Away", "Pred Home"])
    ax.set_yticks([0, 1], ["Actual Away", "Actual Home"])
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    savefig(fig, path)


def plot_roc(frame, path):
    y = frame["actual_home_win"].astype(int)
    if y.nunique() < 2:
        return
    fpr, tpr, _ = roc_curve(y, frame["pred_proba_home_win"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.set_title("ROC Curve")
    savefig(fig, path)


def plot_pr(frame, path):
    y = frame["actual_home_win"].astype(int)
    if y.nunique() < 2:
        return
    precision, recall, _ = precision_recall_curve(y, frame["pred_proba_home_win"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP {average_precision_score(y, frame['pred_proba_home_win']):.3f}")
    ax.axhline(y.mean(), linestyle="--", color="0.6", label="No skill")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.set_title("Precision-Recall Curve")
    savefig(fig, path)


def plot_calibration(frame, path):
    bins = pd.cut(frame["pred_proba_home_win"], np.linspace(0, 1, 11), include_lowest=True)
    cal = frame.groupby(bins, observed=False).agg(pred=("pred_proba_home_win", "mean"), actual=("actual_home_win", "mean"), n=("actual_home_win", "size")).dropna()
    cal.to_csv(path.with_suffix(".csv"), index=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6")
    ax.plot(cal["pred"], cal["actual"], marker="o")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Actual Home Win Rate")
    ax.set_title("Calibration Curve")
    savefig(fig, path)


def plot_confidence_hist(frame, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(frame["confidence"], bins=20, color="#4C78A8")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Games")
    ax.set_title("Confidence Histogram")
    savefig(fig, path)


def plot_accuracy_vs_confidence(frame, path):
    table = confidence_table(frame)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(table["confidence_threshold"], table["winner_accuracy"], marker="o")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy vs Confidence")
    savefig(fig, path)


def plot_coverage_accuracy(frame, path):
    table = coverage_accuracy_table(frame)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(table["coverage"], table["winner_accuracy"], marker="o")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Coverage vs Accuracy")
    savefig(fig, path)


def plot_probability_by_outcome(frame, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([frame.loc[~frame["actual_home_win"], "pred_proba_home_win"], frame.loc[frame["actual_home_win"], "pred_proba_home_win"]], labels=["Away won", "Home won"])
    ax.set_ylabel("Predicted Home Win Probability")
    ax.set_title("Probability by Actual Outcome")
    savefig(fig, path)


def plot_error_vs_confidence(frame, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(frame["confidence"], frame["abs_prob_error"], s=10, alpha=0.45)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Absolute Probability Error")
    ax.set_title("Error vs Confidence")
    savefig(fig, path)


def plot_error_vs_closeness(frame, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(frame["abs_margin"], (~frame["correct"]).astype(int), s=10, alpha=0.35)
    ax.set_xlabel("Actual Margin Absolute Value")
    ax.set_ylabel("Incorrect")
    ax.set_title("Error vs Game Closeness")
    savefig(fig, path)


def plot_cumulative_gains(frame, path):
    ordered = frame.sort_values("pred_proba_home_win", ascending=False).reset_index(drop=True)
    total_positive = max(float(ordered["actual_home_win"].sum()), 1.0)
    ordered["gain"] = ordered["actual_home_win"].cumsum() / total_positive
    ordered["coverage"] = (np.arange(len(ordered)) + 1) / len(ordered)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ordered["coverage"], ordered["gain"], label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", label="Baseline")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Cumulative Positive Capture")
    ax.set_title("Cumulative Gains")
    ax.legend()
    savefig(fig, path)


def plot_brier_decomposition(frame, path):
    decomp = brier_decomposition(frame)
    row = decomp.loc[decomp["bucket"].eq("__decomposition__")]
    if row.empty:
        return
    values = row.iloc[0][["reliability", "resolution", "uncertainty"]].astype(float)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(values.index, values.values)
    ax.set_title("Brier Decomposition")
    savefig(fig, path)


def plot_residual_distribution(frame, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(frame["prob_error"], bins=30, color="#F58518")
    ax.set_xlabel("Actual - Predicted Probability")
    ax.set_ylabel("Games")
    ax.set_title("Probability Error Distribution")
    savefig(fig, path)


def plot_subgroup_grid(frame, path):
    perf = performance_by_group(frame)
    perf = perf.loc[perf["group"].isin(["home_pick", "favorite_won"])].copy()
    if perf.empty:
        return
    perf["label"] = perf["group"].astype(str) + "=" + perf["value"].astype(str)
    fig, ax = plt.subplots(figsize=(7, max(4, len(perf) * 0.35)))
    ax.barh(perf["label"][::-1], perf["winner_accuracy"][::-1])
    ax.set_xlim(0, 1)
    ax.set_title("Performance by Subgroup")
    savefig(fig, path)


def plot_performance_bars(frame, path, group_col):
    if group_col not in frame.columns:
        return
    grouped = frame.groupby(group_col).agg(accuracy=("correct", "mean"), n=("correct", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grouped[group_col], grouped["accuracy"], marker="o")
    ax.set_ylim(0, 1)
    ax.set_xlabel(group_col)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Performance by {group_col}")
    savefig(fig, path)


def plot_feature_importance(importance, path):
    if importance is None or importance.empty or "importance" not in importance.columns:
        return
    d = importance.copy()
    d["importance"] = pd.to_numeric(d["importance"], errors="coerce").fillna(0.0)
    d = d.sort_values("importance", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(8, max(4, len(d) * 0.25)))
    ax.barh(d["feature"][::-1], d["importance"][::-1])
    ax.set_title("Feature Importance")
    savefig(fig, path)


def plot_feature_family_importance(importance, path):
    if importance is None or importance.empty or "importance" not in importance.columns:
        return
    d = importance.copy()
    d["importance"] = pd.to_numeric(d["importance"], errors="coerce").fillna(0.0)
    fam = d.groupby("feature_family", as_index=False)["importance"].sum().sort_values("importance", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(7, max(4, len(fam) * 0.3)))
    ax.barh(fam["feature_family"][::-1], fam["importance"][::-1])
    ax.set_title("Feature Family Importance")
    savefig(fig, path)


def plot_class_separation(row, source_root, importance, path):
    source = Path(row["source_search_row"])
    if not source.exists() or importance is None or importance.empty:
        return
    try:
        source_row = pd.read_csv(source).iloc[0].to_dict()
        frame = load_selected_frame(row=source_row, source_root=source_root)
    except Exception:
        return
    target = "y_next_margin" if "y_next_margin" in frame.columns else None
    if target is None:
        return
    top_features = [f for f in importance.sort_values("importance", ascending=False).get("feature", pd.Series(dtype=str)).head(6).tolist() if f in frame.columns]
    if not top_features:
        return
    n = len(top_features)
    fig, axes = plt.subplots(n, 1, figsize=(6, max(3, n * 1.5)))
    if n == 1:
        axes = [axes]
    y = pd.to_numeric(frame[target], errors="coerce") > 0
    for ax, feature in zip(axes, top_features):
        vals = pd.to_numeric(frame[feature], errors="coerce")
        ax.boxplot([vals.loc[~y].dropna(), vals.loc[y].dropna()], labels=["loss", "win"], vert=False)
        ax.set_title(feature[:80], fontsize=8)
    savefig(fig, path)


def plot_feature_clustering(corr, path):
    # Lightweight clustering proxy: sort features by average correlation.
    if corr.empty:
        return
    order = corr.abs().mean().sort_values(ascending=False).index[:60]
    plot_heatmap(corr.loc[order, order], path, "Fingerprint Feature Clustering Proxy")


def plot_neighbor_distances(distances, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for i in range(1, distances.shape[1]):
        ax.hist(distances[:, i], bins=25, alpha=0.35, label=f"NN {i}")
    ax.set_title("Nearest-Neighbor Distances")
    ax.legend(fontsize=7)
    savefig(fig, path)


def fingerprint_drift(frame, feature_cols):
    if not {"keys_team", "keys_season", "keys_week"}.issubset(frame.columns):
        return pd.DataFrame()
    rows = []
    work = frame[["keys_team", "keys_season", "keys_week", *feature_cols]].copy()
    for _, team_frame in work.sort_values(["keys_team", "keys_season", "keys_week"]).groupby("keys_team"):
        values = (
            team_frame[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        if len(values) < 2:
            continue
        dist = np.sqrt(((values[1:] - values[:-1]) ** 2).sum(axis=1))
        meta = team_frame.iloc[1:][["keys_team", "keys_season", "keys_week"]].copy()
        meta["drift_distance"] = dist
        rows.append(meta)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def plot_drift(drift, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    summary = drift.groupby("keys_week")["drift_distance"].median().reset_index()
    ax.plot(summary["keys_week"], summary["drift_distance"], marker="o")
    ax.set_xlabel("Week")
    ax.set_ylabel("Median Drift")
    ax.set_title("Fingerprint Drift Over Time")
    savefig(fig, path)


def plot_heatmap(frame, path, title):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(frame.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.75)
    savefig(fig, path)


def plot_pca_variance(values, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.arange(1, len(values) + 1), np.cumsum(values), marker="o")
    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_ylim(0, 1.05)
    ax.set_title("PCA Explained Variance")
    savefig(fig, path)


def plot_scatter(frame, path, x, y, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(frame[x], frame[y], s=10, alpha=0.45)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    savefig(fig, path)


def merge_outputs(output_root: Path) -> dict[str, pd.DataFrame]:
    summary = output_root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    tables = {
        "overall_metrics": read_many(output_root.glob("runs/*/*/*/tables/overall_metrics.csv")),
        "confidence_thresholds": read_many(output_root.glob("runs/*/*/*/tables/confidence_thresholds.csv")),
        "coverage_accuracy": read_many(output_root.glob("runs/*/*/*/tables/coverage_accuracy.csv")),
        "memorization_checks": read_many(output_root.glob("runs/*/*/*/tables/memorization_checks.csv")),
        "feature_family_importance": read_many(output_root.glob("runs/*/*/*/tables/feature_family_importance.csv")),
        "requested_output_status": read_many(output_root.glob("runs/*/*/*/tables/requested_output_status.csv")),
    }
    for name, frame in tables.items():
        frame.to_csv(summary / f"{name}.csv", index=False)
    return tables


def read_many(paths) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        parts = Path(path).parts
        try:
            idx = parts.index("runs")
            frame.insert(0, "objective", parts[idx + 1])
            frame.insert(1, "family", parts[idx + 2])
            frame.insert(2, "model", parts[idx + 3])
        except Exception:
            pass
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def market_pick_home(frame):
    if "market_win_probability" in frame.columns:
        prob = pd.to_numeric(frame["market_win_probability"], errors="coerce")
        if prob.notna().any():
            return prob > 0.5
    if "market_spread_close" in frame.columns:
        spread = pd.to_numeric(frame["market_spread_close"], errors="coerce")
        return spread < 0
    return pd.Series(np.nan, index=frame.index)


def favorite_won(frame):
    pick = market_pick_home(frame)
    if pick.notna().any():
        return pick == frame["actual_home_win"]
    return pd.Series(np.nan, index=frame.index)


def first_numeric(frame, names):
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def feature_family(feature: str) -> str:
    text = str(feature).lower()
    for name in ["elo", "graph", "offense", "defense", "statoff", "statdef", "statgen", "statspe", "coach", "roster", "return", "talent", "market", "travel", "target", "next"]:
        if name in text:
            return name
    if text.startswith("opp_adj"):
        return "opponent_adjusted"
    return "other"


def safe(value) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def resolve_job_index(job_index, sge_task_id) -> int:
    if job_index is not None:
        return int(job_index)
    if sge_task_id is None:
        raise ValueError("Provide --job-index or --sge-task-id")
    return int(sge_task_id) - 1


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_requested_status(tables: Path, status: dict[str, str]) -> None:
    pd.DataFrame(
        [{"output": name, "status": status.get(name, "not_started")} for name in REQUESTED_OUTPUTS]
    ).to_csv(tables / "requested_output_status.csv", index=False)


if __name__ == "__main__":
    main()
