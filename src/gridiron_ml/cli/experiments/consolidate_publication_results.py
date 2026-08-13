#!/usr/bin/env python3
"""Build deterministic winner and margin tables from completed publication searches.

The source arrays deliberately retain one compact result fragment per training
task.  This script is the only supported reduction point: it checks each
manifest/result pairing, records completeness, and writes a stable, objective-
specific table suitable for selection and publication diagnostics.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

from gridiron_ml.experiments.publication import read_frame


SUITES = (
    "publication_feature_model_matrix_v1",
    "publication_hps_spline_v1",
    "publication_hps_hist_gradient_boosted_v1",
    "publication_hps_mlp_v1",
    "publication_hps_structured_mlp_v1",
    "publication_hps_kernel_v1",
    "publication_hps_temporal_v1",
)
STABLE_COLUMNS = [
    "source_suite", "objective", "model_family", "model_level",
    "feature_config", "model_config", "outer_fold", "parameter_index",
    "seed", "task_id",
]

# Trial fragments also contain serialized model metadata and, for some model
# families, large diagnostic payloads.  Those fields are useful at the trial
# directory but are not part of deterministic model selection.  Reading only
# this compact projection keeps consolidation bounded when an array contains
# tens of thousands of one-row parquet fragments.
RESULT_COLUMNS = [
    "task_id", "status", "total_loss", "margin_loss", "win_probability_loss",
    "favorite_correctness_loss", "calibration_loss", "mae", "rmse",
    "winner_accuracy", "brier_score", "market_rmse", "market_mae", "n_rows",
    "label", "selected_feature_count", "selected_features_json",
    "prediction_rows", "runtime_seconds", "completed_at_utc",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_suite(root: Path, suite: str) -> tuple[pd.DataFrame, dict]:
    suite_root = root / "experiments" / suite
    manifest = read_frame(suite_root / "job_manifest.parquet")
    fragments: list[dict] = []
    missing: list[int] = []
    malformed: list[int] = []
    for row in manifest.itertuples(index=False):
        path = Path(row.output_path) / "result.parquet"
        if not path.exists():
            missing.append(int(row.task_id))
            continue
        try:
            available = set(pq.ParquetFile(path).schema.names)
            columns = [column for column in RESULT_COLUMNS if column in available]
            frame = pq.read_table(path, columns=columns).to_pandas()
        except Exception:
            malformed.append(int(row.task_id))
            continue
        if len(frame) != 1 or "status" not in frame.columns:
            malformed.append(int(row.task_id))
            continue
        # The manifest is canonical for trial identity/configuration.  Result
        # rows are canonical for metrics/status.  Joining the two here also
        # makes the consolidated table self-contained without copying the
        # bulky model_metadata_json field.
        result = frame.iloc[0].to_dict()
        manifest_row = {column: getattr(row, column) for column in manifest.columns}
        result = {**manifest_row, **result, "source_suite": suite}
        fragments.append(result)
    results = pd.DataFrame(fragments)
    duplicate_task_ids = int(results.duplicated(["source_suite", "task_id"]).sum()) if not results.empty else 0
    report = {
        "source_suite": suite,
        "manifest_rows": int(len(manifest)),
        "result_rows": int(len(results)),
        "missing_rows": len(missing),
        "malformed_rows": len(malformed),
        "duplicate_task_ids": duplicate_task_ids,
        "missing_task_ids": missing[:25],
        "malformed_task_ids": malformed[:25],
    }
    if report["missing_rows"] or report["malformed_rows"] or duplicate_task_ids:
        raise RuntimeError(f"Incomplete or malformed suite: {report}")
    return results, report


def write_diagnostics(frame: pd.DataFrame, objective: str, figures: Path, tables: Path) -> None:
    metric = "brier_score" if objective == "winner" else "mae"
    success = frame.loc[frame["status"].eq("success")].copy()
    ranking = (
        success.groupby(["model_family", "model_level", "feature_config"], as_index=False)
        .agg(**{metric: (metric, "mean")}, folds=("outer_fold", "nunique"), trials=("task_id", "count"))
        .sort_values([metric, "model_family", "model_level", "feature_config"], kind="mergesort")
    )
    ranking.to_csv(tables / f"{objective}_candidate_ranking.csv", index=False)
    plot_data = success.dropna(subset=[metric]).copy()
    labels = (
        plot_data["model_level"].astype(str) + " · " + plot_data["feature_config"].astype(str)
    )
    plot_data = plot_data.assign(candidate=labels)
    ordered = (
        plot_data.groupby("candidate")[metric].mean().sort_values().index.tolist()
    )
    fig, axis = plt.subplots(figsize=(max(9, 0.55 * len(ordered)), 6))
    values = [plot_data.loc[plot_data["candidate"].eq(label), metric].to_numpy() for label in ordered]
    axis.boxplot(values, tick_labels=ordered, showfliers=False)
    axis.set_ylabel("Brier score (lower is better)" if objective == "winner" else "Margin MAE (lower is better)")
    axis.set_title(f"{objective.title()} candidate performance across rolling-origin folds")
    axis.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(figures / f"{objective}_candidate_cv_boxplots.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("TDNET_ARTIFACT_ROOT", "publication_artifacts")))
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments/publication_model_selection"))
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    output_root = args.output_root.resolve()
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    frames, reports = [], []
    for suite in SUITES:
        frame, report = load_suite(artifact_root, suite)
        frames.append(frame)
        reports.append(report)
    all_results = pd.concat(frames, ignore_index=True)
    counts = all_results["status"].value_counts(dropna=False).to_dict()
    # Failed trials remain in the canonical audit table, but cannot enter
    # model selection.  This lets a mostly-complete array produce useful
    # tables while making the recovery scope explicit in the report.
    all_results = all_results.sort_values(
        ["source_suite", "task_id"], kind="mergesort"
    ).reset_index(drop=True)
    all_results.to_parquet(tables / "all_trial_results.parquet", index=False)
    all_results.to_csv(tables / "all_trial_results.csv.gz", index=False, compression="gzip")
    all_results.loc[all_results["status"].ne("success")].to_csv(
        tables / "failed_trials.csv", index=False
    )
    outputs = []
    for objective in ("winner", "margin"):
        result = all_results.loc[
            all_results["objective"].eq(objective)
            & all_results["status"].eq("success")
        ].copy()
        result = result.sort_values(STABLE_COLUMNS, kind="mergesort").reset_index(drop=True)
        path = tables / f"{objective}_search_results.parquet"
        result.to_parquet(path, index=False)
        result.to_csv(path.with_suffix(".csv.gz"), index=False, compression="gzip")
        write_diagnostics(result, objective, figures, tables)
        outputs.append({"objective": objective, "rows": len(result), "sha256": sha256(path)})
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_suites": reports,
        "result_status_counts": counts,
        "failed_trial_rows": all_results.loc[all_results["status"].ne("success")].to_dict(orient="records"),
        "outputs": outputs,
        "selection_scope": "new publication arrays only; legacy model types remain pending leakage-safe CV",
    }
    (output_root / "consolidation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
