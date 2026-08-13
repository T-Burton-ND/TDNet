#!/usr/bin/env python3
"""Merge corrected OOF heatmap evaluations and render the six-metric suite."""

from argparse import ArgumentParser
import json
from pathlib import Path

from gridiron_ml.experiments.publication import atomic_write_json, merge_experiment_chunks
from gridiron_ml.publication.scientific_metric_heatmaps import (
    aggregate_metric_grid,
    render_metric_heatmaps,
    selected_complete_trials,
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compression-results", type=Path)
    parser.add_argument("--require-compression", action="store_true")
    args = parser.parse_args()
    merged = merge_experiment_chunks(job_manifest=args.manifest, output_root=args.output_root)
    master = merged["master"]
    compression = None
    if args.compression_results:
        import pandas as pd
        compression = pd.read_parquet(args.compression_results)
    selected = selected_complete_trials(master, expected_folds=10)
    grid = aggregate_metric_grid(selected)
    if compression is not None:
        if not compression["status"].astype(str).eq("success").all():
            raise ValueError("Compression heatmap input contains unsuccessful rows.")
        grid = pd.concat([grid, aggregate_metric_grid(compression)], ignore_index=True)
        selected = pd.concat([selected, compression], ignore_index=True)
    tables = args.output_root / "summary" / "heatmaps" / "tables"
    figures = args.output_root / "summary" / "heatmaps" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(tables / "selected_complete_fold_results.parquet", index=False)
    grid.to_parquet(tables / "scientific_roster_metric_grid.parquet", index=False)
    grid.to_csv(tables / "scientific_roster_metric_grid.csv", index=False)
    required = [f"F{i}" for i in range(9)] + (["F6-C"] if args.require_compression else [])
    paths = render_metric_heatmaps(grid, figures, required_tiers=required)
    report = {
        "manifest_rows": int(len(master) + (len(compression) if compression is not None else 0)),
        "successful_rows": int(
            master["status"].astype(str).eq("success").sum()
            + (compression["status"].astype(str).eq("success").sum() if compression is not None else 0)
        ),
        "missing_rows": int(len(merged["missing_trials"])),
        "grid_cells": int(len(grid)),
        "figure_files": int(len(paths)),
        "required_tiers": required,
    }
    atomic_write_json(args.output_root / "summary" / "heatmaps" / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
