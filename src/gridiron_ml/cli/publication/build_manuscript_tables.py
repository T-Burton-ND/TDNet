#!/usr/bin/env python3
"""Materialize publication tables and available figures from completed CV outputs.

This builder intentionally consumes only successful historical search results.
Analyses that require predictions, ablations, negative controls, or prospective
2026 outcomes remain explicit empty tables/figures until those artifacts exist.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from gridiron_ml.publication.figures import PublicationFigureBuilder
from gridiron_ml.publication.tables import build_publication_tables


def main() -> None:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--results", type=Path, default=root / "data/experiments/publication_model_selection/tables/all_trial_results.parquet")
    parser.add_argument("--table-root", type=Path, default=root / "docs/publication_2026/preseason/tables")
    parser.add_argument("--figure-root", type=Path, default=root / "docs/publication_2026/preseason/figures")
    args = parser.parse_args()

    results = pd.read_parquet(args.results)
    results = results.loc[results["status"].astype(str).eq("success")].copy()
    feature_tiers = pd.read_csv(args.table_root / "table_02_feature_tiers.csv")
    model_families = pd.read_csv(args.table_root / "table_03_model_families.csv")
    tables = build_publication_tables(
        output_root=args.table_root,
        experiment_results=results,
        feature_tiers=feature_tiers,
        model_families=model_families,
    )

    # Completed CV rows are sufficient for performance, market, historical,
    # distribution, and compute figures. Missing prediction-dependent analyses
    # are recorded as skipped by the non-strict figure builder.
    # Use the compact matrix table for plotting. Passing every trial row here
    # creates an unnecessarily large melted frame and can exhaust memory while
    # rendering the compute/distribution figures.
    figure_manifest = PublicationFigureBuilder(args.figure_root, strict=False).generate_all(
        matrix_summary=tables["table_05_feature_model_matrix"],
        historical_summary=tables["table_04_historical_performance"],
    )
    print(f"wrote {len(tables)} tables and {figure_manifest['generated_count']} figures")


if __name__ == "__main__":
    main()
