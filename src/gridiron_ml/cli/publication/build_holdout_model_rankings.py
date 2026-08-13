#!/usr/bin/env python3
"""Rank a holdout roster using only model-selection evidence available pre-2025."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def build_rankings(
    inventory_path: Path, output_path: Path, selection_path: Path | None = None,
    ranking_protocol: str = "pre-2025 rolling/nested CV selection metric; 2025 excluded from ranking",
) -> pd.DataFrame:
    inventory = pd.read_csv(inventory_path).copy()
    required = {"model_id", "objective", "selection_metric"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Inventory is missing {sorted(missing)}.")

    if selection_path is not None:
        selection = pd.read_csv(selection_path).copy()
        selection["model_id"] = selection["model_id"].astype(str)
        value_columns = [column for column in ("brier_score", "mae") if column in selection]
        inventory = inventory.merge(
            selection[["model_id", *value_columns]].drop_duplicates("model_id"),
            on="model_id",
            how="left",
            validate="one_to_one",
        )

    ranked_parts = []
    for objective, group in inventory.groupby("objective", sort=True):
        group = group.copy()
        metric_column = "brier_score" if str(objective) == "winner" else "mae"
        if metric_column in group:
            group["selection_metric"] = pd.to_numeric(group[metric_column], errors="coerce")
        else:
            group["selection_metric"] = pd.to_numeric(group["selection_metric"], errors="coerce")
        group["ranking_metric"] = metric_column
        group["ranking_eligible"] = group["selection_metric"].notna()
        group["ranking_exclusion_reason"] = np.where(
            group["ranking_eligible"], "", "non_tuned_or_no_pre_2025_cv_score"
        )
        eligible = group.loc[group["ranking_eligible"]].sort_values(
            ["selection_metric", "model_id"], ascending=[True, True], kind="mergesort"
        )
        eligible["preseason_performance_rank"] = np.arange(1, len(eligible) + 1)
        group = group.merge(
            eligible[["model_id", "preseason_performance_rank"]],
            on="model_id",
            how="left",
            validate="one_to_one",
        )
        group["preseason_performance_rank"] = group["preseason_performance_rank"].fillna(
            len(eligible) + 1
        ).astype(int)
        ranked_parts.append(group)

    ranked = pd.concat(ranked_parts, ignore_index=True)
    ranked.insert(0, "selection_stage", "preseason_holdout_frozen")
    ranked.insert(1, "ranking_source", str(inventory_path.resolve()))
    ranked.insert(
        2,
        "ranking_protocol",
        ranking_protocol,
    )
    ranked = ranked.sort_values(["objective", "preseason_performance_rank", "model_id"], kind="mergesort")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)
    return ranked.reset_index(drop=True)


def main() -> None:
    root = project_root()
    parser = ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ranking-protocol", default="pre-2025 rolling/nested CV selection metric; 2025 excluded from ranking")
    args = parser.parse_args()
    output = args.output or args.inventory.with_name("preseason_model_rankings.csv")
    frame = build_rankings(
        args.inventory.resolve(),
        output.resolve(),
        args.selection.resolve() if args.selection else None,
        args.ranking_protocol,
    )
    print(f"wrote {len(frame)} ranking rows: {output.resolve()}")
    print(
        frame[
            [
                "preseason_performance_rank",
                "model_id",
                "objective",
                "selection_metric",
                "ranking_eligible",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
