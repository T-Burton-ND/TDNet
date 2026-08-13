"""Preseason-frozen model rankings used for provenance and audit sidecars."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_preseason_performance_rankings(
    inventory_path: str | Path,
    leaderboard_path: str | Path,
    output_path: str | Path,
    *,
    exclude_baselines: bool = True,
) -> pd.DataFrame:
    """Rank frozen models by historical performance before the season.

    The comparison leaderboard is expected to have one aggregate row per
    source.  Market and deliberately diagnostic baselines are retained in the
    source comparison tables, but are not eligible to become TDNet's lead
    model unless explicitly requested.
    """
    inventory = pd.read_csv(inventory_path).copy()
    leaderboard = pd.read_csv(leaderboard_path).copy()
    required_inventory = {"model_id", "objective"}
    required_leaderboard = {"source", "mean_total_score", "mean_winner_accuracy", "mean_mae", "mean_rmse"}
    if not required_inventory.issubset(inventory):
        raise ValueError(f"Inventory is missing {sorted(required_inventory - set(inventory))}.")
    if not required_leaderboard.issubset(leaderboard):
        raise ValueError(f"Leaderboard is missing {sorted(required_leaderboard - set(leaderboard))}.")

    inventory["__source_key"] = inventory["model_id"].astype(str).str.strip().str.lower()
    leaderboard["__source_key"] = leaderboard["source"].astype(str).str.strip().str.lower()
    metrics = leaderboard[
        ["__source_key", "mean_total_score", "mean_winner_accuracy", "mean_mae", "mean_rmse"]
    ].drop_duplicates("__source_key", keep="first")
    ranked = inventory.merge(metrics, on="__source_key", how="left", validate="one_to_one")
    ranked["historical_performance_score"] = pd.to_numeric(ranked["mean_total_score"], errors="coerce")
    ranked["historical_winner_accuracy"] = pd.to_numeric(ranked["mean_winner_accuracy"], errors="coerce")
    ranked["historical_mae"] = pd.to_numeric(ranked["mean_mae"], errors="coerce")
    ranked["historical_rmse"] = pd.to_numeric(ranked["mean_rmse"], errors="coerce")
    ranked["ranking_eligible"] = ranked["historical_performance_score"].notna()
    if exclude_baselines:
        family = ranked.get("model_family", pd.Series("", index=ranked.index)).astype(str).str.lower()
        source = ranked["model_id"].astype(str).str.lower()
        baseline = family.isin({"knn", "naive"}) | source.str.contains("vegas|homefield", regex=True)
        ranked.loc[baseline, "ranking_eligible"] = False
        ranked["ranking_exclusion_reason"] = np.where(baseline, "baseline_excluded", "")
    else:
        ranked["ranking_exclusion_reason"] = ""

    eligible = ranked.loc[ranked["ranking_eligible"]].copy()
    if eligible.empty:
        raise ValueError("No roster model has a historical performance score.")
    eligible = eligible.sort_values(
        ["historical_performance_score", "historical_winner_accuracy", "historical_mae", "historical_rmse", "model_id"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    eligible["preseason_performance_rank"] = np.arange(1, len(eligible) + 1)
    ranked = ranked.merge(
        eligible[["model_id", "preseason_performance_rank"]], on="model_id", how="left", validate="one_to_one"
    )
    ranked["preseason_performance_rank"] = ranked["preseason_performance_rank"].fillna(len(eligible) + 1).astype(int)
    ranked = ranked.sort_values(["preseason_performance_rank", "model_id"], kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "selection_stage", "preseason_frozen")
    ranked.insert(1, "ranking_source", str(Path(leaderboard_path).resolve()))
    ranked.insert(2, "ranking_protocol", "historical frozen-roster leaderboard; baselines excluded")
    ranked = ranked.drop(columns=["__source_key"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output, index=False)
    return ranked


def load_preseason_performance_rankings(path: str | Path) -> pd.DataFrame:
    """Load and validate the frozen ranking sidecar."""
    frame = pd.read_csv(path)
    required = {"model_id", "preseason_performance_rank"}
    if not required.issubset(frame):
        raise ValueError(f"Preseason rankings are missing {sorted(required - set(frame))}.")
    frame["preseason_performance_rank"] = pd.to_numeric(
        frame["preseason_performance_rank"], errors="coerce"
    )
    if frame["preseason_performance_rank"].isna().any():
        raise ValueError("Preseason rankings contain non-numeric ranks.")
    if frame["model_id"].astype(str).duplicated().any():
        raise ValueError("Preseason rankings contain duplicate model IDs.")
    return frame
