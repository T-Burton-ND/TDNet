"""Deterministic confirmatory-roster selection and Top-1/Top-3 ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def select_confirmatory_roster(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select one best checkpoint per concrete type, then rank by Brier score.

    Candidate rows may provide either ``selection_brier_score`` directly or
    season-specific ``brier_2024`` and ``brier_2025`` values.  The latter are
    averaged with equal season weight, matching the owner-approved lead rule.
    """
    frame = candidates.copy()
    family = next((c for c in ["model_family", "family"] if c in frame), None)
    model_type = next((c for c in ["model_type", "model", "variant"] if c in frame), None)
    if not family or not model_type:
        raise ValueError("Candidates require family/model_family and model_type/model columns.")
    if "selection_brier_score" not in frame:
        brier_columns = [c for c in ["brier_2024", "brier_2025"] if c in frame]
        if not brier_columns:
            raise ValueError("Candidates require selection_brier_score or brier_2024/brier_2025.")
        frame["selection_brier_score"] = frame[brier_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    frame["selection_brier_score"] = pd.to_numeric(frame["selection_brier_score"], errors="coerce")
    if frame["selection_brier_score"].isna().any():
        raise ValueError("Every candidate must have a finite Brier selection score.")
    frame["concrete_model_type"] = frame[family].astype(str) + "/" + frame[model_type].astype(str)
    objective = next((c for c in ["objective", "training_objective"] if c in frame), None)
    frame["registered_model_id"] = (
        frame[objective].astype(str) + "/" + frame["concrete_model_type"]
        if objective else frame["concrete_model_type"]
    )
    tie_columns = ["selection_brier_score"]
    ascending = [True]
    for column, direction in [("margin_mae", True), ("winner_accuracy", False), ("parameter_count", True)]:
        if column in frame:
            tie_columns.append(column)
            ascending.append(direction)
    selected = (
        frame.sort_values(tie_columns, ascending=ascending)
        .drop_duplicates("registered_model_id", keep="first")
        .sort_values(tie_columns, ascending=ascending)
        .reset_index(drop=True)
    )
    selected["roster_rank"] = np.arange(1, len(selected) + 1)
    selected["publication_role"] = np.select(
        [selected["roster_rank"].eq(1), selected["roster_rank"].le(3)],
        ["lead_top_1", "top_3"], default="registered_consensus_member",
    )
    selected["use_in_tdnet_poll"] = True
    selected["use_in_all_model_consensus"] = True
    selected["use_in_top_3_consensus"] = selected["roster_rank"].le(3)
    return selected


def write_confirmatory_roster(candidates_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    source = Path(candidates_path)
    candidates = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    roster = select_confirmatory_roster(candidates)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    roster.to_csv(output, index=False)
    return roster
