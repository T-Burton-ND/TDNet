"""Equal-weight confirmatory consensus products.

Consensus is aggregation, not a learned model: weights are never fitted and a
failed model contributes no vote for that game. Compact selection is restricted
to development OOF rows supplied by the caller and rejects prospective or
consumed-holdout seasons.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def build_equal_weight_consensus(
    predictions: pd.DataFrame,
    *,
    model_column: str = "model_name",
    game_column: str = "game_id",
    margin_column: str = "pred_margin",
    probability_column: str = "pred_probability_home",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate available model votes and return rows plus membership audit."""
    required = {model_column, game_column, margin_column}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Consensus input missing columns: {missing}")
    frame = predictions.copy()
    frame[margin_column] = pd.to_numeric(frame[margin_column], errors="coerce")
    frame = frame.loc[frame[margin_column].notna()].copy()
    if frame.empty:
        raise ValueError("Consensus has no valid model margins.")
    grouped = frame.groupby(game_column, sort=True)
    consensus = grouped[margin_column].mean().rename("consensus_margin").to_frame()
    consensus["consensus_probability_home"] = 1.0 / (1.0 + np.exp(-np.clip(consensus["consensus_margin"] / 10.0, -35, 35)))
    consensus["effective_model_count"] = grouped[model_column].nunique()
    consensus["consensus_models"] = grouped[model_column].agg(lambda values: "|".join(sorted(set(map(str, values)))))
    if probability_column in frame:
        probabilities = pd.to_numeric(frame[probability_column], errors="coerce")
        consensus["consensus_calibrated_probability"] = probabilities.groupby(frame[game_column]).mean()
    membership = frame[[game_column, model_column]].drop_duplicates().sort_values([game_column, model_column])
    membership["included"] = True
    return consensus.reset_index(), membership.reset_index(drop=True)


def select_compact_components(
    development_oof: pd.DataFrame,
    *,
    candidate_models: Iterable[str] | None = None,
    model_column: str = "model_name",
    game_column: str = "game_id",
    season_column: str = "season",
    margin_column: str = "pred_margin",
    actual_margin_column: str = "actual_margin",
    max_size: int = 5,
    minimum_seasons: int = 2,
    minimum_improvement: float = 0.0,
) -> dict[str, object]:
    """Select a small equal-weight set from development OOF predictions only."""
    required = {model_column, game_column, season_column, margin_column, actual_margin_column}
    missing = sorted(required - set(development_oof.columns))
    if missing:
        raise ValueError(f"Compact consensus input missing columns: {missing}")
    frame = development_oof.copy()
    seasons = set(pd.to_numeric(frame[season_column], errors="coerce").dropna().astype(int))
    forbidden = sorted(seasons & {2025, 2026})
    if forbidden:
        raise ValueError(f"Compact selection cannot use consumed/prospective seasons: {forbidden}")
    frame[margin_column] = pd.to_numeric(frame[margin_column], errors="coerce")
    frame[actual_margin_column] = pd.to_numeric(frame[actual_margin_column], errors="coerce")
    frame = frame.dropna(subset=[margin_column, actual_margin_column])
    candidates = sorted(set(map(str, candidate_models)) if candidate_models is not None else set(frame[model_column].astype(str)))
    if len(candidates) < 2:
        raise ValueError("Compact selection requires at least two candidate models.")
    valid_candidates = []
    for model in candidates:
        rows = frame.loc[frame[model_column].astype(str).eq(model)]
        if rows[season_column].nunique() >= minimum_seasons:
            valid_candidates.append(model)
    if not valid_candidates:
        raise ValueError("No candidate model has the declared minimum season coverage.")

    def score(models: list[str]) -> tuple[float, int]:
        subset = frame.loc[frame[model_column].astype(str).isin(models)]
        pivot = subset.pivot_table(index=[game_column, season_column], columns=model_column, values=margin_column, aggfunc="mean")
        actual = subset.groupby([game_column, season_column])[actual_margin_column].first()
        common = pivot.dropna(subset=models).index.intersection(actual.index)
        if len(common) == 0:
            return float("inf"), 0
        pred = pivot.loc[common, models].mean(axis=1)
        return float(np.abs(pred.to_numpy() - actual.loc[common].to_numpy()).mean()), int(common.get_level_values(season_column).nunique())

    selected: list[str] = []
    current_score, current_seasons = score(valid_candidates[:1])
    history = []
    while len(selected) < max_size:
        options = []
        for candidate in valid_candidates:
            if candidate in selected:
                continue
            trial = selected + [candidate]
            trial_score, trial_seasons = score(trial)
            options.append((trial_score, candidate, trial_seasons))
        if not options:
            break
        trial_score, candidate, trial_seasons = min(options, key=lambda item: (item[0], item[1]))
        improvement = current_score - trial_score if np.isfinite(current_score) else float("inf")
        if selected and (improvement <= minimum_improvement or trial_seasons < minimum_seasons):
            break
        selected.append(candidate)
        current_score = trial_score
        current_seasons = trial_seasons
        history.append({"added": candidate, "mae": trial_score, "improvement": improvement, "seasons": trial_seasons})
    if not selected:
        selected = [min(valid_candidates, key=lambda candidate: score([candidate])[0])]
        current_score, current_seasons = score(selected)
    return {
        "selected_models": selected,
        "max_size": int(max_size),
        "minimum_seasons": int(minimum_seasons),
        "minimum_improvement": float(minimum_improvement),
        "development_seasons": sorted(seasons),
        "development_mae": current_score,
        "coverage_seasons": current_seasons,
        "selection_history": history,
        "equal_weights": True,
    }


def leave_one_out_audit(predictions: pd.DataFrame, selected_models: Iterable[str], **kwargs) -> pd.DataFrame:
    """Return compact consensus sensitivity when each component is removed."""
    selected = list(map(str, selected_models))
    rows = []
    for removed in [None, *selected]:
        keep = [model for model in selected if model != removed]
        if not keep:
            continue
        consensus, _ = build_equal_weight_consensus(
            predictions.loc[predictions[kwargs.get("model_column", "model_name")].astype(str).isin(keep)],
            **kwargs,
        )
        rows.append({"removed_model": removed or "none", "models": "|".join(keep), "games": len(consensus)})
    return pd.DataFrame(rows)
