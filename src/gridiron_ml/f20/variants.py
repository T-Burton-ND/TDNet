"""Controlled F20 feature arms; B/C require audited F19 weekly matrices."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

from .model import BASE_COLUMNS, FEATURE_COLUMNS

EXPERT_ONLY = tuple(column for column in FEATURE_COLUMNS
                    if column not in (*BASE_COLUMNS, "market_minus_expert_median"))


def select_compact_tdnet(frame: pd.DataFrame, labels: pd.Series, seasons: pd.Series,
                         manifest: list[dict], *, max_features: int = 20) -> tuple[str, ...]:
    """Select only on 2017–2024 fit rows with a pregame source contract."""
    if len(frame) != len(labels) or len(frame) != len(seasons) or frame.empty:
        raise ValueError("Feature selection rows differ or are empty")
    if seasons.min() < 2017 or seasons.max() > 2024:
        raise ValueError("F20-B selection must exclude 2025/2026 and use 2017–2024")
    if max_features < 1 or max_features > 20:
        raise ValueError("Compact TDNet budget must be 1–20")
    candidates = []
    for item in manifest:
        name = item["name"]
        if (name in frame and not item.get("market_derived", False)
                and not item.get("target_derived", False)
                and not item.get("pregame_win_probability_derived", False)
                and item.get("temporal_cutoff") == "before_target_game"
                and item.get("prospective_allowed", False)):
            candidates.append(name)
    if len(candidates) < max_features or len(candidates) != len(set(candidates)):
        raise ValueError("Too few unique audited prospective TDNet candidates")
    X = frame.loc[:, candidates].apply(pd.to_numeric, errors="raise")
    finite = X.notna().sum().gt(0)
    X = X.loc[:, finite]
    if len(X.columns) < max_features:
        raise ValueError("Too few observed TDNet candidates")
    y = pd.to_numeric(labels, errors="raise")
    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("Feature selection requires binary training-only labels")
    values = SimpleImputer(strategy="median").fit_transform(X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.feature_selection")
        scores, _ = f_classif(values, y)
    if np.isnan(scores).all():
        raise ValueError("All TDNet candidate scores are undefined")
    ranking = sorted(((float(score) if not np.isnan(score) else -np.inf, name)
                      for score, name in zip(scores, X.columns)), key=lambda pair: (-pair[0], pair[1]))
    return tuple(name for _, name in ranking[:max_features])


def assemble_variant(variant: str, expert: pd.DataFrame, *,
                     f19: pd.DataFrame | None = None,
                     selected_tdnet: tuple[str, ...] = ()) -> pd.DataFrame:
    """Pair on game ID and refuse target/outcome columns in F19 input."""
    required = {"game_id", "cutoff_utc", "kickoff_utc", *FEATURE_COLUMNS}
    if required - set(expert) or expert.game_id.duplicated().any():
        raise ValueError("Expert block lacks unique games or required columns")
    cutoff = pd.to_datetime(expert.cutoff_utc, utc=True)
    kickoff = pd.to_datetime(expert.kickoff_utc, utc=True)
    if cutoff.isna().any() or kickoff.isna().any() or not (cutoff < kickoff).all():
        raise ValueError("Expert feature cutoff is not before kickoff")
    if variant == "F20-A":
        return expert[["game_id", *FEATURE_COLUMNS]].copy()
    if variant not in {"F20-B", "F20-C"} or f19 is None:
        raise ValueError("F20-B/C require an audited F19 game matrix")
    if f19.game_id.duplicated().any() or set(f19.game_id) != set(expert.game_id):
        raise ValueError("F19 and expert games are not identical")
    banned = {"actual_margin", "home_win", "home_points", "away_points", "next_game_margin"}
    if banned & set(f19):
        raise ValueError("F19 matrix contains outcome columns")
    if "feature_available_utc" not in f19:
        raise ValueError("F19 matrix lacks point-in-time availability")
    paired = expert[["game_id", "cutoff_utc"]].merge(f19, on="game_id", validate="one_to_one")
    if not (pd.to_datetime(paired.feature_available_utc, utc=True)
            < pd.to_datetime(paired.cutoff_utc, utc=True)).all():
        raise ValueError("F19 feature became available after forecast cutoff")
    if variant == "F20-B":
        if not 1 <= len(selected_tdnet) <= 20 or any(name not in f19 for name in selected_tdnet):
            raise ValueError("F20-B requires 1–20 selected TDNet columns")
        if len(selected_tdnet) != len(set(selected_tdnet)):
            raise ValueError("F20-B selected duplicate TDNet columns")
        columns = ["game_id", *FEATURE_COLUMNS, *selected_tdnet]
    else:
        f19_columns = [name for name in f19 if name not in {"game_id", "feature_available_utc"}]
        columns = ["game_id", *EXPERT_ONLY, *f19_columns]
    if len(columns) != len(set(columns)):
        raise ValueError("F20 arm has duplicate feature names")
    return expert.merge(f19, on="game_id", validate="one_to_one").loc[:, columns]
