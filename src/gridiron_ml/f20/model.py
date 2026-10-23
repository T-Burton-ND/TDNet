"""Fixed expert-consensus features for the F20 winner probability panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

EXPERTS = ("teamrankings", "massey", "espn_fpi", "fei", "sagarin_pred")
BASE_COLUMNS = ("market_margin", "market_missing")
EXPERT_COLUMNS = tuple(f"expert_{name}" for name in EXPERTS)
FEATURE_COLUMNS = (*BASE_COLUMNS, *EXPERT_COLUMNS,
                   *(f"missing_{name}" for name in EXPERTS),
                   "expert_count", "expert_median", "expert_std",
                   "market_minus_expert_median")
HISTORICAL_COLUMNS = {"market_margin": "line", "opening_line": "lineopen",
                      "teamrankings": "lineteamrank", "massey": "linemass",
                      "espn_fpi": "lineespn", "fei": "linefei",
                      "sagarin_pred": "linesagpred"}


def feature_row(values: dict[str, float | None]) -> dict[str, float | None]:
    """Use only a fixed source whitelist; never accept scores or summary models."""
    market = values.get("market_margin")
    experts = [values.get(name) for name in EXPERTS]
    good = np.array([float(x) for x in experts if x is not None and np.isfinite(x)], dtype=float)
    median = float(np.median(good)) if len(good) else None
    result: dict[str, float | None] = {
        "market_margin": float(market) if market is not None and np.isfinite(market) else None,
        "market_missing": float(market is None or not np.isfinite(market)),
        "expert_count": float(len(good)), "expert_median": median,
        "expert_std": float(np.std(good)) if len(good) else None,
        "market_minus_expert_median": float(market - median) if market is not None and median is not None else None,
    }
    for name, value in zip(EXPERTS, experts):
        available = value is not None and np.isfinite(value)
        result[f"expert_{name}"] = float(value) if available else None
        result[f"missing_{name}"] = float(not available)
    return result


def historical_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        values = {}
        for name, column in HISTORICAL_COLUMNS.items():
            raw = row.get(column)
            values[name] = float(raw) if raw is not None and pd.notna(raw) else None
        rows.append(feature_row(values))
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def captured_features(observation: dict) -> pd.DataFrame:
    values = {"market_margin": observation["values"]["current_line"]["home_margin"]}
    values.update({name: observation["values"][name]["home_margin"] for name in EXPERTS})
    return pd.DataFrame([feature_row(values)], columns=FEATURE_COLUMNS)
