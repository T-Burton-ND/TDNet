"""src.gridiron_ml.fingerprints.builders.common.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Build, load, and split time-dependent team fingerprints.
"""

from datetime import datetime

import pandas as pd


def sort_team_week_frame(frame):
    """Run the sort_team_week_frame step and return its normalized result."""
    sort_cols = [c for c in ["keys_season", "keys_team", "keys_week"] if c in frame.columns]
    if sort_cols:
        return frame.sort_values(sort_cols).reset_index(drop=True)
    return frame.reset_index(drop=True)


def normalize_parquet_dtypes(frame):
    """Run the normalize_parquet_dtypes step and return its normalized result."""
    out = frame.copy()
    for col in out.columns:
        if col.endswith("_date") or col == "keys_game_date":
            out[col] = normalize_datetime_column(out[col])
            continue
        if out[col].dtype != "object":
            continue
        non_null = out[col].dropna()
        if non_null.empty:
            continue
        if non_null.map(lambda value: isinstance(value, (pd.Timestamp, datetime))).all():
            out[col] = normalize_datetime_column(out[col])
    return out


def normalize_datetime_column(values):
    """Return timezone-stable datetime64 values for parquet output."""
    try:
        parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    except TypeError:
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def coerce_numeric_columns(frame, columns):
    """Run the coerce_numeric_columns step and return its normalized result."""
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def regular_season_only(frame):
    """Run the regular_season_only step and return its normalized result."""
    if "keys_season_type" not in frame.columns:
        return frame
    return frame.loc[frame["keys_season_type"].astype(str).str.lower().eq("regular")].copy()
