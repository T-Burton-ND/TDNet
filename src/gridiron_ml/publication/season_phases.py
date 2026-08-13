"""Owner-approved, immutable 2026 information-maturity phase contract."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml


REQUIRED_PREDICTION_METADATA = {
    "season", "official_week", "season_phase", "game_date", "kickoff_time_utc",
    "home_team_id", "away_team_id", "model_family_id", "fingerprint_id",
    "frozen_model_role_id", "predicted_margin", "predicted_win_probability",
    "pregame_input_snapshot_id", "pregame_input_snapshot_sha256", "frozen_bundle_id",
    "frozen_bundle_sha256", "execution_code_commit", "prediction_timestamp_utc", "output_sha256",
}


def load_phase_map(path: str | Path) -> dict[int, str]:
    """Load and validate the one canonical official-week-to-phase mapping."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    mapping = {int(week): str(phase) for week, phase in payload["week_to_phase"].items()}
    if any(week < 0 for week in mapping) or any(phase not in {"PHASE_1", "PHASE_2", "PHASE_3", "PHASE_4"} for phase in mapping.values()):
        raise ValueError("invalid season-phase map")
    expected = {week: "PHASE_1" if week <= 3 else "PHASE_2" if week <= 6 else "PHASE_3" if week <= 9 else "PHASE_4" for week in mapping}
    if mapping != expected:
        raise ValueError("season-phase boundaries differ from the owner-approved mapping")
    return mapping


def validate_prediction_phase_metadata(frame: pd.DataFrame, phase_map: Mapping[int, str]) -> None:
    """Fail closed when a prospective prediction lacks an unambiguous phase."""
    missing = REQUIRED_PREDICTION_METADATA - set(frame.columns)
    if missing:
        raise ValueError(f"prediction metadata missing required fields: {sorted(missing)}")
    data = frame.copy()
    week = pd.to_numeric(data["official_week"], errors="coerce")
    if week.isna().any() or not week.astype(int).isin(phase_map).all():
        raise ValueError("prediction has missing or unmapped official week")
    expected = week.astype(int).map(phase_map)
    if data["season_phase"].astype(str).ne(expected).any():
        raise ValueError("prediction season_phase does not match the frozen week mapping")
    if pd.to_numeric(data["season"], errors="coerce").ne(2026).any():
        raise ValueError("season-phase validator is only for prospective 2026 predictions")
