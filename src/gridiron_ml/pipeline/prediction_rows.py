"""Future schedule-row abstractions for TDNet prediction workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gridiron_ml.pipeline.schemas import validate_prediction_rows


@dataclass(frozen=True)
class FuturePredictionRows:
    """Container for schedule-only rows that are not historical training labels."""

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        validate_prediction_rows(self.frame)

    def to_frame(self) -> pd.DataFrame:
        """Return a defensive copy of the validated prediction rows."""
        return self.frame.copy().reset_index(drop=True)


class PredictionRowsBuilder:
    """Build validated future prediction rows from standardized schedule tables."""

    def from_schedule(self, schedule_frame: pd.DataFrame) -> FuturePredictionRows:
        """Wrap a schedule frame as future prediction rows."""
        return FuturePredictionRows(schedule_frame.copy().reset_index(drop=True))
