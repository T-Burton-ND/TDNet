"""Mandatory canonical write and model-input boundaries for F09–F12.

Builders subclass ``NextgenFeatureBuilder`` and cannot override its checked
materialization method. Later matchup, SHAP, and fit code enters through
``NextgenModelBoundary`` so persisted rows are checked again on load.
"""

from __future__ import annotations

import os
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import pandas as pd

from .nextgen_contract import assert_temporal_feature_rows
from gridiron_ml.td_run.matchups.builder import MatchupBuilder


SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def canonical_family_path(root: Path, generation: str, family: str) -> Path:
    if generation not in {"F09", "F10", "F11", "F12"} or not SAFE_NAME.fullmatch(family):
        raise ValueError("Invalid nextgen feature family identity")
    return Path(root) / "feature_families" / generation / family / "canonical.parquet"


def write_canonical_feature_family(root: Path, generation: str, family: str,
                                   frame: pd.DataFrame, feature_columns: Iterable[str]) -> Path:
    """The sole supported canonical family writer; validate before any file exists."""
    columns = tuple(feature_columns)
    assert_temporal_feature_rows(frame, columns)
    path = canonical_family_path(root, generation, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".canonical_", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temp_name, index=False)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return path


class NextgenFeatureBuilder(ABC):
    """Base for every F09–F12 builder; only ``build_frame`` is customizable."""

    generation: str
    family: str
    feature_columns: tuple[str, ...]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "materialize" in cls.__dict__:
            raise TypeError("Nextgen builders cannot bypass canonical temporal validation")

    @abstractmethod
    def build_frame(self) -> pd.DataFrame:
        """Return rows with required temporal provenance."""

    def materialize(self, root: Path) -> Path:
        return write_canonical_feature_family(
            root, self.generation, self.family, self.build_frame(), self.feature_columns)


class NextgenModelBoundary:
    """Validate canonical rows before matchup, SHAP, or fitting input is exposed."""

    def __init__(self, frame: pd.DataFrame, feature_columns: Iterable[str]):
        self.feature_columns = tuple(feature_columns)
        assert_temporal_feature_rows(frame, self.feature_columns)
        self._frame = frame.copy()

    @classmethod
    def from_canonical(cls, path: Path, feature_columns: Iterable[str]) -> "NextgenModelBoundary":
        return cls(pd.read_parquet(path), feature_columns)

    def matchup(self, home_rows: pd.DataFrame, away_rows: pd.DataFrame,
                *, representation: str = "diff") -> pd.DataFrame:
        """Use validated rows only; caller selects equal-length home/away slices."""
        home = self._frame.loc[home_rows.index, self.feature_columns]
        away = self._frame.loc[away_rows.index, self.feature_columns]
        return MatchupBuilder(representation=representation).build(home, away)

    def for_fit(self) -> pd.DataFrame:
        return self._frame.loc[:, self.feature_columns].copy()

    def for_shap(self) -> pd.DataFrame:
        return self._frame.loc[:, self.feature_columns].copy()
