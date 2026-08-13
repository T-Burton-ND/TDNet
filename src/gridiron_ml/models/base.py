"""Unified runtime contract shared by all TDNet model families.

The protocol documents the common surface without forcing legacy checkpoint
classes into a new inheritance tree. Existing TDLinear, TDStat, and TDTree
objects remain pickle-compatible while new families implement the same methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TDModel(Protocol):
    """Structural interface used by training, experiments, and publication."""

    model_family: str
    model_name: str

    def fit(self, X_train, y_train, sample_weight=None, X_val=None, y_val=None): ...

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs): ...

    def predict(self, X, meta_df=None, market_df=None): ...

    def predict_margin(self, X) -> np.ndarray: ...

    def predict_proba(self, X) -> np.ndarray: ...

    def save(self, path: str | Path) -> Path: ...

    @classmethod
    def load(cls, path: str | Path, config=None): ...

    def get_metadata(self) -> dict: ...


REQUIRED_MODEL_METHODS = (
    "fit",
    "train",
    "predict",
    "predict_margin",
    "predict_proba",
    "save",
    "load",
    "get_metadata",
)


def validate_model_contract(model) -> None:
    """Raise a readable error when a registry class violates the contract."""
    missing = [name for name in REQUIRED_MODEL_METHODS if not hasattr(model, name)]
    if missing:
        raise TypeError(
            f"{type(model).__name__} is missing TDNet model methods: {', '.join(missing)}"
        )

