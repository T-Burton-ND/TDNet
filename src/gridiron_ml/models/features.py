"""Shared feature handling for TDNet model wrappers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatch

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelFeatureAdapter:
    """Normalize model feature frames and selected feature names."""

    feature_prefixes: tuple[str, ...] = ()
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    require_selected: bool = False

    def coerce_frame(self, X, *, allow_none: bool = False) -> pd.DataFrame | None:
        """Coerce model features to a numeric pandas dataframe."""

        if X is None:
            if allow_none:
                return None
            raise ValueError("Feature dataframe is required.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Model methods require pandas DataFrame inputs.")

        out = X.copy().reset_index(drop=True)
        for col in out.columns:
            if pd.api.types.is_bool_dtype(out[col]):
                out[col] = out[col].astype(float)
            elif not pd.api.types.is_numeric_dtype(out[col]):
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.astype(float)

    def align_frame(self, X, feature_names: Iterable[object]) -> pd.DataFrame:
        """Coerce and align a feature frame to the trained feature order."""

        out = self.coerce_frame(X)
        features = [str(name) for name in feature_names]
        missing = [name for name in features if name not in out.columns]
        if missing:
            missing_df = pd.DataFrame(np.nan, index=out.index, columns=missing)
            out = pd.concat([out, missing_df], axis=1)
        return out.loc[:, features]

    def select_feature_names(self, names: Iterable[object]) -> list[str]:
        """Select model features from names using prefixes and glob patterns."""

        all_names = [str(name) for name in names]
        selected = [name for name in all_names if self.includes(name)]
        if selected:
            return selected
        if self.require_selected:
            raise ValueError("No configured model feature columns were found.")
        return all_names

    def includes(self, name: object) -> bool:
        """Return whether a feature name matches this adapter's include rules."""

        text = str(name)
        if self._matches_any(text, self.exclude_patterns):
            return False
        has_include_rules = bool(self.feature_prefixes or self.include_patterns)
        if not has_include_rules:
            return True
        return text.startswith(self.feature_prefixes) or self._matches_any(
            text, self.include_patterns
        )

    @staticmethod
    def _matches_any(name: str, patterns: Iterable[str]) -> bool:
        lowered = str(name).lower()
        for pattern in patterns:
            pattern_text = str(pattern).strip()
            if not pattern_text:
                continue
            pattern_lower = pattern_text.lower()
            if fnmatch(lowered, pattern_lower):
                return True
            if "*" not in pattern_lower and "?" not in pattern_lower:
                if lowered.startswith(pattern_lower) or pattern_lower in lowered:
                    return True
        return False
