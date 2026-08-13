"""Canonical prefix-driven feature categorization for TDNet tables.

Usage:
    Import `map_from_dataframe` when splitting team-game tables or fingerprints
    into feature, metadata, target, and market groups.

Logic flow:
    1. Inspect each column name and map known prefixes to canonical categories.
    2. Preserve unknown columns only when the caller explicitly allows them.
    3. Return a FeatureMap that downstream splitters can query consistently.

Expected canonical prefixes:
- keys_        (ID)
- game_        (meta)
- target_
- market_
- offense_
- defense_
- statOff_
- statDef_
- statGen_
- statSpe_
- coach_
- roster_
- travel_

Core categories:
- meta
- target
- market
- offense
- defense
- statOff
- statDef
- statGen
- statSpe
- coach
- roster
- travel

Design goals:
- No exhaustive per-column lists
- Strict: unknown columns should fail fast
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from gridiron_ml.pipeline.contracts.features import (
    COACH_PREFIX,
    GAME_PREFIX,
    KEY_PREFIX,
    LABEL_PREFIX,
    MARKET_PREFIX,
)


# Canonical ID/meta prefixes (excluded from X)
ID_PREFIXES = (KEY_PREFIX,)

# Canonical category prefixes
PREFIX_TO_CATEGORY: Tuple[Tuple[str, str], ...] = (
    (GAME_PREFIX, "meta"),
    ("games", "meta"),
    ("target_", "target"),
    (MARKET_PREFIX, "market"),
    ("offense_", "offense"),
    ("defense_", "defense"),
    ("statOff_", "statOff"),
    ("statDef_", "statDef"),
    ("statGen_", "statGen"),
    ("statSpe_", "statSpe"),
    (COACH_PREFIX, "coach"),
    ("roster_", "roster"),
    ("travel_", "travel"),
    (LABEL_PREFIX, "labels")
)

ALL_CATEGORIES: Tuple[str, ...] = (
    "id",
    "meta",
    "target",
    "market",
    "offense",
    "defense",
    "statOff",
    "statDef",
    "statGen",
    "statSpe",
    "coach",
    "roster",
    "travel",
    "labels",
)


@dataclass(frozen=True)
class FeatureMap:
    """Represent the FeatureMap component and its local behavior."""
    by_category: Dict[str, List[str]] = field(default_factory=dict)
    id_cols: List[str] = field(default_factory=list)
    unknown_cols: List[str] = field(default_factory=list)

    def feature_columns(
        self,
        *,
        include_market: bool = False,
        include_travel: bool = True,
        include_meta: bool = True,
        include_stat_off: bool = True,
        include_stat_def: bool = True,
        include_stat_gen: bool = True,
        include_stat_spe: bool = True,
        include_labels: bool = True,
    ):
        """Return columns eligible for model X before dtype/leakage filters.

        Market and label columns are optional here because splitters and
        validators enforce the training-safe defaults. That keeps this map a
        transparent prefix categorizer rather than a leakage policy engine.
        """
        cols: List[str] = []

        # Core football channels
        for cat in ("offense", "defense", "coach", "roster"):
            cols.extend(self.by_category.get(cat, []))

        # Boxscore stat channels (ablatable)
        if include_stat_off:
            cols.extend(self.by_category.get("statOff", []))
        if include_stat_def:
            cols.extend(self.by_category.get("statDef", []))
        if include_stat_gen:
            cols.extend(self.by_category.get("statGen", []))
        if include_stat_spe:
            cols.extend(self.by_category.get("statSpe", []))
            
        # Optional channels
        if include_market:
            cols.extend(self.by_category.get("market", []))
        if include_travel:
            cols.extend(self.by_category.get("travel", []))
        if include_meta:
            cols.extend(self.by_category.get("meta", []))
        if include_labels:
            cols.extend(self.by_category.get("labels", []))

        return cols

    def report(self, max_per_cat: int = 50):
        """Quick debug report: category counts + example columns."""
        print("Category counts:")
        for cat in ALL_CATEGORIES:
            n = len(self.by_category.get(cat, []))
            print(f"  {cat:>7}: {n}")

        print(f"\nID/meta cols: {len(self.id_cols)}")
        print(f"Unknown cols: {len(self.unknown_cols)}")

        if self.unknown_cols:
            print("\nUncategorized:")
            for c in self.unknown_cols:
                print(" -", c)

        for cat in ALL_CATEGORIES:
            cols = self.by_category.get(cat, [])
            if not cols:
                continue
            shown = cols[:max_per_cat]
            print(f"\n[{cat}] {len(cols)} columns (showing {len(shown)}):")
            for c in shown:
                print(" -", c)



def build_map_from_columns(
    columns: Sequence[str],
    *,
    keep_unknown: bool = True,
) -> FeatureMap:
    """
    Categorize columns using canonical prefix rules.

    Priority:
    1) ID/meta prefixes (keys_, game_)
    2) Category prefixes (target_, market_, ...)
    3) Unknown
    """
    by_cat: Dict[str, List[str]] = {k: [] for k in ALL_CATEGORIES}
    ids: List[str] = []
    unknown: List[str] = []

    def assign(cat: str, c: str) -> None:
        """Run the assign step and return its normalized result."""
        by_cat[cat].append(c)

    for c in columns:
        # Track keys_ columns as IDs (for convenience)
        if c.startswith(ID_PREFIXES):
            ids.append(c)
            assign("id", c)   # keep them visible in report()
            continue

        matched = False
        for prefix, cat in PREFIX_TO_CATEGORY:
            if c.startswith(prefix):
                assign(cat, c)
                matched = True
                break
        if matched:
            continue

        if keep_unknown:
            unknown.append(c)


    return FeatureMap(by_category=by_cat, id_cols=ids, unknown_cols=unknown)


def map_from_dataframe(df, **kwargs) -> FeatureMap:
    """Convenience wrapper for pandas DataFrames."""
    return build_map_from_columns(list(df.columns), **kwargs)
