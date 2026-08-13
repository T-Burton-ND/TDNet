"""Machine-checkable scientific protocol and fingerprint-ladder contracts.

F0--F6 are the nested market-free ladder, F7 is market-only, and F8 is the
explicit F6-plus-market incremental-value representation.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd
import yaml

from gridiron_ml.experiments.publication import expand_feature_registry


CANONICAL_TIERS = tuple(f"F{i}" for i in range(9))
MARKET_FREE_TIERS = frozenset(f"F{i}" for i in range(7))
MARKET_ONLY_TIER = "F7"
MARKET_AWARE_TIER = "F8"
MARKET_TIERS = frozenset({MARKET_ONLY_TIER, MARKET_AWARE_TIER})
LEGACY_TIERS = frozenset()
MARKET_FAMILY = "market"
MARKET_TOKENS = re.compile(
    r"(?:market|vegas|spread|moneyline|over_under|implied|closing_line|bookmaker)",
    flags=re.IGNORECASE,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return value


def _resolve_tier(name: str, tiers: dict[str, Any], visiting: set[str]) -> set[str]:
    if name in visiting:
        raise ValueError(f"Cyclic fingerprint ladder at {name}.")
    if name not in tiers:
        raise ValueError(f"Unknown fingerprint tier {name}.")
    visiting.add(name)
    spec = dict(tiers[name] or {})
    families = {str(x) for x in spec.get("include_families", [])}
    for child in spec.get("union", []) or []:
        families.update(_resolve_tier(str(child), tiers, visiting))
    visiting.remove(name)
    return families


def resolve_tier_families(tier: str, ladder: dict[str, Any]) -> set[str]:
    """Resolve a canonical tier or ``F1+F2`` expression to families."""
    tiers = dict(ladder.get("tiers", {}))
    result: set[str] = set()
    for part in (piece.strip() for piece in str(tier).split("+") if piece.strip()):
        result.update(_resolve_tier(part, tiers, set()))
    return result


def validate_ladder_config(ladder: dict[str, Any]) -> dict[str, Any]:
    """Validate nesting, market boundary, and legacy alias policy."""
    order = tuple(map(str, ladder.get("canonical_order", [])))
    if order != CANONICAL_TIERS:
        raise ValueError(f"canonical_order must be {list(CANONICAL_TIERS)}, got {list(order)}")
    if len(set(order)) != len(order):
        raise ValueError("canonical_order contains duplicate tiers.")
    tiers = dict(ladder.get("tiers", {}))
    resolved = {tier: resolve_tier_families(tier, ladder) for tier in CANONICAL_TIERS}
    nested = tuple(f"F{i}" for i in range(7))
    for previous, current in zip(nested, nested[1:]):
        if not resolved[previous].issubset(resolved[current]):
            raise ValueError(f"Fingerprint ladder is not nested: {previous} is not a subset of {current}.")
    for tier in MARKET_FREE_TIERS:
        if MARKET_FAMILY in resolved[tier] or bool(tiers[tier].get("market", False)):
            raise ValueError(f"Market information crosses the F6 boundary at {tier}.")
    if resolved[MARKET_ONLY_TIER] != {MARKET_FAMILY} or not tiers[MARKET_ONLY_TIER].get("market", False):
        raise ValueError("F7 must be the market-only benchmark.")
    if MARKET_FAMILY not in resolved[MARKET_AWARE_TIER] or not tiers[MARKET_AWARE_TIER].get("market", False):
        raise ValueError("F8 must combine F6 with the market family.")
    if not resolved["F6"].issubset(resolved[MARKET_AWARE_TIER]):
        raise ValueError("F8 must contain the complete F6 representation.")
    return {
        "version": ladder.get("version"),
        "canonical_order": list(CANONICAL_TIERS),
        "resolved_families": {tier: sorted(resolved[tier]) for tier in CANONICAL_TIERS},
        "market_free_tiers": sorted(MARKET_FREE_TIERS),
        "market_only_tier": MARKET_ONLY_TIER,
        "market_aware_tier": MARKET_AWARE_TIER,
        "legacy_tiers": sorted(LEGACY_TIERS.intersection(tiers)),
    }


def _canonical_alias_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _check_duplicate_aliases(columns: Iterable[str]) -> None:
    aliases: dict[str, list[str]] = {}
    for column in map(str, columns):
        aliases.setdefault(_canonical_alias_key(column), []).append(column)
    duplicates = {key: names for key, names in aliases.items() if len(names) > 1}
    if duplicates:
        raise ValueError(f"Duplicate feature aliases detected: {duplicates}")


def _check_market_boundary(columns: Iterable[str], *, tier: str, definitions: dict[str, Any]) -> None:
    hidden = []
    for column in map(str, columns):
        definition = definitions.get(column)
        metadata = getattr(definition, "metadata", {})
        declared_market = bool(metadata.get("market_derived", False)) or getattr(definition, "family", None) == MARKET_FAMILY
        name_market = bool(MARKET_TOKENS.search(column))
        if tier in MARKET_FREE_TIERS and (declared_market or name_market):
            hidden.append(column)
        if tier in MARKET_TIERS and name_market and not declared_market:
            raise ValueError(f"{tier} market-like feature is not declared market-derived: {column}")
    if hidden:
        raise ValueError(f"Market or hidden market-proxy columns in {tier}: {hidden}")


def materialize_feature_manifest(
    columns: Iterable[str],
    *,
    tier: str,
    registry_path: str | Path,
    ladder_path: str | Path,
    strict_registry: bool = True,
    allow_context_sidecars: bool = False,
) -> dict[str, Any]:
    """Return exact ordered feature metadata and a reproducible schema hash."""
    tier = str(tier)
    if tier not in CANONICAL_TIERS:
        raise ValueError(f"Only canonical F0-F8 tiers are accepted; got {tier}.")
    ladder = load_yaml(ladder_path)
    validation = validate_ladder_config(ladder)
    tier_for_resolution = tier
    definitions = expand_feature_registry(
        list(map(str, columns)), registry_path=registry_path, strict=strict_registry
    )
    selected_families = resolve_tier_families(tier_for_resolution, ladder)
    selected = {
        name: definition
        for name, definition in definitions.items()
        if definition.family in selected_families
    }
    selected_names = sorted(selected)
    _check_duplicate_aliases(selected_names)
    # Direct manifest calls are fail-closed: a market column in a market-free
    # input is rejected. The frame validator may explicitly permit sidecars
    # because the production feature frame carries market context separately.
    boundary_columns = selected.keys() if allow_context_sidecars else definitions.keys()
    _check_market_boundary(boundary_columns, tier=tier_for_resolution, definitions=definitions)
    features = []
    for name in selected_names:
        definition = selected[name]
        metadata = dict(definition.metadata)
        features.append(
            {
                "name": name,
                "family": definition.family,
                "source": metadata.get("source"),
                "earliest_known_availability": metadata.get("availability_rule"),
                "temporal_cutoff_rule": metadata.get("temporal_lag"),
                "missingness_rule": metadata.get("missing_policy"),
                "transformation": metadata.get("transformation", "declared_in_builder"),
                "market_derived": bool(metadata.get("market_derived", False)),
                "opponent_adjusted": bool(metadata.get("opponent_adjusted", False)),
                "version": metadata.get("version"),
            }
        )
    schema_payload = {
        "tier": tier,
        "resolved_tier": tier_for_resolution,
        "features": features,
    }
    schema_hash = sha256(
        json.dumps(schema_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "tier": tier,
        "resolved_tier": tier_for_resolution,
        "label": ladder["tiers"][tier_for_resolution].get("label"),
        "feature_count": len(features),
        "feature_names": selected_names,
        "features": features,
        "feature_families": sorted(selected_families),
        "market_derived": tier_for_resolution in MARKET_TIERS,
        "schema_hash": schema_hash,
        "ladder_validation": validation,
    }


def validate_feature_frame(
    frame: pd.DataFrame,
    *,
    registry_path: str | Path,
    ladder_path: str | Path,
    tiers: Iterable[str] = CANONICAL_TIERS,
) -> dict[str, dict[str, Any]]:
    """Materialize every requested tier and fail on hidden market columns."""
    return {
        str(tier): materialize_feature_manifest(
            frame.columns,
            tier=str(tier),
            registry_path=registry_path,
            ladder_path=ladder_path,
            allow_context_sidecars=True,
        )
        for tier in tiers
    }


def protocol_summary(protocol_path: str | Path) -> dict[str, Any]:
    """Load and validate the top-level protocol without reading model data."""
    protocol = load_yaml(protocol_path)
    required = {"study", "fingerprint_ladder", "architecture_matrix", "poll", "calibration", "inference", "prospective"}
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"Protocol is missing required sections: {missing}")
    return protocol
