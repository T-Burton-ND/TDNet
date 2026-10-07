#!/usr/bin/env python3
"""Checked historical union of the saved F12 A/B/C design families.

The extra B/C fields are development candidates only until an independently
checked 2026 construction exists. No 2026 data is loaded here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from constraint_free_search import DATA, digest, load_historical
from gridiron_ml.experiments.constraint_free import (
    FeatureRecord, assert_feature_contract, feature_family,
)
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import source_to_matchup


def load_broad_historical(tier: str):
    base, meta, metadata, evidence = load_historical(tier)
    archive = Path(json.loads((DATA / "prepare_receipt.json").read_text())["source_archive"])
    a_names = set(evidence["source_features"])
    old_names = [record.name for record in metadata]
    game_ids = pd.Index(meta.target_game_id.astype(int), name="target_game_id")
    extra_values: dict[str, np.ndarray] = {}
    extra_evidence: dict[str, dict] = {}

    for design in ("b", "c"):
        folder = archive / "fingerprints" / f"F12_F_{design}"
        provenance = json.loads((folder / "provenance.json").read_text())
        manifest_path, values_path = folder / "feature_manifest.json", folder / "values.parquet"
        if digest(manifest_path) != provenance["manifest_sha256"]:
            raise ValueError(f"F12 {design} manifest changed")
        if digest(values_path) != provenance["data_sha256"]:
            raise ValueError(f"F12 {design} values changed")
        manifest = json.loads(manifest_path.read_text())
        names = [record["name"] for record in manifest]
        if len(names) != len(set(names)):
            raise ValueError(f"F12 {design} duplicate feature name")
        for record in manifest:
            if any(record.get(flag) for flag in (
                "market_derived", "target_derived", "pregame_win_probability_derived"
            )):
                raise ValueError(f"F12 {design} inadmissible source: {record['name']}")
            if record.get("temporal_cutoff") != "before_target_game":
                raise ValueError(f"F12 {design} lacks target cutoff: {record['name']}")
        columns = [
            "target_game_id", "team", "season", "is_home", "target_start_utc",
            "feature_available_utc", "latest_source_game_id", "next_game_margin", *names,
        ]
        frame = pd.read_parquet(values_path, columns=columns)
        frame = frame.loc[frame.season.between(2013, 2025)]
        if frame.duplicated(["target_game_id", "team"]).any():
            raise ValueError(f"F12 {design} duplicate team target")
        if len(frame) != 2 * len(game_ids) or set(frame.target_game_id) != set(game_ids):
            raise ValueError(f"F12 {design} cohort differs from F16 A path")
        target = pd.to_datetime(frame.target_start_utc, utc=True)
        available = pd.to_datetime(frame.feature_available_utc, utc=True)
        if not (available < target).all():
            raise ValueError(f"F12 {design} feature after target kickoff")
        if frame.latest_source_game_id.eq(frame.target_game_id).any():
            raise ValueError(f"F12 {design} target game contributed to itself")
        home = frame.loc[frame.is_home].set_index("target_game_id").reindex(game_ids)
        away = frame.loc[~frame.is_home].set_index("target_game_id").reindex(game_ids)
        if home.team.isna().any() or away.team.isna().any():
            raise ValueError(f"F12 {design} unpaired game")
        if not np.allclose(home.next_game_margin, meta.next_game_margin):
            raise ValueError(f"F12 {design} outcome differs from common cohort")
        if not np.allclose(home.next_game_margin, -away.next_game_margin):
            raise ValueError(f"F12 {design} home/away outcomes differ")
        raw = np.column_stack((home[names].to_numpy(float), away[names].to_numpy(float)))
        matchup = source_to_matchup(raw, manifest)
        for column, record in enumerate(manifest):
            name = record["name"]
            if name in a_names:
                continue
            values = matchup[:, column]
            if name in extra_values:
                if not np.allclose(extra_values[name], values, equal_nan=True):
                    raise ValueError(f"F12 B/C shared source differs: {name}")
                continue
            extra_values[name] = values
            extra_evidence[name] = {
                "design": design, "generation": record["generation"],
                "manifest_sha256": provenance["manifest_sha256"],
                "values_sha256": provenance["data_sha256"],
                "availability_rule": record["availability_rule"],
                "code_path": record["code_path"],
            }

    if len(extra_values) != 98:
        raise ValueError(f"Expected 98 B/C-only columns, got {len(extra_values)}")
    additions = sorted(extra_values)
    insertion = len(old_names) if tier == "F18" else len(old_names) - 5
    columns = [*old_names[:insertion], *(f"matchup__{name}" for name in additions),
               *old_names[insertion:]]
    values = np.column_stack((base[:, :insertion],
                              np.column_stack([extra_values[name] for name in additions]),
                              base[:, insertion:]))
    added_metadata = [FeatureRecord(
        name=f"matchup__{name}", family=feature_family(name),
        market_derived=False, source=f"F12_F_{extra_evidence[name]['design']} checked archive",
        timestamp_semantics="team target state available before kickoff",
        prospective_allowed=False,
    ) for name in additions]
    records = [*metadata[:insertion], *added_metadata, *metadata[insertion:]]
    assert_feature_contract(columns, records, tier)
    if values.shape != (len(meta), len(columns)):
        raise ValueError("Broad historical matrix shape mismatch")
    return values, meta, records, {
        **evidence, "source_features": [*evidence["source_features"], *additions],
        "universe": "F16_A_plus_distinct_F12_BC",
        "extra_source_features": additions, "extra_feature_evidence": extra_evidence,
        "prospective_extra_features_approved": False,
    }
