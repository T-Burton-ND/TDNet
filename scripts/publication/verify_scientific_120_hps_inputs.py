#!/usr/bin/env python3
"""Verify all 20 materialized historical matrices against the common cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1/data")
OUT = Path("/groups/bsavoie2/tburton2/TDNet/scientific_120_hps_v1/input_preflight.json")
MARKET = {"F7", "F8", "F17", "F19"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(data: Path = DATA) -> dict:
    cohort_path = data / "cohort.parquet"
    common_receipt = json.loads((data / "cohort_receipt.json").read_text())
    if sha256(cohort_path) != common_receipt["cohort_sha256"]:
        raise ValueError("Shared cohort hash differs from receipt")
    cohort = pd.read_parquet(cohort_path)
    if len(cohort) != 7358 or not cohort.target_game_id.is_unique:
        raise ValueError("Shared cohort identity changed")
    source_columns = {"target_game_id", "season", "week", "target_start_utc",
                      "home_team", "away_team", "actual_margin"}
    results = []
    for tier in (f"F{i}" for i in range(20)):
        folder = data / tier
        receipt = json.loads((folder / "receipt.json").read_text())
        path = folder / "features.parquet"
        if (receipt["tier"] != tier or receipt["status"] != "materialized" or
                receipt["cohort_sha256"] != common_receipt["cohort_sha256"] or
                sha256(path) != receipt["matrix_sha256"]):
            raise ValueError(f"{tier} source binding changed")
        frame = pd.read_parquet(path)
        if (len(frame) != len(cohort) or not frame.target_game_id.equals(cohort.target_game_id) or
                not frame.season.equals(cohort.season) or
                not np.allclose(frame.actual_margin, cohort.actual_margin) or
                frame.season.max() > 2025 or
                set(frame.columns) != source_columns | set(receipt["feature_names"])):
            raise ValueError(f"{tier} is not aligned to the common target ledger")
        if receipt["feature_count"] != len(receipt["feature_names"]):
            raise ValueError(f"{tier} feature count differs from manifest")
        evidence = receipt["evidence"]
        if tier in MARKET and evidence["pregame_market_quote_timing"] != "unverified":
            raise ValueError(f"{tier} historical market timing label changed")
        if tier not in MARKET and evidence["pregame_market_quote_timing"] != "not_applicable":
            raise ValueError(f"{tier} unexpectedly uses market data")
        if tier in {f"F{i}" for i in range(9)}:
            if (not evidence["source_date_before_target_date"] or
                    not evidence["source_game_not_target_game"] or
                    evidence["prior_target_disagreement_count"] != 10):
                raise ValueError(f"{tier} old-source cutoff or label discrepancy changed")
        results.append({"tier": tier, "games": len(frame),
                        "feature_count": receipt["feature_count"],
                        "matrix_sha256": receipt["matrix_sha256"],
                        "market_interpretation": ("retrospective_untimed_quotes" if tier in MARKET
                                                  else "market_free")})
    return {"status": "ready_for_cpu_smoke", "cells": 120, "tiers": results,
            "cohort_games": len(cohort), "cohort_sha256": common_receipt["cohort_sha256"],
            "earliest_season": int(cohort.season.min()), "latest_season": int(cohort.season.max()),
            "historical_market_quote_timing": "unverified_research_only",
            "retrospective_target_corrections": 10,
            "no_2026_training_labels": True, "no_checkpoints": True}


def main() -> None:
    result = verify()
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tiers"}, indent=2))


if __name__ == "__main__":
    main()
