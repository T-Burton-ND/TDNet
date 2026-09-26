"""Read-only validation and lineage planning for the exploratory fingerprint study.

This module never fetches data or submits training jobs. Later workers should call
``assert_design_years`` before fitting, ranking, or inspecting design evidence.
"""

from __future__ import annotations

import json
import argparse
import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


ID_PATTERN = re.compile(r"^F(06|09|10|11|12)_(F|R|LR|PR)_([abc])$")
FORBIDDEN_INPUT = re.compile(
    r"(?:market|vegas|betting|spread|moneyline|over_?under|pregame_?wp|"
    r"pregame.*prob|win_?prob(?:ability)?|implied_?prob|postgame|"
    r"pregame_elo|excitement_index|(?:^|_)ats(?:_|$))",
    re.IGNORECASE,
)
GENERATIONS = ("F06", "F09", "F10", "F11", "F12")
DESIGNS = ("a", "b", "c")


def load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def parse_fingerprint_id(value: str) -> tuple[str, str, str]:
    match = ID_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid next-generation fingerprint id: {value}")
    generation, variant, design = f"F{match[1]}", match[2], match[3]
    allowed = {"F", "R"} if generation == "F06" else {"F", "LR", "PR"}
    if variant not in allowed:
        raise ValueError(f"Variant {variant} is invalid for {generation}")
    return generation, variant, design


def parent_of(value: str) -> str | None:
    generation, variant, design = parse_fingerprint_id(value)
    if generation == "F06":
        return None if variant == "F" else f"F06_F_{design}"
    previous = GENERATIONS[GENERATIONS.index(generation) - 1]
    if variant == "F":
        return f"{previous}_F_{design}"
    if variant == "LR":
        return f"{generation}_F_{design}"
    prior_variant = "R" if previous == "F06" else "PR"
    return f"{previous}_{prior_variant}_{design}"


def all_fingerprint_ids() -> list[str]:
    ids = [f"F06_{variant}_{design}" for design in DESIGNS for variant in ("F", "R")]
    ids += [
        f"{generation}_{variant}_{design}"
        for generation in GENERATIONS[1:]
        for design in DESIGNS
        for variant in ("F", "LR", "PR")
    ]
    return ids


def lineage_plan() -> list[dict[str, str | None]]:
    """Return the complete lightweight materialization order, without running it."""
    return [{"fingerprint_id": value, "parent": parent_of(value)} for value in all_fingerprint_ids()]


def assert_design_years(years: Iterable[int], *, maximum: int = 2025) -> None:
    rejected = sorted({int(year) for year in years if int(year) > maximum})
    if rejected:
        raise ValueError(f"Design evidence includes quarantined years: {rejected}")


def assert_design_operation_frame(frame, operation: str, *, season_column: str = "season") -> None:
    """Guard any later fitted/design operation at its input boundary."""
    allowed = {"feature_discovery", "equations", "composite_fitting", "scaling",
               "imputation", "missingness_rules", "correlation", "pruning", "shap",
               "hyperparameters", "lineage", "recommendations", "model_evaluation"}
    if operation not in allowed:
        raise ValueError(f"Unknown nextgen design operation: {operation}")
    if season_column not in frame:
        raise ValueError(f"Missing season column for {operation}")
    assert_design_years(int(year) for year in frame[season_column]
                        if year is not None and str(year).lower() != "nan")


def assert_temporal_feature_rows(frame: pd.DataFrame, feature_columns: Iterable[str]) -> None:
    """Validate the common canonical write and model/SHAP load boundary."""
    required = {"season", "season_type", "feature_kind", "target_game_id",
                "target_start_utc", "feature_available_utc", "latest_source_game_id",
                "latest_source_game_utc", "latest_source_season_type",
                "static_availability_documentation"}
    if not required <= set(frame):
        raise ValueError(f"Feature rows lack temporal provenance: {sorted(required-set(frame))}")
    if pd.to_numeric(frame.season, errors="coerce").isna().any():
        raise ValueError("Feature row lacks a valid season")
    assert_design_operation_frame(frame, "feature_discovery")
    if frame.empty or frame.target_game_id.isna().any():
        raise ValueError("Feature row lacks a target game")
    if not frame.season_type.astype(str).str.lower().eq("regular").all():
        raise ValueError("Postseason feature or target row is forbidden")
    if not frame.feature_kind.isin(["dynamic", "static_week0"]).all():
        raise ValueError("Unknown nextgen feature kind")
    available = pd.to_datetime(frame.feature_available_utc, utc=True, errors="coerce")
    starts = pd.to_datetime(frame.target_start_utc, utc=True, errors="coerce")
    if available.isna().any() or starts.isna().any() or not available.lt(starts).all():
        raise ValueError("A feature is not provably available before its target game")
    dynamic = frame.feature_kind.eq("dynamic")
    source_time = pd.to_datetime(frame.latest_source_game_utc, utc=True, errors="coerce")
    if (frame.loc[dynamic, "latest_source_game_id"].isna().any()
            or source_time.loc[dynamic].isna().any()
            or not source_time.loc[dynamic].lt(available.loc[dynamic]).all()
            or not frame.loc[dynamic, "latest_source_season_type"].astype(str).str.lower().eq("regular").all()):
        raise ValueError("Dynamic feature lacks completed regular source-game provenance")
    if frame.loc[dynamic, "latest_source_game_id"].eq(frame.loc[dynamic, "target_game_id"]).any():
        raise ValueError("A target game cannot supply its own pregame feature")
    static = ~dynamic
    if (frame.loc[static, ["latest_source_game_id", "latest_source_game_utc",
                           "latest_source_season_type"]].notna().any().any()
            or frame.loc[static, "static_availability_documentation"].fillna("").astype(str).str.strip().eq("").any()):
        raise ValueError("Week-0 feature lacks documented static availability")
    names = set(feature_columns)
    if (not names or names & required
            or any(name.startswith(("target_", "next_game_")) for name in names)
            or names & {"home_points", "away_points"} or not names <= set(frame)):
        raise ValueError("Target outcome or absent column selected as a feature")
    assert_safe_inputs(names)


def assert_average_reference_years(target_year: int, source_years: Iterable[int]) -> None:
    rejected = sorted({int(year) for year in source_years if int(year) >= target_year})
    if rejected:
        raise ValueError(f"Average-team reference includes unfinished/future seasons: {rejected}")


def assert_safe_inputs(columns: Iterable[str], records: Iterable[Mapping] = ()) -> None:
    rejected = sorted({name for name in columns if FORBIDDEN_INPUT.search(name)})
    for record in records:
        if record.get("market_derived") or record.get("pregame_win_probability_derived"):
            rejected.append(str(record.get("name")))
    if rejected:
        raise ValueError(f"Forbidden model/feature-engineering inputs: {sorted(set(rejected))}")


def assert_pair_closed(selected: Iterable[str], manifest: Iterable[Mapping]) -> None:
    records = {str(record["name"]): record for record in manifest}
    names = set(selected)
    for name in names:
        if name not in records:
            raise ValueError(f"Selected feature absent from manifest: {name}")
        counterpart = records[name]["matchup_counterpart"]
        if counterpart not in names:
            raise ValueError(f"Pair closure violated: {name} requires {counterpart}")


def validate_feature_manifest(records: list[dict], schema: dict) -> None:
    required = set(schema["required"])
    names = [record.get("name") for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate source-feature names")
    for record in records:
        name = record.get("name", "<unnamed>")
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{name}: missing {sorted(missing)}")
        for field, choices in schema["enums"].items():
            value = record[field]
            values = value if isinstance(value, list) else [value]
            if not values or any(item not in choices for item in values):
                raise ValueError(f"{name}: invalid {field}: {value}")
        if "c" in record["designs"] and len(record["source_inputs"]) > schema["c_max_source_inputs"]:
            raise ValueError(f"{name}: C composite exceeds five source inputs")
        if not record["source_inputs"] or not record["equation_excel"] or not record["matchup_formula"]:
            raise ValueError(f"{name}: source inputs and exact equations are required")
        if record["static_or_dynamic"] == "dynamic" and record["temporal_cutoff"] != "before_target_game":
            raise ValueError(f"{name}: dynamic feature must precede target game")
        if record["static_or_dynamic"] == "static_preseason" and not any(
            token in record["availability_rule"] for token in ("week0", "before_season")
        ):
            raise ValueError(f"{name}: static feature requires Week-0 availability")
        if record["market_derived"] or record.get("pregame_win_probability_derived", False):
            raise ValueError(f"{name}: market or win-probability input forbidden")
        counterpart = record["matchup_counterpart"]
        if counterpart != name and counterpart not in names:
            raise ValueError(f"{name}: missing matchup counterpart {counterpart}")
    by_name = {record["name"]: record for record in records}
    for record in records:
        counterpart = by_name[record["matchup_counterpart"]]
        if counterpart["matchup_counterpart"] != record["name"]:
            raise ValueError(f"{record['name']}: matchup counterpart is not reciprocal")
    assert_safe_inputs(names, records)


def validate_contract(config: dict, *, repo_root: Path) -> None:
    root = Path(config["artifact_root"])
    if not root.is_absolute() or not str(root).startswith("/groups/"):
        raise ValueError("Large artifact root must be an absolute /groups path")
    if config["seasons"]["design_max_year"] >= config["quarantine"]["year"]:
        raise ValueError("Design window reaches prospective quarantine")
    if config["seasons"].get("evidence_roles") != {
        "2024": "design_informed_internal_validation",
        "2025": "late_development_design_informed",
        "2026": "untouched_prospective_only",
    } or config["recommendation"].get("2024_2025_metrics_are_unbiased_holdout") is not False:
        raise ValueError("Development metrics must not be labeled as unbiased holdout results")
    if config["source_f6_feature_count"] != 227:
        raise ValueError("F06 canonical baseline count changed")
    source = load_json(repo_root / config["source_f6_manifest"])
    if source["feature_count"] != 227 or source["schema_hash"] != config["source_f6_schema_hash"]:
        raise ValueError("F06 source manifest does not match frozen baseline")
    if source["market_derived"]:
        raise ValueError("F06 source baseline is market-derived")
    for fingerprint in all_fingerprint_ids():
        parent = parent_of(fingerprint)
        if parent is not None and parent not in all_fingerprint_ids():
            raise ValueError(f"Missing parent {parent}")
        if parent and ("F07" in parent or "F08" in parent):
            raise ValueError("Market comparator is an ancestor")
    if config["scheduler"]["max_running_jobs_project_wide"] > 50:
        raise ValueError("Scheduler cap exceeds 50")
    budget = config["cfbd_api_call_budget"]
    if (budget["hard_limit"] != 20000 or budget["minimum_reserve"] != 10000
            or budget["account_allowance"] - budget["hard_limit"] < budget["minimum_reserve"]
            or not budget["ledger"].startswith(str(root) + "/")):
        raise ValueError("Next-generation CFBD budget must cap at 20,000 and reserve 10,000")


def validate_setup(repo_root: Path) -> None:
    base = repo_root / "configs/experiments"
    config = load_json(base / "nextgen_fingerprints_v1.json")
    validate_contract(config, repo_root=repo_root)
    setpoints = load_json(base / "nextgen_screening_setpoints_v1.json")
    acquisition = load_json(base / "nextgen_acquisition_v1.json")
    if acquisition["api_call_budget"]["hard_limit"] != config["cfbd_api_call_budget"]["hard_limit"]:
        raise ValueError("Acquisition budget differs from experiment contract")
    if acquisition["api_call_budget"]["minimum_reserve"] != config["cfbd_api_call_budget"]["minimum_reserve"]:
        raise ValueError("Acquisition reserve differs from experiment contract")
    if acquisition["api_call_budget"]["ledger"] != config["cfbd_api_call_budget"]["ledger"]:
        raise ValueError("Acquisition ledger differs from experiment contract")
    if (acquisition["cache"]["schedule_authority"] !=
            "fresh_success_complete_ledger_backed_games_2010_2025"
            or acquisition["cache"]["legacy_games_reuse"] is not False
            or acquisition["cache"]["games_teams_reuse_requires_authoritative_schedule"] is not True):
        raise ValueError("Fresh schedules must precede legacy team-game reuse")
    inventory = load_json(base / "nextgen_cfbd_endpoint_inventory_v1.json")
    if next(item for item in inventory["endpoints"] if item["endpoint"] == "/games")["stage"] != "A":
        raise ValueError("Fresh /games acquisition must be Stage A")
    if setpoints["seed"] != config["screening"]["seed"]:
        raise ValueError("Setpoint seed differs from contract")
    for architecture in config["screening"]["architectures"]:
        points = setpoints[architecture]
        if len(points) != 10 or len({p["id"] for p in points}) != 10:
            raise ValueError(f"{architecture} requires ten unique setpoints")
    schema = load_json(base / "nextgen_feature_manifest_schema_v1.json")
    records = load_json(base / "nextgen_seed_features_v1.json")
    validate_feature_manifest(records, schema)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate or print next-generation setup; never launches jobs")
    parser.add_argument("--plan", action="store_true", help="Print the 42-variant lineage plan as JSON")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    validate_setup(root)
    if args.plan:
        print(json.dumps(lineage_plan(), indent=2))
    else:
        print(f"Validated {len(all_fingerprint_ids())} next-generation fingerprint variants")
