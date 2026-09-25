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


ID_PATTERN = re.compile(r"^F(06|09|10|11|12)_(F|R|LR|PR)_([abc])$")
FORBIDDEN_INPUT = re.compile(
    r"(?:market|vegas|betting|spread|moneyline|over_?under|pregame_?wp|"
    r"pregame.*prob|win_?prob(?:ability)?|implied_?prob|(?:^|_)ats(?:_|$))",
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
    if budget["hard_limit"] != 20000 or not budget["ledger"].startswith(str(root) + "/"):
        raise ValueError("Next-generation CFBD budget must be 20,000 under artifact root")


def validate_setup(repo_root: Path) -> None:
    base = repo_root / "configs/experiments"
    config = load_json(base / "nextgen_fingerprints_v1.json")
    validate_contract(config, repo_root=repo_root)
    setpoints = load_json(base / "nextgen_screening_setpoints_v1.json")
    acquisition = load_json(base / "nextgen_acquisition_v1.json")
    if acquisition["api_call_budget"]["hard_limit"] != config["cfbd_api_call_budget"]["hard_limit"]:
        raise ValueError("Acquisition budget differs from experiment contract")
    if acquisition["api_call_budget"]["ledger"] != config["cfbd_api_call_budget"]["ledger"]:
        raise ValueError("Acquisition ledger differs from experiment contract")
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
