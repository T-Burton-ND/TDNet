#!/usr/bin/env python3
"""Inventory every available raw/engineered data schema without reading outcomes.

This is a source-discovery ledger, not an eligibility decision. Every field is
catalogued; only a later field-level as-of audit may admit it to F18 or F19.
Parquet footers and CSV headers are read, never data rows or 2026 scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
GROUP = Path("/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen")
DEFAULT_OUTPUT = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/source_audit")


def family(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    parts = relative.parts
    if root == GROUP:
        if parts[:2] == ("raw_cache", "v1") and len(parts) > 2:
            return f"group/raw/{parts[2]}"
        if parts and parts[0] in {"feature_families", "fingerprints"}:
            return f"group/{parts[0]}/{parts[1] if len(parts) > 1 else 'root'}"
        return f"group/{parts[0]}"
    if parts[:2] == ("raw", "cfbd"):
        return f"repo/raw/cfbd/{parts[2] if len(parts) > 2 else 'root'}"
    if parts and parts[0] == "what_if_2026_fingerprints":
        return f"repo/what_if_2026/{parts[1] if len(parts) > 1 else 'root'}"
    if parts and parts[0] == "publication":
        return "repo/publication"
    if parts and parts[0] == "private":
        return "repo/private"
    return f"repo/{parts[0]}"


def source_class(group: str) -> str:
    if group.startswith("group/raw/lines"):
        return "market_raw_requires_quote_time_audit"
    if group.startswith("group/raw/"):
        return "raw_requires_target_asof_construction"
    if group.startswith("group/feature_families/"):
        return "engineered_requires_manifest_and_cutoff_audit"
    if group.startswith("group/fingerprints/"):
        return "engineered_requires_manifest_and_cutoff_audit"
    if group == "group/canonical":
        return "canonical_engineered_requires_cutoff_audit"
    if group in {"group/scratch", "group/progressive_pools"}:
        return "scratch_engineered_requires_cutoff_audit"
    if group == "group/results":
        return "experiment_result_quarantined"
    if group.startswith("group/"):
        return "inspect_source_role"
    if group.startswith("repo/what_if_2026/"):
        return "2026_target_or_result_quarantined_until_freeze"
    if group == "repo/publication" or group == "repo/private":
        return "publication_or_private_quarantined_until_cutoff_audit"
    if group == "repo/nextgen_rounds_2026":
        return "historical_prepared_requires_feature_outcome_separation"
    if group == "repo/raw/cfbd/v2":
        return "raw_requires_target_asof_construction_or_quote_time"
    if group == "repo/fingerprints":
        return "prior_fingerprint_requires_feature_target_separation"
    if group == "repo/derived":
        return "derived_requires_cutoff_audit"
    if group == "repo/team_game_tables":
        return "raw_game_outcomes_require_prior_game_lagging"
    if group == "repo/experiments":
        return "engineered_requires_manifest_and_cutoff_audit"
    return "inspect_source_role"


def flatten_schema(schema: pa.Schema) -> list[dict]:
    fields = []

    def visit(field: pa.Field, prefix: str):
        name = f"{prefix}.{field.name}" if prefix else field.name
        dtype = field.type
        fields.append({"name": name, "type": str(dtype), "nested": bool(prefix)})
        if pa.types.is_struct(dtype):
            for child in dtype:
                visit(child, name)
        elif pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
            value = dtype.value_field
            if pa.types.is_struct(value.type):
                for child in value.type:
                    visit(child, name + "[]")

    for field in schema:
        visit(field, "")
    return fields


def inspect_file(path: Path, root: Path) -> dict:
    group = family(path, root)
    result = {"path": str(path), "group": group, "source_class": source_class(group),
              "suffix": path.suffix.lower(), "bytes": path.stat().st_size}
    if path.suffix.lower() == ".parquet":
        metadata = pq.ParquetFile(path).metadata
        schema = pq.read_schema(path)
        result.update(rows=metadata.num_rows, fields=flatten_schema(schema))
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        result.update(rows=None, fields=[{"name": name, "type": "csv_untyped", "nested": False}
                                         for name in header])
    else:
        raise ValueError(f"Unsupported source suffix: {path}")
    return result


def manifest_records(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob("feature_manifest.json")):
        try:
            content = json.loads(path.read_text())
            if isinstance(content, list):
                names = [item.get("name") for item in content if isinstance(item, dict)]
            elif isinstance(content, dict):
                items = content.get("features", [])
                names = [item.get("name") for item in items if isinstance(item, dict)]
            else:
                names = []
            records.append({"path": str(path), "names": sorted({str(x) for x in names if x}),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        except (OSError, ValueError, TypeError) as exc:
            records.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return records


def run(output: Path, include_group: bool = True) -> dict:
    roots = [ROOT / "data"] + ([GROUP] if include_group else [])
    all_files = sorted(path for root in roots for path in root.rglob("*") if path.is_file())
    files = [path for path in all_files if path.suffix.lower() in {".parquet", ".csv"}]
    non_tabular = [path for path in all_files if path.suffix.lower() not in {".parquet", ".csv"}]
    by_group = defaultdict(lambda: {"files": 0, "bytes": 0, "rows_parquet": 0,
                                  "fields": {}, "errors": []})
    file_rows = []
    for index, path in enumerate(files, 1):
        root = next(root for root in roots if path.is_relative_to(root))
        group = family(path, root)
        try:
            record = inspect_file(path, root)
            fields = record.pop("fields")
            file_rows.append({**record, "field_count": len(fields),
                              "schema_sha256": hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()})
            item = by_group[group]
            item["files"] += 1
            item["bytes"] += record["bytes"]
            item["rows_parquet"] += record.get("rows") or 0
            for field in fields:
                item["fields"][field["name"]] = field["type"]
        except (OSError, ValueError, pa.ArrowException) as exc:
            by_group[group]["errors"].append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        if index % 2000 == 0:
            print(f"inspected {index}/{len(files)} file schemas", flush=True)
    manifests = manifest_records(GROUP) if include_group else []
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Parquet schema/footer and CSV header only; no 2026 data rows or outcomes read",
        "roots": [str(root) for root in roots], "source_files_found": len(files),
        "source_files_inspected": len(file_rows),
        "unparsed_files_found": len(non_tabular),
        "unparsed_suffix_counts": dict(sorted(Counter(
            path.suffix.lower() or "<none>" for path in non_tabular).items())),
        "source_groups": {
            group: {**{key: value for key, value in item.items() if key != "fields"},
                    "source_class": source_class(group),
                    "field_count": len(item["fields"]),
                    "fields": item["fields"]}
            for group, item in sorted(by_group.items())
        },
        "feature_manifests": manifests,
        "interpretation": "Discovery only. All fields remain candidates, but model admission needs source-time, target-time, market-lineage, and outcome-exclusion proof.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "schema_inventory.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    with (output / "source_files.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "group", "source_class", "suffix",
                                                      "bytes", "rows", "field_count", "schema_sha256"])
        writer.writeheader()
        writer.writerows(file_rows)
    with (output / "unparsed_file_paths.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "group", "source_class", "suffix",
                                                      "inspection_status"])
        writer.writeheader()
        for path in non_tabular:
            root = next(root for root in roots if path.is_relative_to(root))
            group = family(path, root)
            writer.writerow({"path": str(path), "group": group,
                             "source_class": source_class(group),
                             "suffix": path.suffix.lower(),
                             "inspection_status": "path_only_no_content_read"})
    return {"files": len(files), "inspected": len(file_rows), "groups": len(by_group),
            "manifests": len(manifests), "output": str(output),
            "errors": sum(len(item["errors"]) for item in by_group.values()),
            "non_tabular_paths": len(non_tabular)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-group", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, include_group=not args.no_group), indent=2), flush=True)


if __name__ == "__main__":
    main()
