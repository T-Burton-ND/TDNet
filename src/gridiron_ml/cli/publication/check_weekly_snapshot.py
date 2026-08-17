#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Certify cached CFBD endpoint completeness for one prospective season."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import yaml

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.pipeline.build_full_pipeline import normalize_raw_endpoint_flags  # noqa: E402
from gridiron_ml.publication.weekly_protocol import (  # noqa: E402
    build_snapshot_completeness,
    write_snapshot_completeness,
)


def _repo_relative_paths(report: dict) -> dict:
    """Keep committed snapshot evidence portable across checkout locations."""
    normalized = dict(report)
    cache = Path(str(normalized["raw_cache_dir"]))
    try:
        normalized["raw_cache_dir"] = str(cache.resolve().relative_to(ROOT))
    except ValueError:
        normalized["raw_cache_dir"] = "EXTERNAL_DURABLE_ROOT/" + cache.name
    endpoints = []
    for endpoint in normalized["endpoints"]:
        item = dict(endpoint)
        path = Path(str(item["path"]))
        try:
            item["path"] = str(path.resolve().relative_to(ROOT))
        except ValueError:
            item["path"] = "EXTERNAL_DURABLE_ROOT/" + path.name
        endpoints.append(item)
    normalized["endpoints"] = endpoints
    return normalized


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fetch/weekly_2026_refresh.yaml")
    parser.add_argument("--raw-cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    raw_cfg = config.get("raw_fetch", {})
    raw_cache = args.raw_cache_dir or ROOT / config.get("paths", {}).get("raw_cache_dir", "data/raw/cfbd/v2")
    report = build_snapshot_completeness(
        raw_cache_dir=raw_cache,
        season=args.season,
        endpoints=normalize_raw_endpoint_flags(raw_cfg.get("endpoints")),
        completeness_config=raw_cfg.get("completeness"),
        required_endpoints=raw_cfg.get("required_endpoints"),
    )
    report = _repo_relative_paths(report)
    output = args.output or ROOT / f"data/publication/{args.season}/weekly_operations/snapshot_completeness.json"
    write_snapshot_completeness(report, output)
    print(output)
    print(report["certification"])
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
