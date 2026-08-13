#!/usr/bin/env python3
"""Scan public-facing source/configuration for machine-local assumptions."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess


ROOT = project_root()
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".json", ".md", ".txt"}
ABSOLUTE_PATTERN = re.compile(r"(?<![A-Za-z0-9_.-])(?:/users/|/groups/|/afs/|/home/|[A-Za-z]:\\)")
ENV_PATTERN = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*|[A-Z][A-Z0-9_]*\b)")


def scan() -> dict:
    findings = []
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    candidates = {ROOT / value for value in tracked}
    # Include mandate-era untracked source/config/docs without traversing raw
    # data, experiment fragments, model bytes, or cache trees.
    for relative_root in ("scripts", "src", "configs", "docs/publication_2026"):
        candidate_root = ROOT / relative_root
        if candidate_root.exists():
            candidates.update(path for path in candidate_root.rglob("*") if path.is_file())
    candidates.update(path for path in ROOT.glob("*.md") if path.is_file())
    for path in sorted(candidates):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name in {"portability_audit.json", "portability_audit.md"}:
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, 1):
            if not ABSOLUTE_PATTERN.search(line):
                continue
            relative = path.relative_to(ROOT)
            suffix = path.suffix.lower()
            source_class = "runtime_or_config" if suffix in {".py", ".sh", ".yaml", ".yml", ".toml"} else "documentation_or_metadata"
            planned_storage_path = (
                str(relative) == "src/gridiron_ml/cli/publication/build_storage_inventory.py"
                or str(relative) == "src/gridiron_ml/cli/publication/run_portability_audit.py"
            )
            severity = "review" if planned_storage_path else ("blocker" if source_class == "runtime_or_config" else "review")
            findings.append({
                "path": str(relative),
                "line": number,
                "severity": severity,
                "source_class": source_class,
                "text": line.strip()[:240],
                "remediation": "replace with project-root discovery, an environment variable, or an explicit cluster profile",
            })
    runtime_findings = [item for item in findings if item["severity"] == "blocker"]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not runtime_findings else "blockers_found",
        "project_root": str(ROOT),
        "portable_environment_policy": "runtime paths must be relative or explicitly supplied through environment/CLI",
        "finding_count": len(findings),
        "runtime_blocker_count": len(runtime_findings),
        "documentation_review_count": len(findings) - len(runtime_findings),
        "findings": findings,
        "clone_verification": "not_run_by_this_static_audit",
    }


def main() -> int:
    root = ROOT
    parser = ArgumentParser()
    parser.add_argument("--json", type=Path, default=root / "docs/publication_2026/portability_audit.json")
    parser.add_argument("--markdown", type=Path, default=root / "docs/publication_2026/portability_audit.md")
    args = parser.parse_args()
    report = scan()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    blockers = [item for item in report["findings"] if item["severity"] == "blocker"]
    lines = [
        "# TDNet portability audit",
        "",
        f"Status: **{report['status']}**.",
        "",
        f"Static scan found {report['finding_count']} absolute-path references, "
        f"including {report['runtime_blocker_count']} runtime/configuration blockers. "
        "This audit does not claim arbitrary-clone execution; that remains a separate gate.",
        "",
        "| Severity | Path | Line | Remediation |",
        "|---|---|---:|---|",
    ]
    for item in report["findings"]:
        lines.append(f"| {item['severity']} | `{item['path']}` | {item['line']} | {item['remediation']} |")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "finding_count", "runtime_blocker_count", "clone_verification")}, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
