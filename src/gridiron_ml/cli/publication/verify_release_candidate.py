from gridiron_ml.cli._paths import project_root
#!/usr/bin/env python3
"""Verify the portable release-candidate manifest and canonical protocol."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.publication.protocol import load_yaml, validate_ladder_config


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "FREEZE_MANIFEST.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    if manifest.get("status") == "final":
        failures.append("final status is not allowed until all release evidence is verified")
    required = [
        "configs/publication/confirmatory_protocol.yaml",
        "configs/features/feature_ladders.yaml",
        "configs/features/feature_registry.yaml",
        "configs/publication/feature_model_matrix.yaml",
    ]
    for relative in required:
        path = ROOT / relative
        if not path.exists():
            failures.append(f"missing required path: {relative}")
    try:
        validate_ladder_config(load_yaml(ROOT / "configs/features/feature_ladders.yaml"))
    except Exception as exc:
        failures.append(f"ladder validation: {exc}")
    for relative, expected in dict(manifest.get("sha256", {})).items():
        if expected.startswith("pending_"):
            continue
        path = ROOT / relative
        if not path.exists():
            failures.append(f"missing hashed path: {relative}")
        elif sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"hash mismatch: {relative}")
    result = {"valid": not failures, "manifest": str(args.manifest), "failures": failures}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
