#!/usr/bin/env python3
from gridiron_ml.cli._paths import project_root
"""Verify local freeze bytes, plus optional tag, DOI, and Sigstore evidence."""

from argparse import ArgumentParser
from pathlib import Path
import json
import subprocess
import sys

ROOT = project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.publication import verify_preseason_freeze


def main():
    parser = ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-tag")
    parser.add_argument("--expected-doi-file", type=Path)
    parser.add_argument("--sigstore-bundle", type=Path)
    args = parser.parse_args()
    result = verify_preseason_freeze(args.bundle)
    failures = list(result["failures"])
    manifest_path = args.bundle / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.expected_tag:
        tagged = subprocess.run(["git", "rev-list", "-n", "1", args.expected_tag], capture_output=True, text=True)
        if tagged.returncode or tagged.stdout.strip() != manifest["git_commit"]:
            failures.append("git commit and expected tag do not match")
    if args.expected_doi_file:
        if not args.expected_doi_file.exists() or "doi" not in json.loads(args.expected_doi_file.read_text(encoding="utf-8")):
            failures.append("Zenodo DOI evidence missing or malformed")
    if args.sigstore_bundle:
        completed = subprocess.run(["cosign", "verify-blob", "--bundle", str(args.sigstore_bundle), str(manifest_path)], capture_output=True, text=True)
        if completed.returncode:
            failures.append("Sigstore verification failed")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("PASS: all manifest-tracked file hashes match")
    print("PASS: manifest self-hash matches")
    print("PASS: all model checkpoints are represented in inventory")


if __name__ == "__main__":
    main()
