from gridiron_ml.cli._paths import project_root
#!/usr/bin/env python3
"""Verify all weekly bundle hashes and pre-kickoff assertions."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.publication import verify_prediction_bundle


def main():
    parser = ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--require-clean-commit", action="store_true")
    args = parser.parse_args()
    report = verify_prediction_bundle(args.bundle, require_clean_commit=args.require_clean_commit)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
