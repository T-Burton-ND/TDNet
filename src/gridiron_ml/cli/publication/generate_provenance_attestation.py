#!/usr/bin/env python3
"""Generate an in-toto/SLSA-shaped provenance statement for a freeze manifest."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json

from gridiron_ml.publication.bundles import sha256_file


def main():
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [{"name": args.manifest.name, "digest": {"sha256": sha256_file(args.manifest)}}],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://github.com/Savoie-Research-Group/TDNet/publication-freeze/v1",
                "externalParameters": {"freeze_version": manifest["freeze_version"]},
                "resolvedDependencies": [{"uri": "git+tdnet", "digest": {"gitCommit": manifest["git_commit"]}}],
            },
            "runDetails": {"builder": {"id": "gridiron_ml.publication.freeze"}, "metadata": {"invocationId": manifest["manifest_sha256"], "startedOn": manifest["created_at_utc"], "finishedOn": datetime.now(timezone.utc).isoformat()}},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(statement, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
