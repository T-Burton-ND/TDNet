"""src.gridiron_ml.fingerprints.build.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Build, load, and split time-dependent team fingerprints.
"""

import sys
from pathlib import Path

import yaml

from gridiron_ml.fingerprints import Fingerprints


def main(argv=None):
    """Run the main step and return its normalized result."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        raise SystemExit("Usage: python -m gridiron_ml.fingerprints.build <config.yaml>")

    config_path = Path(argv[0])
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    version = int(config.get("version", 0))
    root = config.get("root")
    overwrite = bool(config.get("overwrite", False))
    team_game_tables_dir = config.get("team_game_tables_dir")

    fp = Fingerprints(
        version=version,
        postseason=bool(config.get("postseason", False)),
        root=root,
        team_game_tables_dir=team_game_tables_dir,
    )
    out = fp.build(overwrite=overwrite)
    print(out)


if __name__ == "__main__":
    main()
