"""src.gridiron_ml.fingerprints.builders.v1.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Build, load, and split time-dependent team fingerprints.
"""

from .base import BaseFingerprintBuilder


class V1FingerprintBuilder(BaseFingerprintBuilder):
    """Represent the V1FingerprintBuilder component and its local behavior."""
    def _build_from_team_game_tables(self, overwrite=False):
        """Internal helper for the build_from_team_game_tables step."""
        raise NotImplementedError("Version 1 fingerprint builder is not implemented yet.")
