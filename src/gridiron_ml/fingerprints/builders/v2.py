"""src.gridiron_ml.fingerprints.builders.v2.

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


class V2FingerprintBuilder(BaseFingerprintBuilder):
    """Represent the V2FingerprintBuilder component and its local behavior."""
    def _build_from_team_game_tables(self, overwrite=False):
        """Internal helper for the build_from_team_game_tables step."""
        raise NotImplementedError("Version 2 fingerprint builder is not implemented yet.")
