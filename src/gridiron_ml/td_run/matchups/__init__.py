"""src.gridiron_ml.td_run.matchups.__init__.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Convert team fingerprints into matchup feature rows for prediction models.
"""

from .builder import DEFAULT_ZERO_OFFSET, MatchupBuilder
from .unit_matchups import (
    PRIMARY_UNIT_MATCHUP_COUNTERPARTS,
    SECONDARY_UNIT_MATCHUP_COUNTERPARTS,
    UNIT_MATCHUP_DIRECTION_OVERRIDES,
    default_unit_pairing_specs,
    feature_direction,
)

__all__ = [
    "DEFAULT_ZERO_OFFSET",
    "MatchupBuilder",
    "PRIMARY_UNIT_MATCHUP_COUNTERPARTS",
    "SECONDARY_UNIT_MATCHUP_COUNTERPARTS",
    "UNIT_MATCHUP_DIRECTION_OVERRIDES",
    "default_unit_pairing_specs",
    "feature_direction",
]
