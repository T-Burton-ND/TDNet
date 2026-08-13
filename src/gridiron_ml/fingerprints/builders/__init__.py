"""src.gridiron_ml.fingerprints.builders.__init__.

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
from .registry import FINGERPRINT_BUILDERS, get_fingerprint_builder, register_fingerprint_builder
from .v0 import V0FingerprintBuilder
from .v1 import V1FingerprintBuilder
from .v2 import V2FingerprintBuilder

register_fingerprint_builder(0, V0FingerprintBuilder)
register_fingerprint_builder(1, V1FingerprintBuilder)
register_fingerprint_builder(2, V2FingerprintBuilder)

__all__ = [
    "BaseFingerprintBuilder",
    "FINGERPRINT_BUILDERS",
    "V0FingerprintBuilder",
    "V1FingerprintBuilder",
    "V2FingerprintBuilder",
    "get_fingerprint_builder",
    "register_fingerprint_builder",
]
