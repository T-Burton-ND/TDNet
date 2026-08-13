"""src.gridiron_ml.fingerprints.__init__.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Build, load, and split time-dependent team fingerprints.
"""

from .features import DEFAULT_FEATURE_SPEC, FeatureSpec, split_frame
from .fingerprints import Fingerprints

__all__ = ["DEFAULT_FEATURE_SPEC", "FeatureSpec", "Fingerprints", "split_frame"]
