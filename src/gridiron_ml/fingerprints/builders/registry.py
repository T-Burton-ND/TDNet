"""src.gridiron_ml.fingerprints.builders.registry.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Build, load, and split time-dependent team fingerprints.
"""

FINGERPRINT_BUILDERS = {}


def register_fingerprint_builder(version, builder_cls):
    """Run the register_fingerprint_builder step and return its normalized result."""
    FINGERPRINT_BUILDERS[int(version)] = builder_cls
    return builder_cls


def get_fingerprint_builder(version):
    """Run the get_fingerprint_builder step and return its normalized result."""
    version = int(version)
    if version not in FINGERPRINT_BUILDERS:
        supported = ", ".join(str(v) for v in sorted(FINGERPRINT_BUILDERS))
        raise ValueError(f"Unsupported fingerprint version: {version}. Supported: {supported}")
    return FINGERPRINT_BUILDERS[version]
