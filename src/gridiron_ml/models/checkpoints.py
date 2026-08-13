"""src.gridiron_ml.models.checkpoints.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Define model wrappers and checkpoint helpers behind the shared TDNet interface.
"""

from pathlib import Path
import pickle


def load_model_checkpoint(path):
    """Run the load_model_checkpoint step and return its normalized result."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
    with path.open("rb") as f:
        return pickle.load(f)
