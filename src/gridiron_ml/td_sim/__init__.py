"""src.gridiron_ml.td_sim.__init__.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Run recursive season simulations with evolving synthetic fingerprints.
"""

from .bootstrap import append_bootstrap_week0, ensure_schedule_team_game_table
from .checkpoints import discover_model_checkpoints
from .orchestrator import TDSimOrchestrator
from .recursive_simulator import RecursiveSeasonSimulator
from .td_sim import TDSim

__all__ = [
    "TDSim",
    "TDSimOrchestrator",
    "RecursiveSeasonSimulator",
    "append_bootstrap_week0",
    "discover_model_checkpoints",
    "ensure_schedule_team_game_table",
]
