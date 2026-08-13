"""src.gridiron_ml.td_sim.orchestrator.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Run recursive season simulations with evolving synthetic fingerprints.
"""

from .td_sim import TDSim


class TDSimOrchestrator:
    """Config-driven recursive TD Sim runner."""

    def __init__(self, config):
        """Internal helper for the init__ step."""
        self.sim = TDSim(config=config)
        self.config = self.sim.config
        self.output_dir = None

    def run(
        self,
        season=None,
        N=None,
        models=None,
        workflow=None,
        schedule_mode=None,
        as_of_week=None,
        save_debug=None,
        show_progress=None,
        sim_start=0,
        shard_id=None,
        **_,
    ):
        """Run the run step and return its normalized result."""
        result = self.sim.run(
            season=season,
            N=N,
            models=self._normalize_models(models),
            workflow=workflow,
            schedule_mode=schedule_mode,
            as_of_week=as_of_week,
            save_debug=save_debug,
            show_progress=show_progress,
            sim_start=sim_start,
            shard_id=shard_id,
        )
        self.output_dir = result["output_dir"]
        return result

    def _normalize_models(self, models):
        """Internal helper for the normalize_models step."""
        if models is None:
            return None
        if isinstance(models, str):
            text = models.strip()
            return [text] if text else None
        normalized = [str(model).strip() for model in models if str(model).strip()]
        return normalized or None
