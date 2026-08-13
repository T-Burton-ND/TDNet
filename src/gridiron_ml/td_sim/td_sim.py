"""src.gridiron_ml.td_sim.td_sim.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Run recursive season simulations with evolving synthetic fingerprints.
"""

from pathlib import Path

import yaml

from .recursive_simulator import RecursiveSeasonSimulator


PROJECT_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_CONFIG = {
    "season": {
        "target_season": 2026,
    },
    "simulation_mode": {
        "workflow": "single_model",
        "schedule_mode": "full_schedule",
        "as_of_week": None,
    },
    "simulation": {
        "n_sims": 10,
        "random_seed": 42,
        "random_scaling_factor": 1.0,
        "game_noise_std": None,
        "model_residual_std": 14.0,
        "residual_std_multiplier": 1.0,
        "clip_win_probability_min": 0.01,
        "clip_win_probability_max": 0.99,
        "sigmoid_scale": 13.5,
    },
    "recursive": {
        "performance_sampler": "hybrid",
        "historical_seasons": 3,
        "historical_regression_games": 3.0,
        "stat_randomness": 1.0,
        "knn_neighbors": 40,
        "knn_randomness": 1.0,
        "knn_margin_band": 17.5,
        "knn_max_candidates": 1500,
        "knn_noise": 0.05,
        "hybrid_knn_weight": 0.65,
        "final_week": 16,
        "save_debug": False,
        "save_final_fingerprints": False,
    },
    "runtime": {
        "show_progress": True,
    },
    "fingerprints": {
        "version": 0,
        "postseason": False,
        "root": str(PROJECT_ROOT),
        "bootstrap_week0_if_missing": True,
        "bootstrap_persist": True,
        "bootstrap_seasons_back": 3,
        "bootstrap_recency_halflife": 1.5,
    },
    "matchup": {
        "representation": "unit_matchup",
    },
    "data": {
        "schedule_path_template": "data/team_game_tables/team_game_table_{season}_fbs.parquet",
        "raw_cache_dir": "data/raw/cfbd/v2",
        "division": "fbs",
        "auto_bootstrap": True,
    },
    "models": {
        "checkpoint_root": "models",
        "include_models": "all",
        "exclude_models": [],
    },
    "outputs": {
        "output_dir": "data/td_sim/{season}",
        "top_n_teams": 25,
    },
    "figures": {
        "dpi": 220,
        "logo_dir": "data/meta/logos/by_team",
    },
}


class TDSim:
    """Recursive TD Sim workflow.

    The workflow loads trained checkpoints, seeds each team from week 0 or a
    historical three-season prior, predicts each week, samples game outcomes,
    updates synthetic fingerprints, and writes only final projection artifacts.
    """

    def __init__(self, config=None, schedule=None, fingerprint_frame=None, model_specs=None, matchup_builder=None):
        """Internal helper for the init__ step."""
        self.config = self._load_config(config)
        self.engine = RecursiveSeasonSimulator(
            config=self.config,
            schedule=schedule,
            fingerprint_frame=fingerprint_frame,
            model_specs=model_specs,
            matchup_builder=matchup_builder,
        )
        self.tables_ = {}
        self.output_dir = None

    def run(
        self,
        season=None,
        N=None,
        models=None,
        workflow=None,
        schedule_mode=None,
        as_of_week=None,
        output_dir=None,
        sim_start=0,
        shard_id=None,
        save_debug=None,
        show_progress=None,
    ):
        """Run the run step and return its normalized result."""
        season = int(season or self.config.get("season", {}).get("target_season"))
        mode_cfg = self.config.get("simulation_mode", {})
        workflow = _normalize_workflow(workflow or mode_cfg.get("workflow") or "single_model")
        schedule_mode = _normalize_schedule_mode(schedule_mode or mode_cfg.get("schedule_mode") or mode_cfg.get("mode") or "full_schedule")
        as_of_week = as_of_week if as_of_week is not None else mode_cfg.get("as_of_week")
        result = self.engine.run(
            season=season,
            N=N,
            models=models,
            workflow=workflow,
            schedule_mode=schedule_mode,
            as_of_week=as_of_week,
            output_dir=output_dir,
            sim_start=sim_start,
            shard_id=shard_id,
            save_debug=save_debug,
            show_progress=show_progress,
        )
        self.tables_ = {
            "final_records": result["final_records"],
            "average_poll": result["average_poll"],
            "simulation_records": result["simulation_records"],
            "poll_ballots": result["poll_ballots"],
            "models": result["models"],
        }
        self.output_dir = result["output_dir"]
        return result

    def run_single_model(self, season=None, model=None, N=None, **kwargs):
        """Run the run_single_model step and return its normalized result."""
        models = [model] if isinstance(model, str) else model
        return self.run(season=season, N=N, models=models, workflow="single_model", **kwargs)

    def run_multi_model(self, season=None, models=None, N=None, **kwargs):
        """Run the run_multi_model step and return its normalized result."""
        return self.run(season=season, N=N, models=models, workflow="multi_model", **kwargs)

    def sim_all_models(self, season, N=None, models=None, **kwargs):
        """Run the sim_all_models step and return its normalized result."""
        return self.run_multi_model(season=season, models=models, N=N, **kwargs)

    def load_outputs(self, season=None, workflow="single_model", model="all_models"):
        """Run the load_outputs step and return its normalized result."""
        season = int(season or self.config.get("season", {}).get("target_season"))
        base = Path(str(self.config.get("outputs", {}).get("output_dir", "data/td_sim/{season}")).format(season=season))
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        out_dir = base / "recursive" / _normalize_workflow(workflow).replace("recursive_", "") / _safe_name(model)
        records_path = out_dir / "final_records.csv"
        poll_path = out_dir / "average_poll.csv"
        if not records_path.exists() or not poll_path.exists():
            raise FileNotFoundError(f"Recursive TD Sim outputs do not exist under {out_dir}")
        import pandas as pd

        self.output_dir = out_dir
        self.tables_ = {
            "final_records": pd.read_csv(records_path),
            "average_poll": pd.read_csv(poll_path),
        }
        return self.tables_

    def _load_config(self, config):
        """Internal helper for the load_config step."""
        loaded = {}
        if config is None:
            loaded = {}
        elif isinstance(config, (str, Path)):
            with Path(config).open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
        else:
            loaded = dict(config)
        return _deep_merge(DEFAULT_CONFIG, loaded)


def _normalize_workflow(value):
    """Internal helper for the normalize_workflow step."""
    key = str(value).strip().lower()
    aliases = {
        "single": "recursive_single_model",
        "single_model": "recursive_single_model",
        "recursive_single": "recursive_single_model",
        "recursive_single_model": "recursive_single_model",
        "multi": "recursive_multi_model",
        "all": "recursive_multi_model",
        "all_models": "recursive_multi_model",
        "multi_model": "recursive_multi_model",
        "recursive_multi": "recursive_multi_model",
        "recursive_multi_model": "recursive_multi_model",
    }
    if key not in aliases:
        raise ValueError("workflow must be one of: single_model, multi_model.")
    return aliases[key]


def _normalize_schedule_mode(value):
    """Internal helper for the normalize_schedule_mode step."""
    key = str(value).strip().lower()
    aliases = {
        "full": "full_schedule",
        "full_schedule": "full_schedule",
        "remaining": "remaining_schedule",
        "remaining_schedule": "remaining_schedule",
    }
    if key not in aliases:
        raise ValueError("schedule_mode must be one of: full_schedule, remaining_schedule.")
    return aliases[key]


def _deep_merge(base, override):
    """Internal helper for the deep_merge step."""
    out = {}
    for key, value in base.items():
        if isinstance(value, dict):
            out[key] = _deep_merge(value, {})
        else:
            out[key] = value
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _safe_name(value):
    """Internal helper for the safe_name step."""
    text = str(value).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "model"
