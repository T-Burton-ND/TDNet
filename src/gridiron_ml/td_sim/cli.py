"""src.gridiron_ml.td_sim.cli.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Run recursive season simulations with evolving synthetic fingerprints.
"""

import argparse

from .orchestrator import TDSimOrchestrator


def parse_args(argv=None):
    """Run the parse_args step and return its normalized result."""
    parser = argparse.ArgumentParser(description="Run recursive TD Sim season simulations.")
    parser.add_argument("--config", default="configs/sim/tdsim_config.yaml")
    parser.add_argument("--season", type=int)
    parser.add_argument("--n-sims", type=int)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--workflow", choices=["single_model", "multi_model"], default=None)
    parser.add_argument("--schedule-mode", choices=["full_schedule", "remaining_schedule"], default=None)
    parser.add_argument("--as-of-week", type=int)
    parser.add_argument("--sim-start", type=int, default=0)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--performance-sampler", choices=["historical", "knn", "hybrid"])
    parser.add_argument("--knn-neighbors", type=int)
    parser.add_argument("--knn-randomness", type=float)
    parser.add_argument("--knn-margin-band", type=float)
    parser.add_argument("--knn-max-candidates", type=int)
    parser.add_argument("--hybrid-knn-weight", type=float)
    parser.add_argument("--progress", dest="show_progress", action="store_true")
    parser.add_argument("--no-progress", dest="show_progress", action="store_false")
    parser.set_defaults(show_progress=None)
    return parser.parse_args(argv)


def main(argv=None):
    """Run the main step and return its normalized result."""
    args = parse_args(argv)
    orchestrator = TDSimOrchestrator(args.config)
    if args.top_n is not None:
        orchestrator.config.setdefault("outputs", {})["top_n_teams"] = int(args.top_n)
    recursive_cfg = orchestrator.config.setdefault("recursive", {})
    for attr, key in [
        ("performance_sampler", "performance_sampler"),
        ("knn_neighbors", "knn_neighbors"),
        ("knn_randomness", "knn_randomness"),
        ("knn_margin_band", "knn_margin_band"),
        ("knn_max_candidates", "knn_max_candidates"),
        ("hybrid_knn_weight", "hybrid_knn_weight"),
    ]:
        value = getattr(args, attr)
        if value is not None:
            recursive_cfg[key] = value
    orchestrator.run(
        season=args.season,
        N=args.n_sims,
        models=args.models,
        workflow=args.workflow,
        schedule_mode=args.schedule_mode,
        as_of_week=args.as_of_week,
        sim_start=args.sim_start,
        shard_id=args.shard_id,
        save_debug=args.save_debug,
        show_progress=args.show_progress,
    )
    print(f"Recursive TD Sim outputs written to {orchestrator.output_dir}")


if __name__ == "__main__":
    main()
