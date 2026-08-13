"""Command-line entrypoint for config-driven TDNet workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat
from typing import Any

import yaml

from gridiron_ml.td_run.td_run import TDRun


WORKFLOWS = ("pipeline", "training", "evaluation", "blog")


def run_tdnet_config(
    config_path: str | Path, workflows: list[str] | None = None
) -> dict[str, Any]:
    """Run selected TDNet workflows from one YAML config."""

    runner = TDRun.from_config(config_path)
    selected = workflows or infer_workflows(runner.config)
    if not selected:
        raise ValueError("No workflows selected and no enabled workflow blocks found.")

    print("TDNet run")
    print(f"Config: {runner.config_path}")
    print(f"Root  : {runner.root}")
    print("Workflows:", ", ".join(selected))
    print("Resolved configuration:")
    print(pformat(runner.config, sort_dicts=False))

    results: dict[str, Any] = {}
    for workflow in selected:
        print("=" * 72)
        print(f"Starting {workflow}")
        if workflow == "pipeline":
            results[workflow] = runner.run_data_pipeline()
        elif workflow == "training":
            results[workflow] = runner.train_models()
        elif workflow == "evaluation":
            results[workflow] = runner.evaluate_latest_checkpoints()
        elif workflow == "blog":
            results[workflow] = runner.build_weekly_blog()
        else:
            raise ValueError(f"Unknown TDNet workflow: {workflow}")
        print(f"Finished {workflow}")

    print("=" * 72)
    print("TDNet run complete.")
    return results


def infer_workflows(config: dict[str, Any]) -> list[str]:
    """Infer enabled workflows from config blocks."""

    workflows: list[str] = []
    if bool((config.get("pipeline", {}) or {}).get("enabled", False)):
        workflows.append("pipeline")
    if bool((config.get("training", {}) or {}).get("enabled", False)):
        workflows.append("training")
    if bool((config.get("evaluation", {}) or {}).get("enabled", False)):
        workflows.append("evaluation")
    if bool((config.get("blog", {}) or {}).get("enabled", False)):
        workflows.append("blog")
    return workflows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse TDNet run CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run TDNet workflows from a YAML config."
    )
    parser.add_argument(
        "config", type=Path, help="Path to a configs/td_run/*.yaml file."
    )
    parser.add_argument(
        "--workflow",
        choices=WORKFLOWS,
        action="append",
        help="Workflow to run. Repeat to run multiple. Defaults to enabled config blocks.",
    )
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print the resolved YAML config and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the TDNet CLI."""

    args = parse_args(argv)
    if args.print_config_only:
        runner = TDRun.from_config(args.config)
        print(yaml.dump(runner.config, sort_keys=False))
        return
    run_tdnet_config(args.config, workflows=args.workflow)


if __name__ == "__main__":
    main()
