#!/usr/bin/env python
"""Rolling backtest for the best finalized winner model."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = project_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.finalize_fingerprint_hyperparameter_search import load_selected_frame, parse_json_mapping
from gridiron_ml.experiments.hyperparameter_search import get_model_class
from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--search-root", type=Path, default=Path("data/experiments/fingerprint_hyperparameter_search"))
    parser.add_argument("--source-fingerprint-root", type=Path, default=Path("data/experiments/opponent_adjusted_fingerprints"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/fingerprint_hyperparameter_search/backtests/winner_random_forest"))
    parser.add_argument("--start-season", type=int, default=2015)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--min-train-season", type=int, default=2010)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.project_root.resolve()
    search_root = resolve(root, args.search_root)
    source_root = resolve(root, args.source_fingerprint_root)
    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(search_root / "winner" / "final_artifacts" / "selected_best_by_model.csv")
    row = (
        selected.loc[(selected["family"].astype(str) == "tree") & (selected["model"].astype(str) == "random_forest")]
        .iloc[0]
        .to_dict()
    )
    frame = load_selected_frame(row=row, source_root=source_root)
    metrics = []
    predictions = []
    for season in range(int(args.start_season), int(args.end_season) + 1):
        train_years = tuple(range(int(args.min_train_season), season))
        if not train_years:
            continue
        print(f"BACKTEST winner/tree/random_forest train={train_years[0]}-{train_years[-1]} eval={season}", flush=True)
        fingerprints = StaticFrameFingerprints(frame)
        config = parse_json_mapping(row.get("model_config_json"))
        config["model_name"] = f"winner_tree_random_forest_backtest_{season}"
        config["name"] = config["model_name"]
        model = get_model_class("tree")(config)
        evaluator = TDEval(
            config={
                "model": {"family": "tree", **config},
                "eval": {"train_years": list(train_years), "test_years": [season], "artifact_root": str(output_dir / f"season_{season}")},
            },
            fingerprints=fingerprints,
            matchup_builder=MatchupBuilder(representation="unit_matchup"),
            model=model,
        )
        evaluator.train(train_years=train_years, val_years=())
        pred, metric = evaluator.evaluate(years=[season], label=f"eval_{season}")
        metric.insert(0, "eval_season", season)
        metric.insert(1, "train_years_json", json.dumps(list(train_years)))
        metrics.append(metric)
        pred.insert(0, "eval_season", season)
        predictions.append(pred)

    metrics_df = pd.concat(metrics, ignore_index=True, sort=False) if metrics else pd.DataFrame()
    pred_df = pd.concat(predictions, ignore_index=True, sort=False) if predictions else pd.DataFrame()
    metrics_df.to_csv(output_dir / "rolling_backtest_metrics.csv", index=False)
    pred_df.to_csv(output_dir / "rolling_backtest_predictions.csv", index=False)
    print(f"Metrics: {output_dir / 'rolling_backtest_metrics.csv'}")
    print(f"Predictions: {output_dir / 'rolling_backtest_predictions.csv'}")


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
