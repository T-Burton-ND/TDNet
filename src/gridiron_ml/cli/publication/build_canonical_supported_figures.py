#!/usr/bin/env python3
"""Generate every publication figure currently supported by canonical 2025 data."""
from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = project_root()
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.publication.figures import PublicationFigureBuilder  # noqa: E402


def canonical_predictions(path: Path, consensus_root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["game_id"] = frame["keys_game_id"]
    frame["model_name"] = frame["model_role"].astype(str) + "|" + frame["fingerprint_id"].astype(str)
    frame["season"] = pd.to_numeric(frame.get("season", frame.get("keys_season", 2025)), errors="coerce")
    frame["week"] = pd.to_numeric(frame.get("keys_week", frame.get("next_week", 0)), errors="coerce")
    frame["pred_home_win_probability"] = pd.to_numeric(frame["pred_probability_home"], errors="coerce")
    frame["actual_home_win"] = pd.to_numeric(frame["actual_margin"], errors="coerce").gt(0).astype(int)
    frame["pred_winner"] = frame["pred_home_win_probability"].ge(0.5).astype(int)
    frame["home_team"] = frame.get("keys_team_home", "home")
    frame["away_team"] = frame.get("keys_team_away", "away")
    keep = ["game_id", "model_name", "season", "week", "pred_home_win_probability", "actual_home_win", "pred_winner", "home_team", "away_team"]
    products = [frame[keep]]
    for name in ["all_model_consensus", "top10_brier_consensus"]:
        product = pd.read_parquet(consensus_root / f"{name}.parquet").copy()
        product["game_id"] = product["keys_game_id"]
        product["model_name"] = name
        product["pred_home_win_probability"] = product["pred_probability_home"]
        product["actual_home_win"] = pd.to_numeric(product["actual_margin"], errors="coerce").gt(0).astype(int)
        product["pred_winner"] = product["pred_home_win_probability"].ge(0.5).astype(int)
        product["home_team"] = "home"
        product["away_team"] = "away"
        product["season"] = product["season"]
        product["week"] = product.get("keys_week", 0)
        products.append(product[keep])
    return pd.concat(products, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--consensus-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/publication/canonical_2025/supported_figures")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results).rename(columns={"fingerprint_id": "feature_config", "model_role": "model_level", "holdout_season": "test_season"})
    results["objective"] = "margin"
    results["mae"] = results["margin_mae"]
    predictions = canonical_predictions(args.predictions, args.consensus_root)
    builder = PublicationFigureBuilder(args.output_root, strict=False)
    manifest = builder.generate_all(matrix_summary=results, predictions=predictions)
    provenance = {
        "status": "canonical_2025_supported_retrospective_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_results": str(args.results.resolve()),
        "source_predictions": str(args.predictions.resolve()),
        "generated_count": manifest["generated_count"],
        "required_count": manifest["required_count"],
        "generated": manifest["generated"],
        "skipped": manifest["skipped"],
        "note": "Figures requiring pre-2025 learning curves, explicit ablation/negative-control tables, or certified 2026 outcomes remain skipped rather than inferred.",
    }
    (args.output_root / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
