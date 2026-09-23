#!/usr/bin/env python3
"""Build the complete Sunday scorecard and refreshed ranking package."""

from argparse import ArgumentParser
import json
from pathlib import Path

from gridiron_ml.cli._paths import project_root
from gridiron_ml.publication.postgame_full import build_full_postgame_package


ROOT = project_root()


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--completed-week", type=int, required=True)
    parser.add_argument("--state-week", type=int, required=True)
    parser.add_argument("--scored-predictions", type=Path, required=True)
    parser.add_argument("--raw-games", type=Path, required=True)
    parser.add_argument("--margin-poll-dir", type=Path, required=True)
    parser.add_argument("--scientific-poll-dir", type=Path, required=True)
    parser.add_argument("--margin-inventory", type=Path, required=True)
    parser.add_argument("--scientific-inventory", type=Path, required=True)
    parser.add_argument("--reference-poll", type=Path, required=True)
    parser.add_argument("--reference-label", default="Latest available AP Top 25")
    parser.add_argument("--reference-short-label", default="AP")
    parser.add_argument("--api-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logo-dir", type=Path, default=ROOT / "data/meta/logos/by_team")
    args = parser.parse_args()
    manifest = build_full_postgame_package(
        scored_predictions_path=args.scored_predictions,
        raw_games_path=args.raw_games,
        margin_poll_dir=args.margin_poll_dir,
        scientific_poll_dir=args.scientific_poll_dir,
        margin_inventory_path=args.margin_inventory,
        scientific_inventory_path=args.scientific_inventory,
        reference_poll_path=args.reference_poll,
        api_manifest_path=args.api_manifest,
        output_root=args.output_root,
        season=args.season,
        completed_week=args.completed_week,
        state_week=args.state_week,
        logo_dir=args.logo_dir,
        reference_label=args.reference_label,
        reference_short_label=args.reference_short_label,
    )
    print(json.dumps({key: manifest[key] for key in (
        "generated_at_eastern",
        "margin_wide_prediction_models",
        "margin_wide_poll_models",
        "scientific_poll_models",
        "week_1_game_predictions_generated",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
