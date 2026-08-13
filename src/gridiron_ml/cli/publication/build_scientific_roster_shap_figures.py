#!/usr/bin/env python3
"""Render the prespecified scientific-roster F6 SHAP figure suite."""

from argparse import ArgumentParser
import json
from pathlib import Path

from gridiron_ml.cli._paths import project_root
from gridiron_ml.publication.scientific_shap_figures import (
    FigureSettings,
    build_scientific_shap_figures,
    load_feature_contract,
    load_study_config,
    read_table,
)


ROOT = project_root()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/publication/scientific_roster_shap_study.yaml",
    )
    parser.add_argument("--importance", type=Path, required=True)
    parser.add_argument("--effects", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config = load_study_config(args.config)
    figure = dict(config.get("figure_generation", {}))
    contract = load_feature_contract(resolve(config["scope"]["source_feature_manifest"]))
    expected = int(config["scope"]["expected_source_feature_count"])
    if len(contract) != expected:
        raise ValueError(f"Expected {expected} F6 features, manifest contains {len(contract)}.")
    settings = FigureSettings(
        features_per_atlas_page=int(figure.get("features_per_atlas_page", 38)),
        features_per_dependence_page=int(figure.get("features_per_dependence_page", 12)),
        raster_dpi=int(figure.get("raster_dpi", 240)),
        minimum_valid_folds=int(config.get("stability", {}).get("minimum_valid_folds_per_model", 8)),
        bootstrap_resamples=int(config.get("stability", {}).get("bootstrap_resamples", 2000)),
        bootstrap_seed=int(config.get("sampling", {}).get("deterministic_seed", 26084)),
    )
    output = args.output_root or resolve(figure["output_root"])
    report = build_scientific_shap_figures(
        importance=read_table(args.importance),
        effects=read_table(args.effects) if args.effects else None,
        feature_contract=contract,
        output_root=output,
        settings=settings,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
