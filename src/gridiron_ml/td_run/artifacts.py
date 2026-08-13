"""Evaluation artifact policy for TDNet runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


CORE_TABLES = (
    "model_score_matrix",
    "overall_winner_metrics",
    "overall_margin_metrics",
    "overall_vegas_alignment_metrics",
    "margin_diagnostics",
    "winner_breakdown_counts",
)
GAME_PREDICTION_TABLES = ("game_predictions",)
PREDICTION_SANITY_TABLES = ("prediction_sanity",)
WEEKLY_TABLES = (
    "weekly_rmse",
    "weekly_mae",
    "weekly_winner_accuracy",
    "weekly_edge_3_plus_accuracy",
    "weekly_disagreement_accuracy",
    "weekly_against_vegas_edge_3_plus_accuracy",
    "weekly_record_upset_recall",
    "weekly_vegas_alignment_accuracy",
)
BUCKET_TABLES = (
    "model_favorite_bucket_winner_accuracy",
    "model_favorite_bucket_mae",
    "vegas_spread_bucket_winner_accuracy",
    "vegas_spread_bucket_mae",
    "vegas_spread_bucket_rmse",
    "vegas_spread_bucket_upset_recall",
    "vegas_spread_bucket_edge_3_plus_accuracy",
    "vegas_spread_bucket_disagreement_accuracy",
)
CALIBRATION_TABLES = (
    "confidence_bucket_accuracy",
    "confidence_bucket_calibration",
)
ATS_TABLES = (
    "ats_summary",
    "ats_by_edge_bucket",
)
SHAP_TABLES = ("shap_artifacts",)

OPTIONAL_GROUP_LABELS = {
    "weekly_tables": "weekly tables",
    "bucket_tables": "bucket tables",
    "calibration_tables": "calibration tables",
    "ats_tables": "ATS tables",
    "shap": "SHAP",
    "png_plots": "PNG plots",
}


@dataclass(frozen=True)
class ArtifactPolicy:
    """Resolved switches for routine versus deep-dive evaluation artifacts."""

    core_tables: bool = True
    game_predictions: bool = True
    prediction_sanity: bool = True
    shap: bool = False
    shap_summary_plots: bool = False
    shap_bar_plots: bool = False
    weekly_tables: bool = False
    bucket_tables: bool = False
    calibration_tables: bool = False
    ats_tables: bool = False
    png_plots: bool = False

    @classmethod
    def from_config(cls, eval_config: dict[str, Any] | None = None) -> "ArtifactPolicy":
        cfg = dict(eval_config or {})
        artifact_cfg = dict(cfg.get("artifacts", {}) or {})
        shap_cfg = dict(cfg.get("shap", {}) or {})

        if "shap" not in artifact_cfg and "enabled" in shap_cfg:
            artifact_cfg["shap"] = bool(shap_cfg["enabled"])
        if "shap_summary_plots" not in artifact_cfg and "summary_plots" in shap_cfg:
            artifact_cfg["shap_summary_plots"] = bool(shap_cfg["summary_plots"])
        if "shap_bar_plots" not in artifact_cfg and "bar_plots" in shap_cfg:
            artifact_cfg["shap_bar_plots"] = bool(shap_cfg["bar_plots"])

        return cls(
            **{
                key: bool(value)
                for key, value in artifact_cfg.items()
                if key in cls.__dataclass_fields__
            }
        )

    def table_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.core_tables:
            names.extend(CORE_TABLES)
        if self.game_predictions:
            names.extend(GAME_PREDICTION_TABLES)
        if self.prediction_sanity:
            names.extend(PREDICTION_SANITY_TABLES)
        if self.weekly_tables:
            names.extend(WEEKLY_TABLES)
        if self.bucket_tables:
            names.extend(BUCKET_TABLES)
        if self.calibration_tables:
            names.extend(CALIBRATION_TABLES)
        if self.ats_tables:
            names.extend(ATS_TABLES)
        if self.shap:
            names.extend(SHAP_TABLES)
        return tuple(dict.fromkeys(names))

    def enabled_optional_groups(self) -> list[str]:
        return [
            label
            for attr, label in OPTIONAL_GROUP_LABELS.items()
            if bool(getattr(self, attr))
        ]

    def skipped_optional_groups(self) -> list[str]:
        return [
            label
            for attr, label in OPTIONAL_GROUP_LABELS.items()
            if not bool(getattr(self, attr))
        ]


def filter_tables_for_policy(
    tables: dict[str, Any],
    eval_config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return only DataFrame tables enabled by the resolved artifact policy."""

    policy = ArtifactPolicy.from_config(eval_config)
    allowed = set(policy.table_names())
    return {
        name: table
        for name, table in tables.items()
        if name in allowed and isinstance(table, pd.DataFrame)
    }


def print_evaluation_artifact_summary(
    *,
    output_dir: str | Path,
    tables_dir: str | Path,
    eval_config: dict[str, Any] | None = None,
) -> None:
    """Print a concise YARP-style summary of evaluation artifact output."""

    policy = ArtifactPolicy.from_config(eval_config)
    tables_path = Path(tables_dir)
    core_written = sum(
        1 for table_name in CORE_TABLES if (tables_path / f"{table_name}.csv").exists()
    )

    print("Evaluation artifacts written:")
    print(f"- output directory: {Path(output_dir)}")
    print(f"- core tables: {core_written}")
    print(
        f"- game predictions: {'yes' if (tables_path / 'game_predictions.csv').exists() else 'no'}"
    )

    skipped = policy.skipped_optional_groups()
    if skipped:
        print("Skipped optional artifacts:")
        for label in skipped:
            print(f"- {label}")

    enabled = policy.enabled_optional_groups()
    if enabled:
        print("Enabled optional artifacts:")
        for label in enabled:
            print(f"- {label}")
