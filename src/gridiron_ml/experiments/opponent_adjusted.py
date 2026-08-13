"""Opponent-adjusted fingerprint experiment pipeline.

This module is intentionally experiment-scoped. It builds alternate TDNet
fingerprint frames from the existing v0 artifact, appends leak-safe
``opp_adj_*`` features, and runs the current model catalog against each
fingerprint variant without promoting these variants into the normal fingerprint
registry yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import re
import shutil
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.fingerprints.features import DEFAULT_FEATURE_SPEC, split_frame
from gridiron_ml.models import build_model_from_config
from gridiron_ml.pipeline.contracts.features import (
    DEFAULT_TRAINING_TARGET,
    FINGERPRINT_KEY_COLUMNS,
    HAS_NEXT_GAME_COLUMN,
)
from gridiron_ml.pipeline.validation.leakage import assert_default_target_is_next_margin
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.td_run.poll_viz import build_weekly_poll_outputs
from gridiron_ml.td_run.season_vs_vegas import evaluate_models_vs_vegas_season
from gridiron_ml.td_run.training import (
    DEFAULT_MODEL_SPECS,
    ModelRunSpec,
    checkpoint_path,
    model_run_dir,
)


DEFAULT_EXPERIMENT_NAME = "opponent_adjusted_fingerprints"
DEFAULT_TRAIN_YEARS = tuple(range(2010, 2024))
DEFAULT_VAL_YEARS = (2024,)
DEFAULT_TEST_YEARS = (2025,)
DEFAULT_ALL_YEARS = tuple(range(2010, 2026))

DEFAULT_ADJUSTED_STAT_COLUMNS = (
    "offense_ppa",
    "offense_success_rate",
    "offense_explosiveness",
    "offense_power_success",
    "offense_stuff_rate",
    "offense_passing_plays_ppa",
    "offense_passing_plays_success_rate",
    "offense_rushing_plays_ppa",
    "offense_rushing_plays_success_rate",
    "defense_ppa",
    "defense_success_rate",
    "defense_explosiveness",
    "defense_power_success",
    "defense_stuff_rate",
    "defense_havoc_rate",
    "statOff_yards_per_pass",
    "statOff_yards_per_rush_attempt",
    "statGen_turnovers",
    "target_points_for",
    "target_points_against",
    "target_team_margin",
)

EXPERIMENT_EVAL_CONFIG = {
    "artifacts": {
        "core_tables": True,
        "game_predictions": True,
        "prediction_sanity": True,
        "weekly_tables": True,
        "bucket_tables": True,
        "calibration_tables": True,
        "ats_tables": True,
        "shap": False,
        "png_plots": False,
    },
    "probability": {"margin_temperature": 7.0},
    "plotting": {"dpi": 150},
}


@dataclass(frozen=True)
class OpponentAdjustedVersionSpec:
    """One opponent-adjusted fingerprint variant."""

    label: str
    method: str
    description: str

    @property
    def safe_label(self) -> str:
        return safe_label(self.label)

    @property
    def subversion(self) -> int:
        digits = re.sub(r"\D", "", self.label)
        return int(digits) if digits else 0


DEFAULT_VERSION_SPECS = (
    OpponentAdjustedVersionSpec(
        "v1.1",
        "opponent_context",
        "Raw stat residuals against the opponent's prior context.",
    ),
    OpponentAdjustedVersionSpec(
        "v1.2",
        "opponent_ridge",
        "Least-squares/ridge schedule residuals using opponent and venue effects.",
    ),
    OpponentAdjustedVersionSpec(
        "v1.3",
        "joint_ridge",
        "Joint team/opponent ridge residuals for adjusted performance.",
    ),
    OpponentAdjustedVersionSpec(
        "v1.4",
        "elo_context",
        "Paired-comparison/Elo schedule-strength adjusted contributions.",
    ),
    OpponentAdjustedVersionSpec(
        "v1.5",
        "graph_context",
        "Graph centrality schedule-strength adjusted contributions.",
    ),
    OpponentAdjustedVersionSpec(
        "v1.6",
        "dynamic_ridge",
        "Recency-weighted ridge approximation to dynamic opponent adjustment.",
    ),
    OpponentAdjustedVersionSpec(
        "v1.7",
        "ensemble_average",
        "Average of v1.1 through v1.6 adjusted feature families.",
    ),
)


@dataclass(frozen=True)
class ExperimentFrameArtifact:
    """Persisted fingerprint frame for one experiment version."""

    spec: OpponentAdjustedVersionSpec
    frame: pd.DataFrame
    path: Path
    game_contribution_path: Path | None = None


class StaticFrameFingerprints:
    """A Fingerprints-compatible wrapper around an in-memory experiment frame."""

    def __init__(self, frame: pd.DataFrame, postseason: bool = False):
        self._frame = pd.DataFrame(frame).copy()
        self.postseason = bool(postseason)

    def frame(
        self,
        seasons=None,
        season=None,
        week=None,
        team=None,
        columns=None,
    ):
        out = self._frame.copy()
        if not self.postseason and "keys_season_type" in out.columns:
            out = out.loc[out["keys_season_type"].astype(str).str.lower().eq("regular")]
        if season is not None:
            seasons = [season]
        if seasons is not None:
            season_set = {int(value) for value in seasons}
            out = out.loc[
                pd.to_numeric(out["keys_season"], errors="coerce").isin(season_set)
            ].copy()
        if week is not None:
            out = out.loc[
                pd.to_numeric(out["keys_week"], errors="coerce") == int(week)
            ].copy()
        if team is not None:
            team_norm = normalize_team(team)
            out = out.loc[
                out["keys_team"].astype(str).map(normalize_team) == team_norm
            ].copy()
        if columns is not None:
            keep = [col for col in columns if col in out.columns]
            out = out.loc[:, keep].copy()
        return out.reset_index(drop=True)

    def split_frame(self, frame, feature_spec=None):
        return split_frame(frame, feature_spec or DEFAULT_FEATURE_SPEC)

    def training_block(self, years, feature_spec=None):
        spec = feature_spec or DEFAULT_FEATURE_SPEC
        assert_default_target_is_next_margin(spec.target_column)
        frame = self.frame(seasons=years)
        if HAS_NEXT_GAME_COLUMN in frame.columns:
            frame = frame.loc[frame[HAS_NEXT_GAME_COLUMN].fillna(False)].copy()
        elif DEFAULT_TRAINING_TARGET in frame.columns:
            frame = frame.loc[
                pd.to_numeric(frame[DEFAULT_TRAINING_TARGET], errors="coerce").notna()
            ].copy()
        return self.split_frame(frame, feature_spec=spec)

    def prediction_block(self, season, predict_week, scheduled_only=False):
        source_week = int(predict_week) - 1
        frame = self.frame(season=season, week=source_week)
        if scheduled_only and "next_week" in frame.columns:
            frame = frame.loc[
                pd.to_numeric(frame["next_week"], errors="coerce")
                == int(predict_week)
            ].copy()
        x_df, _, meta_df, market_df = self.split_frame(frame)
        return x_df, meta_df, market_df

    def team_fingerprint(self, team, season, week):
        frame = self.frame(season=season, week=week, team=team)
        if frame.empty:
            raise ValueError(
                f"No fingerprint rows found for team={team!r}, season={season}, week={week}."
            )
        return self.split_frame(frame.iloc[[0]].copy())

    def season_snapshot(self, season, week):
        frame = self.frame(season=season, week=week)
        if "keys_team" in frame.columns:
            frame = frame.sort_values(["keys_team"]).drop_duplicates(
                subset=["keys_team"], keep="first"
            )
        return self.split_frame(frame.reset_index(drop=True))

    def average_team(self, season=None, years=None, scope="season"):
        scope = str(scope).strip().lower()
        if scope not in {"season", "all_time"}:
            raise ValueError("scope must be one of: 'season', 'all_time'.")
        if scope == "season":
            if season is None and not years:
                raise ValueError("season average requires season= or years=")
            frame = self.frame(season=season) if season is not None else self.frame(seasons=years)
        else:
            frame = self.frame(seasons=years) if years is not None else self.frame()
        x_df, _, _, _ = self.split_frame(frame)
        if x_df.empty:
            raise ValueError("average_team source frame is empty.")
        avg = pd.DataFrame([x_df.mean(axis=0, numeric_only=True)], columns=x_df.columns)
        avg.index = [0]
        return avg


def build_opponent_adjusted_experiment_frames(
    *,
    project_root: str | Path,
    output_root: str | Path | None = None,
    seasons: tuple[int, ...] | list[int] = DEFAULT_ALL_YEARS,
    version_specs: tuple[OpponentAdjustedVersionSpec, ...] = DEFAULT_VERSION_SPECS,
    stat_columns: tuple[str, ...] | list[str] = DEFAULT_ADJUSTED_STAT_COLUMNS,
    overwrite: bool = False,
) -> dict[str, ExperimentFrameArtifact]:
    """Build and persist all opponent-adjusted experiment fingerprint frames."""

    root = Path(project_root).resolve()
    output_root = Path(output_root or default_output_root(root)).resolve()
    frames_root = output_root / "fingerprints"
    frames_root.mkdir(parents=True, exist_ok=True)

    specs = tuple(version_specs)
    base_specs = tuple(spec for spec in specs if spec.method != "ensemble_average")
    ensemble_specs = tuple(spec for spec in specs if spec.method == "ensemble_average")

    v0 = Fingerprints(version=0, postseason=False, root=root).frame(seasons=seasons)
    v0 = v0.sort_values(list(FINGERPRINT_KEY_COLUMNS)).reset_index(drop=True)
    games = load_team_game_tables(root, seasons=seasons, stat_columns=stat_columns)
    available_stats = tuple(col for col in stat_columns if col in games.columns)
    if not available_stats:
        raise ValueError("None of the requested adjusted stat columns exist in team-game tables.")

    artifacts: dict[str, ExperimentFrameArtifact] = {}
    rolled_by_label: dict[str, pd.DataFrame] = {}
    game_contrib_by_label: dict[str, pd.DataFrame] = {}

    for spec in base_specs:
        print(f"Building adjusted fingerprint {spec.label} ({spec.method})...")
        path = frames_root / spec.safe_label / "canonical_fingerprint.parquet"
        game_path = frames_root / spec.safe_label / "adjusted_game_contributions.parquet"
        if path.exists() and game_path.exists() and not overwrite:
            frame = pd.read_parquet(path)
            artifacts[spec.label] = ExperimentFrameArtifact(
                spec=spec,
                frame=frame,
                path=path,
                game_contribution_path=game_path,
            )
            rolled_by_label[spec.label] = frame.loc[
                :, [c for c in frame.columns if c.startswith("opp_adj_")]
            ].copy()
            print(f"Loaded adjusted fingerprint {spec.label}: {frame.shape[0]} rows, {frame.shape[1]} columns.")
            continue

        game_contrib = build_method_game_contributions(
            games=games,
            stat_columns=available_stats,
            spec=spec,
        )
        rolled = roll_adjusted_game_contributions(game_contrib, spec=spec)
        frame = merge_adjusted_features(v0, rolled, spec=spec)
        write_frame_artifacts(frame, path, metadata=metadata_payload(spec, available_stats))
        game_path.parent.mkdir(parents=True, exist_ok=True)
        game_contrib.to_parquet(game_path, index=False)
        artifacts[spec.label] = ExperimentFrameArtifact(
            spec=spec,
            frame=frame,
            path=path,
            game_contribution_path=game_path,
        )
        rolled_by_label[spec.label] = rolled.loc[
            :, [c for c in rolled.columns if c.startswith("opp_adj_")]
        ].copy()
        game_contrib_by_label[spec.label] = game_contrib
        print(f"Wrote adjusted fingerprint {spec.label}: {frame.shape[0]} rows, {frame.shape[1]} columns.")

    for spec in ensemble_specs:
        print(f"Building adjusted fingerprint {spec.label} ({spec.method})...")
        path = frames_root / spec.safe_label / "canonical_fingerprint.parquet"
        if path.exists() and not overwrite:
            frame = pd.read_parquet(path)
            artifacts[spec.label] = ExperimentFrameArtifact(spec=spec, frame=frame, path=path)
            print(f"Loaded adjusted fingerprint {spec.label}: {frame.shape[0]} rows, {frame.shape[1]} columns.")
            continue
        if len(artifacts) < 2:
            raise ValueError("Ensemble version requires at least two base method frames.")
        rolled = average_adjusted_feature_tables(
            [
                artifacts[base_spec.label].frame
                for base_spec in base_specs
                if base_spec.label in artifacts
            ],
            spec=spec,
        )
        frame = merge_adjusted_features(v0, rolled, spec=spec)
        write_frame_artifacts(frame, path, metadata=metadata_payload(spec, available_stats))
        artifacts[spec.label] = ExperimentFrameArtifact(spec=spec, frame=frame, path=path)
        print(f"Wrote adjusted fingerprint {spec.label}: {frame.shape[0]} rows, {frame.shape[1]} columns.")

    manifest_path = output_root / "fingerprint_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiment": DEFAULT_EXPERIMENT_NAME,
                "built_at": datetime.now(timezone.utc).isoformat(),
                "seasons": [int(season) for season in seasons],
                "stat_columns": list(available_stats),
                "versions": {
                    label: {
                        "method": artifact.spec.method,
                        "description": artifact.spec.description,
                        "path": str(artifact.path),
                        "game_contribution_path": (
                            str(artifact.game_contribution_path)
                            if artifact.game_contribution_path
                            else None
                        ),
                        "rows": int(len(artifact.frame)),
                        "columns": int(artifact.frame.shape[1]),
                    }
                    for label, artifact in artifacts.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifacts


def run_opponent_adjusted_sweep(
    *,
    project_root: str | Path,
    output_root: str | Path | None = None,
    train_years: tuple[int, ...] | list[int] = DEFAULT_TRAIN_YEARS,
    val_years: tuple[int, ...] | list[int] = DEFAULT_VAL_YEARS,
    test_years: tuple[int, ...] | list[int] = DEFAULT_TEST_YEARS,
    version_specs: tuple[OpponentAdjustedVersionSpec, ...] = DEFAULT_VERSION_SPECS,
    model_specs: tuple[ModelRunSpec, ...] | list[ModelRunSpec] = DEFAULT_MODEL_SPECS,
    overwrite_fingerprints: bool = False,
    clear_output_root: bool = False,
    keep_checkpoints: bool = True,
) -> dict[str, Any]:
    """Train/evaluate every selected model over every adjusted fingerprint."""

    root = Path(project_root).resolve()
    output_root = Path(output_root or default_output_root(root)).resolve()
    model_specs = tuple(ModelRunSpec.from_mapping(spec) for spec in model_specs)
    if clear_output_root and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_years = tuple(
        sorted({int(year) for year in [*train_years, *val_years, *test_years]})
    )
    artifacts = build_opponent_adjusted_experiment_frames(
        project_root=root,
        output_root=output_root,
        seasons=all_years,
        version_specs=tuple(version_specs),
        overwrite=overwrite_fingerprints,
    )

    matchup_builder = MatchupBuilder(representation="unit_matchup")
    rows = []
    failures = []

    for spec in version_specs:
        artifact = artifacts[spec.label]
        fingerprints = StaticFrameFingerprints(artifact.frame)
        version_root = output_root / "runs" / spec.safe_label
        models_root = version_root / "models"

        for raw_model_spec in model_specs:
            model_spec = ModelRunSpec.from_mapping(raw_model_spec)
            combo_root = version_root / model_spec.family / model_spec.name
            print(f"Running {spec.label} / {model_spec.family}:{model_spec.name}...")
            status = "success"
            error = None
            model = None
            metrics_row: dict[str, Any] = {
                "fingerprint": spec.label,
                "fingerprint_method": spec.method,
                "model": model_spec.name,
                "family": model_spec.family,
            }
            try:
                model = build_model_from_config(
                    {
                        "family": model_spec.family,
                        "config_path": str(root / model_spec.config_path),
                    }
                )
                evaluator = TDEval(
                    config={
                        "model": {
                            "family": model_spec.family,
                            "config_path": str(root / model_spec.config_path),
                        },
                        "eval": {
                            "train_years": list(train_years),
                            "test_years": list(test_years),
                            "artifact_root": str(combo_root / "train_artifacts"),
                        },
                    },
                    fingerprints=fingerprints,
                    matchup_builder=matchup_builder,
                    model=model,
                )
                model = evaluator.train(train_years=train_years, val_years=val_years)
                checkpoint = model.save(checkpoint_path(model_spec, models_root=models_root))
                _, metrics_df = evaluator.evaluate(years=test_years, label="test")
                train_artifact_root = evaluator.save_outputs(combo_root / "train_artifacts")
                metrics_row.update(first_row(metrics_df))
                metrics_row["checkpoint_path"] = str(checkpoint)
                metrics_row["train_artifact_root"] = str(train_artifact_root)

                vegas_tables = evaluate_models_vs_vegas_season(
                    fingerprints=fingerprints,
                    matchup_builder=matchup_builder,
                    season=int(test_years[-1]),
                    model_specs=[{"name": model_spec.name, "model": model}],
                    output_dir=combo_root / "season_eval",
                    make_plots=False,
                    eval_config=EXPERIMENT_EVAL_CONFIG,
                )
                metrics_row.update(extract_vegas_metrics(vegas_tables, model_spec.name))

                poll_weeks = available_weeks_for_poll(
                    artifact.frame,
                    season=int(test_years[-1]),
                )
                poll_tables = build_weekly_poll_outputs(
                    evaluator=TDEval(
                        config={"model": {"family": model_spec.family}},
                        fingerprints=fingerprints,
                        matchup_builder=matchup_builder,
                        model=model,
                    ),
                    models=[model],
                    season=int(test_years[-1]),
                    weeks=poll_weeks,
                    top_n=25,
                    average_scope="season",
                    output_dir=combo_root / "polls",
                    logo_dir=root / "data" / "meta" / "logos" / "by_team",
                    eval_config=EXPERIMENT_EVAL_CONFIG,
                )
                metrics_row["poll_top25_rows"] = int(
                    len(poll_tables.get("weekly_poll_top25", pd.DataFrame()))
                )
                metrics_row["poll_skipped_weeks"] = int(
                    len(poll_tables.get("weekly_poll_skipped_weeks", pd.DataFrame()))
                )
                metrics_row["poll_figure"] = str(
                    combo_root / "polls" / "plots" / "weekly_poll_top25_table.png"
                )

                if not keep_checkpoints and Path(checkpoint).exists():
                    Path(checkpoint).unlink()
                    metrics_row["checkpoint_path"] = None
            except Exception as exc:
                status = "failed"
                error = str(exc)
                print(f"FAILED {spec.label} / {model_spec.family}:{model_spec.name}: {error}")
                failures.append(
                    {
                        "fingerprint": spec.label,
                        "fingerprint_method": spec.method,
                        "model": model_spec.name,
                        "family": model_spec.family,
                        "error": error,
                        "traceback": traceback.format_exc(),
                    }
                )
            metrics_row["status"] = status
            metrics_row["error"] = error
            rows.append(metrics_row)
            write_incremental_summary(output_root, rows, failures)
            if status == "success":
                print(f"Finished {spec.label} / {model_spec.family}:{model_spec.name}.")

    summary = pd.DataFrame(rows)
    failure_df = pd.DataFrame(failures)
    tables_dir = output_root / "summary" / "tables"
    figures_dir = output_root / "summary" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "model_fingerprint_summary.csv", index=False)
    failure_df.to_csv(tables_dir / "failures.csv", index=False)
    save_summary_tables(summary, tables_dir)
    save_summary_figures(summary, figures_dir)
    write_run_manifest(
        output_root=output_root,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        version_specs=version_specs,
        model_specs=model_specs,
        summary=summary,
    )
    return {
        "output_root": output_root,
        "summary": summary,
        "failures": failure_df,
        "fingerprints": artifacts,
    }


def build_method_game_contributions(
    *,
    games: pd.DataFrame,
    stat_columns: tuple[str, ...],
    spec: OpponentAdjustedVersionSpec,
) -> pd.DataFrame:
    """Create one adjusted contribution row per team-game for one method."""

    rows = []
    games = games.sort_values(["keys_season", "keys_week", "keys_game_id", "keys_team"]).reset_index(drop=True)
    cutoffs = (
        games.loc[:, ["keys_season", "keys_week"]]
        .drop_duplicates()
        .sort_values(["keys_season", "keys_week"])
        .itertuples(index=False, name=None)
    )
    ratings_cache: dict[tuple[int, int, str], dict[str, float]] = {}
    for season, week in cutoffs:
        season = int(season)
        week = int(week)
        current = games.loc[
            (games["keys_season"] == season) & (games["keys_week"] == week)
        ].copy()
        history = games.loc[
            (games["keys_season"] < season)
            | ((games["keys_season"] == season) & (games["keys_week"] < week))
        ].copy()
        if current.empty:
            continue

        if spec.method == "opponent_context":
            contrib = opponent_context_contributions(current, history, stat_columns)
        elif spec.method == "opponent_ridge":
            contrib = ridge_contributions(
                current,
                history,
                stat_columns,
                include_team=False,
                alpha=1.0,
                weighted=False,
            )
        elif spec.method == "joint_ridge":
            contrib = ridge_contributions(
                current,
                history,
                stat_columns,
                include_team=True,
                alpha=25.0,
                weighted=False,
            )
        elif spec.method == "dynamic_ridge":
            contrib = ridge_contributions(
                current,
                history,
                stat_columns,
                include_team=True,
                alpha=35.0,
                weighted=True,
                cutoff_season=season,
                cutoff_week=week,
            )
        elif spec.method == "elo_context":
            ratings = ratings_cache.get((season, week, "elo"))
            if ratings is None:
                ratings = elo_ratings(history)
                ratings_cache[(season, week, "elo")] = ratings
            contrib = rating_context_contributions(
                current,
                history,
                stat_columns,
                ratings=ratings,
                rating_name="elo",
            )
        elif spec.method == "graph_context":
            ratings = ratings_cache.get((season, week, "graph"))
            if ratings is None:
                ratings = graph_centrality_ratings(history)
                ratings_cache[(season, week, "graph")] = ratings
            contrib = rating_context_contributions(
                current,
                history,
                stat_columns,
                ratings=ratings,
                rating_name="graph",
            )
        else:
            raise ValueError(f"Unsupported opponent-adjusted method: {spec.method}")

        keys = current.loc[
            :,
            [
                "keys_season",
                "keys_week",
                "keys_game_id",
                "keys_team",
                "keys_opponent",
                "game_is_home",
            ],
        ].reset_index(drop=True)
        out = pd.concat([keys, contrib.reset_index(drop=True)], axis=1)
        out["fingerprint"] = spec.label
        out["method"] = spec.method
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def opponent_context_contributions(
    current: pd.DataFrame,
    history: pd.DataFrame,
    stat_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Residualize observations against opponent prior means."""

    out = pd.DataFrame(index=current.index)
    history_means = (
        history.groupby("keys_team", observed=True)[list(stat_columns)].mean(numeric_only=True)
        if not history.empty
        else pd.DataFrame()
    )
    global_means = history.loc[:, list(stat_columns)].mean(numeric_only=True)
    for stat in stat_columns:
        counterpart = counterpart_stat(stat, stat_columns)
        global_value = float(global_means.get(counterpart, 0.0))
        if not np.isfinite(global_value):
            global_value = 0.0
        if not history_means.empty and counterpart in history_means.columns:
            expected = current["keys_opponent"].map(history_means[counterpart])
        else:
            expected = pd.Series(np.nan, index=current.index)
        expected = pd.to_numeric(expected, errors="coerce").fillna(global_value)
        observed = pd.to_numeric(current[stat], errors="coerce")
        out[stat] = observed - expected
    return out


def ridge_contributions(
    current: pd.DataFrame,
    history: pd.DataFrame,
    stat_columns: tuple[str, ...],
    *,
    include_team: bool,
    alpha: float,
    weighted: bool,
    cutoff_season: int | None = None,
    cutoff_week: int | None = None,
) -> pd.DataFrame:
    """Multi-output ridge residuals for one cutoff week."""

    out = pd.DataFrame(index=current.index)
    if len(history) < 20:
        return opponent_context_contributions(current, history, stat_columns)

    train_X = ridge_design(history, include_team=include_team)
    pred_X = ridge_design(current, include_team=include_team)
    all_cols = sorted(set(train_X.columns) | set(pred_X.columns))
    train_X = train_X.reindex(columns=all_cols, fill_value=0.0)
    pred_X = pred_X.reindex(columns=all_cols, fill_value=0.0)

    y = history.loc[:, list(stat_columns)].apply(pd.to_numeric, errors="coerce")
    medians = y.median(numeric_only=True).fillna(0.0)
    y_fit = y.fillna(medians)
    sample_weight = None
    if weighted and cutoff_season is not None and cutoff_week is not None:
        age = (
            (int(cutoff_season) - pd.to_numeric(history["keys_season"], errors="coerce"))
            * 20.0
            + (int(cutoff_week) - pd.to_numeric(history["keys_week"], errors="coerce"))
        )
        sample_weight = np.exp(-np.maximum(age.to_numpy(dtype=float), 0.0) / 18.0)
        sample_weight = np.clip(sample_weight, 0.05, 1.0)

    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(train_X.to_numpy(dtype=float), y_fit.to_numpy(dtype=float), sample_weight=sample_weight)
    pred = pd.DataFrame(
        model.predict(pred_X.to_numpy(dtype=float)),
        columns=list(stat_columns),
        index=current.index,
    )
    for stat in stat_columns:
        out[stat] = pd.to_numeric(current[stat], errors="coerce") - pred[stat]
    return out


def rating_context_contributions(
    current: pd.DataFrame,
    history: pd.DataFrame,
    stat_columns: tuple[str, ...],
    *,
    ratings: dict[str, float],
    rating_name: str,
) -> pd.DataFrame:
    """Use a team-strength rating as schedule context for stat residuals."""

    out = pd.DataFrame(index=current.index)
    global_means = history.loc[:, list(stat_columns)].mean(numeric_only=True)
    global_stds = history.loc[:, list(stat_columns)].std(numeric_only=True).replace(0.0, 1.0).fillna(1.0)
    rating_values = pd.Series(ratings, dtype=float)
    scale = float(rating_values.std()) if len(rating_values) > 1 else 1.0
    if not np.isfinite(scale) or math.isclose(scale, 0.0):
        scale = 1.0
    opp_rating = current["keys_opponent"].map(ratings).fillna(0.0).astype(float)
    opp_z = opp_rating / scale

    if "target_team_margin" in stat_columns:
        team_rating = current["keys_team"].map(ratings).fillna(0.0).astype(float)
        home = current["game_is_home"].astype(float).fillna(0.0)
        expected_margin = ((team_rating - opp_rating) / scale) * 7.0 + home * 2.5
    else:
        expected_margin = pd.Series(0.0, index=current.index)

    for stat in stat_columns:
        observed = pd.to_numeric(current[stat], errors="coerce")
        if stat == "target_team_margin":
            out[stat] = observed - expected_margin
            continue
        baseline = float(global_means.get(stat, 0.0))
        stat_scale = float(global_stds.get(stat, 1.0))
        out[stat] = observed - baseline + 0.10 * opp_z * stat_scale
    out[f"{rating_name}_team_rating"] = current["keys_team"].map(ratings).fillna(0.0).astype(float)
    out[f"{rating_name}_opponent_rating"] = opp_rating
    out[f"{rating_name}_rating_edge"] = (
        out[f"{rating_name}_team_rating"] - out[f"{rating_name}_opponent_rating"]
    )
    return out


def roll_adjusted_game_contributions(
    game_contrib: pd.DataFrame,
    *,
    spec: OpponentAdjustedVersionSpec,
    alpha: float = 0.45,
) -> pd.DataFrame:
    """Roll game-level adjusted contributions into team-week fingerprint rows."""

    if game_contrib.empty:
        return pd.DataFrame(columns=list(FINGERPRINT_KEY_COLUMNS))
    base_cols = [
        "keys_season",
        "keys_team",
        "keys_week",
        "keys_opponent",
        "keys_game_id",
        "game_is_home",
        "fingerprint",
        "method",
    ]
    value_cols = [
        col
        for col in game_contrib.columns
        if col not in base_cols and pd.api.types.is_numeric_dtype(game_contrib[col])
    ]
    frame = game_contrib.sort_values(["keys_season", "keys_team", "keys_week", "keys_game_id"]).copy()
    out = frame.loc[:, list(FINGERPRINT_KEY_COLUMNS)].copy()
    prefix = f"opp_adj_{spec.safe_label}"

    grouped = frame.groupby(["keys_season", "keys_team"], sort=False, observed=True)
    out[f"{prefix}_games_played"] = grouped.cumcount() + 1
    unique_counts = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby(["keys_season", "keys_team"], sort=False, observed=True):
        seen: set[str] = set()
        counts = []
        for opponent in group["keys_opponent"].astype(str):
            seen.add(opponent)
            counts.append(float(len(seen)))
        unique_counts.loc[group.index] = counts
    out[f"{prefix}_unique_opponents"] = unique_counts.to_numpy(dtype=float)
    for col in value_cols:
        safe_col = safe_label(col)
        values = pd.to_numeric(frame[col], errors="coerce")
        out[f"{prefix}_{safe_col}_mean_to_date"] = (
            values.groupby([frame["keys_season"], frame["keys_team"]], sort=False)
            .expanding(1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        out[f"{prefix}_{safe_col}_last3"] = (
            values.groupby([frame["keys_season"], frame["keys_team"]], sort=False)
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        out[f"{prefix}_{safe_col}_ewm"] = grouped[col].transform(
            lambda s: pd.to_numeric(s, errors="coerce").ewm(alpha=alpha, adjust=False, min_periods=1).mean()
        )
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return (
        out.sort_values(["keys_season", "keys_team", "keys_week"])
        .drop_duplicates(list(FINGERPRINT_KEY_COLUMNS), keep="last")
        .reset_index(drop=True)
    )


def average_adjusted_feature_tables(
    frames: list[pd.DataFrame],
    *,
    spec: OpponentAdjustedVersionSpec,
) -> pd.DataFrame:
    """Average aligned adjusted feature families into an ensemble version."""

    if not frames:
        raise ValueError("At least one source frame is required for the ensemble.")
    key = list(FINGERPRINT_KEY_COLUMNS)
    source_tables = []
    for frame in frames:
        source_cols = [col for col in frame.columns if col.startswith("opp_adj_")]
        table = frame.loc[:, key + source_cols].copy()
        rename = {}
        for col in source_cols:
            suffix = re.sub(r"^opp_adj_v\d+_\d+_", "", col)
            if suffix != col:
                rename[col] = suffix
        table = table.rename(columns=rename)
        source_tables.append(table)

    base = source_tables[0].loc[:, key].copy()
    suffixes = sorted(
        set().union(
            *[
                {col for col in table.columns if col not in key}
                for table in source_tables
            ]
        )
    )
    prefix = f"opp_adj_{spec.safe_label}"
    for suffix in suffixes:
        pieces = [
            pd.to_numeric(table[suffix], errors="coerce")
            for table in source_tables
            if suffix in table.columns
        ]
        if not pieces:
            continue
        base[f"{prefix}_{suffix}"] = pd.concat(pieces, axis=1).mean(axis=1)
    return base


def merge_adjusted_features(
    base_frame: pd.DataFrame,
    adjusted: pd.DataFrame,
    *,
    spec: OpponentAdjustedVersionSpec,
) -> pd.DataFrame:
    """Merge adjusted features onto the baseline v0 fingerprint frame."""

    key = list(FINGERPRINT_KEY_COLUMNS)
    frame = base_frame.copy()
    drop_cols = [col for col in frame.columns if col.startswith("opp_adj_")]
    if drop_cols:
        frame = frame.drop(columns=drop_cols)
    out = frame.merge(adjusted, on=key, how="left", validate="one_to_one")
    out["fp_version_label"] = spec.label
    out["fp_method"] = spec.method
    out["fp_subversion"] = spec.subversion
    out["fp_experiment"] = DEFAULT_EXPERIMENT_NAME
    return out


def ridge_design(frame: pd.DataFrame, *, include_team: bool) -> pd.DataFrame:
    pieces = []
    if include_team:
        pieces.append(pd.get_dummies(frame["keys_team"].astype(str), prefix="team", dtype=float))
    pieces.append(pd.get_dummies(frame["keys_opponent"].astype(str), prefix="opp", dtype=float))
    home = pd.to_numeric(frame["game_is_home"], errors="coerce").fillna(0.0).astype(float)
    pieces.append(pd.DataFrame({"home_field": home.to_numpy(dtype=float)}, index=frame.index))
    return pd.concat(pieces, axis=1)


def elo_ratings(history: pd.DataFrame) -> dict[str, float]:
    """Sequential Elo ratings from completed historical games."""

    if history.empty:
        return {}
    ratings: dict[str, float] = {}
    games = one_row_per_game(history)
    for row in games.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        home_margin = float(row.home_margin) if pd.notna(row.home_margin) else 0.0
        h = ratings.get(home, 0.0)
        a = ratings.get(away, 0.0)
        expected = 1.0 / (1.0 + 10.0 ** (-((h - a) + 55.0) / 400.0))
        actual = 1.0 if home_margin > 0 else 0.5 if math.isclose(home_margin, 0.0) else 0.0
        margin_mult = min(2.5, max(0.75, math.log(abs(home_margin) + 1.0)))
        delta = 20.0 * margin_mult * (actual - expected)
        ratings[home] = h + delta
        ratings[away] = a - delta
    return ratings


def graph_centrality_ratings(history: pd.DataFrame) -> dict[str, float]:
    """Small PageRank-style team graph rating from completed historical games."""

    if history.empty:
        return {}
    games = one_row_per_game(history)
    teams = sorted(set(games["home_team"].astype(str)) | set(games["away_team"].astype(str)))
    if not teams:
        return {}
    idx = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    matrix = np.zeros((n, n), dtype=float)
    for row in games.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        margin = float(row.home_margin) if pd.notna(row.home_margin) else 0.0
        weight = 1.0 + min(abs(margin), 35.0) / 14.0
        if margin >= 0:
            loser, winner = away, home
        else:
            loser, winner = home, away
        matrix[idx[loser], idx[winner]] += weight
        if abs(margin) <= 7.0:
            matrix[idx[winner], idx[loser]] += 0.25
    row_sums = matrix.sum(axis=1)
    dangling = row_sums == 0
    matrix[~dangling] = matrix[~dangling] / row_sums[~dangling, None]
    matrix[dangling] = 1.0 / n
    scores = np.full(n, 1.0 / n)
    damping = 0.85
    teleport = np.full(n, (1.0 - damping) / n)
    for _ in range(50):
        scores = teleport + damping * matrix.T.dot(scores)
    scores = (scores - scores.mean()) / (scores.std() if scores.std() > 0 else 1.0)
    return {team: float(scores[i]) for team, i in idx.items()}


def one_row_per_game(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    if "target_team_margin" not in frame.columns:
        frame["target_team_margin"] = (
            pd.to_numeric(frame["target_points_for"], errors="coerce")
            - pd.to_numeric(frame["target_points_against"], errors="coerce")
        )
    home = frame.loc[frame["game_is_home"].astype(bool)].copy()
    away = frame.loc[~frame["game_is_home"].astype(bool)].copy()
    keep = ["keys_season", "keys_week", "keys_game_id", "keys_team", "target_team_margin"]
    home = home.loc[:, [col for col in keep if col in home.columns]].rename(
        columns={"keys_team": "home_team", "target_team_margin": "home_margin"}
    )
    away = away.loc[:, [col for col in keep if col in away.columns]].rename(
        columns={"keys_team": "away_team"}
    )
    join_cols = [col for col in ["keys_season", "keys_week", "keys_game_id"] if col in home.columns and col in away.columns]
    if not join_cols:
        return pd.DataFrame(columns=["home_team", "away_team", "home_margin"])
    out = home.merge(away, on=join_cols, how="inner", validate="one_to_one")
    return out.loc[:, ["home_team", "away_team", "home_margin"]].dropna(subset=["home_team", "away_team"])


def load_team_game_tables(
    project_root: Path,
    *,
    seasons: tuple[int, ...] | list[int],
    stat_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Load canonical team-game tables for selected seasons."""

    frames = []
    table_root = project_root / "data" / "team_game_tables"
    for season in seasons:
        parquet_path = table_root / f"team_game_table_{int(season)}_fbs.parquet"
        csv_path = table_root / f"team_game_table_{int(season)}_fbs.csv"
        if parquet_path.exists():
            frame = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            frame = pd.read_csv(csv_path)
        else:
            continue
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No team-game tables found under {table_root}.")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if "keys_season_type" in frame.columns:
        frame = frame.loc[frame["keys_season_type"].astype(str).str.lower().eq("regular")].copy()
    frame["keys_season"] = pd.to_numeric(frame["keys_season"], errors="coerce").astype("Int64")
    frame["keys_week"] = pd.to_numeric(frame["keys_week"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["keys_season", "keys_week", "keys_team", "keys_opponent"])
    frame["keys_season"] = frame["keys_season"].astype(int)
    frame["keys_week"] = frame["keys_week"].astype(int)
    if "game_is_home" in frame.columns:
        if pd.api.types.is_bool_dtype(frame["game_is_home"]):
            frame["game_is_home"] = frame["game_is_home"].astype(bool)
        else:
            text_home = frame["game_is_home"].astype(str).str.lower().str.strip()
            frame["game_is_home"] = text_home.isin({"1", "true", "t", "yes", "y", "home"})
    elif "game_home_away" in frame.columns:
        frame["game_is_home"] = frame["game_home_away"].astype(str).str.lower().eq("home")
    else:
        frame["game_is_home"] = False
    keep = [
        "keys_season",
        "keys_week",
        "keys_game_id",
        "keys_team",
        "keys_opponent",
        "game_is_home",
        *[col for col in stat_columns if col in frame.columns],
    ]
    frame = frame.loc[:, list(dict.fromkeys(keep))].copy()
    for col in stat_columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = (
        frame.sort_values(["keys_season", "keys_week", "keys_game_id", "keys_team"])
        .drop_duplicates(["keys_season", "keys_week", "keys_game_id", "keys_team"], keep="first")
        .reset_index(drop=True)
    )
    return frame


def counterpart_stat(stat: str, stat_columns: tuple[str, ...]) -> str:
    if stat.startswith("offense_"):
        candidate = "defense_" + stat[len("offense_") :]
        if candidate in stat_columns:
            return candidate
    if stat.startswith("defense_"):
        candidate = "offense_" + stat[len("defense_") :]
        if candidate in stat_columns:
            return candidate
    if stat == "target_points_for" and "target_points_against" in stat_columns:
        return "target_points_against"
    if stat == "target_points_against" and "target_points_for" in stat_columns:
        return "target_points_for"
    return stat


def write_frame_artifacts(frame: pd.DataFrame, path: Path, *, metadata: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    (path.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def metadata_payload(
    spec: OpponentAdjustedVersionSpec, stat_columns: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "artifact_kind": "opponent_adjusted_experiment_fingerprint",
        "experiment": DEFAULT_EXPERIMENT_NAME,
        "version": spec.label,
        "method": spec.method,
        "description": spec.description,
        "base_fingerprint_version": 0,
        "stat_columns": list(stat_columns),
        "default_target": DEFAULT_TRAINING_TARGET,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_vegas_metrics(tables: dict[str, pd.DataFrame], source: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    margin = tables.get("overall_margin_metrics", pd.DataFrame())
    winner = tables.get("overall_winner_metrics", pd.DataFrame())
    score = tables.get("model_score_matrix", pd.DataFrame())
    for table_name, table in [("season_margin", margin), ("season_winner", winner)]:
        if table.empty or "metric" not in table.columns or source not in table.columns:
            continue
        for row in table.itertuples(index=False):
            metric = str(getattr(row, "metric"))
            value = getattr(row, source)
            out[f"{table_name}_{metric}"] = value
    if not score.empty:
        model_score = score.loc[score["source"].astype(str) == str(source)]
        if not model_score.empty:
            for col, value in model_score.iloc[0].to_dict().items():
                if col not in {"source", "source_type"}:
                    out[f"score_{col}"] = value
    return out


def save_summary_tables(summary: pd.DataFrame, tables_dir: Path):
    if summary.empty:
        return
    success = summary.loc[summary["status"].eq("success")].copy()
    if success.empty:
        return
    metric_cols = [
        col
        for col in [
            "mae",
            "rmse",
            "winner_accuracy",
            "favorite_correct",
            "upset_correct",
            "season_margin_mae",
            "season_margin_rmse",
            "season_winner_winner_accuracy",
            "season_winner_upset_recall",
            "score_total_score",
        ]
        if col in success.columns
    ]
    if metric_cols:
        by_fp = success.groupby(["fingerprint", "fingerprint_method"], as_index=False)[metric_cols].mean(numeric_only=True)
        by_model = success.groupby(["family", "model"], as_index=False)[metric_cols].mean(numeric_only=True)
        by_fp.to_csv(tables_dir / "summary_by_fingerprint.csv", index=False)
        by_model.to_csv(tables_dir / "summary_by_model.csv", index=False)
    pivot_metric(success, "season_margin_mae").to_csv(tables_dir / "mae_model_by_fingerprint.csv")
    pivot_metric(success, "season_margin_rmse").to_csv(tables_dir / "rmse_model_by_fingerprint.csv")
    pivot_metric(success, "season_winner_upset_recall").to_csv(tables_dir / "upset_recall_model_by_fingerprint.csv")


def save_summary_figures(summary: pd.DataFrame, figures_dir: Path):
    success = summary.loc[summary["status"].eq("success")].copy() if not summary.empty else pd.DataFrame()
    if success.empty:
        return
    heatmap_specs = [
        ("season_margin_mae", "2025 MAE by Model and Fingerprint", "mae_heatmap.png"),
        ("season_margin_rmse", "2025 RMSE by Model and Fingerprint", "rmse_heatmap.png"),
        (
            "season_winner_upset_recall",
            "2025 Upset Recall by Model and Fingerprint",
            "upset_recall_heatmap.png",
        ),
        ("score_total_score", "Composite Score by Model and Fingerprint", "score_heatmap.png"),
    ]
    for metric, title, filename in heatmap_specs:
        table = pivot_metric(success, metric)
        if table.empty:
            continue
        plot_heatmap(table, figures_dir / filename, title=title)

    if "season_margin_mae" in success.columns:
        grouped = (
            success.groupby("fingerprint", as_index=False)
            .agg(
                mae=("season_margin_mae", "mean"),
                rmse=("season_margin_rmse", "mean"),
                upset_recall=("season_winner_upset_recall", "mean"),
                score=("score_total_score", "mean"),
            )
            .sort_values("fingerprint")
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(grouped["fingerprint"], grouped["mae"], marker="o", label="MAE")
        if "rmse" in grouped:
            ax.plot(grouped["fingerprint"], grouped["rmse"], marker="s", label="RMSE")
        ax.set_title("Average Margin Error by Fingerprint")
        ax.set_xlabel("Fingerprint")
        ax.set_ylabel("Points")
        ax.grid(True, color="#DDE2E7", linewidth=0.8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "fingerprint_margin_error_trends.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def pivot_metric(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in summary.columns:
        return pd.DataFrame()
    table = summary.pivot_table(
        index=["family", "model"],
        columns="fingerprint",
        values=metric,
        aggfunc="mean",
    )
    return table.sort_index(axis=0).sort_index(axis=1)


def plot_heatmap(table: pd.DataFrame, path: Path, *, title: str):
    values = table.to_numpy(dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return
    fig_width = max(8.0, 1.1 * table.shape[1] + 4.0)
    fig_height = max(6.0, 0.35 * table.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([f"{fam}/{model}" for fam, model in table.index], fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_incremental_summary(output_root: Path, rows: list[dict], failures: list[dict]):
    tables_dir = output_root / "summary" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables_dir / "model_fingerprint_summary.partial.csv", index=False)
    pd.DataFrame(failures).to_csv(tables_dir / "failures.partial.csv", index=False)


def write_run_manifest(
    *,
    output_root: Path,
    train_years,
    val_years,
    test_years,
    version_specs,
    model_specs,
    summary: pd.DataFrame,
):
    payload = {
        "experiment": DEFAULT_EXPERIMENT_NAME,
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
        "train_years": [int(year) for year in train_years],
        "val_years": [int(year) for year in val_years],
        "test_years": [int(year) for year in test_years],
        "versions": [
            {"label": spec.label, "method": spec.method, "description": spec.description}
            for spec in version_specs
        ],
        "models": [
            {"name": spec.name, "family": spec.family, "config_path": spec.config_path}
            for spec in model_specs
        ],
        "successful_runs": int(summary["status"].eq("success").sum()) if "status" in summary else 0,
        "failed_runs": int(summary["status"].eq("failed").sum()) if "status" in summary else 0,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def available_weeks_for_poll(frame: pd.DataFrame, *, season: int) -> list[int]:
    season_frame = frame.loc[pd.to_numeric(frame["keys_season"], errors="coerce") == int(season)].copy()
    weeks = pd.to_numeric(season_frame["keys_week"], errors="coerce").dropna().astype(int)
    return sorted(weeks.unique().tolist())


def first_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def safe_label(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_team(value: object) -> str:
    s = str(value).casefold()
    return "".join(ch for ch in s if ch.isalnum())


def default_output_root(project_root: Path) -> Path:
    return project_root / "data" / "experiments" / DEFAULT_EXPERIMENT_NAME
