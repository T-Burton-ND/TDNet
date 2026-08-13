"""Config-driven TDNet run orchestration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.pipeline.build_full_pipeline import run_pipeline
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.td_run.season_vs_vegas import evaluate_models_vs_vegas_season
from gridiron_ml.td_run.training import (
    DEFAULT_MODEL_SPECS,
    ModelRunSpec,
    filter_model_specs,
    train_model_specs,
)
from gridiron_ml.td_run.weekly_report import (
    WeeklyReportBuilder,
    discover_latest_checkpoints,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TDRun:
    """Own the high-level TDNet workflows behind notebooks and scripts."""

    def __init__(self, config: dict[str, Any] | str | Path | None = None):
        self.config, self.config_path = load_td_run_config(config)
        self.root = resolve_run_path(
            self.config.get("root", PROJECT_ROOT),
            base=self.config_path.parent if self.config_path else PROJECT_ROOT,
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "TDRun":
        """Build a run orchestrator from a YAML config."""

        return cls(config_path)

    def selected_model_specs(
        self, config: dict[str, Any] | None = None
    ) -> list[ModelRunSpec]:
        """Resolve the configured model families and names from the catalog."""

        cfg = dict(config or self.config.get("models", {}) or {})
        family_values = cfg.get("families", "all")
        families = normalize_filter(family_values)
        names = normalize_filter(cfg.get("names", "all"))
        exclude = set(normalize_filter(cfg.get("exclude", [])) or [])
        specs = cfg.get("catalog", DEFAULT_MODEL_SPECS)

        selected = filter_model_specs(
            specs,
            train_stat=families is None or "stat" in families,
            train_linear=families is None or "linear" in families,
            train_tree=families is None or "tree" in families,
            train_knn=families is None or "knn" in families,
            only_names=names,
        )
        if exclude:
            selected = [spec for spec in selected if spec.name not in exclude]
        return selected

    def run_data_pipeline(
        self, config_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Run the configured raw-data/team-table/fingerprint pipeline."""

        pipeline_cfg = dict(self.config.get("pipeline", {}) or {})
        if not bool(pipeline_cfg.get("enabled", True)):
            return {"status": "skipped", "reason": "pipeline.enabled is false"}

        path = config_path or pipeline_cfg.get("config_path")
        if path is None:
            raise ValueError("TDRun pipeline config requires `pipeline.config_path`.")
        return run_pipeline(resolve_run_path(path, base=self.root))

    def train_models(self, specs: list[ModelRunSpec] | None = None):
        """Train the configured model set and return a TrainingResult."""

        training_cfg = dict(self.config.get("training", {}) or {})
        if not bool(training_cfg.get("enabled", True)):
            return None

        specs = specs if specs is not None else self.selected_model_specs()
        fp_cfg = self._fingerprint_config()
        return train_model_specs(
            specs,
            project_root=self.root,
            fingerprint_version=fp_cfg["version"],
            postseason=fp_cfg["postseason"],
            train_years=required_years(training_cfg, "train_years"),
            val_years=training_cfg.get("val_years", []),
            test_years=training_cfg.get("test_years", []),
            matchup_config=self._matchup_config(),
            models_root=self._models_root(),
            build_fingerprints=bool(
                training_cfg.get("build_fingerprints", fp_cfg.get("build", True))
            ),
            overwrite_fingerprints=bool(
                training_cfg.get(
                    "overwrite_fingerprints",
                    fp_cfg.get("overwrite", False),
                )
            ),
            clear_existing_model_artifacts=bool(
                training_cfg.get("clear_existing_model_artifacts", True)
            ),
        )

    def evaluate_latest_checkpoints(self) -> dict[str, Any]:
        """Evaluate the newest selected checkpoints for the configured seasons."""

        eval_cfg = dict(self.config.get("evaluation", {}) or {})
        if not bool(eval_cfg.get("enabled", True)):
            return {"status": "skipped", "reason": "evaluation.enabled is false"}

        model_cfg = merge_dicts(
            self.config.get("models", {}), eval_cfg.get("models", {})
        )
        include = model_filter_for_checkpoints(model_cfg)
        exclude = normalize_filter(model_cfg.get("exclude", [])) or []
        output_root = resolve_run_path(
            eval_cfg.get("output_dir", "data/comparisons/latest_checkpoints"),
            base=self.root,
        )
        output_root.mkdir(parents=True, exist_ok=True)

        model_entries, checkpoint_inventory = discover_latest_checkpoints(
            models_root=self._models_root(model_cfg),
            include_models=include,
            exclude_models=exclude,
        )
        if not model_entries:
            raise FileNotFoundError(
                f"No selected checkpoints found under {self._models_root(model_cfg)}."
            )

        fp_cfg = self._fingerprint_config()
        fingerprints = Fingerprints(
            version=fp_cfg["version"],
            postseason=fp_cfg["postseason"],
            root=self.root,
        )
        if bool(eval_cfg.get("build_fingerprints", fp_cfg.get("build", False))):
            fingerprints.build(
                overwrite=bool(eval_cfg.get("overwrite_fingerprints", False))
            )

        matchup_builder = MatchupBuilder(**self._matchup_config())
        seasons = required_years(eval_cfg, "seasons")
        eval_config_path = eval_cfg.get(
            "eval_config_path", "configs/eval/model_vs_vegas.yaml"
        )
        season_tables = {}
        for season in seasons:
            season_tables[int(season)] = evaluate_models_vs_vegas_season(
                fingerprints=fingerprints,
                matchup_builder=matchup_builder,
                season=int(season),
                model_specs=model_entries,
                output_dir=output_root / str(int(season)),
                target_column=eval_cfg.get("target_column", "y_next_margin"),
                make_plots=bool(eval_cfg.get("make_plots", True)),
                eval_config=eval_cfg.get("eval_config"),
                eval_config_path=resolve_run_path(eval_config_path, base=self.root),
            )

        poll_tables = {}
        poll_cfg = dict(eval_cfg.get("polls", {}) or {})
        if bool(poll_cfg.get("enabled", True)):
            evaluator = TDEval(
                {
                    "fingerprints": fp_cfg,
                    "matchup": self._matchup_config(),
                    "model": {"family": model_entries[0]["family"]},
                },
                fingerprints=fingerprints,
                matchup_builder=matchup_builder,
                model=model_entries[0]["model"],
            )
            for season in seasons:
                poll_tables[int(season)] = evaluator.build_weekly_poll_outputs(
                    models=[entry["model"] for entry in model_entries],
                    season=int(season),
                    weeks=poll_cfg.get("weeks", range(0, 17)),
                    top_n=int(poll_cfg.get("top_n", 25)),
                    average_scope=poll_cfg.get("average_scope", "season"),
                    output_dir=output_root / str(int(season)) / "polls",
                    logo_dir=resolve_run_path(
                        poll_cfg.get("logo_dir", "data/meta/logos/by_team"),
                        base=self.root,
                    ),
                    manual_ballots=poll_cfg.get("manual_ballots"),
                    merge_existing=bool(poll_cfg.get("merge_existing", True)),
                )

        return {
            "output_dir": output_root,
            "checkpoint_inventory": checkpoint_inventory,
            "model_entries": model_entries,
            "season_tables": season_tables,
            "poll_tables": poll_tables,
        }

    def build_weekly_blog(self) -> dict[str, Any]:
        """Build the configured weekly blog package."""

        blog_cfg = dict(self.config.get("blog", {}) or {})
        if not bool(blog_cfg.get("enabled", True)):
            return {"status": "skipped", "reason": "blog.enabled is false"}

        fp_cfg = self._fingerprint_config()
        builder = WeeklyReportBuilder(
            root=self.root,
            fingerprint_version=fp_cfg["version"],
            postseason=fp_cfg["postseason"],
            matchup_config=self._matchup_config(blog_cfg.get("matchup")),
            models_root=self._models_root(blog_cfg.get("models")),
            output_root=resolve_run_path(
                blog_cfg.get(
                    "output_root",
                    f"publication/{required_int(blog_cfg, 'season')}/weekly_predictions",
                ),
                base=self.root,
            ),
            logo_dir=resolve_run_path(
                blog_cfg.get("logo_dir", "data/meta/logos/by_team"),
                base=self.root,
            ),
        )
        return builder.run(
            season=required_int(blog_cfg, "season"),
            target_week=required_int(blog_cfg, "target_week"),
            include_models=blog_cfg.get("include_models", "all"),
            exclude_models=blog_cfg.get("exclude_models", []),
            top_n=int(blog_cfg.get("top_n", 25)),
            average_scope=blog_cfg.get("average_scope", "season"),
            manual_ballots=blog_cfg.get("manual_ballots"),
            team_a=blank_to_none(blog_cfg.get("team_a")),
            team_b=blank_to_none(blog_cfg.get("team_b")),
            scheduled_only=bool(blog_cfg.get("scheduled_only", True)),
            default_total=float(blog_cfg.get("default_total", 52.5)),
            rebuild_poll_history=bool(blog_cfg.get("rebuild_poll_history", False)),
        )

    def _fingerprint_config(self) -> dict[str, Any]:
        cfg = dict(self.config.get("fingerprints", {}) or {})
        return {
            "version": int(cfg.get("version", 0)),
            "postseason": bool(cfg.get("postseason", False)),
            "root": str(self.root),
            "build": bool(cfg.get("build", False)),
            "overwrite": bool(cfg.get("overwrite", False)),
        }

    def _matchup_config(
        self, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return merge_dicts(
            {"representation": "unit_matchup", "safe_math": True},
            self.config.get("matchup", {}),
            overrides or {},
        )

    def _models_root(self, model_cfg: dict[str, Any] | None = None) -> Path:
        cfg = dict(model_cfg or self.config.get("models", {}) or {})
        return resolve_run_path(cfg.get("root", "models"), base=self.root)


def load_td_run_config(
    config: dict[str, Any] | str | Path | None
) -> tuple[dict[str, Any], Path | None]:
    """Load a TDRun config mapping and remember its source path."""

    if config is None:
        return {}, None
    if isinstance(config, dict):
        return deepcopy(config), None
    path = Path(config).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}, path


def resolve_run_path(value: str | Path, *, base: str | Path) -> Path:
    """Resolve project-relative paths from run configs."""

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base) / path).resolve()


def normalize_filter(value) -> list[str] | None:
    """Normalize all/list/scalar config filters."""

    if value in (None, "all"):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else None
    out = [str(item).strip() for item in value if str(item).strip()]
    return out or None


def model_filter_for_checkpoints(model_cfg: dict[str, Any]):
    names = normalize_filter(model_cfg.get("names", "all"))
    if names is not None:
        return names
    families = normalize_filter(model_cfg.get("families", "all"))
    if families is None:
        return "all"
    selected = [
        spec.name
        for spec in filter_model_specs(
            DEFAULT_MODEL_SPECS,
            train_stat="stat" in families,
            train_linear="linear" in families,
            train_tree="tree" in families,
        )
    ]
    return selected or []


def required_years(config: dict[str, Any], key: str) -> list[int]:
    values = config.get(key)
    if values is None:
        raise ValueError(f"TDRun config requires `{key}`.")
    return [int(value) for value in values]


def required_int(config: dict[str, Any], key: str) -> int:
    if config.get(key) is None:
        raise ValueError(f"TDRun config requires `{key}`.")
    return int(config[key])


def merge_dicts(*values: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        merged.update(dict(value or {}))
    return merged


def blank_to_none(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def result_table(value) -> pd.DataFrame:
    """Coerce common TDRun result objects to a displayable dataframe."""

    if isinstance(value, pd.DataFrame):
        return value
    if value is None:
        return pd.DataFrame()
    return pd.DataFrame(value)
