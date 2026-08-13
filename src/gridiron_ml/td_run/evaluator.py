"""src.gridiron_ml.td_run.evaluator.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Evaluate model outputs, compare predictions to market baselines, and build reporting artifacts.
"""

from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gridiron_ml.fingerprints import DEFAULT_FEATURE_SPEC, FeatureSpec, Fingerprints
from gridiron_ml.td_run.market import (
    DEFAULT_VEGAS_CONVENTION,
    market_home_margin,
    normalize_vegas_frame,
)
from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.models import build_model_from_config, model_label
from gridiron_ml.pipeline.schemas import validate_matchup_frame
from gridiron_ml.pipeline.validation.leakage import (
    assert_disjoint_years,
    training_allows_market_features,
)
from .poll_viz import build_weekly_poll_outputs as _build_weekly_poll_outputs
from .season_vs_vegas import (
    evaluate_models_vs_vegas_season as _evaluate_models_vs_vegas_season,
    save_evaluation_plots as _save_evaluation_plots,
    save_metric_tables as _save_metric_tables,
)


class TDEval:
    """Represent the TDEval component and its local behavior."""

    def __init__(
        self, config=None, fingerprints=None, matchup_builder=None, model=None
    ):
        """Internal helper for the init__ step."""
        self.config = self._load_config(config)
        self.fingerprints = fingerprints or self._build_fingerprints()
        self.matchup_builder = matchup_builder or self._build_matchup_builder()
        self.model = model or self._build_model()

        self.metrics_history_ = []
        self.predictions_ = pd.DataFrame()
        self.poll_history_ = []
        self.poll_ballots_ = pd.DataFrame()

    def train(self, train_years=None, val_years=None):
        """Run the train step and return its normalized result."""
        train_years = self._coalesce_years(train_years, key="train_years")
        val_years = self._coalesce_years(val_years, key="test_years")
        assert_disjoint_years(train_years, val_years)
        feature_spec = self._feature_spec()

        X_train_block, y_train_block, meta_train, market_train = (
            self.fingerprints.training_block(train_years, feature_spec=feature_spec)
        )
        X_train, y_train, meta_train, market_train = self._build_matchups(
            X_train_block,
            y_train_block,
            meta_train,
            market_train,
        )

        X_val = None
        y_val = None
        if val_years:
            X_val_block, y_val_block, meta_val, market_val = (
                self.fingerprints.training_block(val_years, feature_spec=feature_spec)
            )
            X_val, y_val, meta_val, market_val = self._build_matchups(
                X_val_block,
                y_val_block,
                meta_val,
                market_val,
            )

        self._assert_training_features_are_safe(X_train, X_val)

        train_kwargs = {"X_val": X_val, "y_val": y_val}
        train_params = signature(self.model.train).parameters
        if "market_train" in train_params:
            train_kwargs["market_train"] = market_train
        if "market_val" in train_params:
            train_kwargs["market_val"] = market_val if val_years else None
        if "meta_train" in train_params:
            train_kwargs["meta_train"] = meta_train
        if "meta_val" in train_params:
            train_kwargs["meta_val"] = meta_val if val_years else None
        self.model.train(X_train, y_train, **train_kwargs)
        return self.model

    def evaluate(
        self, years=None, X=None, y=None, meta_df=None, market_df=None, label="eval"
    ):
        """Run the evaluate step and return its normalized result."""
        if X is None:
            years = self._coalesce_years(years, key="test_years")
            feature_spec = self._feature_spec()
            X_block, y_block, meta_df, market_df = self.fingerprints.training_block(
                years,
                feature_spec=feature_spec,
            )
            X, y, meta_df, market_df = self._build_matchups(
                X_block, y_block, meta_df, market_df
            )

        pred_df = self.model.predict(X, meta_df=meta_df, market_df=market_df)
        result_df = pred_df.copy()
        result_df["y"] = pd.to_numeric(
            pd.Series(y).reset_index(drop=True), errors="coerce"
        )
        metrics = self._metrics_from_result(result_df)
        metrics["label"] = label

        self.predictions_ = result_df
        self.metrics_history_.append(metrics)
        return result_df, pd.DataFrame([metrics])

    def evaluate_vs_vegas_season(
        self, season, output_dir=None, model_name=None, **kwargs
    ):
        """Run the evaluate_vs_vegas_season step and return its normalized result."""
        model_name = model_name or self._model_label(self.model, 0)
        kwargs = self._with_default_eval_config(kwargs)
        return _evaluate_models_vs_vegas_season(
            fingerprints=self.fingerprints,
            matchup_builder=self.matchup_builder,
            season=season,
            model_specs=[{"name": model_name, "model": self.model}],
            output_dir=output_dir,
            **kwargs,
        )

    def evaluate_season_vs_vegas(
        self, season, model_specs=None, models=None, output_dir=None, **kwargs
    ):
        """Run the evaluate_season_vs_vegas step and return its normalized result."""
        kwargs = self._with_default_eval_config(kwargs)
        return _evaluate_models_vs_vegas_season(
            fingerprints=self.fingerprints,
            matchup_builder=self.matchup_builder,
            season=season,
            model_specs=model_specs,
            models=models,
            output_dir=output_dir,
            **kwargs,
        )

    def build_weekly_poll_outputs(
        self,
        models,
        season,
        weeks=range(1, 17),
        top_n=25,
        average_scope="season",
        output_dir=None,
        logo_dir=None,
        eval_config=None,
        manual_ballots=None,
        merge_existing=False,
    ):
        """Run the build_weekly_poll_outputs step and return its normalized result."""
        return _build_weekly_poll_outputs(
            evaluator=self,
            models=models,
            season=season,
            weeks=weeks,
            top_n=top_n,
            average_scope=average_scope,
            output_dir=output_dir,
            logo_dir=logo_dir,
            eval_config=eval_config,
            manual_ballots=manual_ballots,
            merge_existing=merge_existing,
        )

    def save_metric_tables(self, tables, output_dir, eval_config=None):
        """Run the save_metric_tables step and return its normalized result."""
        return _save_metric_tables(tables, output_dir, eval_config=eval_config)

    def save_evaluation_plots(self, tables, output_dir, eval_config=None):
        """Run the save_evaluation_plots step and return its normalized result."""
        return _save_evaluation_plots(tables, output_dir, eval_config=eval_config)

    @staticmethod
    def train_model_specs(*args, **kwargs):
        """Train a set of model specs through the shared eval-layer workflow."""
        from gridiron_ml.td_run.training import train_model_specs

        return train_model_specs(*args, **kwargs)

    def predict(self, X, meta_df=None, market_df=None):
        """Run the predict step and return its normalized result."""
        return self.model.predict(X, meta_df=meta_df, market_df=market_df)

    def predict_block(self, season_week_df, meta_df=None, market_df=None):
        """Run the predict_block step and return its normalized result."""
        return self.model.predict_block(
            season_week_df, meta_df=meta_df, market_df=market_df
        )

    def predict_matchup(self, season, week, home_team, away_team):
        """Run the predict_matchup step and return its normalized result."""
        home_X, _, home_meta, home_market = self.fingerprints.team_fingerprint(
            home_team, season, week
        )
        away_X, _, away_meta, away_market = self.fingerprints.team_fingerprint(
            away_team, season, week
        )

        matchup_X = self.matchup_builder.build_many(home_X, away_X)
        meta_df = pd.DataFrame(
            {
                "keys_season": [int(season)],
                "keys_week": [int(week)],
                "keys_team_home": [
                    (
                        str(home_meta.loc[0, "keys_team"])
                        if "keys_team" in home_meta.columns
                        else str(home_team)
                    )
                ],
                "keys_team_away": [
                    (
                        str(away_meta.loc[0, "keys_team"])
                        if "keys_team" in away_meta.columns
                        else str(away_team)
                    )
                ],
            }
        )

        market_df = pd.DataFrame(index=meta_df.index)
        for market_source in [home_market, away_market]:
            if market_source is None or market_source.empty:
                continue
            keep = [c for c in market_source.columns if c.startswith("market_")]
            keep += [
                c
                for c in ["keys_season", "keys_week", "keys_game_id"]
                if c in market_source.columns
            ]
            market_df = pd.concat(
                [
                    market_df.reset_index(drop=True),
                    market_source.loc[[0], keep].reset_index(drop=True),
                ],
                axis=1,
            )
            market_df = market_df.loc[:, ~market_df.columns.duplicated()].copy()

        return self.model.predict(matchup_X, meta_df=meta_df, market_df=market_df)

    def predict_week(self, season, predict_week, scheduled_only=False):
        """Run the predict_week step and return its normalized result."""
        X_block, meta_df, market_df = self.fingerprints.prediction_block(
            season=season,
            predict_week=predict_week,
            scheduled_only=scheduled_only,
        )
        matchup_X, matchup_meta, matchup_market = self.matchup_builder.matchups(
            X_block,
            meta_df,
            market_df=market_df,
        )
        return self.model.predict_block(
            matchup_X, meta_df=matchup_meta, market_df=matchup_market
        )

    def total_rank(self, season, week, average_scope="season"):
        """Run the total_rank step and return its normalized result."""
        X_week, _, meta_week, market_week = self.fingerprints.season_snapshot(
            season, week
        )
        average_team = self.fingerprints.average_team(
            season=season, scope=average_scope
        )
        X_rank, meta_rank, market_rank = self.matchup_builder.team_vs_average(
            X_week,
            meta_week,
            average_team_df=average_team,
            market_df=market_week,
        )
        return self.model.total_rank(X_rank, meta_df=meta_rank)

    def top25(self, season, week, average_scope="season"):
        """Run the top25 step and return its normalized result."""
        X_week, _, meta_week, market_week = self.fingerprints.season_snapshot(
            season, week
        )
        average_team = self.fingerprints.average_team(
            season=season, scope=average_scope
        )
        X_rank, meta_rank, market_rank = self.matchup_builder.team_vs_average(
            X_week,
            meta_week,
            average_team_df=average_team,
            market_df=market_week,
        )
        return self.model.top25(X_rank, meta_df=meta_rank)

    def poll(
        self,
        models,
        season,
        week,
        average_scope="season",
        top_n=25,
        manual_ballots=None,
    ):
        """Run the poll step and return its normalized result."""
        if not models:
            if manual_ballots is None:
                raise ValueError("poll() requires at least one model or manual ballot.")

        X_week, _, meta_week, market_week = self.fingerprints.season_snapshot(
            season, week
        )
        average_team = self.fingerprints.average_team(
            season=season, scope=average_scope
        )
        X_rank, meta_rank, market_rank = self.matchup_builder.team_vs_average(
            X_week,
            meta_week,
            average_team_df=average_team,
            market_df=market_week,
        )

        ballot_frames = []
        self._last_poll_rank_scores_ = {}
        self.poll_model_failures_ = []
        for idx, model in enumerate(models):
            try:
                rank_df = model.total_rank(X_rank, meta_df=meta_rank).reset_index(drop=True)
                key_col = self._team_key_column(rank_df)
                # Ranking must use an uncapped/appropriate score.  Classifier
                # margins can be capped at +/-30, and neural margin models can
                # expose the same cap; both create large artificial ties that
                # are then resolved by input order (often alphabetically).
                # Use an uncapped fitted score for ranking.  In particular,
                # TDLinear/TDSpline classifiers expose useful logits through
                # _predict_margin_with_pipeline, while predict_proba() calls
                # _predict_margin_array() and therefore applies the public
                # +/-30 prediction cap.  Sorting those capped probabilities
                # creates large artificial ties that fall through to the
                # alphabetical team-key tie-breaker.
                score_values = None
                for candidate in (model, getattr(model, "delegate_", None)):
                    if candidate is None:
                        continue
                    if hasattr(candidate, "_raw_output"):
                        score_values = np.asarray(candidate._raw_output(X_rank), dtype=float).reshape(-1)
                        break
                    if (
                        hasattr(candidate, "_predict_margin_with_pipeline")
                        and getattr(candidate, "pipeline_", None) is not None
                    ):
                        score_values = np.asarray(
                            candidate._predict_margin_with_pipeline(X_rank, candidate.pipeline_), dtype=float
                        ).reshape(-1)
                        break
                if score_values is None and getattr(model, "_is_classifier_objective", lambda: False)():
                    probabilities = np.asarray(model.predict_proba(X_rank), dtype=float)
                    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
                        raise ValueError("classifier did not return two-column probabilities")
                    score_values = probabilities[:, 1]
                if score_values is not None:
                    source_key_col = self._team_key_column(meta_rank)
                    source_keys = meta_rank[source_key_col].astype(str).to_numpy()
                    if len(score_values) != len(source_keys):
                        raise ValueError("ranking score length does not match ranking metadata")
                    score_lookup = pd.Series(score_values, index=source_keys)
                    rank_df["score"] = rank_df[key_col].astype(str).map(score_lookup)
                    rank_df = rank_df.sort_values(
                        ["score", key_col], ascending=[False, True], kind="mergesort"
                    ).reset_index(drop=True)
            except Exception as exc:
                self.poll_model_failures_.append(
                    {"model": self._model_label(model, idx), "reason": str(exc)}
                )
                continue
            if rank_df.empty:
                continue
            score_values = pd.to_numeric(rank_df.get("score"), errors="coerce")
            self._last_poll_rank_scores_[self._model_label(model, idx)] = {
                "rows": int(len(rank_df)),
                "non_null_scores": int(score_values.notna().sum()),
                "unique_scores": int(score_values.dropna().nunique()),
            }
            if score_values.dropna().nunique() <= 1:
                self.poll_model_failures_.append(
                    {
                        "model": self._model_label(model, idx),
                        "reason": "non-informative rank scores (constant or all missing)",
                    }
                )
                continue

            ballot = rank_df.drop_duplicates(subset=[key_col], keep="first").copy()
            ballot["keys_team"] = ballot[key_col].astype(str)
            ballot["ballot_model"] = self._model_label(model, idx)
            ballot["ballot_rank"] = np.arange(1, len(ballot) + 1)
            ballot["poll_points"] = np.maximum(
                int(top_n) + 1 - ballot["ballot_rank"], 0
            )
            ballot["top25_vote"] = ballot["ballot_rank"] <= int(top_n)
            ballot["first_place_vote"] = ballot["ballot_rank"] == 1
            ballot_frames.append(
                ballot.loc[
                    :,
                    [
                        "keys_team",
                        "ballot_model",
                        "ballot_rank",
                        "poll_points",
                        "top25_vote",
                        "first_place_vote",
                    ],
                ]
            )

        ballot_frames.extend(
            self._manual_poll_ballots(
                manual_ballots=manual_ballots,
                season=season,
                week=week,
                top_n=top_n,
            )
        )

        if not ballot_frames:
            raise ValueError("poll() could not build any ranking ballots.")

        ballots_df = pd.concat(ballot_frames, ignore_index=True)
        poll_df = (
            ballots_df.groupby("keys_team", as_index=False)
            .agg(
                poll_points=("poll_points", "sum"),
                ballots_seen=("ballot_model", "nunique"),
                top25_votes=("top25_vote", "sum"),
                first_place_votes=("first_place_vote", "sum"),
                average_rank=("ballot_rank", "mean"),
                best_rank=("ballot_rank", "min"),
                worst_rank=("ballot_rank", "max"),
            )
            .sort_values(
                ["poll_points", "average_rank", "best_rank", "keys_team"],
                ascending=[False, True, True, True],
            )
            .reset_index(drop=True)
        )
        poll_df.insert(0, "rank", np.arange(1, len(poll_df) + 1))
        poll_df["average_rank"] = poll_df["average_rank"].astype(float).round(3)
        self.poll_ballots_ = ballots_df
        self.poll_history_.append(poll_df)
        return poll_df

    def save_outputs(self, output_root=None):
        """Run the save_outputs step and return its normalized result."""
        output_root = Path(output_root or self._output_root())
        output_root.mkdir(parents=True, exist_ok=True)

        if not self.predictions_.empty:
            pred_path = output_root / "predictions" / "predictions.csv"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            self.predictions_.to_csv(pred_path, index=False)

        neighbor_audit = getattr(self.model, "neighbor_audit_", None)
        if neighbor_audit is not None and not pd.DataFrame(neighbor_audit).empty:
            audit_path = output_root / "predictions" / "neighbor_audit.csv"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(neighbor_audit).to_csv(audit_path, index=False)

        if self.metrics_history_:
            metrics_path = output_root / "metrics" / "metrics_summary.csv"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.metrics_history_).to_csv(metrics_path, index=False)

        if self.poll_history_:
            poll_path = output_root / "polls" / "latest_poll.csv"
            poll_path.parent.mkdir(parents=True, exist_ok=True)
            self.poll_history_[-1].to_csv(poll_path, index=False)
        if not self.poll_ballots_.empty:
            ballots_path = output_root / "polls" / "latest_poll_ballots.csv"
            ballots_path.parent.mkdir(parents=True, exist_ok=True)
            self.poll_ballots_.to_csv(ballots_path, index=False)

        if (
            getattr(self.model, "training_history_", None) is not None
            and len(self.model.training_history_) > 0
        ):
            history_path = output_root / "history" / "training_history.csv"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.training_history_.to_csv(history_path, index=False)
            self._save_training_curve(
                self.model.training_history_,
                history_path.parent / "training_curve.png",
            )

        feature_importance = self._model_feature_importance()
        if feature_importance is not None and not feature_importance.empty:
            importance_path = (
                output_root / "feature_importance" / "feature_importance.csv"
            )
            importance_path.parent.mkdir(parents=True, exist_ok=True)
            feature_importance.to_csv(importance_path, index=False)

        return output_root

    def _model_feature_importance(self):
        """Internal helper for the model_feature_importance step."""
        if hasattr(self.model, "get_feature_importance"):
            return pd.DataFrame(self.model.get_feature_importance())
        importance = getattr(self.model, "feature_importances_", None)
        if importance is None:
            return None
        return pd.DataFrame(importance)

    def _save_training_curve(self, history_df, path):
        """Internal helper for the save_training_curve step."""
        history = pd.DataFrame(history_df).copy()
        if history.empty:
            return None

        metric_col = next(
            (
                col
                for col in ["optimized_loss", "total_loss", "rmse", "mae"]
                if col in history.columns
                and pd.to_numeric(history[col], errors="coerce").notna().any()
            ),
            None,
        )
        if metric_col is None:
            return None

        import matplotlib.pyplot as plt

        colors = self._training_curve_colors()
        title_model = (
            getattr(self.model, "model_name", None)
            or getattr(self.model, "model_type", None)
            or "model"
        )
        history["split"] = (
            history.get("split", pd.Series("train", index=history.index))
            .fillna("train")
            .astype(str)
        )
        if "epoch" in history.columns:
            history["step"] = pd.to_numeric(history["epoch"], errors="coerce")
            x_label = "Epoch"
        else:
            history["step"] = history.groupby("split").cumcount() + 1
            x_label = "Step"

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        split_styles = {
            "train": {"color": colors["train"], "linestyle": "-", "marker": "o"},
            "val": {"color": colors["val"], "linestyle": "--", "marker": "s"},
            "validation": {"color": colors["val"], "linestyle": "--", "marker": "s"},
            "test": {"color": colors["aux"], "linestyle": "-.", "marker": "^"},
        }

        for split_name, split_df in history.groupby("split", sort=False):
            series = split_df.loc[:, ["step", metric_col]].copy()
            series["step"] = pd.to_numeric(series["step"], errors="coerce")
            series[metric_col] = pd.to_numeric(series[metric_col], errors="coerce")
            series = series.dropna(subset=["step", metric_col]).sort_values("step")
            if series.empty:
                continue
            style = split_styles.get(
                split_name.lower(),
                {"color": colors["aux"], "linestyle": "-", "marker": "o"},
            )
            ax.plot(
                series["step"],
                series[metric_col],
                label=split_name,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=4.5,
            )

        if not ax.lines:
            plt.close(fig)
            return None

        ylabel = metric_col.replace("_", " ").title()
        ax.set_title(f"{title_model} Training Curve")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, color=colors["grid"], linewidth=0.8, alpha=0.85)
        ax.set_facecolor(colors["panel"])
        for spine in ax.spines.values():
            spine.set_color(colors["spine"])
            spine.set_linewidth(1.0)
        ax.legend(title="Split", frameon=True)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    def _training_curve_colors(self):
        """Internal helper for the training_curve_colors step."""
        repo_root = Path(__file__).resolve().parents[3]
        model_name = (
            str(
                getattr(
                    self.model, "model_name", getattr(self.model, "model_type", "model")
                )
            )
            .strip()
            .lower()
        )
        model_colors_path = (
            repo_root / "style" / "color_palettes" / "tdnet_model_colors.csv"
        )
        palette_path = repo_root / "style" / "color_palettes" / "tdnet_palette.csv"

        train = "#1EA7FF"
        val = "#5FF2D2"
        aux = "#D4B56E"
        grid = "#DDE2E7"
        panel = "#F3F5F8"
        spine = "#3A4450"

        if model_colors_path.exists():
            model_colors = pd.read_csv(model_colors_path)
            match = model_colors.loc[
                model_colors["model"].astype(str).str.strip().str.lower() == model_name,
                "hex",
            ]
            if not match.empty:
                train = str(match.iloc[0])

        if palette_path.exists():
            palette = pd.read_csv(palette_path)
            palette_map = {
                str(row["name"]).strip(): str(row["hex"]).strip()
                for _, row in palette.iterrows()
                if "name" in row and "hex" in row
            }
            val = palette_map.get("Soft Mint", val)
            aux = palette_map.get("Brass", aux)
            grid = palette_map.get("Fog Silver", grid)
            panel = palette_map.get("Mist Panel", panel)
            spine = palette_map.get("Slate Line", spine)

        return {
            "train": train,
            "val": val,
            "aux": aux,
            "grid": grid,
            "panel": panel,
            "spine": spine,
        }

    def _build_matchups(self, X_block, y_block, meta_df, market_df):
        """Internal helper for the build_matchups step."""
        matchups_df, paired_meta, paired_market = self.matchup_builder.matchups(
            X_block,
            meta_df,
            market_df=market_df,
            y=y_block,
        )
        y = (
            pd.to_numeric(paired_meta.pop("y"), errors="coerce")
            if "y" in paired_meta.columns
            else pd.Series(dtype=float)
        )
        return matchups_df, y, paired_meta, paired_market

    def _feature_spec(self):
        """Return the configured fingerprint feature-selection contract."""
        spec_cfg = self.config.get("feature_spec")
        if spec_cfg is None:
            spec_cfg = self.config.get("fingerprints", {}).get("feature_spec")
        if isinstance(spec_cfg, FeatureSpec):
            return spec_cfg
        if not spec_cfg:
            return DEFAULT_FEATURE_SPEC
        allowed = set(FeatureSpec.__dataclass_fields__)
        values = {key: value for key, value in dict(spec_cfg).items() if key in allowed}
        return FeatureSpec(**values)

    def _metrics_from_result(self, result_df):
        """Internal helper for the metrics_from_result step."""
        result_df = normalize_vegas_frame(result_df, DEFAULT_VEGAS_CONVENTION)
        y_true = pd.to_numeric(result_df["y"], errors="coerce")
        y_pred = pd.to_numeric(result_df["pred_margin"], errors="coerce")
        model_pick_home = y_pred > 0
        actual_home_win = y_true > 0
        model_correct = model_pick_home == actual_home_win

        if hasattr(self.model, "loss_breakdown"):
            metrics = self.model.loss_breakdown(y_pred, y_true)
        else:
            metrics = {
                "loss_function": getattr(self.model, "loss_function", None),
                "total_loss": np.nan,
                "margin_loss": np.nan,
                "win_probability_loss": np.nan,
                "favorite_correctness_loss": np.nan,
                "calibration_loss": np.nan,
                "mae": float(np.nanmean(np.abs(y_true - y_pred))),
                "rmse": float(np.sqrt(np.nanmean((y_true - y_pred) ** 2))),
                "winner_accuracy": float(model_correct.mean()),
                "brier_score": np.nan,
            }
        for key, value in self._model_favorite_bucket_metrics(y_pred, y_true).items():
            metrics.setdefault(key, value)

        favorite_correct = np.nan
        upset_correct = np.nan
        market_rmse = np.nan
        market_mae = np.nan
        ats_accuracy = np.nan
        ats_n = 0

        market_margin = market_home_margin(result_df, DEFAULT_VEGAS_CONVENTION)
        market_pick_home = market_margin > 0
        line_mask = market_margin.notna() & y_true.notna()
        if line_mask.any():
            market_rmse = float(
                np.sqrt(np.nanmean((market_margin[line_mask] - y_true[line_mask]) ** 2))
            )
            market_mae = float(
                np.nanmean(np.abs(market_margin[line_mask] - y_true[line_mask]))
            )

            favorite_won = market_pick_home[line_mask] == actual_home_win[line_mask]
            model_correct_masked = model_correct[line_mask]
            if (~favorite_won).any():
                upset_correct = float(model_correct_masked[~favorite_won].mean())
            if favorite_won.any():
                favorite_correct = float(model_correct_masked[favorite_won].mean())

            model_edge = y_pred[line_mask] - market_margin[line_mask]
            actual_edge = y_true[line_mask] - market_margin[line_mask]
            ats_mask = model_edge.ne(0.0) & actual_edge.ne(0.0)
            ats_n = int(ats_mask.sum())
            if ats_n:
                ats_accuracy = float(
                    ((model_edge[ats_mask] > 0) == (actual_edge[ats_mask] > 0)).mean()
                )

        metrics["bias"] = float(np.nanmean(y_pred - y_true))
        metrics["accuracy"] = float(model_correct.mean())
        metrics["favorite_correct"] = favorite_correct
        metrics["upset_correct"] = upset_correct
        metrics["market_rmse"] = market_rmse
        metrics["market_mae"] = market_mae
        metrics["ats_accuracy"] = ats_accuracy
        metrics["ats_n"] = ats_n
        metrics["n_rows"] = int(len(result_df))
        return metrics

    def _model_favorite_bucket_metrics(self, y_pred, y_true):
        """Internal helper for the model_favorite_bucket_metrics step."""
        pred = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy(dtype=float)
        actual = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(pred) & np.isfinite(actual)
        pred = pred[valid]
        actual = actual[valid]
        abs_margin = np.abs(pred)
        correct = ((pred > 0) == (actual > 0)).astype(float)
        buckets = [
            ("0_3", 0.0, 3.0),
            ("3_7", 3.0, 7.0),
            ("7_14", 7.0, 14.0),
            ("14_21", 14.0, 21.0),
            ("21_plus", 21.0, np.inf),
        ]

        out = {}
        for label, low, high in buckets:
            if np.isinf(high):
                mask = abs_margin >= low
            else:
                mask = (abs_margin >= low) & (abs_margin < high)
            out[f"favorite_{label}_count"] = int(mask.sum())
            out[f"favorite_{label}_accuracy"] = (
                float(correct[mask].mean()) if mask.any() else np.nan
            )
        return out

    def _build_fingerprints(self):
        """Internal helper for the build_fingerprints step."""
        fp_cfg = dict(self.config.get("fingerprints", {}))
        version = fp_cfg.get("version", self.config.get("data", {}).get("version", 0))
        postseason = bool(fp_cfg.get("postseason", False))
        root = fp_cfg.get("root")
        return Fingerprints(version=version, postseason=postseason, root=root)

    def _build_matchup_builder(self):
        """Internal helper for the build_matchup_builder step."""
        matchup_cfg = dict(self.config.get("matchup", {}))
        return MatchupBuilder(
            representation=matchup_cfg.get("representation", "unit_matchup"),
            eps=matchup_cfg.get("eps"),
            blocks=matchup_cfg.get("blocks"),
            safe_math=matchup_cfg.get("safe_math", True),
            zero_offset=matchup_cfg.get("zero_offset"),
            unit_pairings=matchup_cfg.get("unit_pairings"),
            unit_pairings_path=matchup_cfg.get("unit_pairings_path"),
            include_unit_secondary=matchup_cfg.get("include_unit_secondary", False),
        )

    def _build_model(self):
        """Internal helper for the build_model step."""
        model_cfg = dict(self.config.get("model", {}) or {})
        if training_allows_market_features(
            self.config
        ) and not training_allows_market_features(model_cfg):
            model_cfg["allow_market_features_for_training"] = True
        return build_model_from_config(model_cfg)

    def _coalesce_years(self, years, key):
        """Internal helper for the coalesce_years step."""
        if years is not None:
            return list(years)
        eval_cfg = self.config.get("eval", {})
        return list(eval_cfg.get(key, []))

    def _assert_training_features_are_safe(self, X_train, X_val=None):
        allow_market = bool(
            getattr(self.model, "allow_market_features_for_training", False)
            or training_allows_market_features(self.config)
        )
        validate_matchup_frame(
            X_train,
            allow_market_features_for_training=allow_market,
        )
        if X_val is not None:
            validate_matchup_frame(
                X_val,
                allow_market_features_for_training=allow_market,
            )

    def _output_root(self):
        """Internal helper for the output_root step."""
        eval_cfg = self.config.get("eval", {})
        return eval_cfg.get("artifact_root", "data/artifacts")

    def _with_default_eval_config(self, kwargs):
        """Carry artifact-layout eval settings into season comparison helpers."""
        return dict(kwargs)

    def _team_key_column(self, rank_df):
        """Internal helper for the team_key_column step."""
        if "keys_team" in rank_df.columns:
            return "keys_team"
        for candidate in rank_df.columns:
            if "team" in str(candidate).lower():
                return candidate
        return rank_df.columns[0]

    def _model_label(self, model, idx):
        """Internal helper for the model_label step."""
        return model_label(model, idx)

    def _manual_poll_ballots(self, manual_ballots, season, week, top_n):
        """Normalize user-provided top-25 ballots into poll ballot rows."""
        if manual_ballots is None:
            return []

        ballot_specs = self._manual_ballot_specs_for_week(manual_ballots, week)
        frames = []
        for idx, spec in enumerate(ballot_specs):
            if isinstance(spec, dict):
                teams = spec.get(
                    "teams", spec.get("top25", spec.get("ranking", spec.get("ballot")))
                )
                ballot_model = (
                    str(
                        spec.get("ballot_model", spec.get("name", "manual_poll"))
                    ).strip()
                    or "manual_poll"
                )
            else:
                teams = spec
                ballot_model = (
                    "manual_poll"
                    if len(ballot_specs) == 1
                    else f"manual_poll_{idx + 1}"
                )

            if teams is None:
                continue
            rows = []
            for rank, team in enumerate(list(teams), start=1):
                if rank > int(top_n):
                    break
                team_name = str(team).strip()
                if not team_name:
                    continue
                rows.append(
                    {
                        "keys_team": team_name,
                        "ballot_model": ballot_model,
                        "ballot_rank": int(rank),
                        "poll_points": max(int(top_n) + 1 - int(rank), 0),
                        "top25_vote": True,
                        "first_place_vote": int(rank) == 1,
                    }
                )
            if rows:
                frames.append(pd.DataFrame(rows))
        return frames

    def _manual_ballot_specs_for_week(self, manual_ballots, week):
        """Select manual ballot specs that apply to a single poll week."""
        if isinstance(manual_ballots, dict):
            direct_keys = [
                week,
                int(week),
                str(int(week)),
                str(week),
                f"week_{int(week)}",
                f"W{int(week)}",
            ]
            for key in direct_keys:
                if key in manual_ballots:
                    return self._as_ballot_spec_list(manual_ballots[key])
            if any(
                key in manual_ballots for key in ["teams", "top25", "ranking", "ballot"]
            ):
                return self._as_ballot_spec_list(manual_ballots)
            return []
        return self._as_ballot_spec_list(manual_ballots)

    def _as_ballot_spec_list(self, value):
        """Coerce manual ballot input into a list of ballot specs."""
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        if isinstance(value, (list, tuple)):
            if value and all(
                isinstance(item, dict)
                and any(k in item for k in ["teams", "top25", "ranking", "ballot"])
                for item in value
            ):
                return list(value)
            return [list(value)]
        return []

    @staticmethod
    def _load_config(config):
        """Internal helper for the load_config step."""
        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        path = Path(config)
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
