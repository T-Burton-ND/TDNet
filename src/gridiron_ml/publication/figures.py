"""Canonical publication figure suite generated only from merged tables."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


FIGURE_SPECS = {
    "figure_01_feature_model_heatmap": "Feature-complexity by model-complexity performance heatmap.",
    "figure_02_winner_feature_ladder": "Winner performance as feature tiers are cumulatively added.",
    "figure_03_margin_feature_ladder": "Margin error as feature tiers are cumulatively added.",
    "figure_04_model_complexity": "Model complexity versus performance within each feature tier.",
    "figure_05_calibration": "Reliability curves for finalist winner-probability models.",
    "figure_06_learning_curves": "Training and validation loss by epoch or training size.",
    "figure_07_ablation_deltas": "Paired feature-family ablation performance deltas.",
    "figure_08_importance_stability": "Feature-family importance stability across folds and seeds.",
    "figure_09_performance_distributions": "Fold/seed/season performance distributions.",
    "figure_10_negative_controls": "Random and shuffled-feature negative controls.",
    "figure_11_market_incremental_value": "Market-only, football-only, and combined performance.",
    "figure_12_compute_tradeoff": "Predictive performance versus training cost.",
    "figure_13_model_disagreement": "Pairwise model prediction disagreement matrix.",
    "figure_14_historical_seasons": "Historical season-by-season performance.",
    "figure_15_prospective_2026": "Prospective cumulative 2026 performance.",
}


class PublicationFigureBuilder:
    """Generate the full paper/blog figure suite from canonical input tables."""

    def __init__(self, output_root: str | Path, *, dpi=200, strict=True):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.dpi = int(dpi)
        self.strict = bool(strict)

    def generate_all(
        self,
        *,
        matrix_summary=None,
        historical_summary=None,
        predictions=None,
        learning_curves=None,
        ablations=None,
        importance_stability=None,
        negative_controls=None,
    ) -> dict[str, object]:
        """Generate all required figures, returning explicit generated/skipped state."""
        inputs = {
            "matrix_summary": self._frame(matrix_summary),
            "historical_summary": self._frame(historical_summary),
            "predictions": self._frame(predictions),
            "learning_curves": self._frame(learning_curves),
            "ablations": self._frame(ablations),
            "importance_stability": self._frame(importance_stability),
            "negative_controls": self._frame(negative_controls),
        }
        calls = {
            "figure_01_feature_model_heatmap": lambda: self._heatmap(inputs["matrix_summary"]),
            "figure_02_winner_feature_ladder": lambda: self._feature_ladder(inputs["matrix_summary"], "winner"),
            "figure_03_margin_feature_ladder": lambda: self._feature_ladder(inputs["matrix_summary"], "margin"),
            "figure_04_model_complexity": lambda: self._model_complexity(inputs["matrix_summary"]),
            "figure_05_calibration": lambda: self._calibration(inputs["predictions"]),
            "figure_06_learning_curves": lambda: self._learning_curves(inputs["learning_curves"]),
            "figure_07_ablation_deltas": lambda: self._ablation(inputs["ablations"]),
            "figure_08_importance_stability": lambda: self._importance(inputs["importance_stability"]),
            "figure_09_performance_distributions": lambda: self._distributions(inputs["matrix_summary"]),
            "figure_10_negative_controls": lambda: self._controls(inputs["negative_controls"]),
            "figure_11_market_incremental_value": lambda: self._market(inputs["matrix_summary"]),
            "figure_12_compute_tradeoff": lambda: self._compute(inputs["matrix_summary"]),
            "figure_13_model_disagreement": lambda: self._disagreement(inputs["predictions"]),
            "figure_14_historical_seasons": lambda: self._historical(
                inputs["historical_summary"] if not inputs["historical_summary"].empty else inputs["matrix_summary"]
            ),
            "figure_15_prospective_2026": lambda: self._prospective(inputs["predictions"]),
        }
        generated = {}
        skipped = {}
        for name, function in calls.items():
            try:
                figure = function()
                generated[name] = str(self._save(figure, name))
            except (KeyError, ValueError) as exc:
                if self.strict:
                    raise
                skipped[name] = str(exc)
        manifest = {
            "required_count": len(FIGURE_SPECS),
            "generated_count": len(generated),
            "generated": generated,
            "skipped": skipped,
        }
        (self.output_root / "figure_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        (self.output_root / "figure_captions.md").write_text(
            "# Figure captions\n\n"
            + "\n\n".join(
                f"- **{name}.** {FIGURE_SPECS[name]}" for name in FIGURE_SPECS
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest

    def _heatmap(self, frame):
        data = self._metric_rows(frame)
        metric = self._preferred_metric(data)
        selected = data.loc[data["metric"] == metric]
        pivot = selected.pivot_table(
            index="feature_config", columns="model_level", values="value", aggfunc="mean"
        )
        if pivot.empty:
            raise ValueError("No feature/model matrix rows for heatmap.")
        fig, ax = plt.subplots(figsize=(9, 6))
        image = ax.imshow(pivot.values, cmap="viridis_r" if self._lower_better(metric) else "viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                if np.isfinite(pivot.iloc[i, j]):
                    ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        ax.set_title(f"Feature × model complexity ({metric})")
        fig.colorbar(image, ax=ax, label=metric)
        return fig

    def _feature_ladder(self, frame, objective):
        data = self._metric_rows(frame)
        data = data.loc[data["objective"].astype(str) == objective]
        metric = "brier_score" if objective == "winner" and "brier_score" in set(data["metric"]) else (
            "winner_accuracy" if objective == "winner" else "mae"
        )
        data = data.loc[data["metric"] == metric]
        if data.empty:
            raise ValueError(f"No {objective} feature-ladder rows.")
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, group in data.groupby("model_level"):
            means = group.groupby("feature_config", sort=False)["value"].mean()
            ax.plot(means.index, means.values, marker="o", label=model)
        ax.set_title(f"{objective.title()} performance across the feature ladder")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=35)
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.25)
        return fig

    def _model_complexity(self, frame):
        data = self._metric_rows(frame)
        metric = self._preferred_metric(data)
        data = data.loc[data["metric"] == metric]
        if data.empty:
            raise ValueError("No model complexity rows.")
        fig, ax = plt.subplots(figsize=(9, 5))
        for feature, group in data.groupby("feature_config"):
            means = group.groupby("model_level", sort=True)["value"].mean()
            ax.plot(means.index, means.values, marker="o", label=feature)
        ax.set_title(f"Model complexity within feature tiers ({metric})")
        ax.legend(ncol=2, fontsize=7)
        ax.grid(alpha=0.25)
        return fig

    def _calibration(self, frame):
        required = {"model_name", "pred_home_win_probability", "actual_home_win"}
        self._require(frame, required, "calibration predictions")
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        for model, group in frame.groupby("model_name"):
            valid = group.dropna(subset=["pred_home_win_probability", "actual_home_win"])
            if len(valid) < 20:
                continue
            observed, predicted = calibration_curve(
                valid["actual_home_win"].astype(int),
                valid["pred_home_win_probability"].astype(float),
                n_bins=min(10, max(3, len(valid) // 20)),
                strategy="quantile",
            )
            ax.plot(predicted, observed, marker="o", label=model)
        ax.plot([0, 1], [0, 1], "--", color="#555", label="perfect")
        ax.set(xlabel="Predicted home-win probability", ylabel="Observed home-win rate", title="Historical reliability")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        return fig

    def _learning_curves(self, frame):
        self._require(frame, {"model_name", "epoch", "train_loss", "validation_loss"}, "learning curves")
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, group in frame.groupby("model_name"):
            mean = group.groupby("epoch")[["train_loss", "validation_loss"]].mean()
            ax.plot(mean.index, mean["train_loss"], alpha=0.55, label=f"{model} train")
            ax.plot(mean.index, mean["validation_loss"], label=f"{model} validation")
        ax.set(xlabel="Epoch", ylabel="Loss", title="Neural learning curves")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.25)
        return fig

    def _ablation(self, frame):
        family = self._column(frame, ["feature_family", "ablation", "family_removed"])
        delta = self._column(frame, ["delta", "metric_delta", "paired_delta"])
        data = frame.groupby(family, as_index=False)[delta].mean().sort_values(delta)
        if data.empty:
            raise ValueError("No ablation rows.")
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(data))))
        ax.barh(data[family].astype(str), data[delta], color=np.where(data[delta] > 0, "#A33B20", "#2B6F77"))
        ax.axvline(0, color="#333", lw=1)
        ax.set(title="Feature-family ablation deltas", xlabel="Paired metric delta")
        return fig

    def _importance(self, frame):
        family = self._column(frame, ["feature_family", "feature"])
        importance = self._column(frame, ["importance", "mean_abs_shap", "permutation_importance"])
        data = frame.groupby(family)[importance].agg(["mean", "std"]).sort_values("mean").tail(20)
        if data.empty:
            raise ValueError("No importance stability rows.")
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(data.index.astype(str), data["mean"], xerr=data["std"].fillna(0), color="#355C7D")
        ax.set(title="Feature importance stability", xlabel="Mean importance across folds/seeds")
        return fig

    def _distributions(self, frame):
        data = self._metric_rows(frame)
        metric = self._preferred_metric(data)
        data = data.loc[data["metric"] == metric]
        groups = [g["value"].dropna().values for _, g in data.groupby("model_level")]
        labels = [str(name) for name, _ in data.groupby("model_level")]
        if not groups:
            raise ValueError("No performance distribution rows.")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(groups, tick_labels=labels, showmeans=True)
        ax.set(title=f"Fold/seed performance distributions ({metric})", ylabel=metric)
        return fig

    def _controls(self, frame):
        control = self._column(frame, ["control", "negative_control", "experiment"])
        metric = self._column(frame, ["value", "metric_value", "score"])
        data = frame.groupby(control, as_index=False)[metric].mean().sort_values(metric)
        if data.empty:
            raise ValueError("No negative-control rows.")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(data[control].astype(str), data[metric], color="#8064A2")
        ax.tick_params(axis="x", rotation=30)
        ax.set(title="Negative controls", ylabel="Metric value")
        return fig

    def _market(self, frame):
        data = self._metric_rows(frame)
        data = data.loc[data["feature_config"].isin(["F6", "F7", "F8"])]
        metric = self._preferred_metric(data)
        data = data.loc[data["metric"] == metric]
        if data.empty:
            raise ValueError("No F6/F7/F8 market comparison rows.")
        pivot = data.pivot_table(index="model_level", columns="feature_config", values="value", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set(title=f"Football, market, and combined representations ({metric})", ylabel=metric)
        return fig

    def _compute(self, frame):
        data = self._metric_rows(frame)
        runtime_column = next((c for c in ["runtime_seconds", "training_seconds", "compute_seconds"] if c in frame.columns), None)
        if runtime_column is None:
            raise ValueError("Compute table lacks runtime_seconds/training_seconds.")
        metric = self._preferred_metric(data)
        values = data.loc[data["metric"] == metric]
        base = frame.merge(
            values[["objective", "feature_config", "model_level", "value"]].drop_duplicates(),
            on=["objective", "feature_config", "model_level"],
            how="inner",
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        for level, group in base.groupby("model_level"):
            ax.scatter(group[runtime_column], group["value"], label=level, alpha=0.7)
        ax.set(xscale="log", xlabel="Training time (seconds, log scale)", ylabel=metric, title="Performance/compute tradeoff")
        ax.legend()
        return fig

    def _disagreement(self, frame):
        self._require(frame, {"game_id", "model_name", "pred_winner"}, "model disagreement")
        pivot = frame.pivot_table(index="game_id", columns="model_name", values="pred_winner", aggfunc="first")
        names = list(pivot.columns)
        matrix = np.full((len(names), len(names)), np.nan)
        for i, first in enumerate(names):
            for j, second in enumerate(names):
                if i == j:
                    matrix[i, j] = 0.0
                    continue
                valid = pivot[[first, second]].dropna()
                matrix[i, j] = (valid[first] != valid[second]).mean() if len(valid) else np.nan
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title("Pairwise model disagreement")
        fig.colorbar(image, ax=ax, label="Disagreement rate")
        return fig

    def _historical(self, frame):
        season = self._column(frame, ["test_season", "season"])
        data = self._metric_rows(frame)
        metric = self._preferred_metric(data)
        if season not in data.columns:
            raise ValueError("Historical performance table lacks season/test_season.")
        data = data.loc[data["metric"] == metric]
        fig, ax = plt.subplots(figsize=(9, 5))
        for level, group in data.groupby("model_level"):
            mean = group.groupby(season)["value"].mean()
            ax.plot(mean.index, mean.values, marker="o", label=level)
        ax.set(title=f"Historical season performance ({metric})", xlabel="Season", ylabel=metric)
        ax.legend()
        ax.grid(alpha=0.25)
        return fig

    def _prospective(self, frame):
        required = {"season", "week", "model_name", "pred_winner", "home_team", "away_team", "actual_home_win"}
        self._require(frame, required, "prospective predictions")
        data = frame.loc[pd.to_numeric(frame["season"], errors="coerce") == 2026].copy()
        if data.empty:
            raise ValueError("No scored prospective 2026 predictions.")
        actual = data["home_team"].where(data["actual_home_win"].astype(bool), data["away_team"])
        data["correct"] = data["pred_winner"].eq(actual)
        weekly = data.groupby(["model_name", "week"], as_index=False)["correct"].agg(["sum", "count"]).reset_index()
        weekly["cumulative_correct"] = weekly.groupby("model_name")["sum"].cumsum()
        weekly["cumulative_games"] = weekly.groupby("model_name")["count"].cumsum()
        weekly["cumulative_accuracy"] = weekly["cumulative_correct"] / weekly["cumulative_games"]
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, group in weekly.groupby("model_name"):
            ax.plot(group["week"], group["cumulative_accuracy"], marker="o", label=model)
        ax.set(title="Prospective cumulative 2026 winner accuracy", xlabel="Week", ylabel="Cumulative accuracy", ylim=(0, 1))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        return fig

    def _metric_rows(self, frame):
        if frame.empty:
            raise ValueError("Matrix summary is empty.")
        if {"metric", "value"}.issubset(frame.columns):
            return frame.copy()
        value_columns = [c for c in ["brier_score", "winner_accuracy", "accuracy", "mae", "rmse"] if c in frame.columns]
        if not value_columns:
            raise ValueError("Matrix summary has no supported metric columns.")
        identifiers = [c for c in ["objective", "feature_config", "model_level", "outer_fold", "seed", "season", "test_season"] if c in frame.columns]
        return frame.melt(id_vars=identifiers, value_vars=value_columns, var_name="metric", value_name="value")

    @staticmethod
    def _preferred_metric(frame):
        metrics = set(frame["metric"].astype(str))
        for metric in ["brier_score", "mae", "winner_accuracy", "accuracy", "rmse"]:
            if metric in metrics:
                return metric
        raise ValueError("No preferred publication metric is available.")

    @staticmethod
    def _lower_better(metric):
        return metric in {"brier_score", "log_loss", "mae", "rmse"}

    @staticmethod
    def _frame(value):
        if value is None:
            return pd.DataFrame()
        if isinstance(value, pd.DataFrame):
            return value.copy()
        path = Path(value)
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    @staticmethod
    def _require(frame, columns, label):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")

    @staticmethod
    def _column(frame, choices):
        for choice in choices:
            if choice in frame.columns:
                return choice
        raise ValueError(f"Table lacks any of columns {choices}.")

    def _save(self, figure, name):
        path = self.output_root / f"{name}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(figure)
        return path
