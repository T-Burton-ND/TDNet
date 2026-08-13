#!/usr/bin/env python
"""Regenerate easy-to-find final-model diagnostic plots and all-year accuracy.

This script intentionally uses only saved best-model artifacts. It does not
retrain models and it ignores any in-progress hyperparameter-search chunks.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = project_root()
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.finalize_fingerprint_hyperparameter_search import selected_features
from gridiron_ml.experiments.hyperparameter_search import always_keep_columns, default_source_fingerprint_root, safe_label
from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints
from gridiron_ml.models.checkpoints import load_model_checkpoint
from gridiron_ml.td_run.evaluator import TDEval
from gridiron_ml.td_run.matchups import MatchupBuilder


OBJECTIVES = ("winner", "balanced", "margin")
FAMILY_MARKERS = {"stat": "o", "tree": "^", "linear": "s"}
FAMILY_COLORS = {"stat": "#2a9d8f", "tree": "#e76f51", "linear": "#457b9d"}
MODEL_PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#e6550d",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("data/experiments/fingerprint_hyperparameter_search"),
    )
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=Path("data/experiments/fingerprint_hyperparameter_search/final_model_diagnostics"),
    )
    parser.add_argument("--source-fingerprint-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/final_model_diagnostics_easy_find"),
    )
    parser.add_argument("--objectives", nargs="*", default=list(OBJECTIVES))
    parser.add_argument("--max-prediction-years", nargs="*", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.project_root.resolve()
    search_root = resolve(root, args.search_root)
    diagnostics_root = resolve(root, args.diagnostics_root)
    source_root = (
        args.source_fingerprint_root.resolve()
        if args.source_fingerprint_root
        else default_source_fingerprint_root(root)
    )
    output_root = resolve(root, args.output_root)
    tables_dir = output_root / "tables"
    plots_dir = output_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_overall_metrics(diagnostics_root, search_root, tuple(args.objectives))
    model_colors = model_color_map(metrics)
    metrics.to_csv(tables_dir / "saved_best_model_objective_metrics.csv", index=False)
    for objective in args.objectives:
        plot_objective(
            metrics,
            objective=objective,
            model_colors=model_colors,
            output_path=plots_dir / f"{objective}_saved_best_model_metrics.png",
        )

    inventory = load_inventory(search_root, tuple(args.objectives))
    inventory.to_csv(tables_dir / "saved_best_model_inventory.csv", index=False)
    year_metrics, failures = all_year_accuracy(
        inventory=inventory,
        source_root=source_root,
        years=args.max_prediction_years,
        output_dir=tables_dir,
    )
    year_metrics.to_csv(tables_dir / "saved_best_model_accuracy_by_year.csv", index=False)
    failures.to_csv(tables_dir / "saved_best_model_accuracy_by_year_failures.csv", index=False)
    plot_accuracy_by_year(year_metrics, plots_dir / "saved_best_model_accuracy_by_year.png", model_colors=model_colors)

    write_readme(output_root, metrics, year_metrics, failures)
    print(f"Easy-find diagnostics: {output_root}")
    print(f"Objective metric rows: {len(metrics)}")
    print(f"All-year metric rows: {len(year_metrics)}")
    print(f"All-year failures: {len(failures)}")


def load_overall_metrics(diagnostics_root: Path, search_root: Path, objectives: tuple[str, ...]) -> pd.DataFrame:
    path = diagnostics_root / "summary" / "overall_metrics.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frames = []
        for objective in objectives:
            for path in (diagnostics_root / "runs" / objective).glob("*/*/tables/overall_metrics.csv"):
                frames.append(pd.read_csv(path))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        raise FileNotFoundError(f"No overall metrics found under {diagnostics_root}")

    inventory = load_inventory(search_root, objectives)
    keep_cols = [
        "objective",
        "family",
        "model",
        "fingerprint",
        "top_k_features",
        "tuning_score",
        "eval_loss_function",
        "eval_mae",
        "eval_rmse",
        "eval_market_mae",
        "eval_market_rmse",
    ]
    have = [col for col in keep_cols if col in inventory.columns]
    if have:
        frame = frame.merge(inventory.loc[:, have].drop_duplicates(), on=["objective", "family", "model"], how="left")
    return frame.sort_values(["objective", "family", "model"]).reset_index(drop=True)


def load_inventory(search_root: Path, objectives: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for objective in objectives:
        path = search_root / objective / "final_artifacts" / "final_model_inventory.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["objective"] = objective
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No final_model_inventory.csv files found under {search_root}")
    return pd.concat(frames, ignore_index=True, sort=False)


def model_color_map(frame: pd.DataFrame) -> dict[str, str]:
    models = sorted(frame["model"].dropna().astype(str).unique())
    return {model: MODEL_PALETTE[idx % len(MODEL_PALETTE)] for idx, model in enumerate(models)}


def plot_objective(metrics: pd.DataFrame, *, objective: str, model_colors: dict[str, str], output_path: Path) -> None:
    frame = metrics.loc[metrics["objective"].astype(str).eq(objective)].copy()
    if frame.empty:
        return
    metric_col = objective_metric_column(objective, frame)
    frame[metric_col] = pd.to_numeric(frame[metric_col], errors="coerce")
    frame = frame.loc[frame[metric_col].notna()].copy()
    higher_is_better = objective_metric_higher_is_better(objective)
    frame = frame.sort_values(metric_col, ascending=not higher_is_better).reset_index(drop=True)
    labels = [f"{row.family}/{row.model}" for row in frame.itertuples(index=False)]

    fig_height = max(7.5, 0.36 * len(frame) + 3.0)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    y = np.arange(len(frame))
    for idx, row in frame.iterrows():
        family = str(row["family"])
        model = str(row["model"])
        ax.scatter(
            row[metric_col],
            y[idx],
            marker=FAMILY_MARKERS.get(family, "o"),
            s=95,
            color=model_colors.get(model, "#555555"),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    baseline_value, baseline_label = objective_baseline(frame, objective)
    if baseline_value is not None and np.isfinite(baseline_value):
        ax.axvline(
            baseline_value,
            color="black",
            linestyle="--",
            linewidth=1.8,
            alpha=0.8,
            label=baseline_label,
            zorder=2,
        )
        ax.text(
            baseline_value,
            -0.8,
            baseline_label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel(pretty_metric(metric_col))
    ax.set_title(f"{objective.title()} objective: saved best models")
    add_legends(fig, ax, frame, model_colors, include_vegas=baseline_value is not None)
    fig.tight_layout(rect=(0, 0, 0.79, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def all_year_accuracy(
    *,
    inventory: pd.DataFrame,
    source_root: Path,
    years: list[int] | None,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    failures = []
    rows_path = output_dir / "saved_best_model_accuracy_by_year.partial.csv"
    failures_path = output_dir / "saved_best_model_accuracy_by_year_failures.partial.csv"
    for idx, row in enumerate(inventory.to_dict("records"), start=1):
        context = {
            "objective": row.get("objective"),
            "family": row.get("family"),
            "model": row.get("model"),
            "fingerprint": row.get("fingerprint"),
            "top_k_features": row.get("top_k_features"),
        }
        label = f"{context['objective']}/{context['family']}/{context['model']}"
        print(f"[{idx}/{len(inventory)}] SCORE {label}", flush=True)
        try:
            checkpoint_path = Path(str(row.get("checkpoint_path", "")))
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            model = load_model_checkpoint(checkpoint_path)
            source_row = load_source_search_row(row)
            frame = load_fast_selected_frame(row=source_row, source_root=source_root)
            available_years = sorted(pd.to_numeric(frame["keys_season"], errors="coerce").dropna().astype(int).unique())
            if years:
                wanted = {int(year) for year in years}
                available_years = [year for year in available_years if year in wanted]
            fingerprints = StaticFrameFingerprints(frame)
            evaluator = TDEval(
                config={"eval": {"artifact_root": ""}},
                fingerprints=fingerprints,
                matchup_builder=MatchupBuilder(representation="unit_matchup"),
                model=model,
            )
            for year in available_years:
                print(f"[{idx}/{len(inventory)}]   year={year}", flush=True)
                try:
                    pred, _ = evaluator.evaluate(years=[year], label=str(year))
                    if pred.empty:
                        continue
                    metric_row = prediction_metrics(pred)
                    if not metric_row:
                        continue
                    rows.append(
                        {
                            **context,
                            "year": int(year),
                            "n_rows": int(len(pred)),
                            "winner_accuracy": metric_row.get("winner_accuracy"),
                            "mae": metric_row.get("mae"),
                            "rmse": metric_row.get("rmse"),
                            "brier_score": metric_row.get("brier_score"),
                            "market_accuracy": metric_row.get("market_accuracy"),
                            "market_mae": metric_row.get("market_mae"),
                            "market_rmse": metric_row.get("market_rmse"),
                            "favorite_correct": metric_row.get("favorite_correct"),
                            "upset_correct": metric_row.get("upset_correct"),
                        }
                    )
                    write_partial(rows_path, pd.DataFrame(rows))
                except Exception as exc:
                    failures.append({**context, "year": int(year), "error": repr(exc)})
                    write_partial(failures_path, pd.DataFrame(failures))
                    print(f"[{idx}/{len(inventory)}]   SKIP year={year}: {exc!r}", flush=True)
        except Exception as exc:
            failures.append({**context, "error": repr(exc)})
            print(f"[{idx}/{len(inventory)}] FAIL {label}: {exc!r}", flush=True)
        write_partial(rows_path, pd.DataFrame(rows))
        write_partial(failures_path, pd.DataFrame(failures))
    return pd.DataFrame(rows), pd.DataFrame(failures)


def write_partial(path: Path, frame: pd.DataFrame) -> None:
    if not frame.empty:
        frame.to_csv(path, index=False)


def prediction_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty or not {"y", "pred_margin"}.issubset(frame.columns):
        return {}
    y = pd.to_numeric(frame["y"], errors="coerce")
    pred_margin = pd.to_numeric(frame["pred_margin"], errors="coerce")
    keep = y.notna() & pred_margin.notna()
    if not keep.any():
        return {}
    y = y.loc[keep]
    pred_margin = pred_margin.loc[keep]
    actual_home = y > 0
    pred_home = pred_margin > 0
    error = pred_margin - y
    out = {
        "winner_accuracy": float((actual_home == pred_home).mean()),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
    }
    if "pred_proba_home_win" in frame.columns:
        proba = pd.to_numeric(frame.loc[keep, "pred_proba_home_win"], errors="coerce").clip(1e-8, 1 - 1e-8)
        if proba.notna().any():
            out["brier_score"] = float(np.square(proba - actual_home.astype(float)).mean())
    market_frame = frame.loc[keep]
    favorite_home = market_pick_home(market_frame)
    if favorite_home.notna().any():
        favorite_won = favorite_home == actual_home
        correct = pred_home == actual_home
        out["market_accuracy"] = float((favorite_home == actual_home).mean())
        out["favorite_correct"] = float(correct.loc[favorite_won].mean()) if favorite_won.any() else np.nan
        out["upset_correct"] = float(correct.loc[~favorite_won].mean()) if (~favorite_won).any() else np.nan
    if "market_spread_close" in market_frame.columns:
        spread = pd.to_numeric(market_frame["market_spread_close"], errors="coerce")
        if spread.notna().any():
            market_margin = -spread
            market_error = market_margin - y
            out["market_mae"] = float(market_error.abs().mean())
            out["market_rmse"] = float(np.sqrt(np.square(market_error).mean()))
    return out


def market_pick_home(frame: pd.DataFrame) -> pd.Series:
    if "market_win_probability" in frame.columns:
        prob = pd.to_numeric(frame["market_win_probability"], errors="coerce")
        if prob.notna().any():
            return prob > 0.5
    if "market_spread_close" in frame.columns:
        spread = pd.to_numeric(frame["market_spread_close"], errors="coerce")
        if spread.notna().any():
            return spread < 0
    return pd.Series(np.nan, index=frame.index)


def objective_baseline(frame: pd.DataFrame, objective: str) -> tuple[float | None, str | None]:
    if objective == "margin":
        for col in ("eval_market_mae", "market_mae"):
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors="coerce").dropna()
                if not values.empty:
                    return float(values.mean()), "Vegas MAE"
        return None, None
    if "market_accuracy" in frame.columns:
        values = pd.to_numeric(frame["market_accuracy"], errors="coerce").dropna()
        if not values.empty:
            return float(values.mean()), "Vegas accuracy"
    return None, None


def add_legends(fig, ax, frame: pd.DataFrame, model_colors: dict[str, str], *, include_vegas: bool) -> None:
    family_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS.get(family, "o"),
            color="black",
            linestyle="",
            label=family,
            markersize=8,
        )
        for family in sorted(frame["family"].dropna().astype(str).unique())
    ]
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=model_colors.get(model, "#555555"),
            linestyle="",
            label=model,
            markersize=7,
        )
        for model in sorted(frame["model"].dropna().astype(str).unique())
    ]
    if include_vegas:
        model_handles.append(
            plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Vegas baseline")
        )
    family_legend = ax.legend(handles=family_handles, title="Shape = family", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(family_legend)
    fig.legend(handles=model_handles, title="Color = model", loc="center right", bbox_to_anchor=(0.995, 0.5), fontsize=8)


def load_source_search_row(inventory_row: dict) -> dict:
    checkpoint_path = Path(str(inventory_row.get("checkpoint_path", "")))
    model_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else Path(str(inventory_row.get("artifact_root", ""))).parent
    source_path = model_dir / "source_search_row.csv"
    if source_path.exists():
        try:
            frame = pd.read_csv(source_path)
            if not frame.empty:
                out = frame.iloc[0].to_dict()
                out.update({key: value for key, value in inventory_row.items() if key not in out or pd.isna(out[key])})
                return out
        except pd.errors.EmptyDataError:
            pass
    return dict(inventory_row)


def load_fast_selected_frame(*, row: dict, source_root: Path) -> pd.DataFrame:
    fp_path = Path(str(row.get("fingerprint_path") or ""))
    if not fp_path.exists():
        fp_path = source_root / "fingerprints" / safe_label(str(row["fingerprint"])) / "canonical_fingerprint.parquet"
    frame = pd.read_parquet(fp_path)
    features = selected_features(row)
    if features:
        keep = list(dict.fromkeys(always_keep_columns(frame) + [feature for feature in features if feature in frame.columns]))
        frame = frame.loc[:, keep].copy()
    return frame


def plot_accuracy_by_year(year_metrics: pd.DataFrame, output_path: Path, *, model_colors: dict[str, str]) -> None:
    if year_metrics.empty:
        return
    frame = year_metrics.copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["winner_accuracy"] = pd.to_numeric(frame["winner_accuracy"], errors="coerce")
    frame = frame.loc[frame["year"].notna() & frame["winner_accuracy"].notna()].copy()

    objectives = [obj for obj in OBJECTIVES if obj in set(frame["objective"].astype(str))]
    if not objectives:
        objectives = sorted(frame["objective"].astype(str).unique())
    fig, axes = plt.subplots(len(objectives), 1, figsize=(16, max(5, 4.3 * len(objectives))), sharex=True, sharey=True)
    if len(objectives) == 1:
        axes = [axes]
    for ax, objective in zip(axes, objectives):
        sub = frame.loc[frame["objective"].astype(str).eq(objective)].copy()
        for (family, model), group in sub.groupby(["family", "model"], sort=True):
            group = group.sort_values("year")
            ax.plot(
                group["year"],
                group["winner_accuracy"],
                color=model_colors.get(str(model), "#555555"),
                alpha=0.42,
                linewidth=1.1,
            )
            ax.scatter(
                group["year"],
                group["winner_accuracy"],
                marker=FAMILY_MARKERS.get(str(family), "o"),
                s=30,
                color=model_colors.get(str(model), "#555555"),
                alpha=0.7,
            )
        if "market_accuracy" in sub.columns:
            market = weighted_market_accuracy_by_year(sub)
            if not market.empty:
                ax.plot(
                    market["year"],
                    market["market_accuracy"],
                    color="black",
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.85,
                    label="Vegas accuracy",
                )
        ax.set_title(f"{objective.title()} saved best models")
        ax.set_ylabel("Winner accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Season")
    family_handles = [
        plt.Line2D([0], [0], marker=marker, color="black", linestyle="", label=family, markersize=8)
        for family, marker in FAMILY_MARKERS.items()
    ]
    model_handles = [
        plt.Line2D([0], [0], marker="o", color=model_colors.get(model, "#555555"), linestyle="", label=model, markersize=7)
        for model in sorted(frame["model"].dropna().astype(str).unique())
    ]
    model_handles.append(plt.Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, label="Vegas accuracy"))
    fig.legend(handles=family_handles, title="Shape = family", loc="upper center", ncol=3, bbox_to_anchor=(0.38, 0.98))
    fig.legend(handles=model_handles, title="Color = model", loc="center right", bbox_to_anchor=(0.995, 0.5), fontsize=8)
    fig.suptitle("Accuracy by year using saved best models on all available games", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.80, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def weighted_market_accuracy_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prepared = frame.assign(
        market_accuracy=pd.to_numeric(frame["market_accuracy"], errors="coerce"),
        n_rows=pd.to_numeric(frame["n_rows"], errors="coerce").fillna(1.0),
    ).dropna(subset=["market_accuracy"])
    for year, group in prepared.groupby("year"):
        rows.append({"year": year, "market_accuracy": weighted_market_accuracy(group)})
    return pd.DataFrame(rows).sort_values("year") if rows else pd.DataFrame(columns=["year", "market_accuracy"])


def weighted_market_accuracy(group: pd.DataFrame) -> float:
    weights = pd.to_numeric(group["n_rows"], errors="coerce").fillna(1.0)
    values = pd.to_numeric(group["market_accuracy"], errors="coerce")
    keep = values.notna() & weights.notna() & (weights > 0)
    if not keep.any():
        return np.nan
    return float(np.average(values.loc[keep], weights=weights.loc[keep]))


def objective_metric_column(objective: str, frame: pd.DataFrame) -> str:
    if objective == "margin":
        for col in ("mae", "eval_mae"):
            if col in frame.columns:
                return col
    for col in ("winner_accuracy", "eval_winner_accuracy"):
        if col in frame.columns:
            return col
    return "winner_accuracy"


def objective_metric_higher_is_better(objective: str) -> bool:
    return objective != "margin"


def pretty_metric(column: str) -> str:
    return column.replace("_", " ").title()


def write_readme(output_root: Path, metrics: pd.DataFrame, year_metrics: pd.DataFrame, failures: pd.DataFrame) -> None:
    text = f"""# Easy-Find Final Model Diagnostics

Generated from saved best final-model artifacts only. No models were retrained.

## Outputs

- `plots/winner_saved_best_model_metrics.png`
- `plots/balanced_saved_best_model_metrics.png`
- `plots/margin_saved_best_model_metrics.png`
- `plots/saved_best_model_accuracy_by_year.png`
- `tables/saved_best_model_objective_metrics.csv`
- `tables/saved_best_model_inventory.csv`
- `tables/saved_best_model_accuracy_by_year.csv`
- `tables/saved_best_model_accuracy_by_year_failures.csv`

## Row Counts

- objective metric rows: {len(metrics)}
- all-year accuracy rows: {len(year_metrics)}
- all-year failure rows: {len(failures)}

The all-year accuracy table deliberately includes in-sample seasons when the saved
model was trained on that season, so it is diagnostic/illustrative rather than a
clean holdout estimate.
"""
    (output_root / "README.md").write_text(text)


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
