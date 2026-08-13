#!/usr/bin/env python
"""Plot weekly winner and upset performance by fingerprint version."""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = project_root()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data" / "experiments" / "fingerprint_weekly_accuracy"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--v0-predictions",
        type=Path,
        default=Path("data/eval_23JUN2026/2025/tables/game_predictions.csv"),
    )
    parser.add_argument(
        "--v1-runs-root",
        type=Path,
        default=Path("data/experiments/opponent_adjusted_fingerprints/runs"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.project_root.resolve()
    v0_path = resolve(root, args.v0_predictions)
    v1_root = resolve(root, args.v1_runs_root)
    output_root = resolve(root, args.output_root)

    model_metrics = []
    if v0_path.exists():
        model_metrics.append(load_wide_prediction_file(v0_path, fingerprint="v0"))
    else:
        raise FileNotFoundError(f"v0 predictions not found: {v0_path}")

    v1_paths = sorted(v1_root.glob("v1_*/*/*/season_eval/tables/game_predictions.csv"))
    if not v1_paths:
        raise FileNotFoundError(f"No v1 prediction files found under: {v1_root}")
    for path in v1_paths:
        fingerprint = path.parts[path.parts.index("runs") + 1].replace("_", ".")
        model_metrics.append(load_single_model_prediction_file(path, fingerprint=fingerprint))

    model_frame = pd.concat(model_metrics, ignore_index=True, sort=False)
    weekly = (
        model_frame.groupby(["fingerprint", "week"], as_index=False)
        .agg(
            winner_accuracy=("winner_accuracy", "mean"),
            upset_recall=("upset_recall", "mean"),
            model_count=("model", "nunique"),
            game_count=("game_count", "mean"),
            actual_upsets=("actual_upsets", "mean"),
        )
        .sort_values(["fingerprint", "week"])
    )

    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_frame.to_csv(tables_dir / "weekly_model_metrics.csv", index=False)
    weekly.to_csv(tables_dir / "weekly_fingerprint_metrics.csv", index=False)
    plot_weekly_fingerprint_metrics(
        weekly,
        figures_dir / "winner_upset_accuracy_by_week.png",
    )
    print(f"Model-week rows: {len(model_frame)}")
    print(f"Fingerprint-week rows: {len(weekly)}")
    print(f"Table: {tables_dir / 'weekly_fingerprint_metrics.csv'}")
    print(f"Figure: {figures_dir / 'winner_upset_accuracy_by_week.png'}")


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_wide_prediction_file(path: Path, *, fingerprint: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    models = sorted(
        col[: -len("__correct")]
        for col in frame.columns
        if col.endswith("__correct") and not col.startswith("Vegas__")
    )
    rows = []
    for model in models:
        rows.append(metrics_by_week(frame, fingerprint=fingerprint, model=model))
    return pd.concat(rows, ignore_index=True, sort=False)


def load_single_model_prediction_file(path: Path, *, fingerprint: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    models = sorted(
        col[: -len("__correct")]
        for col in frame.columns
        if col.endswith("__correct") and not col.startswith("Vegas__")
    )
    if len(models) != 1:
        raise ValueError(f"Expected one model prediction prefix in {path}, found {models}")
    return metrics_by_week(frame, fingerprint=fingerprint, model=models[0])


def metrics_by_week(frame: pd.DataFrame, *, fingerprint: str, model: str) -> pd.DataFrame:
    required = ["week", "actual_is_upset", f"{model}__correct", f"{model}__called_upset"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns for {fingerprint}/{model}: {missing}")

    data = frame.loc[:, required].copy()
    data["week"] = pd.to_numeric(data["week"], errors="coerce").astype("Int64")
    data[f"{model}__correct"] = coerce_bool(data[f"{model}__correct"])
    data[f"{model}__called_upset"] = coerce_bool(data[f"{model}__called_upset"])
    data["actual_is_upset"] = coerce_bool(data["actual_is_upset"])
    data = data.dropna(subset=["week"])

    rows = []
    for week, group in data.groupby("week", dropna=True):
        actual_upsets = group["actual_is_upset"].fillna(False)
        upset_mask = actual_upsets.eq(True)
        winner_accuracy = group[f"{model}__correct"].mean()
        upset_recall = (
            group.loc[upset_mask, f"{model}__called_upset"].mean()
            if upset_mask.any()
            else pd.NA
        )
        rows.append(
            {
                "fingerprint": fingerprint,
                "model": model,
                "week": int(week),
                "winner_accuracy": winner_accuracy,
                "upset_recall": upset_recall,
                "game_count": int(len(group)),
                "actual_upsets": int(upset_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    text = series.astype("string").str.strip().str.lower()
    out = text.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    )
    return out.astype("boolean")


def plot_weekly_fingerprint_metrics(weekly: pd.DataFrame, path: Path) -> None:
    colors = {
        "v0": "#222222",
        "v1.1": "#0072B2",
        "v1.2": "#E69F00",
        "v1.3": "#009E73",
        "v1.4": "#D55E00",
        "v1.5": "#CC79A7",
        "v1.6": "#56B4E9",
        "v1.7": "#7F7F7F",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for fingerprint, group in weekly.groupby("fingerprint"):
        group = group.sort_values("week")
        color = colors.get(fingerprint)
        axes[0].plot(
            group["week"],
            group["winner_accuracy"],
            marker="o",
            linewidth=2.0 if fingerprint == "v0" else 1.6,
            label=fingerprint,
            color=color,
        )
        axes[1].plot(
            group["week"],
            group["upset_recall"],
            marker="o",
            linewidth=2.0 if fingerprint == "v0" else 1.6,
            label=fingerprint,
            color=color,
        )

    axes[0].set_title("Winner Accuracy by Week")
    axes[0].set_ylabel("Winner accuracy")
    axes[1].set_title("Upset Recall by Week")
    axes[1].set_ylabel("Upset recall")
    axes[1].set_xlabel("Week")
    for ax in axes:
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncol=4, frameon=False)
    fig.suptitle("2025 Model-Averaged Accuracy by Fingerprint Version", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
