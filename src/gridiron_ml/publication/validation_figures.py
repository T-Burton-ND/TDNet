"""Cross-validation summaries and publication-ready fold-distribution plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "brier_score": (["brier_score", "season_winner_brier_score"], "Brier score (lower is better)"),
    "winner_accuracy": (["winner_accuracy", "season_winner_winner_accuracy"], "Winner accuracy"),
    "margin_mae": (["mae", "season_margin_mae"], "Margin MAE (points; lower is better)"),
    "calibration_slope": (["season_margin_calibration_slope", "calibration_slope"], "Calibration slope (0.9–1.1 target)"),
}


def normalize_cv_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for canonical, (candidates, _) in METRICS.items():
        source = next((column for column in candidates if column in out), None)
        out[canonical] = pd.to_numeric(out[source], errors="coerce") if source else pd.NA
    return out


def plot_cv_metric_boxplots(frame: pd.DataFrame, path: str | Path, *, title: str, dpi: int = 180) -> Path:
    """Plot the selected configuration's outer-fold distribution by model type."""
    data = normalize_cv_metrics(frame)
    label_column = next((c for c in ["concrete_model_type", "model", "model_level"] if c in data), None)
    if not label_column:
        raise ValueError("CV metrics need concrete_model_type, model, or model_level.")
    labels = list(dict.fromkeys(data[label_column].astype(str)))
    fig, axes = plt.subplots(2, 2, figsize=(max(13, len(labels) * 0.7), 10))
    for axis, (metric, (_, ylabel)) in zip(axes.flat, METRICS.items()):
        values = [data.loc[data[label_column].astype(str).eq(label), metric].dropna().to_numpy() for label in labels]
        axis.boxplot(values, labels=labels, showmeans=True, meanline=True, patch_artist=True)
        for patch in axis.artists:
            patch.set_facecolor("#DCE6F1")
        if metric == "calibration_slope":
            axis.axhspan(0.9, 1.1, color="#DCEFE1", alpha=0.65, zorder=0)
            axis.axhline(1.0, color="#3E7C59", lw=1)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=55, labelsize=8)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path
