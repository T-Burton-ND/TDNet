"""Selection-aware metric tables and heatmaps for the scientific roster."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = ("M1", "M2", "M3", "M4", "M5", "M10")
TIER_ORDER = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F6-C", "F7", "F8")
METRICS = {
    "mae": {"label": "MAE (points)", "higher_better": False, "weight": "n_rows"},
    "upset_correct": {"label": "Upset recall", "higher_better": True, "weight": None},
    "favorite_correct": {"label": "Chalk recall", "higher_better": True, "weight": None},
    "winner_accuracy": {"label": "Winner accuracy", "higher_better": True, "weight": "n_rows"},
    "brier_score": {"label": "Brier score", "higher_better": False, "weight": "n_rows"},
    "ats_accuracy": {"label": "ATS accuracy", "higher_better": True, "weight": "ats_n"},
}


def selected_complete_trials(results: pd.DataFrame, expected_folds: int = 10) -> pd.DataFrame:
    success = results.loc[results["status"].astype(str).eq("success")].copy()
    config_keys = [
        "objective", "feature_config", "model_level", "model_family",
        "model_config", "params_json", "seed",
    ]
    selected = []
    for cell, group in success.groupby(["objective", "feature_config", "model_level"], sort=False):
        objective = str(cell[0])
        metric = "brier_score" if objective == "winner" else "mae"
        summary = (
            group.groupby(config_keys, dropna=False)
            .agg(cv_mean=(metric, "mean"), fold_count=("outer_fold", "nunique"))
            .reset_index()
        )
        summary = summary.loc[summary["fold_count"].eq(int(expected_folds))].sort_values("cv_mean")
        if summary.empty:
            continue
        best = summary.iloc[0]
        mask = pd.Series(True, index=group.index)
        for key in config_keys:
            mask &= group[key].astype(str).eq(str(best[key]))
        chosen = group.loc[mask].copy()
        chosen["selected_cv_mean"] = float(best["cv_mean"])
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def aggregate_metric_grid(selected_trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["objective", "feature_config", "model_level"]
    for keys, group in selected_trials.groupby(group_keys, sort=False):
        row = dict(zip(group_keys, keys))
        row["fold_count"] = int(group["outer_fold"].nunique())
        for metric, spec in METRICS.items():
            values = pd.to_numeric(group.get(metric), errors="coerce")
            valid = values.notna()
            weight_col = spec["weight"]
            if not valid.any():
                row[metric] = np.nan
                continue
            if weight_col and weight_col in group:
                weights = pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0)
                use = valid & weights.gt(0)
                row[metric] = float(np.average(values[use], weights=weights[use])) if use.any() else float(values[valid].mean())
            else:
                row[metric] = float(values[valid].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def render_metric_heatmaps(table: pd.DataFrame, output_root: Path, *, required_tiers=()) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    missing = sorted(set(required_tiers) - set(table["feature_config"].astype(str)))
    if missing:
        raise ValueError(f"Metric grid is missing required fingerprints: {missing}.")
    output_root.mkdir(parents=True, exist_ok=True)
    tiers = [tier for tier in TIER_ORDER if tier in set(table["feature_config"].astype(str))]
    extra = sorted(set(table["feature_config"].astype(str)) - set(tiers))
    tiers.extend(extra)
    paths = []
    for metric, spec in METRICS.items():
        finite = pd.to_numeric(table[metric], errors="coerce")
        finite = finite[np.isfinite(finite)]
        vmin = float(finite.min()) if len(finite) else 0.0
        vmax = float(finite.max()) if len(finite) else 1.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-9
        fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.05 * len(tiers) + 5), 5.2), sharey=True)
        for ax, objective in zip(axes, ("margin", "winner")):
            panel = (
                table.loc[table["objective"].astype(str).eq(objective)]
                .pivot(index="model_level", columns="feature_config", values=metric)
                .reindex(index=MODEL_ORDER, columns=tiers)
            )
            labels = panel.map(lambda value: "" if pd.isna(value) else (f"{value:.2f}" if metric == "mae" else f"{value:.3f}"))
            sns.heatmap(
                panel, ax=ax, annot=labels, fmt="", cmap="viridis" if spec["higher_better"] else "viridis_r",
                vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="#dddddd",
                cbar=ax is axes[-1], cbar_kws={"label": spec["label"]},
                mask=panel.isna(),
            )
            ax.set_title(f"{objective.capitalize()} objective")
            ax.set_xlabel("Fingerprint")
            ax.set_ylabel("Scientific model" if ax is axes[0] else "")
        fig.suptitle(f"Scientific roster: {spec['label']}", y=1.01)
        fig.tight_layout()
        stem = output_root / f"scientific_roster_{metric}_heatmap"
        for suffix in ("png", "svg", "pdf"):
            path = stem.with_suffix(f".{suffix}")
            fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    return paths
