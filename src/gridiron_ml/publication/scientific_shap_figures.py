"""Publication figures for the scientific-roster F6 SHAP study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import yaml


MODEL_ORDER = ("M1", "M2", "M3", "M4", "M5", "M10")
OBJECTIVE_ORDER = ("margin", "winner")
MODEL_LABELS = {
    "M1": "Linear",
    "M2": "Spline",
    "M3": "Random forest",
    "M4": "Boosted tree",
    "M5": "MLP",
    "M10": "KNN",
}
FAMILY_ORDER = (
    "structural", "sample_size", "preseason_prior", "box_score",
    "box_score_offense", "box_score_defense", "box_score_general",
    "box_score_special_teams", "efficiency", "opponent_adjusted",
    "situational", "returning_production", "coaching", "temporal",
    "schedule_graph",
)
FAMILY_COLORS = {
    family: plt.get_cmap("tab20")(index % 20)
    for index, family in enumerate(FAMILY_ORDER)
}


@dataclass(frozen=True)
class FigureSettings:
    features_per_atlas_page: int = 38
    features_per_dependence_page: int = 12
    raster_dpi: int = 240
    minimum_valid_folds: int = 8
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 26084


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_study_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("SHAP study config must be a mapping.")
    return value


def load_feature_contract(path: str | Path) -> pd.DataFrame:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        {
            "source_feature": str(row["name"]),
            "feature_family": str(row["family"]),
        }
        for row in manifest.get("features", [])
    ]
    frame = pd.DataFrame(rows).drop_duplicates("source_feature")
    if len(frame) != int(manifest.get("feature_count", len(frame))):
        raise ValueError("Feature manifest count does not match its feature records.")
    return frame


def prepare_importance(
    frame: pd.DataFrame,
    feature_contract: pd.DataFrame,
    *,
    objectives: tuple[str, ...] = OBJECTIVE_ORDER,
    models: tuple[str, ...] = MODEL_ORDER,
    require_complete: bool = True,
    minimum_valid_folds: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"objective", "model_level", "outer_fold", "source_feature", "mean_abs_shap"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"SHAP importance table is missing {missing}.")
    out = frame.copy()
    out["objective"] = out["objective"].astype(str)
    out["model_level"] = out["model_level"].astype(str)
    out["source_feature"] = out["source_feature"].astype(str)
    out["mean_abs_shap"] = pd.to_numeric(out["mean_abs_shap"], errors="coerce")
    out = out.loc[
        out["objective"].isin(objectives) & out["model_level"].isin(models)
    ].copy()
    unknown = sorted(set(out["source_feature"]) - set(feature_contract["source_feature"]))
    if unknown:
        raise ValueError(f"Importance table has unknown F6 source features: {unknown[:10]}")
    if (~np.isfinite(out["mean_abs_shap"])).any() or out["mean_abs_shap"].lt(0).any():
        raise ValueError("mean_abs_shap must be finite and nonnegative in every row.")
    keys = ["objective", "model_level", "outer_fold", "source_feature"]
    duplicates = out.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(
            "Importance table must contain one source-aggregated row per model-fold-feature; "
            f"found {int(duplicates.sum())} duplicate rows."
        )
    out = out.merge(feature_contract, on="source_feature", how="left", suffixes=("", "_contract"))
    if "feature_family_contract" in out:
        out["feature_family"] = out["feature_family_contract"]
        out = out.drop(columns="feature_family_contract")
    totals = out.groupby(["objective", "model_level", "outer_fold"])["mean_abs_shap"].transform("sum")
    if totals.le(0).any():
        raise ValueError("Every model-fold must have positive total absolute SHAP.")
    out["importance_share"] = np.where(totals > 0, out["mean_abs_shap"] / totals, np.nan)
    out["fold_rank"] = out.groupby(
        ["objective", "model_level", "outer_fold"]
    )["importance_share"].rank(method="average", ascending=False)

    expected = set(feature_contract["source_feature"])
    coverage_rows = []
    for objective in objectives:
        for model in models:
            observed = set(out.loc[
                out["objective"].eq(objective) & out["model_level"].eq(model),
                "source_feature",
            ])
            absent = sorted(expected - observed)
            cell = out.loc[
                out["objective"].eq(objective) & out["model_level"].eq(model)
            ]
            fold_feature_counts = cell.groupby("outer_fold")["source_feature"].nunique()
            incomplete_folds = int((fold_feature_counts < len(expected)).sum())
            coverage_rows.append(
                {
                    "objective": objective,
                    "model_level": model,
                    "expected_features": len(expected),
                    "observed_features": len(observed),
                    "missing_features": len(absent),
                    "missing_features_json": json.dumps(absent),
                    "valid_folds": int(cell["outer_fold"].nunique()),
                    "incomplete_folds": incomplete_folds,
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    invalid = (
        coverage["missing_features"].gt(0)
        | coverage["incomplete_folds"].gt(0)
        | coverage["valid_folds"].lt(int(minimum_valid_folds))
    )
    if require_complete and invalid.any():
        bad = coverage.loc[invalid, [
            "objective", "model_level", "missing_features", "valid_folds", "incomplete_folds"
        ]]
        raise ValueError("Incomplete all-feature SHAP coverage:\n" + bad.to_string(index=False))
    return out, coverage


def _clustered_bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Fold-cluster bootstrap of the cross-model median importance."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for objective, objective_rows in frame.groupby("objective", sort=False):
        fold_feature = objective_rows.groupby(
            ["outer_fold", "source_feature"], as_index=False
        )["importance_share"].median()
        pivot = fold_feature.pivot(
            index="outer_fold", columns="source_feature", values="importance_share"
        ).sort_index()
        values = pivot.to_numpy(dtype=float)
        if not len(values):
            continue
        draws = rng.integers(0, len(values), size=(int(resamples), len(values)))
        boot = np.nanmedian(values[draws, :], axis=1)
        low, high = np.nanquantile(boot, [0.025, 0.975], axis=0)
        rows.extend(
            {
                "objective": objective,
                "source_feature": feature,
                "importance_ci_low": float(low[index]),
                "importance_ci_high": float(high[index]),
            }
            for index, feature in enumerate(pivot.columns)
        )
    return pd.DataFrame(rows)


def summarize_importance(
    frame: pd.DataFrame,
    *,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 26084,
) -> pd.DataFrame:
    per_model = (
        frame.groupby(["objective", "model_level", "source_feature", "feature_family"], as_index=False)
        .agg(
            importance_median=("importance_share", "median"),
            importance_q25=("importance_share", lambda x: x.quantile(0.25)),
            importance_q75=("importance_share", lambda x: x.quantile(0.75)),
            valid_folds=("outer_fold", "nunique"),
            fold_rank_median=("fold_rank", "median"),
            fold_rank_q25=("fold_rank", lambda x: x.quantile(0.25)),
            fold_rank_q75=("fold_rank", lambda x: x.quantile(0.75)),
            top_10_frequency=("fold_rank", lambda x: float((x <= 10).mean())),
            top_25_frequency=("fold_rank", lambda x: float((x <= 25).mean())),
            top_50_frequency=("fold_rank", lambda x: float((x <= 50).mean())),
            top_100_frequency=("fold_rank", lambda x: float((x <= 100).mean())),
        )
    )
    ranks = per_model.groupby(["objective", "model_level"])["importance_median"].rank(
        method="average", ascending=False
    )
    per_model["model_rank"] = ranks
    consensus = (
        per_model.groupby(["objective", "source_feature", "feature_family"], as_index=False)
        .agg(
            consensus_importance=("importance_median", "median"),
            consensus_rank_median=("model_rank", "median"),
            consensus_rank_q25=("model_rank", lambda x: x.quantile(0.25)),
            consensus_rank_q75=("model_rank", lambda x: x.quantile(0.75)),
            model_count=("model_level", "nunique"),
        )
    )
    consensus["consensus_rank"] = consensus.groupby("objective")["consensus_importance"].rank(
        method="first", ascending=False
    )
    across_cells = frame.groupby(
        ["objective", "source_feature", "feature_family"], as_index=False
    ).agg(
        cell_rank_median=("fold_rank", "median"),
        cell_rank_q25=("fold_rank", lambda x: x.quantile(0.25)),
        cell_rank_q75=("fold_rank", lambda x: x.quantile(0.75)),
        cell_top_10_frequency=("fold_rank", lambda x: float((x <= 10).mean())),
        cell_top_25_frequency=("fold_rank", lambda x: float((x <= 25).mean())),
        cell_top_50_frequency=("fold_rank", lambda x: float((x <= 50).mean())),
        cell_top_100_frequency=("fold_rank", lambda x: float((x <= 100).mean())),
    )
    intervals = _clustered_bootstrap_intervals(
        frame, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    consensus = consensus.merge(
        across_cells, on=["objective", "source_feature", "feature_family"], how="left"
    ).merge(intervals, on=["objective", "source_feature"], how="left")
    return per_model.merge(
        consensus,
        on=["objective", "source_feature", "feature_family"],
        how="left",
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#D8DEE7", linewidth=0.6, alpha=0.7)


def _save_figure(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def plot_family_allocation(summary: pd.DataFrame, path: Path, *, dpi: int) -> list[dict[str, Any]]:
    data = (
        summary.groupby(["objective", "model_level", "feature_family"], as_index=False)["importance_median"]
        .sum()
    )
    families = [x for x in FAMILY_ORDER if x in set(data["feature_family"])]
    families += sorted(set(data["feature_family"]) - set(families))
    fig, axes = plt.subplots(1, 2, figsize=(15, max(7, 0.35 * len(families) + 2)), sharey=True)
    for axis, objective in zip(axes, OBJECTIVE_ORDER):
        pivot = data.loc[data["objective"].eq(objective)].pivot(
            index="feature_family", columns="model_level", values="importance_median"
        ).reindex(index=families, columns=MODEL_ORDER).fillna(0.0)
        image = axis.imshow(pivot.to_numpy() * 100.0, aspect="auto", cmap="Blues")
        axis.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[x] for x in MODEL_ORDER], rotation=35, ha="right")
        axis.set_yticks(range(len(families)), [x.replace("_", " ") for x in families])
        axis.set_title(f"{objective.title()} objective")
        axis.set_xlabel("Scientific model")
        for row in range(len(families)):
            for column in range(len(MODEL_ORDER)):
                value = pivot.iat[row, column] * 100.0
                axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=7,
                          color="white" if value > 8 else "#172033")
        fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02, label="Normalized absolute SHAP (%)")
    fig.suptitle("F6 information allocation across the scientific roster", fontsize=16)
    fig.tight_layout()
    _save_figure(fig, path, dpi=dpi)
    plt.close(fig)
    return [{"figure_type": "family_allocation", "objective": "both", "page": 1, "features": families, "path": str(path)}]


def plot_model_concordance(summary: pd.DataFrame, output: Path, *, dpi: int) -> list[dict[str, Any]]:
    records = []
    for objective in OBJECTIVE_ORDER:
        pivot = summary.loc[summary["objective"].eq(objective)].pivot(
            index="source_feature", columns="model_level", values="model_rank"
        ).reindex(columns=MODEL_ORDER)
        corr = pivot.corr(method="spearman")
        fig, axis = plt.subplots(figsize=(8.3, 7.2))
        image = axis.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        labels = [MODEL_LABELS[x] for x in MODEL_ORDER]
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(f"Architecture agreement on F6 feature ranks — {objective}")
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column, row, f"{corr.iat[row, column]:.2f}", ha="center", va="center",
                          color="white" if abs(corr.iat[row, column]) > 0.55 else "#172033")
        fig.colorbar(image, ax=axis, label="Spearman rank correlation")
        fig.tight_layout()
        path = output / f"figure_02_model_concordance_{objective}.png"
        _save_figure(fig, path, dpi=dpi)
        plt.close(fig)
        records.append({"figure_type": "model_concordance", "objective": objective, "page": 1,
                        "features": list(pivot.index), "path": str(path)})
    return records


def _feature_pages(summary: pd.DataFrame, objective: str, page_size: int) -> list[list[str]]:
    ordered = (
        summary.loc[summary["objective"].eq(objective), ["source_feature", "consensus_rank"]]
        .drop_duplicates()
        .sort_values("consensus_rank")["source_feature"]
        .tolist()
    )
    return [ordered[start : start + page_size] for start in range(0, len(ordered), page_size)]


def plot_feature_atlas(
    summary: pd.DataFrame,
    output: Path,
    *,
    page_size: int,
    dpi: int,
) -> list[dict[str, Any]]:
    records = []
    cmap = LinearSegmentedColormap.from_list("tdnet_importance", ["#F5F7FA", "#B8D5EA", "#2166AC"])
    pdf_path = output / "supplement_complete_feature_atlas.pdf"
    with PdfPages(pdf_path) as pdf:
        for objective in OBJECTIVE_ORDER:
            objective_rows = summary.loc[summary["objective"].eq(objective)].copy()
            for page_number, features in enumerate(_feature_pages(summary, objective, page_size), start=1):
                page = objective_rows.loc[objective_rows["source_feature"].isin(features)].copy()
                pivot = page.pivot(index="source_feature", columns="model_level", values="importance_median").reindex(
                    index=features, columns=MODEL_ORDER
                )
                consensus = page.drop_duplicates("source_feature").set_index("source_feature").reindex(features)
                fig, (stripe, axis, rank_axis) = plt.subplots(
                    1, 3, figsize=(15.5, max(8, 0.31 * len(features) + 2.2)),
                    gridspec_kw={"width_ratios": [0.22, 6.0, 1.55]}, sharey=True,
                )
                family_values = [FAMILY_COLORS.get(str(x), (0.65, 0.65, 0.65, 1.0)) for x in consensus["feature_family"]]
                stripe.imshow(np.asarray(family_values).reshape(len(features), 1, 4), aspect="auto")
                stripe.set_xticks([]); stripe.set_yticks([]); stripe.set_ylabel("Feature family")
                values = pivot.to_numpy() * 100.0
                vmax = max(float(np.nanpercentile(values, 98)), 0.01)
                image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
                axis.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[x] for x in MODEL_ORDER], rotation=35, ha="right")
                axis.set_yticks(range(len(features)), features, fontsize=7.5)
                axis.set_title(f"Complete F6 feature atlas — {objective} — page {page_number}")
                axis.set_xlabel("Median normalized absolute SHAP across outer folds (%)")
                fig.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
                rank_axis.set_xlim(0, 1); rank_axis.set_xticks([]); rank_axis.set_yticks([])
                rank_axis.set_title("Consensus")
                for row, feature in enumerate(features):
                    item = consensus.loc[feature]
                    rank_axis.text(0.02, row, f"#{int(item['consensus_rank'])}", va="center", fontsize=7.5)
                    rank_axis.text(0.34, row, f"{100*item['consensus_importance']:.2f}%", va="center", fontsize=7.5)
                    rank_axis.text(0.69, row, f"IQR {item['consensus_rank_q25']:.0f}–{item['consensus_rank_q75']:.0f}", va="center", fontsize=7)
                rank_axis.set_ylim(len(features) - 0.5, -0.5)
                fig.tight_layout()
                path = output / f"figure_03_feature_atlas_{objective}_page_{page_number:02d}.png"
                _save_figure(fig, path, dpi=dpi)
                pdf.savefig(fig, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                records.append({"figure_type": "feature_atlas", "objective": objective,
                                "page": page_number, "features": features, "path": str(path)})
    records.append({"figure_type": "feature_atlas_pdf", "objective": "both", "page": 0,
                    "features": [], "path": str(pdf_path)})
    return records


def plot_rank_stability(
    summary: pd.DataFrame,
    output: Path,
    *,
    page_size: int,
    dpi: int,
) -> list[dict[str, Any]]:
    records = []
    pdf_path = output / "supplement_rank_stability_atlas.pdf"
    with PdfPages(pdf_path) as pdf:
        for objective in OBJECTIVE_ORDER:
            consensus = summary.loc[summary["objective"].eq(objective)].drop_duplicates("source_feature").set_index("source_feature")
            for page_number, features in enumerate(_feature_pages(summary, objective, page_size), start=1):
                page = consensus.reindex(features)
                y = np.arange(len(features))
                fig, axis = plt.subplots(figsize=(12.5, max(8, 0.31 * len(features) + 2)))
                low = page["cell_rank_median"] - page["cell_rank_q25"]
                high = page["cell_rank_q75"] - page["cell_rank_median"]
                colors = [FAMILY_COLORS.get(str(x), "#777777") for x in page["feature_family"]]
                for row in range(len(page)):
                    axis.errorbar(page["cell_rank_median"].iloc[row], y[row],
                                  xerr=[[low.iloc[row]], [high.iloc[row]]], fmt="o",
                                  color=colors[row], ecolor=colors[row], capsize=2, markersize=4)
                axis.set_yticks(y, features, fontsize=7.5)
                axis.invert_yaxis()
                axis.set_xlim(0, len(consensus) + 1)
                axis.set_xlabel("Feature rank across model-fold cells (median and interquartile range)")
                axis.set_title(f"F6 feature-rank stability — {objective} — page {page_number}")
                _style_axis(axis)
                fig.tight_layout()
                path = output / f"figure_04_rank_stability_{objective}_page_{page_number:02d}.png"
                _save_figure(fig, path, dpi=dpi)
                pdf.savefig(fig, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                records.append({"figure_type": "rank_stability", "objective": objective,
                                "page": page_number, "features": features, "path": str(path)})
    records.append({"figure_type": "rank_stability_pdf", "objective": "both", "page": 0,
                    "features": [], "path": str(pdf_path)})
    return records


def prepare_effects(frame: pd.DataFrame, feature_contract: pd.DataFrame) -> pd.DataFrame:
    required = {"objective", "model_level", "outer_fold", "source_feature", "feature_value_z", "shap_value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"SHAP effect table is missing {missing}.")
    out = frame.copy()
    out["objective"] = out["objective"].astype(str)
    out["model_level"] = out["model_level"].astype(str)
    out["source_feature"] = out["source_feature"].astype(str)
    out["feature_value_z"] = pd.to_numeric(out["feature_value_z"], errors="coerce")
    out["shap_value"] = pd.to_numeric(out["shap_value"], errors="coerce")
    unknown = sorted(set(out["source_feature"]) - set(feature_contract["source_feature"]))
    if unknown:
        raise ValueError(f"Effect table has unknown F6 source features: {unknown[:10]}")
    out = out.merge(feature_contract, on="source_feature", how="inner", suffixes=("", "_contract"))
    if "feature_family_contract" in out:
        out["feature_family"] = out["feature_family_contract"]
        out = out.drop(columns="feature_family_contract")
    return out


def direction_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in effects.groupby(["objective", "model_level", "outer_fold", "source_feature", "feature_family"]):
        valid = group[["feature_value_z", "shap_value"]].dropna()
        rho = valid["feature_value_z"].corr(valid["shap_value"], method="spearman") if len(valid) >= 8 else np.nan
        rows.append(dict(zip(["objective", "model_level", "outer_fold", "source_feature", "feature_family"], keys)) | {"direction_rho": rho, "n": len(valid)})
    fold = pd.DataFrame(rows)
    return fold.groupby(["objective", "model_level", "source_feature", "feature_family"], as_index=False).agg(
        direction_median=("direction_rho", "median"), valid_folds=("direction_rho", "count")
    )


def plot_direction_atlas(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    output: Path,
    *,
    page_size: int,
    dpi: int,
) -> list[dict[str, Any]]:
    direction = direction_summary(effects)
    records = []
    pdf_path = output / "supplement_direction_atlas.pdf"
    with PdfPages(pdf_path) as pdf:
        for objective in OBJECTIVE_ORDER:
            for page_number, features in enumerate(_feature_pages(summary, objective, page_size), start=1):
                pivot = direction.loc[
                    direction["objective"].eq(objective) & direction["source_feature"].isin(features)
                ].pivot(index="source_feature", columns="model_level", values="direction_median").reindex(
                    index=features, columns=MODEL_ORDER
                )
                fig, axis = plt.subplots(figsize=(12.5, max(8, 0.31 * len(features) + 2)))
                image = axis.imshow(pivot.to_numpy(), aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
                axis.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[x] for x in MODEL_ORDER], rotation=35, ha="right")
                axis.set_yticks(range(len(features)), features, fontsize=7.5)
                axis.set_title(f"Held-out SHAP direction atlas — {objective} — page {page_number}")
                axis.set_xlabel("Spearman correlation of standardized feature value with SHAP; blank = indeterminate")
                fig.colorbar(image, ax=axis, label="Direction correlation")
                fig.tight_layout()
                path = output / f"figure_05_direction_atlas_{objective}_page_{page_number:02d}.png"
                _save_figure(fig, path, dpi=dpi)
                pdf.savefig(fig, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                records.append({"figure_type": "direction_atlas", "objective": objective,
                                "page": page_number, "features": features, "path": str(path)})
    records.append({"figure_type": "direction_atlas_pdf", "objective": "both", "page": 0,
                    "features": [], "path": str(pdf_path)})
    return records


def _binned_curve(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    data = frame[["feature_value_z", "shap_value"]].dropna().sort_values("feature_value_z")
    if len(data) < bins:
        return data
    groups = pd.qcut(data["feature_value_z"], q=min(bins, data["feature_value_z"].nunique()), duplicates="drop")
    return data.groupby(groups, observed=True).median(numeric_only=True).reset_index(drop=True)


def plot_dependence_supplement(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    output: Path,
    *,
    page_size: int,
    dpi: int,
) -> list[dict[str, Any]]:
    model_colors = dict(zip(MODEL_ORDER, plt.get_cmap("tab10").colors[: len(MODEL_ORDER)]))
    records = []
    pdf_path = output / "supplement_all_feature_dependence.pdf"
    with PdfPages(pdf_path) as pdf:
        for objective in OBJECTIVE_ORDER:
            for page_number, features in enumerate(_feature_pages(summary, objective, page_size), start=1):
                columns = 3
                rows = int(np.ceil(len(features) / columns))
                fig, axes = plt.subplots(rows, columns, figsize=(15, 3.4 * rows), squeeze=False)
                for axis, feature in zip(axes.flat, features):
                    feature_data = effects.loc[
                        effects["objective"].eq(objective) & effects["source_feature"].eq(feature)
                    ]
                    for model in MODEL_ORDER:
                        model_data = feature_data.loc[feature_data["model_level"].eq(model)].dropna(
                            subset=["feature_value_z", "shap_value"]
                        )
                        if len(model_data) > 1200:
                            model_data = model_data.sample(1200, random_state=26084)
                        axis.scatter(model_data["feature_value_z"], model_data["shap_value"], s=5,
                                     alpha=0.08, color=model_colors[model], rasterized=True)
                        curve = _binned_curve(model_data)
                        if len(curve) >= 2:
                            axis.plot(curve["feature_value_z"], curve["shap_value"], color=model_colors[model], lw=1.2,
                                      label=MODEL_LABELS[model])
                    axis.axhline(0, color="#8792A2", lw=0.6)
                    axis.set_title(feature, fontsize=8)
                    axis.set_xlabel("Standardized held-out value", fontsize=7)
                    output_label = "margin points" if objective == "margin" else "home-win probability"
                    axis.set_ylabel(f"SHAP contribution ({output_label})", fontsize=7)
                    axis.tick_params(labelsize=7)
                    _style_axis(axis)
                for axis in axes.flat[len(features):]:
                    axis.axis("off")
                handles, labels = axes.flat[0].get_legend_handles_labels()
                if handles:
                    fig.legend(
                        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
                        ncol=6, frameon=False,
                    )
                fig.suptitle(
                    f"All-feature SHAP dependence — {objective} — page {page_number}",
                    fontsize=15, y=0.998,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.925])
                path = output / f"figure_06_dependence_{objective}_page_{page_number:02d}.png"
                _save_figure(fig, path, dpi=dpi)
                pdf.savefig(fig, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                records.append({"figure_type": "dependence", "objective": objective,
                                "page": page_number, "features": features, "path": str(path)})
    records.append({"figure_type": "dependence_pdf", "objective": "both", "page": 0,
                    "features": [], "path": str(pdf_path)})
    return records


def plot_new_block_focus(summary: pd.DataFrame, output: Path, *, dpi: int) -> list[dict[str, Any]]:
    block = summary.loc[summary["feature_family"].isin(["temporal", "schedule_graph"])].copy()
    records = []
    for objective in OBJECTIVE_ORDER:
        page = block.loc[block["objective"].eq(objective)]
        pivot = page.pivot(index="source_feature", columns="model_level", values="model_rank").reindex(columns=MODEL_ORDER)
        order = page.drop_duplicates("source_feature").sort_values("consensus_rank")["source_feature"]
        pivot = pivot.reindex(order)
        fig, axis = plt.subplots(figsize=(12.5, max(9, 0.25 * len(pivot) + 2)))
        image = axis.imshow(pivot.to_numpy(), aspect="auto", vmin=1, vmax=227, cmap="viridis_r")
        axis.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABELS[x] for x in MODEL_ORDER], rotation=35, ha="right")
        axis.set_yticks(range(len(pivot)), pivot.index, fontsize=7)
        family_counts = page.drop_duplicates("source_feature")["feature_family"].value_counts()
        axis.set_title(
            "Ranks of the temporal and schedule-graph features — "
            f"{objective} ({int(family_counts.get('temporal', 0))} + "
            f"{int(family_counts.get('schedule_graph', 0))})"
        )
        axis.set_xlabel("Rank within all 227 F6 features; lower is more important")
        fig.colorbar(image, ax=axis, label="Feature rank")
        fig.tight_layout()
        path = output / f"figure_07_new_block_focus_{objective}.png"
        _save_figure(fig, path, dpi=dpi)
        plt.close(fig)
        records.append({"figure_type": "new_block_focus", "objective": objective, "page": 1,
                        "features": list(pivot.index), "path": str(path)})
    return records


def plot_explainer_audit(importance: pd.DataFrame, output: Path, *, dpi: int) -> list[dict[str, Any]]:
    optional = [x for x in [
        "explainer_method", "n_explained", "runtime_seconds", "additivity_error",
        "common_permutation_rank_rho",
    ] if x in importance]
    cells = importance.groupby(["objective", "model_level", "outer_fold"], as_index=False).agg(
        explained_features=("source_feature", "nunique"),
        **{column: (column, "first" if column == "explainer_method" else "median") for column in optional},
    )
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.4))
    metrics = [
        ("explained_features", "Explained source features"),
        ("runtime_seconds", "Runtime per fold (seconds)"),
        ("additivity_error", "Absolute additivity error"),
        ("common_permutation_rank_rho", "Native/common rank agreement"),
    ]
    for axis, (column, label) in zip(axes, metrics):
        if column not in cells:
            axis.text(0.5, 0.5, "Not supplied", ha="center", va="center")
            axis.set_axis_off()
            continue
        values = [pd.to_numeric(cells.loc[cells["model_level"].eq(model), column], errors="coerce").dropna() for model in MODEL_ORDER]
        axis.boxplot(values, tick_labels=[MODEL_LABELS[x] for x in MODEL_ORDER], showfliers=False)
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylabel(label)
        _style_axis(axis)
    fig.suptitle("SHAP computation and additivity audit")
    fig.tight_layout()
    path = output / "figure_08_explainer_audit.png"
    _save_figure(fig, path, dpi=dpi)
    plt.close(fig)
    cells.to_csv(output / "explainer_cell_audit.csv", index=False)
    return [{"figure_type": "explainer_audit", "objective": "both", "page": 1,
             "features": [], "path": str(path)}]


def build_scientific_shap_figures(
    *,
    importance: pd.DataFrame,
    feature_contract: pd.DataFrame,
    output_root: str | Path,
    settings: FigureSettings,
    effects: pd.DataFrame | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    prepared, coverage = prepare_importance(
        importance,
        feature_contract,
        require_complete=require_complete,
        minimum_valid_folds=settings.minimum_valid_folds,
    )
    summary = summarize_importance(
        prepared,
        bootstrap_resamples=settings.bootstrap_resamples,
        bootstrap_seed=settings.bootstrap_seed,
    )
    summary.to_csv(output / "all_feature_importance_summary.csv", index=False)
    coverage.to_csv(output / "feature_coverage_audit.csv", index=False)

    records = []
    records += plot_family_allocation(summary, output / "figure_01_family_allocation.png", dpi=settings.raster_dpi)
    records += plot_model_concordance(summary, output, dpi=settings.raster_dpi)
    records += plot_feature_atlas(summary, output, page_size=settings.features_per_atlas_page, dpi=settings.raster_dpi)
    records += plot_rank_stability(summary, output, page_size=settings.features_per_atlas_page, dpi=settings.raster_dpi)
    records += plot_new_block_focus(summary, output, dpi=settings.raster_dpi)
    records += plot_explainer_audit(prepared, output, dpi=settings.raster_dpi)

    if effects is not None and not effects.empty:
        prepared_effects = prepare_effects(effects, feature_contract)
        records += plot_direction_atlas(summary, prepared_effects, output,
                                        page_size=settings.features_per_atlas_page, dpi=settings.raster_dpi)
        records += plot_dependence_supplement(summary, prepared_effects, output,
                                              page_size=settings.features_per_dependence_page, dpi=settings.raster_dpi)

    manifest_rows = []
    for record in records:
        features = list(record.pop("features"))
        manifest_rows.append(
            {
                **record,
                "feature_count": len(features),
                "first_feature": features[0] if features else "",
                "last_feature": features[-1] if features else "",
                "features_json": json.dumps(features),
            }
        )
    importance_hash = hashlib.sha256(
        pd.util.hash_pandas_object(prepared.sort_values(
            ["objective", "model_level", "outer_fold", "source_feature"]
        ), index=False).values.tobytes()
    ).hexdigest()
    effects_hash = ""
    if effects is not None:
        effects_hash = hashlib.sha256(
            pd.util.hash_pandas_object(effects, index=False).values.tobytes()
        ).hexdigest()
    figure_manifest = pd.DataFrame(manifest_rows)
    figure_manifest["importance_input_sha256"] = importance_hash
    figure_manifest["effects_input_sha256"] = effects_hash
    figure_manifest.to_csv(output / "figure_manifest.csv", index=False)
    report = {
        "status": "complete",
        "source_feature_count": int(feature_contract["source_feature"].nunique()),
        "importance_rows": int(len(prepared)),
        "effect_rows": int(0 if effects is None else len(effects)),
        "figure_files": int(len(figure_manifest)),
        "complete_coverage": bool(
            coverage["missing_features"].eq(0).all()
            and coverage["incomplete_folds"].eq(0).all()
            and coverage["valid_folds"].ge(settings.minimum_valid_folds).all()
        ),
        "importance_input_sha256": importance_hash,
        "effects_input_sha256": effects_hash,
        "output_root": str(output.resolve()),
    }
    (output / "figure_generation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
