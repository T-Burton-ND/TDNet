#!/usr/bin/env python3
"""Render 2026 scientific roster performance rotations and PCA rankings."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from gridiron_ml.publication.figure_theme import TDNET_COLORS, apply_tdnet_theme

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "publication/2026/week_05/post_game/scientific/current_season_f0_f17_market"
FIGURES = PUB / "figures"
STAGES = [*(f"F{i}" for i in range(17)), "F17-market"]
MODELS = ["M1", "M2", "M3", "M4", "M5", "M10"]
MODEL_LABELS = {
    "M1": "M1 · Linear", "M2": "M2 · Spline", "M3": "M3 · Tree",
    "M4": "M4 · Boosted", "M5": "M5 · Neural", "M10": "M10 · KNN",
}
FAMILY_TO_MODEL = {
    "linear": "M1", "spline": "M2", "tree": "M3",
    "boosted": "M4", "neural": "M5", "knn": "M10",
}
METRICS = [
    ("margin_mae", "Margin MAE", "Points · lower is better", False),
    ("su_accuracy", "Straight-up winner accuracy", "Accuracy · higher is better", True),
    ("brier_score", "Brier score", "Score · lower is better", False),
    ("ats_accuracy", "ATS accuracy", "Accuracy · higher is better", True),
]


def _palette() -> dict[str, str]:
    return TDNET_COLORS


def _pair_matrix(scorecard: pd.DataFrame) -> pd.DataFrame:
    models = scorecard.loc[scorecard.series_type.eq("model")].copy()
    models["model_id"] = models.model_family.map(FAMILY_TO_MODEL)
    models["generation"] = models.fingerprint.map(
        lambda stage: 17 if stage == "F17-market" else int(str(stage)[1:])
    )
    expected = pd.MultiIndex.from_product([STAGES, MODELS], names=["fingerprint", "model_id"]).to_frame(index=False)
    models = expected.merge(models, on="fingerprint", how="left", validate="one_to_many")
    models = models.loc[models.model_id_y.notna()] if "model_id_y" in models else models
    if "model_id_x" in models:
        models["model_id"] = models.model_id_x
        models = models.drop(columns=["model_id_x", "model_id_y"], errors="ignore")
    models["generation"] = models.fingerprint.map(
        lambda stage: 17 if stage == "F17-market" else int(str(stage)[1:])
    )
    models["coverage_status"] = models.coverage_status.fillna("no_2026_predictions")
    return models


def _pca_ranking(scorecard: pd.DataFrame, output: Path) -> dict:
    metrics = ["margin_mae", "su_accuracy", "brier_score"]
    all_frame = scorecard.loc[scorecard.series_type.eq("model")].copy()
    all_frame["model_id"] = all_frame.model_family.map(FAMILY_TO_MODEL)
    all_frame["generation"] = all_frame.fingerprint.map(lambda stage: 17 if stage == "F17-market" else int(str(stage)[1:]))
    frame = all_frame.loc[all_frame.coverage_status.isin(
        ["scored_2026", "scored_2026_reconstructed_pregame_inputs"]
    )].copy()
    values = frame[metrics].to_numpy(float)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=1)
    standardized = (values - means) / scales
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(standardized, rowvar=False))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    pc1 = eigenvectors[:, 0]
    if not (pc1[0] < 0 and pc1[1] > 0 and pc1[2] < 0):
        pc1 *= -1
    frame["pc1_favorable_sd"] = standardized @ pc1 / np.sqrt(eigenvalues[0])

    pair = pd.MultiIndex.from_product([STAGES, MODELS], names=["fingerprint", "model_id"]).to_frame(index=False)
    pair = pair.merge(
        all_frame[["fingerprint", "model_id", "model_name", "coverage_status", *metrics]],
        on=["fingerprint", "model_id"], how="left", validate="one_to_one",
    )
    pair["coverage_status"] = pair.coverage_status.fillna("no_2026_predictions")
    pair = pair.merge(frame[["fingerprint", "model_id", "pc1_favorable_sd"]],
                      on=["fingerprint", "model_id"], how="left", validate="one_to_one")
    pair["generation"] = pair.fingerprint.map(lambda stage: 17 if stage == "F17-market" else int(stage[1:]))
    pair["model_name"] = pair.model_name.fillna(pair.apply(lambda row: f"scientific_{row.fingerprint}_{row.model_id}", axis=1))
    pair["rank_within_observed_pairs"] = pair.pc1_favorable_sd.rank(method="min", ascending=False).astype("Int64")
    pair = pair.sort_values(["rank_within_observed_pairs", "generation", "model_id"], na_position="last")
    pair.to_csv(output / "scientific_2026_pca_pair_ranking.csv", index=False, float_format="%.6f")

    model_rank = frame.groupby("model_id", as_index=False).agg(
        median_pc1_favorable_sd=("pc1_favorable_sd", "median"),
        scored_fingerprints=("fingerprint", "nunique"),
    )
    model_rank["model_name"] = model_rank.model_id.map(MODEL_LABELS)
    model_rank = model_rank.sort_values("median_pc1_favorable_sd", ascending=False)
    model_rank.insert(0, "rank", range(1, len(model_rank) + 1))
    model_rank.to_csv(output / "scientific_2026_pca_model_ranking.csv", index=False, float_format="%.6f")

    fingerprint_rank = frame.groupby(["fingerprint", "generation"], as_index=False).agg(
        median_pc1_favorable_sd=("pc1_favorable_sd", "median"),
        scored_models=("model_id", "nunique"),
    )
    fingerprint_rank = fingerprint_rank.sort_values("median_pc1_favorable_sd", ascending=False)
    fingerprint_rank.insert(0, "rank", range(1, len(fingerprint_rank) + 1))
    all_fingerprints = pd.DataFrame({"fingerprint": STAGES})
    all_fingerprints["generation"] = all_fingerprints.fingerprint.map(lambda s: 17 if s == "F17-market" else int(s[1:]))
    fingerprint_rank = all_fingerprints.merge(fingerprint_rank, on=["fingerprint", "generation"], how="left", validate="one_to_one")
    fingerprint_rank["coverage_status"] = np.where(
        fingerprint_rank.fingerprint.eq("F17-market"), "scored_2026_pregame_snapshot_partial_market",
        np.where(fingerprint_rank.fingerprint.isin(["F7", "F8"]),
                 "scored_2026_reconstructed_pregame_inputs",
                 np.where(fingerprint_rank.scored_models.notna(), "scored_2026", "no_2026_predictions")),
    )
    fingerprint_rank["partial_pregame_market_models"] = np.where(
        fingerprint_rank.fingerprint.eq("F17-market"), len(MODELS), 0
    )
    fingerprint_rank["reconstructed_pregame_models"] = np.where(
        fingerprint_rank.fingerprint.isin(["F7", "F8"]), len(MODELS), 0
    )
    fingerprint_rank["scored_models"] = fingerprint_rank.scored_models.fillna(0).astype(int)
    fingerprint_rank["rank"] = fingerprint_rank["rank"].astype("Int64")
    fingerprint_rank.to_csv(output / "scientific_2026_pca_fingerprint_ranking.csv", index=False, float_format="%.6f")
    return {
        "method": "PC1 on standardized MAE, straight-up accuracy, and Brier; oriented toward lower MAE/Brier and higher accuracy",
        "scored_pairs": len(frame),
        "explained_variance_ratio_pc1": float(eigenvalues[0] / eigenvalues.sum()),
        "pc1_loadings": dict(zip(metrics, pc1.tolist())),
        "training_scope": "2026 season-to-date only; available F0–F16 forecasts only",
        "partial_market_input_fingerprints_excluded_from_pca": ["F17-market"],
    }


def _gif(frame: pd.DataFrame, *, metric: str, title: str, axis_label: str,
         percent: bool, better_high: bool, output: Path, colors: dict[str, str]) -> None:
    data = frame.dropna(subset=[metric]).copy()
    if data.empty:
        raise ValueError(f"No 2026 observations for {metric}")
    generation_cmap = LinearSegmentedColormap.from_list(
        "tdnet_2026_fingerprint_generation",
        [colors["edge_pink"], colors["gridiron_violet"], colors["figure_primary"]],
    )
    norm = Normalize(vmin=0, vmax=17)
    marker_map = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "P", "M10": "X"}
    apply_tdnet_theme()
    fig = plt.figure(figsize=(10.5, 8.4), facecolor=colors["parchment"])
    ax = fig.add_subplot(111, projection="3d", facecolor=colors["parchment_panel"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(colors["parchment_panel"])
        axis.pane.set_edgecolor(colors["polar_mist"])
        axis._axinfo["grid"]["color"] = colors["polar_mist"]
        axis._axinfo["grid"]["linewidth"] = 0.8
    for model_id, points in data.groupby("model_id", sort=False):
        ax.scatter(
            points.generation, points.model_y, points[metric],
            c=points.generation, cmap=generation_cmap, norm=norm,
            marker=marker_map[model_id], s=70, alpha=0.94,
            edgecolors=colors["parchment"],
            linewidths=0.65, depthshade=False,
        )
    z = data[metric].astype(float)
    lo, hi = float(z.min()), float(z.max())
    pad = max((hi - lo) * 0.12, 0.015 if percent else 0.1)
    ax.set_xlim(-0.5, 17.5)
    ax.set_ylim(-0.5, len(MODELS) - 0.5)
    ax.set_zlim(lo - pad, hi + pad)
    ax.set_xticks([0, 4, 8, 12, 17])
    ax.set_xticklabels(["F0", "F4", "F8", "F12", "F17M"])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([m for m in MODELS])
    ax.set_xlabel("Fingerprint generation", labelpad=12)
    ax.set_ylabel("Scientific architecture", labelpad=11)
    ax.set_zlabel(axis_label, labelpad=9)
    ax.tick_params(colors=colors["midnight_gridiron"], labelsize=9)
    ax.view_init(elev=24, azim=-58)
    fig.suptitle(f"2026 scientific roster · {title}", y=0.96, fontsize=17,
                 fontweight="bold", color=colors["midnight_gridiron"])
    scored_cells = int(frame[["fingerprint", "model_id"]].drop_duplicates().shape[0])
    fig.text(.5, .915,
             f"{scored_cells} pregame-input-checked pairs · F17-market excluded · no Vegas series",
             ha="center", fontsize=10, color=colors["slate"])
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=generation_cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, shrink=.66, pad=.12, aspect=24)
    cbar.set_ticks([0, 4, 8, 12, 17])
    cbar.set_ticklabels(["F0", "F4", "F8", "F12", "F17M"])
    cbar.set_label("Fingerprint generation · red → blue", color=colors["midnight_gridiron"])
    handles = [Line2D([0], [0], marker=marker_map[m], linestyle="", color=colors["slate"],
                      markerfacecolor=colors["gridiron_violet"], markeredgecolor=colors["parchment"], markersize=7,
                      label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .07),
               ncol=3, frameon=False, fontsize=9)
    direction = "Higher is better" if better_high else "Lower is better"
    fig.text(.5, .025,
             f"{direction}. Rotations and PCA use F0–F16; F17-market appears in companion surfaces using archived pregame snapshots where available, with partial target market inputs.",
             ha="center", fontsize=9, color=colors["slate"])
    animation = FuncAnimation(
        fig, lambda k: ax.view_init(elev=24, azim=-58 + 360 * k / 72),
        frames=72, interval=167, blit=False,
    )
    animation.save(output, writer=PillowWriter(fps=6), dpi=90,
                   savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    scorecard = pd.read_csv(PUB / "scientific_2026_current_season_scorecard.csv")
    frame = scorecard.loc[scorecard.series_type.eq("model")].copy()
    frame["model_id"] = frame.model_family.map(FAMILY_TO_MODEL)
    frame["generation"] = frame.fingerprint.map(lambda s: 17 if s == "F17-market" else int(str(s)[1:]))
    frame["model_y"] = frame.model_id.map({model: idx for idx, model in enumerate(MODELS)})
    frame = frame.loc[frame.coverage_status.isin(
        ["scored_2026", "scored_2026_reconstructed_pregame_inputs"]
    )].copy()
    if len(frame) != 102 or frame[["fingerprint", "model_id"]].duplicated().any():
        raise ValueError("Expected 102 distinct pregame-input-checked 2026 score rows (F0–F16).")
    colors = _palette()
    for metric, title, axis_label, high in METRICS:
        _gif(
            frame, metric=metric, title=title, axis_label=axis_label,
            percent=metric in {"su_accuracy", "ats_accuracy"}, better_high=high,
            output=FIGURES / f"scientific_2026_{metric}_3d_rotation.gif", colors=colors,
        )
    pca = _pca_ranking(scorecard, FIGURES)
    pca["rotation_gifs"] = [f"scientific_2026_{metric}_3d_rotation.gif" for metric, *_ in METRICS]
    (FIGURES / "scientific_2026_pca_ranking_manifest.json").write_text(json.dumps(pca, indent=2) + "\n")
    (FIGURES / "point_rotations_README.md").write_text(
        "# 2026 scientific 3D views\n\n"
        "The point-based rotations and PCA use 102 pairs with pregame inputs (F0–F16). The 12 F7/F8 Week 1 forecasts were reconstructed from an archived September 2 fingerprint snapshot that predates the four kickoffs. F17-market uses archived pregame target summaries where available, but eight Aug. 29 openers have no pregame market snapshot and quote-level fields are missing; those six rows remain visible in companion surfaces and pair rankings without PCA scores. Vegas is omitted.\n\n"
        "PCA rankings use standardized margin MAE, straight-up accuracy, and Brier across the 102 pregame-input-checked pairs, oriented so higher scores favor lower MAE/Brier and higher accuracy. Model and fingerprint rankings are medians of the verified pair score.\n"
    )
    print(json.dumps({"gifs": pca["rotation_gifs"], "pca_pairs": pca["scored_pairs"],
                      "pc1_explained_variance_ratio": pca["explained_variance_ratio_pc1"]}, indent=2))


if __name__ == "__main__":
    main()
