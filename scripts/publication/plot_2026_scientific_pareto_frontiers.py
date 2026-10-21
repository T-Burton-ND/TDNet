"""Poster-ready resource Pareto plots for the frozen F0–F6 scientific roster.

The two measured costs are input feature count and serialized checkpoint size.
Checkpoint size is a deployment-footprint proxy, not a universal measure of
statistical model capacity. All comparisons use identical game IDs per metric.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "publication/2026/week_05/post_game/scientific/current_season_f0_f19"
OUTPUT = ROOT / "publication/2026/research/fingerprint_model_pareto"
MANIFESTS = ROOT / "docs/publication_2026/feature_manifests"
CHECKPOINTS = Path(
    "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
    "scientific_roster_refits/f0_f8_margin_through_2025_v1/cells"
)
MODELS = ("M1", "M2", "M3", "M4", "M5", "M10")
MODEL_LABELS = {
    "M1": "Linear", "M2": "Spline", "M3": "Tree",
    "M4": "Boosted", "M5": "Neural", "M10": "KNN",
}
MARKERS = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "P", "M10": "X"}
FINGERPRINTS = tuple(f"F{i}" for i in range(7))
PAPER = "#F7F4ED"
NAVY = "#233349"
GOLD = "#B88416"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def costs() -> pd.DataFrame:
    path = OUTPUT / "model_costs.csv"
    features = {}
    for fingerprint in FINGERPRINTS:
        manifest = MANIFESTS / f"{fingerprint}.json"
        record = json.loads(manifest.read_text())
        if record["feature_count"] != len(record["feature_names"]):
            raise ValueError(f"Feature manifest count mismatch: {fingerprint}")
        features[fingerprint] = int(record["feature_count"])
    if path.exists():
        frame = pd.read_csv(path)
    else:
        rows = []
        for fingerprint in FINGERPRINTS:
            for model_id in MODELS:
                status_path = CHECKPOINTS / fingerprint / model_id / "status.json"
                status = json.loads(status_path.read_text())
                if status["feature_config"] != fingerprint or status["model_level"] != model_id:
                    raise ValueError(f"Checkpoint identity mismatch: {status_path}")
                if status.get("prospective_2026_used_for_fit") is not False:
                    raise ValueError(f"2026 fitting detected: {status_path}")
                rows.append({
                    "fingerprint": fingerprint,
                    "model_id": model_id,
                    "feature_count": features[fingerprint],
                    "checkpoint_bytes": int(status["checkpoint_bytes"]),
                    "checkpoint_sha256": status["checkpoint_sha256"],
                    "status_sha256": digest(status_path),
                })
        frame = pd.DataFrame(rows)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    if len(frame) != 42 or frame.duplicated(["fingerprint", "model_id"]).any():
        raise ValueError("Expected 42 unique frozen model costs")
    if set(zip(frame.fingerprint, frame.model_id)) != {
        (fingerprint, model_id) for fingerprint in FINGERPRINTS for model_id in MODELS
    }:
        raise ValueError("Cost roster does not match F0–F6 scientific panel")
    if any(frame.loc[frame.fingerprint.eq(fingerprint), "feature_count"].ne(count).any()
           for fingerprint, count in features.items()):
        raise ValueError("Frozen cost ledger disagrees with feature manifests")
    if frame.checkpoint_bytes.le(0).any():
        raise ValueError("Checkpoint size must be positive")
    return frame


def scores() -> tuple[pd.DataFrame, dict[str, float | int]]:
    source = SOURCE / "scientific_2026_model_game_results.parquet"
    frozen_manifest = json.loads((SOURCE / "manifest.json").read_text())
    if digest(source) != frozen_manifest["outputs"]["model_games_parquet"]["sha256"]:
        raise ValueError("Scientific model-game source changed")
    frame = pd.read_parquet(source)
    all_games = frame.drop_duplicates("game_id")
    spread_games = set(all_games.loc[all_games.market_home_spread.notna(), "game_id"])
    probability_games = set(all_games.loc[all_games.market_home_probability.notna(), "game_id"])
    if len(spread_games) != 263 or len(probability_games) != 259:
        raise ValueError("Archived market cohort changed")
    if not probability_games <= spread_games:
        raise ValueError("Market Brier cohort is not inside MAE cohort")
    targets = all_games.set_index("game_id")
    spread = targets.loc[sorted(spread_games)]
    probability = targets.loc[sorted(probability_games)]
    vegas_mae = (spread.actual_home_margin + spread.market_home_spread).abs().mean()
    vegas_brier = (
        probability.market_home_probability
        - probability.actual_home_margin.gt(0).astype(float)
    ).pow(2).mean()
    rows = []
    for (fingerprint, model_id), group in frame.loc[frame.fingerprint.isin(FINGERPRINTS)].groupby(
        ["fingerprint", "model_id"]
    ):
        mae_rows = group.loc[group.game_id.isin(spread_games)]
        brier_rows = group.loc[group.game_id.isin(probability_games)]
        if len(mae_rows) != 263 or len(brier_rows) != 259 or not group.forecast_status.eq("available").all():
            raise ValueError(f"Incomplete paired cohort: {fingerprint}/{model_id}")
        rows.append({
            "fingerprint": fingerprint,
            "model_id": model_id,
            "mae_games": len(mae_rows),
            "mae": float(mae_rows.absolute_margin_error.mean()),
            "brier_games": len(brier_rows),
            "brier": float(brier_rows.brier_error.mean()),
        })
    if len(rows) != 42:
        raise ValueError("Expected 42 performance rows")
    market = {"mae": float(vegas_mae), "brier": float(vegas_brier),
              "mae_games": len(spread_games), "brier_games": len(probability_games)}
    return pd.DataFrame(rows), market


def pareto_flags(frame: pd.DataFrame, metric: str) -> np.ndarray:
    """Nondominance under fewer features, fewer bytes, and lower error."""
    values = frame[["feature_count", "checkpoint_bytes", metric]].to_numpy(float)
    result = np.ones(len(values), dtype=bool)
    for i, candidate in enumerate(values):
        dominated = np.all(values <= candidate, axis=1) & np.any(values < candidate, axis=1)
        result[i] = not dominated.any()
    return result


def _format_score(value: float, metric: str) -> str:
    return f"{value:.3f}" if metric == "mae" else f"{value:.4f}"


def plot(frame: pd.DataFrame, market: dict[str, float | int], metric: str, path: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": PAPER, "axes.facecolor": "#FFFCF7",
    })
    fig = plt.figure(figsize=(18, 8.5), facecolor=PAPER)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.13, 1), wspace=0.22,
                            left=0.07, right=0.98, top=0.81, bottom=0.23)
    map_axis, budget_axis = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    feature_levels = sorted(frame.feature_count.unique())
    feature_position = {count: index for index, count in enumerate(feature_levels)}
    norm = Normalize(frame[metric].min(), frame[metric].max())
    cmap = plt.get_cmap("viridis_r")
    for model_id in MODELS:
        subset = frame.loc[frame.model_id.eq(model_id)]
        map_axis.scatter(subset.feature_count.map(feature_position), subset.checkpoint_bytes / 2**20,
                         c=subset[metric], cmap=cmap, norm=norm, marker=MARKERS[model_id],
                         s=116, edgecolor="white", linewidth=1.1, zorder=3)
    efficient = frame.loc[frame[f"pareto_{metric}"]]
    map_axis.scatter(efficient.feature_count.map(feature_position), efficient.checkpoint_bytes / 2**20,
                     facecolors="none", edgecolors=NAVY, linewidth=1.4, s=235, zorder=4)
    for fingerprint, model_id in (("F3", "M4"), ("F6", "M4")):
        row = frame.loc[frame.fingerprint.eq(fingerprint) & frame.model_id.eq(model_id)].iloc[0]
        map_axis.annotate(f"{fingerprint}/{model_id}",
                          (feature_position[row.feature_count], row.checkpoint_bytes / 2**20),
                          xytext=(-48 if fingerprint == "F6" else 10, 8),
                          textcoords="offset points", fontsize=9,
                          color=NAVY, weight="bold")
    map_axis.set(yscale="log", xlabel="Fingerprint / declared input features",
                 ylabel="Serialized checkpoint size (MiB)")
    map_axis.set_xticks(range(7))
    map_axis.set_xticklabels([f"F{i}\n{count}" for i, count in enumerate(feature_levels)])
    map_axis.set_xlim(-0.35, 6.35)
    map_axis.set_title("Cost map · outlined points are Pareto efficient", loc="left", weight="bold", color=NAVY)
    map_axis.grid(alpha=0.18)
    map_axis.spines[["top", "right"]].set_visible(False)
    marker_legend = [Line2D([0], [0], marker=MARKERS[model_id], linestyle="",
                            markerfacecolor="#647A91", markeredgecolor="white",
                            markersize=8, label=f"{model_id} {MODEL_LABELS[model_id]}")
                     for model_id in MODELS]
    map_axis.legend(handles=marker_legend, loc="upper center", bbox_to_anchor=(.5, -.17),
                    ncol=3, fontsize=8, frameon=False)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=map_axis,
                           fraction=0.04, pad=0.02)
    colorbar.set_label("MAE (points)" if metric == "mae" else "Brier score")

    budgets = (("≤0.1 MiB", 0.1, "#2B617E", ":"),
               ("≤0.5 MiB", 0.5, "#80549D", "--"),
               ("≤3 MiB", 3.0, "#D04486", "-"),
               ("Any model size", float("inf"), NAVY, "-"))
    for label, cap_mib, color, style in budgets:
        best = []
        for count in feature_levels:
            eligible = frame.loc[frame.feature_count.le(count)
                                 & frame.checkpoint_bytes.le(cap_mib * 2**20)]
            best.append(float(eligible[metric].min()) if not eligible.empty else np.nan)
        budget_axis.step(range(7), best, where="post", color=color,
                         lw=2.6 if np.isinf(cap_mib) else 1.9, ls=style,
                         marker="o", markersize=4, label=label)
    budget_axis.axhline(float(market[metric]), color=GOLD, lw=2.2,
                         ls="--", label=f"Vegas · {_format_score(float(market[metric]), metric)}")
    budget_axis.set_xticks(range(7))
    budget_axis.set_xticklabels([f"F{i}\n{count}" for i, count in enumerate(feature_levels)])
    budget_axis.set_xlim(-0.35, 6.35)
    budget_axis.set_xlabel("Maximum available input features · fingerprint / count")
    budget_axis.set_ylabel("Best observed MAE (points)" if metric == "mae" else "Best observed Brier score")
    budget_axis.set_title("Best observed score within each resource budget", loc="left", weight="bold", color=NAVY)
    budget_axis.grid(alpha=0.18)
    budget_axis.spines[["top", "right"]].set_visible(False)
    budget_axis.legend(loc="lower left", bbox_to_anchor=(0.02, 0.15),
                       ncol=2, frameon=False, fontsize=9)
    f0 = frame.loc[frame.fingerprint.eq("F0"), metric].min()
    f6 = frame.loc[frame.fingerprint.eq("F6"), metric].min()
    fig.suptitle("TDNet 2026 · information and model-footprint Pareto frontier",
                 fontsize=21, color=NAVY, weight="bold", y=0.965)
    fig.text(0.5, 0.898,
             f"Market-free F0–F6 · 42 cells · {int(market[metric + '_games'])} paired Week 1–5 games · "
             f"+225 features: best {_format_score(float(f0), metric)}→{_format_score(float(f6), metric)} "
             f"(Vegas {_format_score(float(market[metric]), metric)})",
             ha="center", fontsize=11, color=NAVY)
    fig.text(0.5, 0.055,
             "Pareto efficiency minimizes input count, checkpoint bytes, and error among the 42 tested cells. "
             "Checkpoint size measures deployment footprint, not statistical capacity.\n"
             "Vegas is a same-game comparator, not a trained-model cost point. Historical quote-level timing is unverified. "
             "Frontiers were selected after viewing 2026 results.",
             ha="center", fontsize=8.6, color="#4E5660", wrap=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    values, market = scores()
    table = values.merge(costs(), on=["fingerprint", "model_id"], validate="one_to_one")
    for metric in ("mae", "brier"):
        table[f"pareto_{metric}"] = pareto_flags(table, metric)
    table = table.sort_values(["feature_count", "model_id"]).reset_index(drop=True)
    metrics_path = OUTPUT / "paired_model_frontier.csv"
    table.to_csv(metrics_path, index=False)
    outputs = {"costs": OUTPUT / "model_costs.csv", "paired_metrics": metrics_path}
    for metric in ("mae", "brier"):
        for extension in ("png", "svg"):
            path = OUTPUT / f"pareto_{metric}.{extension}"
            plot(table, market, metric, path)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            outputs[f"{metric}_{extension}"] = path
    readme = OUTPUT / "README.md"
    f0 = table.loc[table.fingerprint.eq("F0")]
    f6 = table.loc[table.fingerprint.eq("F6")]
    readme.write_text(
        "# F0–F6 information × model-footprint Pareto figures\n\n"
        "The MAE figure compares all 42 frozen market-free scientific cells and the archived "
        "market spread on the same 263 Week 1–5 2026 games. The Brier figure restricts every "
        "series to the same 259 games with an archived market probability. The seven input "
        "counts come from the declared F0–F6 feature manifests; checkpoint byte counts "
        "come from the frozen through-2025 model statuses. Point outlines mark the three-cost "
        "Pareto set under fewer features, smaller serialized checkpoint, and lower error. "
        "The right-hand curves show the best observed score within cumulative feature and "
        "checkpoint-size budgets. Checkpoint size is a reproducible deployment-footprint "
        "measure, not a universal statistical-complexity scale. The curves are selected on "
        "the same 2026 outcomes and are descriptive; do not read them as prospectively "
        "optimized strategies or an irreducible-error estimate. The Vegas quote-level "
        "timestamps for historical market inputs remain unverified. The full F0–F19 "
        "roster has separate market-inclusive tiers and is excluded from this market-free "
        "resource frontier.\n\n"
        f"On these paired cohorts, best F0-to-F6 MAE moves from {f0.mae.min():.4f} "
        f"to {f6.mae.min():.4f} as declared inputs grow from 2 to 227; Vegas is "
        f"{market['mae']:.4f}. Best Brier moves from {f0.brier.min():.5f} to "
        f"{f6.brier.min():.5f}; Vegas is {market['brier']:.5f}. The best cell at "
        "each fingerprint was selected using the same 2026 outcomes, so these "
        "values are descriptive and do not establish diminishing marginal returns.\n"
    )
    outputs["readme"] = readme
    source = SOURCE / "scientific_2026_model_game_results.parquet"
    manifest = {
        "season": 2026,
        "cohort_weeks": [1, 2, 3, 4, 5],
        "models": 42,
        "market_free": True,
        "mae_games": market["mae_games"],
        "brier_games": market["brier_games"],
        "vegas_mae": market["mae"],
        "vegas_brier": market["brier"],
        "pareto_cells": {metric: int(table[f"pareto_{metric}"].sum()) for metric in ("mae", "brier")},
        "source_sha256": {str(source.relative_to(ROOT)): digest(source),
                          **{str((MANIFESTS / f"{fingerprint}.json").relative_to(ROOT)):
                             digest(MANIFESTS / f"{fingerprint}.json") for fingerprint in FINGERPRINTS}},
        "outputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
                    for name, path in outputs.items()},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "vegas": market,
                      "pareto_cells": manifest["pareto_cells"]}, indent=2))


if __name__ == "__main__":
    main()
