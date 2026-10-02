#!/usr/bin/env python3
"""Harvest fingerprint scores and plot all trained scientific models, F0–F17."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/nextgen_rounds_2026"
OUTPUT_ROOT = ROOT / "docs/nextgen_fingerprints/figures"
MODEL_ORDER = ("M1", "M2", "M3", "M4", "M5", "M10")
MODEL_LABELS = {
    "M1": "Linear", "M2": "Spline", "M3": "Random forest",
    "M4": "Boosted trees", "M5": "Neural net", "M10": "KNN",
}
STAGES = ("F09", "F10", "F11", "F12_corrected", "F13", "F14", "F15", "F16", "F17_market")
METRICS = {
    "mae": ("Margin MAE", "points · lower is better", 1.0),
    "upset_recall": ("Upset recall", "% · higher is better", 100.0),
    "winner_accuracy": ("Winner accuracy", "% · higher is better", 100.0),
    "brier_score": ("Brier score", "0–1 · lower is better", 1.0),
}
HISTORICAL_PATH = Path(
    "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
    "scientific_roster_heatmaps/corrected_full_ladder_v1/summary/heatmaps/tables/"
    "selected_complete_fold_results.parquet"
)
ARCHIVED_PATH = ROOT / ".llm-wiki/evidence/Nextgen-Model-Results-2026-09-29.csv"
HISTORICAL_GAMES_PATH = Path(
    "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
    "fingerprint_ladder_v3/canonical_fingerprint.parquet"
)
MARKET_SIDECAR_PATH = Path(
    "/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/canonical/"
    "evaluation_market_sidecar.parquet"
)
BROAD_PREDICTIONS_PATH = Path(
    "/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/experiments/"
    "F06/F06_F_a__M2__m2_01/predictions.parquet"
)
NARROW_PREDICTIONS_PATH = DATA_ROOT / "experiments/F12_original/M2/m2_01/predictions.parquet"
MODEL_COLORS = {
    "M1": "Brass", "M2": "Ion Blue", "M3": "Electric Emerald",
    "M4": "Gridiron Violet", "M5": "Soft Mint", "M10": "Slate Line",
}
MARKET_STAGES = {"F17_market"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_colors() -> dict[str, str]:
    palette = pd.read_csv(ROOT / "docs/style/color_palettes/tdnet_palette.csv")
    return dict(zip(palette.name, palette.hex))


def load_historical(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    folds = pd.read_parquet(path)
    folds = folds.loc[
        folds.objective.eq("margin") & folds.feature_config.isin([f"F{i}" for i in range(9)])
    ].copy()
    if set(folds.model_level.unique()) != set(MODEL_ORDER):
        raise ValueError("Historical F0–F8 source does not contain all six trained models")
    if not folds.status.eq("success").all():
        raise ValueError("Historical F0–F8 source contains an unsuccessful selected fold")
    counts = folds.groupby(["feature_config", "model_level"]).outer_fold.nunique()
    if len(counts) != 9 * len(MODEL_ORDER) or not counts.eq(10).all():
        raise ValueError("Expected ten complete historical folds per F0–F8 model cell")
    rows = []
    for row in folds.itertuples():
        rows.append({
            "generation": int(row.feature_config[1:]), "stage": row.feature_config,
            "model": row.model_level, "setting": "historical rolling folds",
            "test_year": 2015 + int(row.outer_fold), "n_games": int(row.n_rows),
            "mae": float(row.mae), "upset_recall": float(row.upset_correct),
            "winner_accuracy": float(row.winner_accuracy),
            "brier_score": float(row.brier_score), "replicates": 10,
        })
    result = pd.DataFrame(rows)
    result["stage"] = result.generation.map(lambda n: f"F{n}")
    result["source"] = "selected historical rolling-fold results"
    metrics = list(METRICS)
    summary = result.groupby(["generation", "stage", "model", "setting", "source"],
                             as_index=False).agg(
                                 **{metric: (metric, "median") for metric in metrics},
                                 n_games=("n_games", "median"),
                                 replicates=("test_year", "nunique"))
    return summary, result, file_sha256(path)


def _verify_prediction(path: Path, result: dict) -> pd.DataFrame:
    if result.get("status") != "success":
        raise ValueError(f"Unsuccessful run: {path}")
    predictions = path.with_name("predictions.parquet")
    if not predictions.exists() or file_sha256(predictions) != result.get("predictions_sha256"):
        raise ValueError(f"Prediction file missing or hash mismatch: {predictions}")
    frame = pd.read_parquet(predictions).sort_values(["season", "target_game_id"])
    if frame.target_game_id.duplicated().any():
        raise ValueError(f"Duplicate target game in {predictions}")
    return frame


def load_nextgen(data_root: Path) -> pd.DataFrame:
    rows = []
    reference_ids = None
    reference_outcomes = None
    for stage in STAGES:
        generation = int(stage.split("_")[0][1:])
        for model in MODEL_ORDER:
            model_root = data_root / "experiments" / stage / model
            if model in {"M2", "M4"}:
                run_paths = sorted(model_root.glob("*/result.json"))
                if len(run_paths) != 10:
                    raise ValueError(f"Expected ten completed {stage}/{model} setpoints, found {len(run_paths)}")
            else:
                model_root = data_root / "scientific_model_runs/experiments" / stage / model
                run_paths = sorted(model_root.glob("seed_*/result.json"))
                if len(run_paths) != 3:
                    raise ValueError(f"Expected three completed seeds for {stage}/{model}, found {len(run_paths)}")

            metric_replicates = {metric: [] for metric in METRICS}
            n_games_values = []
            for result_path in run_paths:
                result = json.loads(result_path.read_text())
                pred = _verify_prediction(result_path, result)
                eval_rows = pred.loc[pred.season.isin([2024, 2025])]
                season25 = eval_rows.loc[eval_rows.season.eq(2025)]
                if len(season25) != 553:
                    raise ValueError(f"Unexpected 2025 evaluation size for {stage}/{model}: {len(season25)}")
                ids = tuple(season25.target_game_id.astype(int))
                outcomes = tuple(season25.actual_margin.astype(float))
                if reference_ids is None:
                    reference_ids, reference_outcomes = ids, outcomes
                elif ids != reference_ids or outcomes != reference_outcomes:
                    raise ValueError(f"Shared 2025 game IDs/outcomes differ for {stage}/{model}")
                metrics = result.get("metrics_2025")
                if not metrics:
                    raise ValueError(f"Missing 2025 metrics in {result_path}")
                values = {
                    "mae": metrics["mae"], "upset_recall": metrics["upset_accuracy"],
                    "winner_accuracy": metrics["winner_accuracy"],
                    "brier_score": metrics["brier"],
                }
                for metric, value in values.items():
                    metric_replicates[metric].append(float(value))
                n_games_values.append(int(metrics["n_games"]))
            if len(set(n_games_values)) != 1 or n_games_values[0] != 553:
                raise ValueError(f"The 2025 evaluation cohort changed for {stage}/{model}")
            rows.append({
                "generation": generation, "stage": stage, "model": model,
                "setting": "nextgen 2025 common-cohort A",
                **{metric: float(np.median(values)) for metric, values in metric_replicates.items()},
                "n_games": 553, "replicates": len(run_paths),
                "source": "M2/M4 setpoint median" if model in {"M2", "M4"} else "seed median",
            })
    return pd.DataFrame(rows)


def load_f06(archived_path: Path) -> pd.DataFrame:
    archived = pd.read_csv(archived_path)
    frame = archived.loc[
        archived.fingerprint.eq("F06_F_a") & archived.model.isin(["M2", "M4"])
    ].copy()
    if set(frame.model) != {"M2", "M4"}:
        raise ValueError("F06 full-A M2/M4 archived results are missing")
    rows = []
    for row in frame.itertuples():
        rows.append({
            "generation": 6, "stage": "F06", "model": row.model,
            "setting": "nextgen 2025 broad A", "mae": float(row.mae_2025),
            "upset_recall": float(row.upset_accuracy_2025),
            "winner_accuracy": float(row.winner_accuracy_2025),
            "brier_score": float(row.brier_2025), "n_games": int(row.n_games_2025),
            "replicates": int(row.successes), "source": "archived A setpoint median",
        })
    return pd.DataFrame(rows)


def score_vegas(games: pd.DataFrame, lines: pd.DataFrame, setting: str) -> dict:
    joined = games.merge(lines[["target_game_id", "home_spread"]], on="target_game_id",
                         how="left", validate="one_to_one")
    if joined.home_spread.isna().any() or joined.actual_margin.isna().any():
        raise ValueError(f"Incomplete Vegas line/outcome coverage in {setting}")
    actual = joined.actual_margin.to_numpy(float)
    spread = joined.home_spread.to_numpy(float)
    favorite_home = -spread > 0
    actual_home = actual > 0
    meaningful = (spread != 0) & (actual != 0)
    actual_upset = meaningful & (favorite_home != actual_home)
    brier = np.square(favorite_home.astype(float) - actual_home.astype(float)).mean()
    return {
        "setting": setting, "games": len(joined), "actual_upsets": int(actual_upset.sum()),
        "mae": float(np.abs(actual + spread).mean()),
        "winner_accuracy": float((favorite_home == actual_home).mean()),
        "upset_recall": 0.0, "brier_score": float(brier),
        "brier_method": "0/1 probability assigned to the Vegas spread favorite",
    }


def load_vegas(historical_folds: pd.DataFrame, games_path: Path, sidecar_path: Path,
               broad_predictions: Path, narrow_predictions: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    lines = pd.read_parquet(sidecar_path, columns=["target_game_id", "season", "home_spread"])
    if lines.target_game_id.duplicated().any():
        raise ValueError("Vegas market sidecar contains duplicate target game IDs")
    canonical = pd.read_parquet(
        games_path, columns=["keys_season", "keys_team", "next_game_id",
                             "next_game_is_home", "y_next_margin"])
    expected = historical_folds.loc[historical_folds.generation.eq(0)].drop_duplicates(
        "test_year").set_index("test_year").n_games
    annual_rows = []
    for year in range(2015, 2025):
        season = canonical.loc[
            canonical.keys_season.eq(year) & canonical.next_game_id.notna()
            & canonical.y_next_margin.notna()
        ]
        pairs = season.groupby("next_game_id").agg(
            rows=("keys_team", "size"), home=("next_game_is_home", "sum"))
        ids = pairs.index[pairs.rows.eq(2) & pairs.home.eq(1)]
        cohort = season.loc[season.next_game_id.isin(ids) & season.next_game_is_home.eq(True),
                            ["next_game_id", "y_next_margin"]].rename(
                                columns={"next_game_id": "target_game_id", "y_next_margin": "actual_margin"})
        if len(cohort) != int(expected.loc[year]):
            raise ValueError(f"Historical Vegas cohort mismatch for {year}")
        row = score_vegas(cohort, lines.loc[lines.season.eq(year)], f"historical {year}")
        row["test_year"] = year
        annual_rows.append(row)
    annual = pd.DataFrame(annual_rows)
    historical_summary = {"setting": "historical 2015–2024 fold median",
                          "games": int(annual.games.median()),
                          "actual_upsets": int(annual.actual_upsets.median()),
                          **{metric: float(annual[metric].median()) for metric in METRICS},
                          "brier_method": "0/1 probability assigned to Vegas spread favorite"}
    rows = [historical_summary]
    for setting, path, expected_n in (("nextgen 2025 broad", broad_predictions, 757),
                                      ("nextgen 2025 narrow", narrow_predictions, 553)):
        predictions = pd.read_parquet(path, columns=["target_game_id", "season", "actual_margin"])
        cohort = predictions.loc[predictions.season.eq(2025), ["target_game_id", "actual_margin"]]
        if len(cohort) != expected_n:
            raise ValueError(f"Unexpected {setting} cohort size: {len(cohort)}")
        rows.append(score_vegas(cohort, lines.loc[lines.season.eq(2025)], setting))
    return annual, pd.DataFrame(rows)


def configure_style(colors: dict[str, str]) -> None:
    plt.style.use(ROOT / "docs/style/color_palettes/gridiron_light.mplstyle")
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF", "figure.autolayout": False,
        "font.size": 10.5, "axes.titlesize": 11.5, "svg.fonttype": "none",
    })


def metric_axis(metric: str, points: pd.DataFrame, vegas: pd.DataFrame) -> tuple[tuple[float, float], np.ndarray]:
    """Use one rounded, focused vertical scale across every model panel."""
    factor = METRICS[metric][2]
    values = pd.concat([points[metric], vegas[metric]], ignore_index=True).to_numpy(float) * factor
    low, high = float(values.min()), float(values.max())
    span = high - low
    if metric == "upset_recall":
        upper = float(np.ceil((high + max(2.0, span * .1)) / 5.0) * 5.0)
        ticks = np.arange(0.0, upper + .1, 5.0)
        return (0.0, upper), ticks
    if metric == "winner_accuracy":
        pad = max(1.5, span * .12)
        lower = float(np.floor((low - pad) / 5.0) * 5.0)
        upper = float(np.ceil((high + pad) / 5.0) * 5.0)
        return (lower, upper), np.arange(lower, upper + .1, 5.0)
    if metric == "brier_score":
        pad = max(.008, span * .12)
        lower = float(np.floor((low - pad) / .01) * .01)
        upper = float(np.ceil((high + pad) / .01) * .01)
        return (lower, upper), np.arange(lower, upper + .001, .02)
    pad = max(.35, span * .1)
    lower = float(np.floor((low - pad) * 2.0) / 2.0)
    upper = float(np.ceil((high + pad) * 2.0) / 2.0)
    return (lower, upper), np.arange(lower, upper + .01, 1.0)


def render_figures(points: pd.DataFrame, vegas: pd.DataFrame,
                   colors: dict[str, str], output: Path) -> None:
    for metric, (label, direction, factor) in METRICS.items():
        ylim, yticks = metric_axis(metric, points, vegas)
        fig, axes = plt.subplots(3, 2, figsize=(17.6, 13.1), sharex=True)
        for ax, model in zip(axes.flat, MODEL_ORDER):
            base = colors[MODEL_COLORS[model]]
            ax.axvspan(11.7, 16.38, color=colors["Neon Mint"], alpha=.14, zorder=0)
            ax.axvline(8.5, color=colors["Slate Line"], ls="--", lw=1.05, alpha=.45, zorder=0)
            # Same-game Vegas lines: historical folds, F06 broad cohort, F09–F17 narrow cohort.
            baselines = vegas.set_index("setting")[metric]
            for setting, start, end in (
                ("historical 2015–2024 fold median", -.25, 8.28),
                ("nextgen 2025 broad", 5.58, 6.42),
                ("nextgen 2025 narrow", 8.72, 17.38),
            ):
                ax.hlines(float(baselines.loc[setting]) * factor, start, end,
                          color=colors["Midnight Gridiron"], lw=1.8,
                          linestyles=(0, (5, 3)), alpha=.9, zorder=5)

            model_points = points.loc[points.model.eq(model)]
            historical = model_points.loc[model_points.setting.eq("historical rolling folds")]
            for row in historical.itertuples():
                generation = int(row.generation)
                is_market = generation in (7, 8)
                ax.scatter(generation, float(getattr(row, metric)) * factor,
                           s=116 if is_market else 57, marker="*" if is_market else "o",
                           color=colors["Edge Pink"] if is_market else base,
                           edgecolor=colors["Midnight Gridiron"] if is_market else "white",
                           linewidth=.65 if is_market else .8, zorder=4)
            f06 = model_points.loc[model_points.stage.eq("F06")]
            for row in f06.itertuples():
                ax.scatter(6.18, float(getattr(row, metric)) * factor, s=72,
                           marker="D", color=base, edgecolor="white", linewidth=.9, zorder=5)
            recent = model_points.loc[model_points.setting.str.startswith("nextgen")
                                      & ~model_points.stage.eq("F06")]
            for row in recent.itertuples():
                generation = int(row.generation)
                is_market = row.stage in MARKET_STAGES
                ax.scatter(generation, float(getattr(row, metric)) * factor,
                           s=136 if is_market else 72, marker="*" if is_market else "D",
                           color=colors["Edge Pink"] if is_market else base,
                           edgecolor=colors["Midnight Gridiron"] if is_market else "white",
                           linewidth=.75, zorder=7 if is_market else 6)

            ax.set_ylim(*ylim)
            ax.set_yticks(yticks)
            ax.set_xlim(-.55, 17.5)
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.set_ylabel("points" if metric == "mae" else "%" if factor == 100 else "score")
            if metric == "upset_recall":
                ax.text(17.34, 1.3, "Vegas 0%", ha="right", va="bottom",
                        fontsize=8.2, color=colors["Midnight Gridiron"],
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 1.2},
                        zorder=8)
            ax.set_title(f"{model}  /  {MODEL_LABELS[model]}", loc="left",
                         fontsize=12, fontweight="bold", color=colors["Midnight Gridiron"])
        for ax in axes[-1]:
            ax.set_xticks(range(18), [f"F{i}" for i in range(18)], fontsize=8.5)
            ax.set_xlabel("Fingerprint generation  →", labelpad=7)
        fig.suptitle(f"F0 → F17 MARKET  /  {label.upper()}",
                     x=.055, y=.99, ha="left", fontsize=19, fontweight="bold",
                     color=colors["Midnight Gridiron"])
        fig.text(.055, .95,
                 f"All trained scientific models · {label} · {direction} · shared focused scale across model panels",
                 ha="left", fontsize=10.6, color=colors["Slate Line"])
        fig.text(.057, .075,
                 "● F0–F8 historical 10-fold medians   ◆ F06/F09–F17 A-screen medians   ☆ Market-bearing F7, F8, F17",
                 ha="left", fontsize=9.1, color=colors["Slate Line"])
        vegas_note = ("Vegas Brier uses a transparent hard favorite forecast (0/1); spread files contain no implied probabilities."
                      if metric == "brier_score" else
                      "Navy dashed segments: Vegas consensus spread on each exact evaluation cohort; Vegas upset recall is 0%.")
        fig.text(.057, .055, vegas_note, ha="left", fontsize=9.1, color=colors["Slate Line"])
        fig.text(.057, .035,
                 "Cohorts differ: F0–F8 rolling test folds; F06 broad 757-game 2025 A cohort; F09–F17 common 553-game 2025 cohort.",
                 ha="left", fontsize=9.1, color=colors["Slate Line"])
        fig.subplots_adjust(left=.055, right=.99, top=.92, bottom=.13,
                            hspace=.35, wspace=.16)
        stem = output / f"all_fingerprints_f0_f17_{metric}_all_architectures"
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=.2)
        svg = stem.with_suffix(".svg")
        fig.savefig(svg, bbox_inches="tight", pad_inches=.2)
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--historical", type=Path, default=HISTORICAL_PATH)
    parser.add_argument("--archived", type=Path, default=ARCHIVED_PATH)
    parser.add_argument("--historical-games", type=Path, default=HISTORICAL_GAMES_PATH)
    parser.add_argument("--market-sidecar", type=Path, default=MARKET_SIDECAR_PATH)
    parser.add_argument("--broad-predictions", type=Path, default=BROAD_PREDICTIONS_PATH)
    parser.add_argument("--narrow-predictions", type=Path, default=NARROW_PREDICTIONS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    historical, historical_folds, historical_sha = load_historical(args.historical)
    points = pd.concat([historical, load_f06(args.archived), load_nextgen(args.data_root)],
                       ignore_index=True)
    expected = 9 * len(MODEL_ORDER) + 2 + len(STAGES) * len(MODEL_ORDER)
    if len(points) != expected or points.duplicated(["generation", "stage", "model"]).any():
        raise ValueError(f"Unexpected point coverage: {len(points)} rows, expected {expected}")
    annual_vegas, vegas = load_vegas(historical_folds, args.historical_games, args.market_sidecar,
                                    args.broad_predictions, args.narrow_predictions)
    points.to_csv(args.output / "all_architectures_scatter_data.csv", index=False,
                  float_format="%.12g")
    historical_folds.to_csv(args.output / "all_architectures_historical_folds.csv", index=False,
                            float_format="%.12g")
    vegas.to_csv(args.output / "all_architectures_vegas_baseline_data.csv", index=False,
                 float_format="%.12g")
    annual_vegas.to_csv(args.output / "all_architectures_vegas_historical_folds.csv", index=False,
                        float_format="%.12g")
    colors = load_colors()
    configure_style(colors)
    render_figures(points, vegas, colors, args.output)
    outputs = sorted(
        [*args.output.glob("all_fingerprints_f0_f17_*_all_architectures.png"),
         *args.output.glob("all_fingerprints_f0_f17_*_all_architectures.svg"),
         args.output / "all_architectures_scatter_data.csv",
         args.output / "all_architectures_historical_folds.csv",
         args.output / "all_architectures_vegas_baseline_data.csv",
         args.output / "all_architectures_vegas_historical_folds.csv"]
    )
    run_sources = []
    for stage in STAGES:
        for model in MODEL_ORDER:
            base = (args.data_root / "experiments" if model in {"M2", "M4"}
                    else args.data_root / "scientific_model_runs/experiments")
            run_sources.extend(sorted((base / stage / model).glob("*/result.json")))
    run_digest = hashlib.sha256()
    for source in run_sources:
        run_digest.update(str(source.relative_to(args.data_root)).encode("utf-8"))
        run_digest.update(bytes.fromhex(file_sha256(source)))
    receipt = {
        "status": "complete", "plotted_points": len(points),
        "models": list(MODEL_ORDER), "stages": list(STAGES),
        "historical_folds_sha256": historical_sha,
        "archived_f06_sha256": file_sha256(args.archived),
        "nextgen_result_files": len(run_sources),
        "nextgen_results_digest_sha256": run_digest.hexdigest(),
        "nextgen_stage_points": int(points.setting.str.startswith("nextgen").sum()),
        "vegas_baselines": vegas.to_dict("records"),
        "output_sha256": {path.name: file_sha256(path) for path in outputs if path.exists()},
        "output": str(args.output.resolve()),
    }
    (args.output / "all_architectures_harvest_receipt.json").write_text(
        json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
