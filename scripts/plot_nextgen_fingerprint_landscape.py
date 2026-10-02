#!/usr/bin/env python3
"""Render one all-generation M2/M4 fingerprint scatter figure.

The archived table is already a median over successful frozen setpoints. The
new A-screen file is run-level, so it is reduced to the same statistic here.
Historical rolling folds and the two 2025 evaluation cohorts retain separate
Vegas benchmarks, each scored on its own game IDs.
"""

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
OLD_DEFAULT = ROOT / ".llm-wiki/evidence/Nextgen-Model-Results-2026-09-29.csv"
NEW_DEFAULT = ROOT / "data/nextgen_rounds_2026/run_results.csv"
OUTPUT_DEFAULT = ROOT / "docs/nextgen_fingerprints/figures"
HISTORICAL_DEFAULT = Path(
    "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
    "scientific_roster_heatmaps/corrected_full_ladder_v1"
)
HISTORICAL_GAMES_DEFAULT = Path(
    "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
    "fingerprint_ladder_v3/canonical_fingerprint.parquet"
)
MARKET_SIDECAR_DEFAULT = Path(
    "/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/"
    "canonical/evaluation_market_sidecar.parquet"
)
BROAD_PREDICTIONS_DEFAULT = Path(
    "/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/experiments/"
    "F06/F06_F_a__M2__m2_01/predictions.parquet"
)
NARROW_PREDICTIONS_DEFAULT = (
    ROOT / "data/nextgen_rounds_2026/experiments/F12_original/M2/m2_01/predictions.parquet"
)
METRICS = (
    ("mae", "Margin MAE", "points · lower is better"),
    ("winner_accuracy", "Winner accuracy", "percent · higher is better"),
    ("upset_accuracy", "Upset recall", "percent · higher is better"),
)
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_palette() -> dict[str, str]:
    table = pd.read_csv(ROOT / "docs/style/color_palettes/tdnet_palette.csv")
    return dict(zip(table["name"], table["hex"]))


def load_metrics(old_path: Path, new_path: Path) -> pd.DataFrame:
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    required = [f"{m}_{year}" for m, _, _ in METRICS for year in (2024, 2025)]
    if old[required].isna().any().any() or new[required].isna().any().any():
        raise ValueError("Missing score in a plotted metric")
    if set(new.status) != {"success"}:
        raise ValueError("The new A screen has incomplete runs")

    match = old.fingerprint.str.extract(r"^(F\d+)_(F|R)_([abc])$")
    if match.isna().any().any():
        raise ValueError("Unexpected archived fingerprint identifier")
    old = old.assign(generation=match[0], variant=match[1], design=match[2],
                     stage=match[0], source="archived")
    if not set(old.generation).issubset({"F06", "F09", "F10", "F11", "F12"}):
        raise ValueError("Unexpected archived generation")
    old["cohort"] = np.where(old.generation.isin(["F06", "F09", "F10"]),
                              "746/757 games", "626/553 games")
    for year in (2024, 2025):
        expected = 746 if year == 2024 else 757
        other = 626 if year == 2024 else 553
        actual = np.where(old.cohort.eq("746/757 games"), expected, other)
        if not np.array_equal(old[f"n_games_{year}"].to_numpy(), actual):
            raise ValueError(f"Archived {year} cohort changed")

    group = new.groupby(["stage", "model"], sort=False)
    if not group.size().eq(10).all():
        raise ValueError("Every new stage/architecture needs ten setpoints")
    fresh = group[required].median().reset_index()
    fresh["generation"] = fresh.stage.str.extract(r"^(F\d+)")[0]
    fresh["variant"] = "F"
    fresh["design"] = "a"
    fresh["source"] = "new common-cohort A screen"
    fresh["cohort"] = "626/553 games"
    fresh["successes"] = 10
    fresh["failures"] = 0
    fresh["n_games_2024"] = 626
    fresh["n_games_2025"] = 553
    keep = ["stage", "generation", "variant", "design", "model", "source",
            "cohort", "successes", "failures", "n_games_2024", "n_games_2025", *required]
    frame = pd.concat([old[keep], fresh[keep]], ignore_index=True)
    if not ((frame.loc[frame.source.eq("archived"), "generation"] == "F12") &
            frame.loc[frame.source.eq("archived"), "design"].eq("a") &
            frame.loc[frame.source.eq("archived"), "variant"].eq("F")).sum() == 2:
        raise ValueError("The archived F12 A bridge is absent")
    return frame


def load_historical(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Read the frozen F0–F8 margin-objective M2/M4 rolling test folds."""
    manifest_path = root / "job_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    tasks = manifest[manifest.objective.eq("margin") &
                     manifest.model_level.isin(["M2", "M4"])].copy()
    if len(tasks) != 180 or set(tasks.feature_config) != {f"F{i}" for i in range(9)}:
        raise ValueError("Historical M2/M4 task manifest is incomplete")
    rows = []
    digest = hashlib.sha256()
    digest.update(manifest_path.read_bytes())
    for task in tasks.itertuples():
        source = root / "runs" / f"task_{task.task_id:05d}" / "result.parquet"
        digest.update(bytes.fromhex(sha256(source)))
        result = pd.read_parquet(source)
        if len(result) != 1:
            raise ValueError(f"Unexpected historical result length: {source}")
        row = result.iloc[0]
        if (row.status != "success" or row.feature_config != task.feature_config or
                row.model_level != task.model_level or row.outer_fold != task.outer_fold or
                row.objective != "margin"):
            raise ValueError(f"Historical result/manifest mismatch: {source}")
        years = json.loads(row.test_seasons_json)
        if years != [2015 + int(task.outer_fold)]:
            raise ValueError(f"Unexpected historical test year: {source}")
        rows.append({"fingerprint": row.feature_config, "model": row.model_level,
                     "fold": int(row.outer_fold), "test_year": years[0],
                     "n_games": int(row.n_rows), "mae": float(row.mae),
                     "winner_accuracy": float(row.winner_accuracy),
                     "upset_accuracy": float(row.upset_correct)})
    folds = pd.DataFrame(rows)
    if not folds.groupby(["fingerprint", "model"]).fold.nunique().eq(10).all():
        raise ValueError("Historical cell lacks ten distinct rolling folds")
    summary = folds.groupby(["fingerprint", "model"], as_index=False)[
        ["mae", "winner_accuracy", "upset_accuracy"]].median()
    summary["folds"] = 10
    summary["test_years"] = "2015–2024"
    summary["source"] = "historical scientific F0–F8 margin-objective replay"
    return folds, summary, digest.hexdigest()


def _vegas_metrics(games: pd.DataFrame, lines: pd.DataFrame, setting: str) -> dict:
    if games.target_game_id.duplicated().any():
        raise ValueError(f"Duplicate game in {setting} Vegas cohort")
    joined = games.merge(lines[["target_game_id", "home_spread"]],
                         on="target_game_id", how="left", validate="one_to_one")
    if joined.home_spread.isna().any() or joined.actual_margin.isna().any():
        raise ValueError(f"Incomplete Vegas line/outcome coverage: {setting}")
    actual = joined.actual_margin.to_numpy(float)
    spread = joined.home_spread.to_numpy(float)
    implied = -spread
    meaningful = (spread != 0) & (actual != 0)
    actual_upset = meaningful & ((actual > 0) != (implied > 0))
    if not actual_upset.any():
        raise ValueError(f"No actual underdog wins in {setting}")
    return {"setting": setting, "games": len(joined),
            "actual_upsets": int(actual_upset.sum()),
            "mae": float(np.abs(actual - implied).mean()),
            "winner_accuracy": float(((implied > 0) == (actual > 0)).mean()),
            "upset_accuracy": 0.0}


def load_vegas_baselines(
    historical_games_path: Path, sidecar_path: Path,
    broad_predictions_path: Path, narrow_predictions_path: Path,
    historical_folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the same archived consensus spread on each figure cohort."""
    sidecar = pd.read_parquet(sidecar_path,
                              columns=["target_game_id", "season", "home_spread"])
    if sidecar.target_game_id.duplicated().any():
        raise ValueError("Market sidecar has duplicate game IDs")
    canonical = pd.read_parquet(
        historical_games_path,
        columns=["keys_season", "keys_team", "next_game_id",
                 "next_game_is_home", "y_next_margin"],
    )
    historical_rows = []
    expected = historical_folds[historical_folds.fingerprint.eq("F0") &
                                historical_folds.model.eq("M2")].set_index("test_year")
    for year in range(2015, 2025):
        season = canonical[canonical.keys_season.eq(year) &
                           canonical.next_game_id.notna() &
                           canonical.y_next_margin.notna()]
        pairs = season.groupby("next_game_id").agg(
            rows=("keys_team", "size"), home=("next_game_is_home", "sum"))
        ids = pairs[pairs.rows.eq(2) & pairs.home.eq(1)].index
        games = season[season.next_game_id.isin(ids) &
                       season.next_game_is_home.eq(True)][
                           ["next_game_id", "y_next_margin"]].rename(
                               columns={"next_game_id": "target_game_id",
                                        "y_next_margin": "actual_margin"})
        if len(games) != int(expected.loc[year, "n_games"]):
            raise ValueError(f"Historical {year} Vegas games differ from test fold")
        scored = _vegas_metrics(games, sidecar[sidecar.season.eq(year)],
                                f"historical {year} rolling fold")
        scored["test_year"] = year
        historical_rows.append(scored)
    annual = pd.DataFrame(historical_rows)
    historical = {"setting": "historical 2015–2024 fold median",
                  "games": int(annual.games.median()),
                  "actual_upsets": int(annual.actual_upsets.median()),
                  **{metric: float(annual[metric].median())
                     for metric, _, _ in METRICS}}
    rows = [historical]
    for setting, path, count in (
        ("nextgen 2025 broad", broad_predictions_path, 757),
        ("nextgen 2025 narrow", narrow_predictions_path, 553),
    ):
        games = pd.read_parquet(path,
                                columns=["target_game_id", "season", "actual_margin"])
        games = games[games.season.eq(2025)][["target_game_id", "actual_margin"]]
        if len(games) != count:
            raise ValueError(f"Unexpected {setting} game count")
        rows.append(_vegas_metrics(games, sidecar[sidecar.season.eq(2025)],
                                   setting))
    return annual, pd.DataFrame(rows)


def setup_style(colors: dict[str, str]) -> None:
    plt.style.use(ROOT / "docs/style/color_palettes/gridiron_light.mplstyle")
    plt.rcParams.update({"figure.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF",
                         "axes.facecolor": "#FFFFFF", "figure.autolayout": False,
                         "font.size": 11,
                         "axes.titlesize": 13, "svg.fonttype": "none"})


def plot_all_generations(frame, historical, vegas, colors, output) -> int:
    """One scatter map: F0–F8 rolling folds plus F06/F09–F17 A screens."""
    archived_a = frame[frame.source.eq("archived") & frame.variant.eq("F") &
                       frame.design.eq("a") &
                       frame.generation.isin(["F06", "F09", "F10", "F11"])].copy()
    fresh_a = frame[frame.source.ne("archived") &
                    frame.stage.isin(["F12_corrected", "F13", "F14", "F15",
                                     "F16", "F17_market"])].copy()
    if len(archived_a) != 8 or len(fresh_a) != 12 or len(historical) != 18:
        raise ValueError("Unified figure is missing a generation/architecture cell")
    plotted = []
    for record in historical.itertuples():
        plotted.append({"generation": record.fingerprint, "stage": record.fingerprint,
                        "model": record.model, "setting": "historical rolling folds",
                        "mae": record.mae,
                        "winner_accuracy": record.winner_accuracy,
                        "upset_accuracy": record.upset_accuracy})
    for group, setting in ((archived_a, "nextgen 2025 archived A"),
                           (fresh_a, "nextgen 2025 new A")):
        for record in group.itertuples():
            plotted.append({"generation": record.generation, "stage": record.stage,
                            "model": record.model, "setting": setting,
                            "mae": record.mae_2025,
                            "winner_accuracy": record.winner_accuracy_2025,
                            "upset_accuracy": record.upset_accuracy_2025})
    pd.DataFrame(plotted).to_csv(output / "all_fingerprints_scatter_data.csv",
                                 index=False, float_format="%.12g")
    fig, axes = plt.subplots(3, 2, figsize=(17.6, 11.2), sharex=True)
    for row, (metric, title, direction) in enumerate(METRICS):
        all_values = (list(historical[metric].astype(float)) +
                      list(archived_a[f"{metric}_2025"].astype(float)) +
                      list(fresh_a[f"{metric}_2025"].astype(float)) +
                      list(vegas[metric].astype(float)))
        factor = 1 if metric == "mae" else 100
        lo, hi = min(all_values) * factor, max(all_values) * factor
        pad = max((hi - lo) * .12, .22 if metric == "mae" else 1.8)
        for col, model in enumerate(("M2", "M4")):
            ax = axes[row, col]
            base = colors["Ion Blue"] if model == "M2" else colors["Gridiron Violet"]
            ax.axvspan(11.7, 16.38, color=colors["Neon Mint"], alpha=.14, zorder=0)
            ax.axvline(8.5, color=colors["Slate Line"], ls="--", lw=1.1,
                       alpha=.48, zorder=0)
            vegas_values = vegas.set_index("setting")[metric]
            for setting, start, end in (
                ("historical 2015–2024 fold median", -.18, 8.2),
                ("nextgen 2025 broad", 8.65, 10.25),
                ("nextgen 2025 narrow", 10.75, 17.32),
            ):
                ax.hlines(float(vegas_values.loc[setting]) * factor, start, end,
                          color=colors["Midnight Gridiron"], lw=1.75,
                          linestyles=(0, (5, 3)), alpha=.82, zorder=2)
            history = historical[historical.model.eq(model)]
            for record in history.itertuples():
                generation = int(record.fingerprint[1:])
                market = generation in (7, 8)
                ax.scatter(generation - (.12 if generation == 6 else 0),
                           float(getattr(record, metric)) * factor,
                           s=170 if market else 72, marker="*" if market else "o",
                           color=colors["Edge Pink"] if market else base,
                           edgecolor=colors["Midnight Gridiron"] if market else "white",
                           linewidth=.65 if market else 1, zorder=4)
            past = archived_a[archived_a.model.eq(model)]
            for record in past.itertuples():
                generation = int(record.generation[1:])
                ax.scatter(generation + (.12 if generation == 6 else 0),
                           float(getattr(record, f"{metric}_2025")) * factor,
                           s=82, marker="D", color=base, edgecolor="white",
                           linewidth=1, zorder=5)
            recent = fresh_a[fresh_a.model.eq(model)]
            for record in recent.itertuples():
                market = record.stage == "F17_market"
                generation = int(record.generation[1:])
                ax.scatter(generation,
                           float(getattr(record, f"{metric}_2025")) * factor,
                           s=195 if market else 90, marker="*" if market else "D",
                           color=colors["Edge Pink"] if market else base,
                           edgecolor=colors["Midnight Gridiron"] if market else "white",
                           linewidth=.8, zorder=7 if market else 6)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-.55, 17.5)
            ax.grid(axis="y", alpha=.17)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.set_ylabel("points" if metric == "mae" else "%")
            ax.set_title(f"{model}  /  {title}  ·  {direction}", loc="left",
                         fontsize=12, fontweight="bold",
                         color=colors["Midnight Gridiron"])
    for ax in axes[-1]:
        ax.set_xticks(range(18), [f"F{i}" for i in range(18)], fontsize=9)
        ax.set_xlabel("Fingerprint generation  →", labelpad=8)
    fig.suptitle("F0 → F17 MARKET  /  WHERE THE FINGERPRINTS STAND",
                 x=.055, y=.993, ha="left", fontsize=19, fontweight="bold",
                 color=colors["Midnight Gridiron"])
    fig.text(.055, .963,
             "One scatter map for M2 and M4  ·  MAE, winner accuracy and upset recall"
             "  ·  F12–F16 market-free additions have reached a measured shelf",
             ha="left", fontsize=10.8, color=colors["Slate Line"])
    fig.text(.06, .061,
             "● Historical F0–F8: median of 10 rolling folds (2015–24)   ◆ New"
             " F06/F09–F17 A screens: 2025 setpoint median   ☆ Market-bearing"
             " F7, F8 and F17",
             ha="left", fontsize=9.3, color=colors["Slate Line"])
    fig.text(.06, .042,
             "Dark navy dashed segments: Vegas consensus spread scored on each"
             " evaluation setting (historical folds, 757-game broad, 553-game narrow).",
             ha="left", fontsize=9.3, color=colors["Slate Line"])
    fig.text(.06, .023,
             "Studies differ: F0–F8 rolling tests; F06–F10 757 development"
             " games; F11–F17 553 development games. Scores across these"
             " boundaries are descriptive, not paired gains.",
             ha="left", fontsize=9.3, color=colors["Slate Line"])
    fig.subplots_adjust(left=.056, right=.99, top=.916, bottom=.13,
                        hspace=.35, wspace=.16)
    save(fig, output / "all_fingerprints_f0_f17_scatter")
    return len(plotted)


def save(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=.22)
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight", pad_inches=.22)
    # Matplotlib indents path data with trailing spaces; keep the export Git clean.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived", type=Path, default=OLD_DEFAULT)
    parser.add_argument("--new", type=Path, default=NEW_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--historical-root", type=Path, default=HISTORICAL_DEFAULT)
    parser.add_argument("--historical-games", type=Path,
                        default=HISTORICAL_GAMES_DEFAULT)
    parser.add_argument("--market-sidecar", type=Path,
                        default=MARKET_SIDECAR_DEFAULT)
    parser.add_argument("--broad-predictions", type=Path,
                        default=BROAD_PREDICTIONS_DEFAULT)
    parser.add_argument("--narrow-predictions", type=Path,
                        default=NARROW_PREDICTIONS_DEFAULT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    colors = load_palette()
    setup_style(colors)
    frame = load_metrics(args.archived, args.new)
    folds, historical, historical_digest = load_historical(args.historical_root)
    folds.to_csv(args.output / "historical_f0_f8_folds.csv", index=False,
                 float_format="%.12g")
    historical.to_csv(args.output / "historical_f0_f8_summary.csv", index=False,
                      float_format="%.12g")
    annual_vegas, vegas = load_vegas_baselines(
        args.historical_games, args.market_sidecar,
        args.broad_predictions, args.narrow_predictions, folds)
    annual_vegas.to_csv(args.output / "vegas_historical_folds.csv", index=False,
                        float_format="%.12g")
    vegas.to_csv(args.output / "vegas_baseline_data.csv", index=False,
                 float_format="%.12g")
    count = plot_all_generations(frame, historical, vegas, colors, args.output)
    print(f"Archived source SHA-256: {sha256(args.archived)}")
    print(f"New run source SHA-256: {sha256(args.new)}")
    print(f"Historical manifest/results digest: {historical_digest}")
    print(f"Vegas sidecar SHA-256: {sha256(args.market_sidecar)}")
    print(f"Plotted {count} M2/M4 generation points in {args.output}")


if __name__ == "__main__":
    main()
