#!/usr/bin/env python3
"""Build publication tables and heat maps for the scientific F0–F17 fingerprint archive."""
from __future__ import annotations

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
HISTORICAL = Path(
    "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
    "scientific_roster_heatmaps/corrected_full_ladder_v1/summary/heatmaps/tables/"
    "selected_complete_fold_results.parquet"
)
CANONICAL = Path(
    "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
    "fingerprint_ladder_v3/canonical_fingerprint.parquet"
)
MARKET = Path(
    "/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/canonical/"
    "evaluation_market_sidecar.parquet"
)
ARCHIVED = ROOT / ".llm-wiki/evidence/Nextgen-Model-Results-2026-09-29.csv"
FIGURE_DATA = ROOT / "docs/nextgen_fingerprints/figures"
OUT = ROOT / "publication/scientific_fingerprint_archive/f0_f17_market"

MODELS = ("M1", "M2", "M3", "M4", "M5", "M10")
STAGES = ("F09", "F10", "F11", "F12_corrected", "F13", "F14", "F15", "F16", "F17_market")
STAGE_ORDER = [f"F{i}" for i in range(9)] + ["F06 broad", "F9", "F10", "F11", "F12 corrected",
                                                "F13", "F14", "F15", "F16", "F17 market"]
MODEL_LABELS = {
    "M1": "Linear", "M2": "Spline", "M3": "Random forest",
    "M4": "Boosted trees", "M5": "Neural net", "M10": "KNN",
}
PARQUET_SHA = "predictions_sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression,
                 float_format="%.8g")


def summarize_predictions(frame: pd.DataFrame) -> dict[str, float | int]:
    actual = frame.actual_margin.to_numpy(float)
    pred = frame.predicted_margin.to_numpy(float)
    prob = frame.home_win_probability.to_numpy(float)
    winner = ((pred > 0) == (actual > 0))
    if "home_spread" in frame:
        spread = frame.home_spread.to_numpy(float)
        ats_actual = actual + spread
        ats_pred = pred + spread
        ats_valid = (np.isfinite(ats_actual) & np.isfinite(ats_pred)
                     & (np.abs(ats_actual) > 1e-8) & (np.abs(ats_pred) > 1e-8))
        ats_accuracy = float(((ats_pred[ats_valid] > 0) == (ats_actual[ats_valid] > 0)).mean()) if ats_valid.any() else np.nan
        ats_games = int(ats_valid.sum())
        meaningful = np.isfinite(spread) & (spread != 0) & (actual != 0)
        actual_chalk = (actual > 0) == (spread < 0)
        actual_upset = meaningful & ~actual_chalk
        upset_recall = float(winner[actual_upset].mean()) if actual_upset.any() else np.nan
    else:
        valid_ats = frame.ats_correct.notna()
        ats_accuracy = float(frame.loc[valid_ats, "ats_correct"].mean()) if valid_ats.any() else np.nan
        ats_games = int(valid_ats.sum())
        upset_recall = np.nan
    return {
        "n_games": int(len(frame)),
        "margin_mae": float(np.abs(pred - actual).mean()),
        "winner_accuracy": float(winner.mean()),
        "upset_recall": upset_recall,
        "brier_score": float(np.square(prob - (actual > 0).astype(float)).mean()),
        "ats_games": ats_games,
        "ats_accuracy": ats_accuracy,
    }


def load_historical() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not HISTORICAL.is_file():
        raise FileNotFoundError(HISTORICAL)
    raw = pd.read_parquet(HISTORICAL)
    folds = raw.loc[
        raw.objective.eq("margin")
        & raw.feature_config.isin([f"F{i}" for i in range(9)])
        & raw.model_level.isin(MODELS)
        & raw.status.eq("success")
    ].copy()
    counts = folds.groupby(["feature_config", "model_level", "outer_fold"]).size()
    if len(counts) != 9 * len(MODELS) * 10 or not counts.eq(1).all():
        raise ValueError("Historical selected-fold data are not a complete unique F0–F8 matrix")
    folds["stage"] = folds.feature_config
    folds["model"] = folds.model_level
    folds["season"] = folds.outer_fold.astype(int) + 2015
    folds["period_type"] = "heldout_season"
    folds["n_games"] = folds.n_rows.astype(int)
    folds["margin_mae"] = folds.mae.astype(float)
    folds["winner_accuracy"] = folds.winner_accuracy.astype(float)
    folds["upset_recall"] = folds.upset_correct.astype(float)
    folds["brier_score"] = folds.brier_score.astype(float)
    folds["ats_accuracy"] = folds.ats_accuracy.astype(float)
    return raw, folds


def historical_cumulative(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (stage, model), group in folds.groupby(["stage", "model"], sort=False):
        cumulative: list[pd.Series] = []
        for season, annual in group.sort_values("season").groupby("season", sort=True):
            cumulative.append(annual.iloc[0])
            part = pd.DataFrame(cumulative)
            n = part.n_games.to_numpy(float)
            ats_n = part.ats_n.fillna(0).to_numpy(float)
            row = {
                "stage": stage, "model": model, "model_label": MODEL_LABELS[model],
                "cohort": "historical rolling-origin test seasons",
                "through_season": int(season), "period_type": "cumulative_heldout_seasons",
                "n_games": int(n.sum()),
                "margin_mae": float(np.average(part.margin_mae, weights=n)),
                "winner_accuracy": float(np.average(part.winner_accuracy, weights=n)),
                "upset_recall": float(part.upset_recall.mean()),
                "brier_score": float(np.average(part.brier_score, weights=n)),
                "ats_games": int(ats_n.sum()),
                "ats_accuracy": float(np.average(part.ats_accuracy, weights=ats_n)) if ats_n.sum() else np.nan,
                "upset_aggregation": "unweighted mean of held-out-season recall values",
            }
            rows.append(row)
    return pd.DataFrame(rows)


def load_nextgen() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not all(p.is_file() for p in (CANONICAL, MARKET)):
        raise FileNotFoundError("Required canonical game-week lookup or evaluation market sidecar is missing")
    canonical = pd.read_parquet(CANONICAL, columns=["keys_season", "next_game_id", "next_week", "next_game_is_home"])
    lookup = canonical.loc[canonical.next_game_id.notna() & canonical.next_game_is_home.eq(True),
                           ["keys_season", "next_game_id", "next_week"]].copy()
    lookup["target_game_id"] = lookup.next_game_id.astype("int64")
    lookup = lookup.rename(columns={"keys_season": "season", "next_week": "week"})
    lookup = lookup[["target_game_id", "season", "week"]].drop_duplicates()
    if lookup.target_game_id.duplicated().any():
        raise ValueError("Canonical target game/week lookup is not unique")
    market = pd.read_parquet(MARKET)
    market = market.loc[market["selection"].eq("median_available_providers"),
                        ["target_game_id", "home_spread"]].copy()
    if market.target_game_id.duplicated().any():
        raise ValueError("Selected market sidecar has duplicate game IDs")

    predictions: list[pd.DataFrame] = []
    replicate_scores: list[dict] = []
    run_receipts: list[dict] = []
    for stage in STAGES:
        for model in MODELS:
            model_root = DATA_ROOT / "experiments" / stage / model
            if model in {"M2", "M4"}:
                result_paths = sorted(model_root.glob("*/result.json"))
                expected = 10
            else:
                model_root = DATA_ROOT / "scientific_model_runs/experiments" / stage / model
                result_paths = sorted(model_root.glob("seed_*/result.json"))
                expected = 3
            if len(result_paths) != expected:
                raise ValueError(f"Expected {expected} successful replicates for {stage}/{model}; found {len(result_paths)}")
            for result_path in result_paths:
                result = json.loads(result_path.read_text())
                if result.get("status") != "success":
                    raise ValueError(f"Unsuccessful training output: {result_path}")
                pred_path = result_path.with_name("predictions.parquet")
                if not pred_path.is_file() or sha256(pred_path) != result.get(PARQUET_SHA):
                    raise ValueError(f"Prediction hash mismatch: {pred_path}")
                rep = result.get("setpoint") or result_path.parent.name
                frame = pd.read_parquet(pred_path)
                frame = frame.loc[frame.season.isin([2024, 2025])].copy()
                frame["stage"] = stage
                frame["model"] = model
                frame["replicate"] = str(rep)
                frame = frame.merge(lookup, on=["target_game_id", "season"], how="left", validate="many_to_one")
                frame = frame.merge(market, on="target_game_id", how="left", validate="many_to_one")
                if frame.week.isna().any() or frame.home_spread.isna().any():
                    raise ValueError(f"Week or market coverage is incomplete for {pred_path}")
                frame["week"] = frame.week.astype(int)
                frame["cohort"] = "common 2024–2025 development cohort"
                frame["ats_margin_actual"] = frame.actual_margin + frame.home_spread
                frame["ats_margin_predicted"] = frame.predicted_margin + frame.home_spread
                actual_nonpush = frame.ats_margin_actual.abs().gt(1e-8)
                model_nonpush = frame.ats_margin_predicted.abs().gt(1e-8)
                frame["ats_pick"] = np.select(
                    [~model_nonpush, frame.ats_margin_predicted.gt(0)],
                    ["no_pick", "home"], default="away")
                frame["ats_correct"] = np.where(
                    actual_nonpush & model_nonpush,
                    frame.ats_margin_actual.gt(0).eq(frame.ats_margin_predicted.gt(0)).astype(float),
                    np.nan)
                predictions.append(frame[[
                    "stage", "model", "replicate", "season", "week", "target_game_id",
                    "predicted_margin", "home_win_probability", "actual_margin", "home_spread",
                    "ats_pick", "ats_correct", "cohort",
                ]])
                run_receipts.append({
                    "stage": stage, "model": model, "replicate": str(rep),
                    "source_prediction_sha256": result[PARQUET_SHA],
                    "source_result_sha256": sha256(result_path),
                    "n_games_2024": int(frame.loc[frame.season.eq(2024)].shape[0]),
                    "n_games_2025": int(frame.loc[frame.season.eq(2025)].shape[0]),
                })
                for season in (2024, 2025):
                    scored = frame.loc[frame.season.eq(season)]
                    if scored.empty:
                        raise ValueError(f"No {season} test rows in {pred_path}")
                    metrics = summarize_predictions(scored)
                    stored = result.get(f"metrics_{season}", {})
                    # The run's published metrics are authoritative; this check guards our scoring formulas.
                    if abs(metrics["margin_mae"] - float(stored["mae"])) > 1e-7:
                        raise ValueError(f"MAE check failed for {result_path} in {season}")
                    if abs(metrics["winner_accuracy"] - float(stored["winner_accuracy"])) > 1e-7:
                        raise ValueError(f"Winner-accuracy check failed for {result_path} in {season}")
                    if abs(metrics["brier_score"] - float(stored["brier"])) > 1e-7:
                        raise ValueError(f"Brier check failed for {result_path} in {season}")
                    if abs(metrics["upset_recall"] - float(stored["upset_accuracy"])) > 1e-7:
                        raise ValueError(f"Upset-recall check failed for {result_path} in {season}")
                    if abs(metrics["ats_accuracy"] - float(stored["ats_accuracy"])) > 1e-7:
                        raise ValueError(f"ATS check failed for {result_path} in {season}")
                    replicate_scores.append({
                        "stage": stage, "model": model, "model_label": MODEL_LABELS[model],
                        "replicate": str(rep), "season": season,
                        "period_type": "heldout_week", **metrics,
                    })
    pred = pd.concat(predictions, ignore_index=True)
    expected_pairs = len(STAGES) * len(MODELS)
    if pred.groupby(["stage", "model"]).replicate.nunique().size != expected_pairs:
        raise ValueError("F09–F17 prediction matrix is incomplete")
    if pred.duplicated(["stage", "model", "replicate", "target_game_id"]).any():
        raise ValueError("Duplicate prediction row found")
    scores = pd.DataFrame(replicate_scores)
    replicate_weekly: list[dict] = []
    replicate_cumulative: list[dict] = []
    for (stage, model, rep, season), group in pred.groupby(["stage", "model", "replicate", "season"], sort=False):
        running: list[pd.DataFrame] = []
        for week, week_rows in group.sort_values("week").groupby("week", sort=True):
            running.append(week_rows)
            weekly = summarize_predictions(week_rows)
            cumulative = summarize_predictions(pd.concat(running, ignore_index=True))
            base = {"stage": stage, "model": model, "model_label": MODEL_LABELS[model],
                    "replicate": rep, "season": int(season), "week": int(week)}
            replicate_weekly.append({**base, **weekly})
            replicate_cumulative.append({**base, **cumulative})
    replicate_weekly_df = pd.DataFrame(replicate_weekly)
    replicate_cumulative_df = pd.DataFrame(replicate_cumulative)
    metric_cols = ["n_games", "margin_mae", "winner_accuracy", "upset_recall", "brier_score", "ats_games", "ats_accuracy"]
    group_cols = ["stage", "model", "model_label", "season", "week"]
    weekly = replicate_weekly_df.groupby(group_cols, as_index=False).agg(
        **{col: (col, "median") for col in metric_cols}, replicates=("replicate", "nunique"))
    cumulative = replicate_cumulative_df.groupby(group_cols, as_index=False).agg(
        **{col: (col, "median") for col in metric_cols}, replicates=("replicate", "nunique"))
    weekly["cohort"] = "common 2024–2025 development cohort; replicate-median weekly performance"
    cumulative["cohort"] = "common 2024–2025 development cohort; replicate-median season-to-date performance"
    year_scorecard = scores.groupby(["stage", "model", "model_label", "season"], as_index=False).agg(
        **{col: (col, "median") for col in metric_cols}, replicates=("replicate", "nunique"))
    year_scorecard["cohort"] = "common 2024–2025 development cohort; median across frozen fits"
    return pred, weekly, cumulative, year_scorecard, pd.DataFrame(run_receipts)


def make_scorecard(historical_folds: pd.DataFrame, year_scorecard: pd.DataFrame) -> pd.DataFrame:
    metrics = ["margin_mae", "winner_accuracy", "upset_recall", "brier_score", "ats_accuracy"]
    rows: list[dict] = []
    for (stage, model), group in historical_folds.groupby(["stage", "model"], sort=False):
        rows.append({
            "stage": stage, "model": model, "model_label": MODEL_LABELS[model],
            "fingerprint_family": "historical F0–F8; separate market F7/F8 retained",
            "evaluation_cohort": "2015–2024 rolling-origin held-out seasons",
            "n_games_median_per_fold": int(group.n_games.median()), "replicates": int(group.season.nunique()),
            "margin_mae": float(group.margin_mae.median()),
            "winner_accuracy": float(group.winner_accuracy.median()),
            "upset_recall": float(group.upset_recall.median()),
            "brier_score": float(group.brier_score.median()),
            "ats_accuracy": float(group.ats_accuracy.median()),
        })
    archived = pd.read_csv(ARCHIVED)
    for row in archived.loc[archived.fingerprint.eq("F06_F_a") & archived.model.isin(["M2", "M4"])].itertuples():
        rows.append({
            "stage": "F06 broad", "model": row.model, "model_label": MODEL_LABELS[row.model],
            "fingerprint_family": "F06 full-A broad development screen",
            "evaluation_cohort": "2025 broad A cohort", "n_games_median_per_fold": int(row.n_games_2025),
            "replicates": int(row.successes), "margin_mae": float(row.mae_2025),
            "winner_accuracy": float(row.winner_accuracy_2025), "upset_recall": float(row.upset_accuracy_2025),
            "brier_score": float(row.brier_2025), "ats_accuracy": float(row.ats_accuracy_2025),
        })
    for row in year_scorecard.loc[year_scorecard.season.eq(2025)].itertuples():
        stage_label = row.stage.replace("_market", " market").replace("_corrected", " corrected")
        if stage_label.startswith("F0") and stage_label[1:3].isdigit():
            stage_label = f"F{int(stage_label[1:3])}" + stage_label[3:]
        rows.append({
            "stage": stage_label,
            "model": row.model, "model_label": row.model_label,
            "fingerprint_family": "candidate A; F17 market is retrospective",
            "evaluation_cohort": "2025 common narrow development cohort",
            "n_games_median_per_fold": int(row.n_games), "replicates": int(row.replicates),
            "margin_mae": float(row.margin_mae), "winner_accuracy": float(row.winner_accuracy),
            "upset_recall": np.nan, "brier_score": float(row.brier_score),
            "ats_accuracy": float(row.ats_accuracy),
        })
    return pd.DataFrame(rows)


def load_colors() -> dict[str, str]:
    palette = json.loads((ROOT / "gridiron.palette.json").read_text())["colors"]
    return palette


def prediction_curves(predictions: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Make equal-architecture model forecasts, six-model consensus, and weekly/S2D scores."""
    keys = ["stage", "model", "season", "week", "target_game_id"]
    model_games = predictions.groupby(keys, as_index=False).agg(
        predicted_margin=("predicted_margin", "mean"),
        home_win_probability=("home_win_probability", "mean"),
        actual_margin=("actual_margin", "first"), home_spread=("home_spread", "first"),
    )
    consensus_games = model_games.groupby(["stage", "season", "week", "target_game_id"], as_index=False).agg(
        predicted_margin=("predicted_margin", "mean"),
        home_win_probability=("home_win_probability", "mean"),
        actual_margin=("actual_margin", "first"), home_spread=("home_spread", "first"),
        architecture_count=("model", "nunique"),
    )
    if not consensus_games.architecture_count.eq(len(MODELS)).all():
        raise ValueError("Scientific consensus does not have all six architectures for every game")
    consensus_games["model"] = "Scientific consensus"
    consensus_games["model_label"] = "Scientific consensus"
    model_games["model_label"] = model_games.model.map(MODEL_LABELS)

    def score_trajectory(frame: pd.DataFrame, identity: list[str]) -> pd.DataFrame:
        weekly_rows: list[dict] = []
        cumulative_rows: list[dict] = []
        for group_key, group in frame.groupby(identity, sort=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            metadata = dict(zip(identity, group_key))
            prior: list[pd.DataFrame] = []
            for week, games in group.sort_values("week").groupby("week", sort=True):
                prior.append(games)
                weekly_rows.append({**metadata, "week": int(week), **summarize_predictions(games)})
                cumulative_rows.append({**metadata, "week": int(week),
                                        **summarize_predictions(pd.concat(prior, ignore_index=True))})
        return pd.DataFrame(weekly_rows), pd.DataFrame(cumulative_rows)

    ids = ["stage", "model", "model_label", "season"]
    model_weekly, model_cumulative = score_trajectory(model_games, ids)
    consensus_weekly, consensus_cumulative = score_trajectory(consensus_games, ids)
    model_weekly["aggregation"] = "within-architecture mean forecast across saved fits"
    model_cumulative["aggregation"] = "within-architecture mean forecast across saved fits"
    consensus_weekly["aggregation"] = "equal-weight mean of the six architecture-level forecasts"
    consensus_cumulative["aggregation"] = "equal-weight mean of the six architecture-level forecasts"
    return model_games, consensus_games, model_weekly, model_cumulative, consensus_weekly, consensus_cumulative


def vegas_cumulative(predictions: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for season in (2024, 2025):
        games = predictions.loc[predictions.season.eq(season)].drop_duplicates("target_game_id").copy()
        games["vegas_predicted_margin"] = -games.home_spread
        games["vegas_probability_home"] = (games.vegas_predicted_margin > 0).astype(float)
        games["vegas_ats_correct"] = np.nan
        games = games.sort_values(["week", "target_game_id"])
        rows = []
        prior: list[pd.DataFrame] = []
        for week, part in games.groupby("week", sort=True):
            prior.append(part)
            total = pd.concat(prior, ignore_index=True)
            actual = total.actual_margin.to_numpy(float)
            pred = total.vegas_predicted_margin.to_numpy(float)
            rows.append({
                "season": season, "week": int(week), "n_games": int(len(total)),
                "margin_mae": float(np.abs(pred - actual).mean()),
                "winner_accuracy": float(((pred > 0) == (actual > 0)).mean()),
                "upset_recall": 0.0,
                "brier_score": float(np.square(total.vegas_probability_home.to_numpy(float)
                                                 - (actual > 0).astype(float)).mean()),
            })
        outputs.append(pd.DataFrame(rows))
    return pd.concat(outputs, ignore_index=True)


def render_cumulative_curves(historical_cumulative_df: pd.DataFrame,
                             model_cumulative: pd.DataFrame,
                             consensus_cumulative: pd.DataFrame,
                             vegas_curves: pd.DataFrame) -> list[Path]:
    colors = load_colors()
    background = colors["parchment"]
    panel = colors["parchmentPanel"]
    ink = colors["midnightGridiron"]
    axis = colors["slateLine"]
    outdir = OUT / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("margin_mae", "Cumulative margin MAE", "Lower is better", 1.0),
        ("upset_recall", "Cumulative upset recall", "Higher is better", 100.0),
        ("winner_accuracy", "Cumulative winner accuracy", "Higher is better", 100.0),
        ("brier_score", "Cumulative Brier score", "Lower is better", 1.0),
        ("ats_accuracy", "Cumulative ATS accuracy", "Higher is better · pushes excluded", 100.0),
    ]
    gen_colors = {f"F{i}": plt.get_cmap("coolwarm_r")(i / 17) for i in range(18)}
    gen_colors["F12 corrected"] = gen_colors["F12"]
    gen_colors["F17 market"] = gen_colors["F17"]
    historical_stages = [f"F{i}" for i in range(9)]
    nextgen_stages = [f"F{i}" for i in range(9, 12)] + ["F12 corrected"] + [
        f"F{i}" for i in range(13, 17)
    ] + ["F17 market"]

    def display_stage(value: str) -> str:
        label = value.replace("_market", " market").replace("_corrected", " corrected")
        if label.startswith("F0") and label[1:3].isdigit():
            label = f"F{int(label[1:3])}" + label[3:]
        return label

    model_curves = model_cumulative.copy()
    consensus_curves = consensus_cumulative.copy()
    model_curves["stage"] = model_curves.stage.map(display_stage)
    consensus_curves["stage"] = consensus_curves.stage.map(display_stage)
    paths: list[Path] = []

    # Annual held-out-year curves for the full historical F0–F8 model packs.
    vegas_hist = pd.read_csv(FIGURE_DATA / "all_architectures_vegas_historical_folds.csv")
    for metric, title, preference, scale in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True,
                                 facecolor=background)
        for ax, model in zip(axes.flat, MODELS):
            ax.set_facecolor(panel)
            view = historical_cumulative_df.loc[historical_cumulative_df.model.eq(model)]
            for stage in historical_stages:
                line = view.loc[view.stage.eq(stage)].sort_values("through_season")
                if not line.empty:
                    ax.plot(line.through_season, line[metric] * scale, color=gen_colors[stage],
                            lw=1.7, label=stage)
            if metric in {"margin_mae", "winner_accuracy", "brier_score"}:
                baseline = vegas_hist.sort_values("test_year").copy()
                games = baseline.games.to_numpy(float)
                vegas_metric = {"margin_mae": "mae", "winner_accuracy": "winner_accuracy",
                                "brier_score": "brier_score"}[metric]
                baseline[metric + "_cum"] = np.cumsum(baseline[vegas_metric] * games) / np.cumsum(games)
                ax.plot(baseline.test_year, baseline[metric + "_cum"] * scale, color=ink,
                        lw=2.0, ls="--", label="Vegas")
            elif metric == "upset_recall":
                ax.axhline(0.0, color=ink, lw=1.5, ls="--", label="Vegas favorite · 0%")
            else:
                ax.axhline(50.0, color=ink, lw=1.5, ls="--", label="50% break-even")
            ax.set_title(f"{model} · {MODEL_LABELS[model]}", color=ink, loc="left", fontsize=11)
            ax.grid(axis="y", color=colors["steelGrey"], alpha=.22, lw=.7)
            ax.tick_params(colors=axis, labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.suptitle(f"F0–F8 · {title} by held-out season", color=ink, fontsize=18,
                     fontweight="bold", x=.06, ha="left")
        fig.text(.06, .94, "Annual rolling-origin performance accumulated through each test season · historical folds only",
                 color=axis, fontsize=10)
        fig.text(.06, .035, "Vegas spread reference for MAE/winners, 0/1 favorite Brier, 0% upset recall; ATS uses 50% break-even. F7/F8 remain separate market branches.",
                 color=axis, fontsize=8.5)
        for ax in axes[-1, :]:
            ax.set_xlabel("Held-out season", color=axis)
        ylabel = {"margin_mae": "MAE (points)", "brier_score": "Brier score",
                  "upset_recall": "Upset recall (%)", "winner_accuracy": "Accuracy (%)",
                  "ats_accuracy": "ATS accuracy (%)"}[metric]
        axes[0, 0].set_ylabel(ylabel, color=axis)
        axes[1, 0].set_ylabel(ylabel, color=axis)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=10, loc="lower center", bbox_to_anchor=(.5, .075),
                   frameon=False, fontsize=8.5)
        fig.subplots_adjust(left=.07, right=.99, top=.88, bottom=.18, hspace=.26, wspace=.12)
        path = outdir / f"historical_f0_f8_cumulative_{metric}.png"
        fig.savefig(path, dpi=200, facecolor=background, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # Per-model F09–F17 weekly curves, plus a separately scored six-architecture consensus.
    for season in (2024, 2025):
        vegas_season = vegas_curves.loc[vegas_curves.season.eq(season)]
        game_count = int(vegas_season.n_games.max())
        for metric, title, preference, scale in metrics:
            ylabel = {"margin_mae": "MAE (points)", "brier_score": "Brier score",
                      "upset_recall": "Upset recall (%)", "winner_accuracy": "Accuracy (%)",
                      "ats_accuracy": "ATS accuracy (%)"}[metric]
            fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True,
                                     facecolor=background)
            for ax, model in zip(axes.flat, MODELS):
                ax.set_facecolor(panel)
                view = model_curves.loc[(model_curves.model.eq(model)) & (model_curves.season.eq(season))]
                for stage in nextgen_stages:
                    line = view.loc[view.stage.eq(stage)].sort_values("week")
                    if not line.empty:
                        ax.plot(line.week, line[metric] * scale, color=gen_colors[stage], lw=1.7, label=stage)
                if metric in {"margin_mae", "winner_accuracy", "brier_score"}:
                    ax.plot(vegas_season.week, vegas_season[metric] * scale, color=ink, lw=2.0,
                            ls="--", label="Vegas")
                elif metric == "upset_recall":
                    ax.axhline(0.0, color=ink, lw=1.5, ls="--", label="Vegas favorite · 0%")
                else:
                    ax.axhline(50.0, color=ink, lw=1.5, ls="--", label="50% break-even")
                ax.set_title(f"{model} · {MODEL_LABELS[model]}", color=ink, loc="left", fontsize=11)
                ax.grid(axis="y", color=colors["steelGrey"], alpha=.22, lw=.7)
                ax.tick_params(colors=axis, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            fig.suptitle(f"F09–F17 market · full scientific roster · {title}", color=ink,
                         fontsize=18, fontweight="bold", x=.06, ha="left")
            fig.text(.06, .94, f"{season} common {game_count}-game development cohort · each model prediction averages its saved fits before scoring",
                     color=axis, fontsize=10)
            fig.text(.06, .035, "F17 market is retrospective. Vegas references: spread MAE/winners, 0/1 favorite Brier, 0% upset recall; ATS uses 50% break-even.",
                     color=axis, fontsize=8.5)
            for ax in axes[-1, :]:
                ax.set_xlabel("Week", color=axis)
            axes[0, 0].set_ylabel(ylabel, color=axis)
            axes[1, 0].set_ylabel(ylabel, color=axis)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=10, loc="lower center", bbox_to_anchor=(.5, .075),
                       frameon=False, fontsize=8.5)
            fig.subplots_adjust(left=.07, right=.99, top=.88, bottom=.18, hspace=.26, wspace=.12)
            path = outdir / f"nextgen_full_roster_cumulative_{metric}_{season}.png"
            fig.savefig(path, dpi=200, facecolor=background, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

            consensus = consensus_curves.loc[consensus_curves.season.eq(season)]
            fig, ax = plt.subplots(figsize=(13, 7), facecolor=background)
            ax.set_facecolor(panel)
            for stage in nextgen_stages:
                line = consensus.loc[consensus.stage.eq(stage)].sort_values("week")
                if not line.empty:
                    ax.plot(line.week, line[metric] * scale, color=gen_colors[stage], lw=2.1,
                            label=stage, ls="--" if stage == "F17 market" else "-")
            if metric in {"margin_mae", "winner_accuracy", "brier_score"}:
                ax.plot(vegas_season.week, vegas_season[metric] * scale, color=ink, lw=2.3,
                        ls=(0, (5, 2)), label="Vegas")
            elif metric == "upset_recall":
                ax.axhline(0.0, color=ink, lw=2.0, ls=(0, (5, 2)), label="Vegas favorite · 0%")
            else:
                ax.axhline(50.0, color=ink, lw=2.0, ls=(0, (5, 2)), label="50% break-even")
            ax.set_title(f"Six-model consensus · {title}", color=ink, loc="left", fontsize=17, fontweight="bold")
            ax.set_xlabel("Week", color=axis)
            ax.set_ylabel(ylabel, color=axis)
            ax.grid(axis="y", color=colors["steelGrey"], alpha=.22, lw=.7)
            ax.tick_params(colors=axis)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.legend(ncol=5, frameon=False, loc="best", fontsize=9)
            fig.text(.02, .015,
                     "Consensus averages each architecture across its saved fits, then weights the six architectures equally. F17 market is retrospective.",
                     color=axis, fontsize=9)
            fig.tight_layout(rect=(0, .04, 1, 1))
            path = outdir / f"nextgen_consensus_cumulative_{metric}_{season}.png"
            fig.savefig(path, dpi=200, facecolor=background, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths


def render_heatmaps(scorecard: pd.DataFrame, vegas: pd.DataFrame) -> list[Path]:
    colors = load_colors()
    background = colors["parchment"]
    panel = colors["parchmentPanel"]
    ink = colors["midnightGridiron"]
    axes = colors["slateLine"]
    heatmap_dir = OUT / "figures"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    scorecard = scorecard.copy()
    # Keep chart labels concise while retaining complete source names in scorecard.csv.
    scorecard["heat_stage"] = scorecard.stage.astype(str)
    stages = [s for s in STAGE_ORDER if s in set(scorecard.heat_stage)]
    model_order = list(MODELS)
    output_paths: list[Path] = []
    specs = [
        ("margin_mae", "Margin MAE", "RdYlGn_r", "Lower is better", ".1f"),
        ("winner_accuracy", "Winner accuracy", "RdYlGn", "Higher is better", ".1%"),
        ("ats_accuracy", "ATS accuracy", "RdYlGn", "Higher is better · pushes excluded", ".1%"),
    ]
    vegas_metric = {"margin_mae": "mae", "winner_accuracy": "winner_accuracy",
                    "ats_accuracy": "ats_accuracy"}
    for metric, title, cmap, descriptor, fmt in specs:
        panel_data = scorecard.pivot_table(index="model", columns="heat_stage", values=metric, aggfunc="first")
        panel_data = panel_data.reindex(index=model_order, columns=stages)
        # Vegas is cohort-specific: historical folds, the broad F06 cohort, or the common narrow cohort.
        vegas_values: list[float] = []
        for stage in stages:
            if metric == "ats_accuracy":
                vegas_values.append(np.nan)
                continue
            if stage in [f"F{i}" for i in range(9)]:
                cohort = "historical 2015–2024 fold median"
            elif stage == "F06 broad":
                cohort = "nextgen 2025 broad"
            else:
                cohort = "nextgen 2025 narrow"
            column = vegas_metric[metric]
            value = vegas.loc[vegas.setting.eq(cohort), column]
            vegas_values.append(float(value.iloc[0]) if len(value) and column in vegas.columns else np.nan)
        # A line-only market reference has no measured ATS pick accuracy.
        panel_data.loc["Vegas reference"] = vegas_values
        vals = panel_data.to_numpy(float)
        finite = vals[np.isfinite(vals)]
        vmin = float(finite.min())
        vmax = float(finite.max())
        if metric.endswith("accuracy"):
            vmin = max(0.0, vmin - .03)
            vmax = min(1.0, vmax + .03)
        else:
            span = vmax - vmin
            vmin -= span * .08
            vmax += span * .08
        fig, ax = plt.subplots(figsize=(19, 7.7), facecolor=background)
        ax.set_facecolor(panel)
        im = ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(stages)), stages, rotation=35, ha="right", color=axes, fontsize=10)
        ylabels = [MODEL_LABELS[m] for m in model_order] + ["Vegas reference"]
        ax.set_yticks(np.arange(len(ylabels)), ylabels, color=ink, fontsize=10.5)
        ax.set_xticks(np.arange(-.5, len(stages), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(ylabels), 1), minor=True)
        ax.grid(which="minor", color=background, linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                value = vals[i, j]
                if np.isfinite(value):
                    label = format(value, fmt)
                    rgba = im.cmap(im.norm(value))
                    luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                    ax.text(j, i, label, ha="center", va="center",
                            color="#10131B" if luminance > .58 else "white", fontsize=8.2,
                            fontweight="bold" if i == len(ylabels)-1 else "normal")
        ax.set_title(f"Scientific fingerprint roster · {title}", loc="left", color=ink,
                     fontsize=18, fontweight="bold", pad=18)
        subtitle = "All six architectures · historical F0–F8 and full-A F06/F09–F17 development cohorts shown separately"
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=axes, fontsize=10, va="bottom")
        cbar = fig.colorbar(im, ax=ax, fraction=.022, pad=.018)
        cbar.set_label(f"{title} · {descriptor}", color=ink)
        cbar.ax.tick_params(colors=axes)
        ax.spines[:].set_visible(False)
        fig.text(.065, .035,
                 "F06 broad is measured only for M2/M4. F0–F8 cells are medians across 2015–2024 rolling folds; F09–F17 are 2025 common-cohort replicate medians. Vegas ATS is not applicable to a line-only reference.",
                 color=axes, fontsize=8.5, ha="left")
        fig.subplots_adjust(left=.13, right=.93, top=.88, bottom=.18)
        path = heatmap_dir / f"scientific_fingerprint_{metric}_heatmap.png"
        fig.savefig(path, dpi=220, facecolor=background, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(path)
    return output_paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, historical_folds = load_historical()
    historical_cumulative_df = historical_cumulative(historical_folds)
    predictions, weekly, cumulative, year_scorecard, run_receipts = load_nextgen()
    scorecard = make_scorecard(historical_folds, year_scorecard)
    model_games, consensus_games, model_weekly, model_cumulative, consensus_weekly, consensus_cumulative = prediction_curves(predictions)
    vegas_curves = vegas_cumulative(predictions)

    write_csv(predictions.drop(columns=["home_spread"]).sort_values(
        ["stage", "model", "season", "week", "target_game_id", "replicate"]),
              OUT / "predictions/nextgen_game_predictions_2024_2025.csv.gz")
    write_csv(model_games.drop(columns=["home_spread"]).sort_values(
        ["stage", "model", "season", "week", "target_game_id"]),
              OUT / "predictions/nextgen_roster_model_mean_predictions_2024_2025.csv.gz")
    write_csv(consensus_games.drop(columns=["home_spread"]).sort_values(
        ["stage", "season", "week", "target_game_id"]),
              OUT / "predictions/nextgen_scientific_consensus_predictions_2024_2025.csv.gz")
    write_csv(weekly.sort_values(["stage", "model", "season", "week"]), OUT / "performance/nextgen_weekly_performance.csv")
    write_csv(cumulative.sort_values(["stage", "model", "season", "week"]), OUT / "performance/nextgen_season_to_date_performance.csv")
    write_csv(year_scorecard.sort_values(["stage", "model", "season"]), OUT / "performance/nextgen_year_scorecards.csv")
    write_csv(historical_folds[[
        "stage", "model", "season", "period_type", "n_games", "margin_mae", "winner_accuracy",
        "upset_recall", "brier_score", "ats_n", "ats_accuracy", "selected_cv_mean",
    ]].sort_values(["stage", "model", "season"]), OUT / "performance/historical_annual_fold_performance.csv")
    write_csv(historical_cumulative_df.sort_values(["stage", "model", "through_season"]),
              OUT / "performance/historical_cumulative_performance.csv")
    write_csv(scorecard.sort_values(["stage", "model"]), OUT / "performance/all_fingerprint_scorecard.csv")
    write_csv(run_receipts.sort_values(["stage", "model", "replicate"]), OUT / "provenance/nextgen_run_receipts.csv")
    write_csv(model_weekly.sort_values(["stage", "model", "season", "week"]),
              OUT / "performance/nextgen_model_mean_weekly_performance.csv")
    write_csv(model_cumulative.sort_values(["stage", "model", "season", "week"]),
              OUT / "performance/nextgen_model_mean_cumulative_performance.csv")
    write_csv(consensus_weekly.sort_values(["stage", "season", "week"]),
              OUT / "performance/nextgen_consensus_weekly_performance.csv")
    write_csv(consensus_cumulative.sort_values(["stage", "season", "week"]),
              OUT / "performance/nextgen_consensus_cumulative_performance.csv")
    legacy_vegas_file = OUT / "performance/vegas_cumulative_performance_2025.csv"
    if legacy_vegas_file.exists():
        legacy_vegas_file.unlink()
    write_csv(vegas_curves, OUT / "performance/vegas_cumulative_performance_2024_2025.csv")

    vegas = pd.read_csv(FIGURE_DATA / "all_architectures_vegas_baseline_data.csv")
    render_heatmaps(scorecard, vegas)
    render_cumulative_curves(historical_cumulative_df, model_cumulative,
                             consensus_cumulative, vegas_curves)
    artifact_files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "package": "scientific_fingerprint_archive",
        "fingerprint_stages": STAGE_ORDER,
        "models": list(MODELS),
        "f0_f8_evaluation": "2015-2024 rolling-origin fold summaries; individual historical game prediction rows were not retained in the selected results artifact",
        "f06_broad_evaluation": "2025 broad cohort; M2 and M4 only",
        "f09_f17_evaluation": "2024 and 2025 game-level development predictions; F17_market is retrospective",
        "nextgen_prediction_rows": int(sum(1 for _ in predictions.itertuples())),
        "nextgen_model_mean_prediction_rows": int(len(model_games)),
        "nextgen_consensus_prediction_rows": int(len(consensus_games)),
        "nextgen_weekly_performance_rows": int(len(weekly)),
        "nextgen_cumulative_performance_rows": int(len(cumulative)),
        "historical_fold_rows": int(len(historical_folds)),
        "scorecard_rows": int(len(scorecard)),
        "input_sha256": {
            "historical_selected_folds": sha256(HISTORICAL),
            "nextgen_canonical_week_lookup": sha256(CANONICAL),
            "nextgen_market_sidecar": sha256(MARKET),
            "f06_archived_scorecard": sha256(ARCHIVED),
            "vegas_baselines": sha256(FIGURE_DATA / "all_architectures_vegas_baseline_data.csv"),
        },
        "files": {str(p.relative_to(OUT)): sha256(p) for p in artifact_files},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
