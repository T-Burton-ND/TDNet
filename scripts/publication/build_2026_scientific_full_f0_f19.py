#!/usr/bin/env python3
"""Build the complete 2026 F0–F19 weekly scientific research scorecards.

This postgame export joins archived F0–F17-market forecasts to the immutable,
score-free F18/F19 prediction receipt. It does not refit or select a model and
does not alter any frozen weekly publication bundle.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "data/what_if_2026_fingerprints/2026_f0_f17_market_scored_model_games.parquet"
FROZEN = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/broad_search/final_fit/prospective")
OUTPUT = ROOT / "publication/2026/week_05/post_game/scientific/current_season_f0_f19"
MODEL_IDS = ("M1", "M2", "M3", "M4", "M5", "M10")
FAMILY = {"M1": "linear", "M2": "spline", "M3": "tree", "M4": "boosted",
          "M5": "neural", "M10": "knn"}
STAGES = (*[f"F{i}" for i in range(17)], "F17-market", "F18", "F19")
WEEKS = (1, 2, 3, 4, 5)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _sources() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    receipt_path = FROZEN / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    prediction_path = Path(receipt["predictions"])
    if prediction_path != FROZEN / "predictions.parquet" or digest(prediction_path) != receipt["predictions_sha256"]:
        raise ValueError("Frozen F18/F19 prediction export changed")
    freeze_path = Path(receipt["freeze_manifest"])
    if digest(freeze_path) != receipt["freeze_manifest_sha256"] or not receipt["no_2026_outcomes_read"]:
        raise ValueError("F18/F19 predictions lack a verified score-free freeze")
    old = pd.read_parquet(PRIOR)
    new = pd.read_parquet(prediction_path)
    if len(old) != 108 * 271 or len(new) != 26 * 271:
        raise ValueError("F0–F17 or F18/F19 archive coverage changed")
    return old, new, receipt


def _model_games(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = old.copy()
    old["model_id"] = old.model_name.str.rsplit("_", n=1).str[-1]
    if (old.duplicated(["game_id", "fingerprint", "model_id"]).any()
            or set(old.fingerprint) != set(STAGES[:-2])
            or set(old.model_id) != set(MODEL_IDS)):
        raise ValueError("Prior scientific model roster changed")
    unique = old.drop_duplicates("game_id").sort_values("game_id").copy()
    for name in ("week", "home_team", "away_team", "actual_home_margin",
                 "market_spread_close", "vegas_home_probability"):
        if old.groupby("game_id")[name].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Prior game-level field differs across models: {name}")
    if len(unique) != 271 or unique.week.value_counts().sort_index().to_dict() != {
            1: 51, 2: 49, 3: 57, 4: 58, 5: 56}:
        raise ValueError("Canonical completed-game cohort changed")
    games = unique[["game_id", "week", "home_team", "away_team", "actual_home_margin",
                    "market_spread_close", "vegas_home_probability"]].rename(columns={
                        "market_spread_close": "market_home_spread",
                        "vegas_home_probability": "market_home_probability"})
    games["game_id"] = games.game_id.astype(int)
    if games.market_home_spread.notna().sum() != 263 or games.market_home_probability.notna().sum() != 259:
        raise ValueError("Archived market baseline coverage changed")
    core = old[["game_id", "fingerprint", "model_id", "pred_home_margin",
                "pred_home_win_probability"]].rename(columns={
                    "pred_home_margin": "pred_margin",
                    "pred_home_win_probability": "home_win_probability"})
    core["game_id"] = core.game_id.astype(int)
    core["forecast_status"] = "available"
    core["source_scope"] = "F0-F17-market archived research forecast"
    if core[["pred_margin", "home_win_probability"]].isna().any().any():
        raise ValueError("An older scientific model forecast is missing")
    selected = new.loc[new.model_id.isin(MODEL_IDS)].copy()
    selected = selected.rename(columns={"tier": "fingerprint", "target_game_id": "game_id",
                                        "status": "forecast_status"})
    selected["source_scope"] = "F18-F19 hash-verified frozen forecast"
    if selected.duplicated(["game_id", "fingerprint", "model_id"]).any() or len(selected) != 12 * 271:
        raise ValueError("F18/F19 scientific roster changed")
    selected = selected[["game_id", "fingerprint", "model_id", "pred_margin",
                         "home_win_probability", "forecast_status", "source_scope",
                         "home_team", "away_team", "week"]]
    checked = selected.merge(games[["game_id", "home_team", "away_team", "week"]],
                             on="game_id", suffixes=("", "_canonical"), validate="many_to_one")
    for name in ("home_team", "away_team", "week"):
        if not checked[name].eq(checked[f"{name}_canonical"]).all():
            raise ValueError(f"Frozen forecast {name} differs from canonical schedule")
    selected = selected.drop(columns=["home_team", "away_team", "week"])
    model = pd.concat([core, selected], ignore_index=True)
    model = model.merge(games, on="game_id", validate="many_to_one")
    model["model_name"] = "scientific_" + model.fingerprint + "_" + model.model_id
    model["model_family"] = model.model_id.map(FAMILY)
    model["series_id"] = model.model_name
    model["series_type"] = "model"
    model["ballots_expected"] = 1
    model["ballots_available"] = model.forecast_status.eq("available").astype(int)
    _assert_model_coverage(model)
    return model, games


def _assert_model_coverage(model: pd.DataFrame) -> None:
    if len(model) != 120 * 271 or model.duplicated(["game_id", "fingerprint", "model_id"]).any():
        raise ValueError("Expected exactly 120 distinct model × fingerprint cells on 271 games")
    count = model.groupby(["fingerprint", "model_id"]).game_id.nunique()
    if len(count) != 120 or not count.eq(271).all():
        raise ValueError("Model/fingerprint game IDs are incomplete")
    available = model.forecast_status.eq("available")
    coverage = model.assign(available=available).groupby(["fingerprint", "model_id"]).available.sum()
    if not coverage.loc[coverage.index.get_level_values(0) != "F19"].eq(271).all():
        raise ValueError("A non-F19 model lacks a forecast")
    if not coverage.loc["F19"].eq(263).all():
        raise ValueError("F19 market-covered model coverage changed")
    if model.loc[~available, ["pred_margin", "home_win_probability"]].notna().any().any():
        raise ValueError("Unavailable F19 forecast was silently filled")
    if not np.isfinite(model.loc[available, ["pred_margin", "home_win_probability"]].to_numpy(float)).all():
        raise ValueError("Available model forecast is not finite")


def _aggregate(group: pd.DataFrame, series_id: str, expected: int,
               *, strict: bool = False, override: pd.DataFrame | None = None) -> pd.DataFrame:
    valid = group.loc[group.forecast_status.eq("available")]
    out = valid.groupby("game_id", as_index=False).agg(
        pred_margin=("pred_margin", "mean"),
        home_win_probability=("home_win_probability", "mean"),
        ballots_available=("model_name", "nunique"))
    if strict:
        out = out.loc[out.ballots_available.eq(expected)].copy()
    if override is not None:
        special = override[["target_game_id", "pred_margin", "home_win_probability", "status"]].rename(
            columns={"target_game_id": "game_id", "pred_margin": "frozen_margin",
                     "home_win_probability": "frozen_probability"})
        out = out.merge(special, on="game_id", validate="one_to_one")
        if not out.status.eq("available").all() or not np.allclose(
                out.pred_margin, out.frozen_margin, atol=1e-8):
            raise ValueError(f"Frozen equal consensus differs from six-model margin mean: {series_id}")
        out["home_win_probability"] = out.frozen_probability
        out = out.drop(columns=["frozen_margin", "frozen_probability", "status"])
    out["series_id"] = series_id
    out["series_type"] = "consensus"
    out["ballots_expected"] = expected
    out["forecast_status"] = "available"
    out["source_scope"] = "mean of frozen scientific model forecasts"
    return out


def _consensus(model: pd.DataFrame, games: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _aggregate(model, "full_available_120", 120),
        _aggregate(model, "full_common_120", 120, strict=True),
        _aggregate(model.loc[model.fingerprint.isin(STAGES[:-2])],
                   "legacy_f0_f17_market_108", 108),
    ]
    for stage in STAGES:
        frozen = (new.loc[new.tier.eq(stage) & new.model_id.eq("equal")]
                  if stage in ("F18", "F19") else None)
        rows.append(_aggregate(model.loc[model.fingerprint.eq(stage)],
                               f"fingerprint_{stage}", 6, strict=True,
                               override=frozen))
    for architecture in MODEL_IDS:
        rows.append(_aggregate(model.loc[model.model_id.eq(architecture)],
                               f"model_{architecture}_across_fingerprints", 20))
    output = pd.concat(rows, ignore_index=True)
    output = output.merge(games, on="game_id", validate="many_to_one")
    counts = output.groupby("series_id").game_id.nunique()
    if len(counts) != 29 or counts["full_available_120"] != 271 or counts["full_common_120"] != 263:
        raise ValueError("Full consensus coverage changed")
    if not counts["fingerprint_F18"] == 271 or not counts["fingerprint_F19"] == 263:
        raise ValueError("Frozen tier consensus coverage changed")
    full = output.loc[output.series_id.eq("full_available_120")]
    if full.ballots_available.value_counts().to_dict() != {120: 263, 114: 8}:
        raise ValueError("Adaptive full-roster ballot coverage changed")
    return output


def _scored(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    available = frame.forecast_status.eq("available")
    truth = frame.actual_home_margin.gt(0)
    valid_prob = available & frame.home_win_probability.notna()
    frame["absolute_margin_error"] = (frame.pred_margin - frame.actual_home_margin).abs()
    frame["brier_error"] = (frame.home_win_probability - truth.astype(float)) ** 2
    frame["winner_correct"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame.loc[valid_prob, "winner_correct"] = (
        frame.loc[valid_prob, "home_win_probability"].ge(.5).to_numpy()
        == truth.loc[valid_prob].to_numpy())
    spread = frame.market_home_spread
    line = spread.notna() & spread.ne(0)
    cover = frame.actual_home_margin + spread
    choice = frame.pred_margin + spread
    frame["ats_result"] = "no_line_or_forecast"
    frame.loc[available & line & cover.eq(0), "ats_result"] = "push"
    contest = available & line & cover.ne(0) & choice.ne(0)
    frame.loc[contest, "ats_result"] = np.where(
        np.sign(choice[contest]) == np.sign(cover[contest]), "win", "loss")
    favorite_home = spread.lt(0)
    upset = line & favorite_home.ne(truth)
    frame["actual_upset"] = upset
    frame["upset_recalled"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame.loc[upset & valid_prob, "upset_recalled"] = (
        frame.loc[upset & valid_prob, "home_win_probability"].ge(.5).to_numpy()
        != favorite_home.loc[upset & valid_prob].to_numpy())
    return frame


def _metric(group: pd.DataFrame, *, week: int, scope: str, series_id: str,
            series_type: str) -> dict:
    available = group.forecast_status.eq("available")
    margin = group.loc[available, "absolute_margin_error"].to_numpy(float)
    probability = group.loc[available & group.brier_error.notna(), "brier_error"].to_numpy(float)
    winner = group.loc[available & group.winner_correct.notna(), "winner_correct"].to_numpy(bool)
    upset = group.loc[available & group.actual_upset & group.upset_recalled.notna(),
                      "upset_recalled"].to_numpy(bool)
    ats = group.loc[available, "ats_result"]
    wins, losses, pushes = (int(ats.eq(value).sum()) for value in ("win", "loss", "push"))
    return {
        "scope": scope, "week": week, "series_type": series_type,
        "series_id": series_id, "fingerprint": (str(group.fingerprint.iloc[0])
            if "fingerprint" in group else ""),
        "model_id": (str(group.model_id.iloc[0]) if "model_id" in group else ""),
        "cohort_games": int(group.game_id.nunique()), "games": int(available.sum()),
        "missing_games": int((~available).sum()),
        "margin_mae": float(margin.mean()) if len(margin) else np.nan,
        "margin_rmse": float(np.sqrt(np.mean(margin ** 2))) if len(margin) else np.nan,
        "brier_games": len(probability),
        "brier_score": float(probability.mean()) if len(probability) else np.nan,
        "winner_games": len(winner), "winner_correct": int(winner.sum()),
        "winner_accuracy": float(winner.mean()) if len(winner) else np.nan,
        "ats_wins": wins, "ats_losses": losses, "ats_pushes": pushes,
        "ats_accuracy": wins / (wins + losses) if wins + losses else np.nan,
        "upset_games": len(upset),
        "upset_recalled": int(upset.sum()),
        "upset_recall": float(upset.mean()) if len(upset) else np.nan,
        "ballots_expected": int(group.ballots_expected.iloc[0]),
        "ballots_available_min": int(group.loc[available, "ballots_available"].min()) if available.any() else 0,
        "ballots_available_max": int(group.loc[available, "ballots_available"].max()) if available.any() else 0,
    }


def _performance(model: pd.DataFrame, consensus: pd.DataFrame,
                 games: pd.DataFrame) -> pd.DataFrame:
    market = games.copy()
    market["series_type"] = "market"
    market["series_id"] = "archived_pregame_market"
    market["forecast_status"] = np.where(market.market_home_spread.notna(),
                                           "available", "unavailable_market_snapshot")
    market["pred_margin"] = -market.market_home_spread
    market["home_win_probability"] = market.market_home_probability
    market["ballots_expected"] = 1
    market["ballots_available"] = market.forecast_status.eq("available").astype(int)
    frames = [model, consensus, market]
    rows = []
    for source in frames:
        source = _scored(source)
        for series_id, group in source.groupby("series_id", sort=False):
            kind = str(group.series_type.iloc[0])
            for week in WEEKS:
                for scope, selection in (("weekly", group.week.eq(week)),
                                         ("cumulative", group.week.le(week))):
                    current = group.loc[selection]
                    if not current.empty:
                        rows.append(_metric(current, week=week, scope=scope,
                                            series_id=series_id, series_type=kind))
    table = pd.DataFrame(rows)
    if table.loc[table.series_type.eq("model")].shape[0] != 120 * len(WEEKS) * 2:
        raise ValueError("Week-by-week model scorecard is incomplete")
    return table.sort_values(["scope", "week", "series_type", "series_id"])


def _plot_consensus(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#F8F4EA")
    settings = [("margin_mae", "Margin MAE", False), ("brier_score", "Brier", False),
                ("winner_accuracy", "Winner accuracy", True),
                ("ats_accuracy", "ATS accuracy", True)]
    palette = {"full_available_120": "#D24F88", "full_common_120": "#9D3970",
               "legacy_f0_f17_market_108": "#7557A4", "fingerprint_F18": "#257E83",
               "fingerprint_F19": "#BD7A44", "archived_pregame_market": "#AD8C40"}
    for ax, (metric, title, percent) in zip(axes.flat, settings, strict=True):
        ax.set_facecolor("#FFFCF5")
        for series, color in palette.items():
            frame = table.loc[table.scope.eq("cumulative") & table.series_id.eq(series)]
            if not frame.empty and frame[metric].notna().any():
                ax.plot(frame.week, frame[metric] * (100 if percent else 1),
                        label=series.replace("_", " "), color=color, marker="o",
                        linewidth=2 if series.startswith("full") else 1.5)
        ax.set_title(title)
        ax.set_xticks(WEEKS)
        ax.set_xlabel("Through 2026 week")
        ax.set_ylabel("Percent" if percent else title)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .055),
               ncol=3, frameon=False)
    fig.suptitle("2026 scientific consensus · F0–F19", fontsize=18)
    fig.text(.5, .014, "Full available: 120 votes on 263 games and 114 on eight market-missing games; strict full: 263 common games. Market probabilities cover 259 games.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .18, 1, .95])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_roster(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True,
                             facecolor="#F8F4EA")
    cmap = plt.get_cmap("coolwarm_r")
    highlighted = {"F17-market": "#7557A4", "F18": "#257E83", "F19": "#BD7A44"}
    for ax, architecture in zip(axes.flat, MODEL_IDS, strict=True):
        ax.set_facecolor("#FFFCF5")
        for number, stage in enumerate(STAGES):
            frame = table.loc[table.scope.eq("cumulative")
                              & table.series_type.eq("model")
                              & table.fingerprint.eq(stage)
                              & table.model_id.eq(architecture)].sort_values("week")
            if len(frame) != 5:
                raise ValueError(f"Missing roster curve: {stage} {architecture}")
            special = stage in ("F17-market", "F18", "F19")
            ax.plot(frame.week, frame.margin_mae,
                    color=highlighted.get(stage, cmap(number / 19)),
                    linewidth=2.1 if special else .8,
                    alpha=1 if special else .55,
                    marker="o" if special else None, markersize=3)
        ax.set_title(f"{architecture} · {FAMILY[architecture]}")
        ax.set_xticks(WEEKS)
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Through 2026 week")
    for ax in axes[:, 0]:
        ax.set_ylabel("Margin MAE")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color=color, linewidth=2.4, label=label)
                        for label, color in highlighted.items()],
               loc="lower center", bbox_to_anchor=(.5, .06), ncol=3, frameon=False)
    fig.suptitle("All 120 F0–F19 scientific model curves", fontsize=18)
    fig.text(.5, .015, "F19 curves use only market-covered games (263 total); the other 114 cells cover all 271 games. Earlier generations are thin lines.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .12, 1, .95])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build(output: Path = OUTPUT) -> dict:
    if (output / "manifest.json").exists():
        raise FileExistsError("F0–F19 full-season export is immutable")
    old, new, receipt = _sources()
    model, games = _model_games(old, new)
    consensus = _consensus(model, games, new)
    scored_model, scored_consensus = _scored(model), _scored(consensus)
    table = _performance(model, consensus, games)
    legacy = table.loc[table.scope.eq("cumulative") & table.week.eq(5)
                       & table.series_id.eq("legacy_f0_f17_market_108")].iloc[0]
    # The earlier output directory can be retired; bind validation to its
    # immutable source-data digest and recorded score instead of that directory.
    if digest(PRIOR) != "02b5a917d1f2259a7d179888d9b2aaf66c9702f92270002553e15b814f8f8241":
        raise ValueError("Archived F0–F17-market source data changed")
    if not np.isclose(legacy.margin_mae, 13.237191613873375, atol=1e-8) or not np.isclose(
            legacy.brier_score, 0.165770, atol=1e-6):
        raise ValueError("Rebuilt F0–F17 consensus differs from its archived score")
    score_receipt = json.loads((FROZEN / "evaluation/score_receipt.json").read_text())
    frozen_metrics = pd.read_csv(FROZEN / "evaluation/metrics.csv")
    for tier in ("F18", "F19"):
        computed = table.loc[table.scope.eq("cumulative") & table.week.eq(5)
                             & table.series_id.eq(f"fingerprint_{tier}")].iloc[0]
        frozen = frozen_metrics.loc[frozen_metrics.scope.eq("available")
                                    & frozen_metrics.tier.eq(tier)
                                    & frozen_metrics.model_id.eq("equal")].iloc[0]
        if not (computed.games == frozen.games and np.isclose(computed.margin_mae, frozen.mae, atol=1e-8)
                and np.isclose(computed.brier_score, frozen.brier, atol=1e-8)):
            raise ValueError(f"Rebuilt {tier} equal consensus differs from frozen score")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_games_parquet": output / "scientific_2026_model_game_results.parquet",
        "model_games_csv": output / "scientific_2026_model_game_results.csv",
        "consensus_games_parquet": output / "scientific_2026_consensus_game_results.parquet",
        "consensus_games_csv": output / "scientific_2026_consensus_game_results.csv",
        "model_weekly": output / "scientific_2026_model_weekly_performance.csv",
        "model_cumulative": output / "scientific_2026_model_cumulative_performance.csv",
        "consensus_weekly": output / "scientific_2026_consensus_weekly_performance.csv",
        "consensus_cumulative": output / "scientific_2026_consensus_cumulative_performance.csv",
        "scorecard": output / "scientific_2026_current_season_scorecard.csv",
        "consensus_figure": output / "scientific_2026_f0_f19_weekly_consensus.png",
        "roster_figure": output / "scientific_2026_f0_f19_roster_cumulative_mae.png",
    }
    scored_model.to_parquet(paths["model_games_parquet"], index=False, compression="zstd")
    scored_model.to_csv(paths["model_games_csv"], index=False, float_format="%.8f")
    scored_consensus.to_parquet(paths["consensus_games_parquet"], index=False, compression="zstd")
    scored_consensus.to_csv(paths["consensus_games_csv"], index=False, float_format="%.8f")
    for key, condition in (
        ("model_weekly", table.scope.eq("weekly") & table.series_type.eq("model")),
        ("model_cumulative", table.scope.eq("cumulative") & table.series_type.eq("model")),
        ("consensus_weekly", table.scope.eq("weekly") & table.series_type.isin(["consensus", "market"])),
        ("consensus_cumulative", table.scope.eq("cumulative") & table.series_type.isin(["consensus", "market"])),
        ("scorecard", table.scope.eq("cumulative") & table.week.eq(5)),
    ):
        table.loc[condition].to_csv(paths[key], index=False, float_format="%.8f")
    _plot_consensus(table, paths["consensus_figure"])
    _plot_roster(table, paths["roster_figure"])
    readme = output / "README.md"
    readme.write_text(
        "# 2026 F0–F19 full scientific roster\n\n"
        "Retrospective research export through Week 5: 271 completed FBS-vs-FBS games, "
        "20 fingerprints (F0–F16, F17-market, F18, F19), and six scientific models "
        "(M1, M2, M3, M4, M5, M10), for 120 model cells. The model-game tables "
        "include explicit unavailable F19 rows on eight games without an archived pregame "
        "market snapshot; F19 scores use the 263 available games.\n\n"
        "`full_available_120` averages all available model forecasts: 120 votes on 263 "
        "games, 114 on the eight market-missing games. `full_common_120` contains only "
        "the 263 games with every vote. `legacy_f0_f17_market_108` reproduces the prior "
        "consensus on all 271 games. Each `fingerprint_*` series averages its six models; "
        "F18/F19 use the probability calibration from their frozen equal consensus. "
        "Each `model_*_across_fingerprints` series averages that model across available "
        "fingerprints. The full-roster consensus averages individual calibrated model "
        "probabilities; it was assembled after scoring and is descriptive, with no "
        "2026 outcome-based weight optimization.\n\n"
        "Weekly tables score each week's games; cumulative tables score all games "
        "through that week. `games` counts available scored forecasts, while "
        "`cohort_games` includes explicitly unavailable F19 rows. ATS excludes missing "
        "and zero spreads, actual pushes, and exact predicted cover ties. Market Brier "
        "uses only 259 archived valid probabilities. The F18/F19 forecasts were frozen "
        "before this new scoring operation, but broader TDNet work had previously "
        "analyzed 2026 outcomes; this is an archived-pregame replay, not a fully "
        "blinded prospective study. Historical F19 quote-level timing remains "
        "unverified. All files here are separate from published weekly bundles.\n")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": 2026, "through_week": 5,
        "fingerprints": list(STAGES), "scientific_models": list(MODEL_IDS),
        "model_cells": 120, "games": 271, "market_eligible_games": 263,
        "f19_unavailable_games": 8,
        "legacy_forecasts_sha256": digest(PRIOR),
        "frozen_prediction_receipt": str(FROZEN / "receipt.json"),
        "frozen_prediction_receipt_sha256": digest(FROZEN / "receipt.json"),
        "frozen_prediction_sha256": receipt["predictions_sha256"],
        "frozen_score_receipt_sha256": digest(FROZEN / "evaluation/score_receipt.json"),
        "legacy_consensus_reference": {"margin_mae": 13.237191613873375,
                                       "brier_rounded_6dp": 0.165770},
        "script_sha256": digest(Path(__file__)),
        "consensus_policies": {
            "full_available_120": "equal mean of available individual model margins and probabilities; 120 votes on 263 games, 114 on eight F19-unavailable games",
            "full_common_120": "same means, restricted to the 263 games with all 120 votes",
            "legacy_f0_f17_market_108": "equal mean of the 108 archived model forecasts on all 271 games",
            "fingerprint_F18_F19": "mean of six margins with historical-OOF frozen equal-consensus probability calibration",
        },
        "not_a_new_model_or_tuning_run": True,
        "broader_project_2026_outcomes_previously_analyzed": True,
        "outputs": {key: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
                    for key, path in paths.items()},
        "readme": {"path": str(readme.relative_to(ROOT)), "sha256": digest(readme)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output": str(output), "model_rows": len(model),
            "consensus_rows": len(consensus), "model_cells": 120,
            "consensus_series": 29, "games": 271,
            "strict_full_games": 263,
            "legacy_consensus_mae": float(legacy.margin_mae)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
