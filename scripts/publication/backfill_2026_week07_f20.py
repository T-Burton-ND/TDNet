#!/usr/bin/env python3
"""Score retrospective CFBD Week 7 F20 with the newly extended tracker archive.

This never issues a pregame forecast or changes an earlier immutable release.
The raw tracker CSV and game-level model predictions stay in ignored private
data; the public artifact contains aggregate 126-cell scorecards and figures.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.f20.model import FEATURE_COLUMNS, HISTORICAL_COLUMNS, historical_features
from gridiron_ml.f20.scientific import MODEL_IDS, predict_margin_panel, predict_panel

from fit_f20_baselines import cohort
from build_2026_week08_scientific_full_scorecards import (
    MODEL_COLUMNS, build_scorecard, load_inputs, render_summary,
)

RAW = ROOT / "data/private/f20/historical_2026_20261025/ncaa2026.csv"
OLD_RAW = ROOT / "data/private/f20/historical_2026/ncaa2026.csv"
SCHEDULE = ROOT / "data/raw/cfbd/v2/games/2026.parquet"
FROZEN = ROOT / "data/private/f20/models/f20_scientific_2017_2024_v2_margin"
PRIVATE = ROOT / "data/private/f20/week07_backfill_20261025"
PUBLIC = ROOT / "publication/2026/week_07/post_game/retrospective_replay/scientific/expanded_f0_f20"
PRIOR = ROOT / "data/private/postgame_2026_week08/prior_f0_f19_model_games.parquet"
MARKET = ROOT / "data/private/postgame_2026_week08/prior_market_games.csv"
SAVED_WEEK7_120 = ROOT / "publication/2026/research/scientific_f0_f20_posthoc_scorecards_week01_07/weekly_scorecard.csv"
EXPECTED_RAW_SHA = "4248cdb3ffbe2b8c07f928cbd32e3571f007659bf96be94ea9bef2e348820c61"
ALIASES = {
    ("Coastal Carolina", "Appalachian St."): (401869844, "Coastal Carolina", "App State", 32, 35),
    ("Oregon St.", "Washington St."): (401860903, "Oregon State", "Washington State", 48, 30),
}
COLORS = {"roster_consensus": "#b94172", "fingerprint_consensus": "#2d7192",
          "vegas_reference": "#d4a928"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_week7() -> tuple[pd.DataFrame, dict]:
    if sha(RAW) != EXPECTED_RAW_SHA:
        raise ValueError("The extended tracker archive differs from the audited copy")
    raw = pd.read_csv(RAW)
    old = pd.read_csv(OLD_RAW)
    if len(raw) != 445 or len(old) != 329 or len(raw.loc[raw.week.eq(8)]) != 59:
        raise ValueError("Unexpected tracker archive size or Week 7 label")
    keys = ["Home", "Road", "week", "hscore", "vscore"]
    frozen_fields = list(dict.fromkeys(HISTORICAL_COLUMNS.values()))
    earlier = old[keys + frozen_fields].merge(
        raw.loc[raw.week.le(7), keys + frozen_fields], on=keys,
        suffixes=("_old", "_new"), validate="one_to_one")
    if len(earlier) != 329:
        raise ValueError("Updated archive changed an earlier game key")
    for field in frozen_fields:
        a, b = earlier[f"{field}_old"], earlier[f"{field}_new"]
        if not (a.isna().eq(b.isna()) & np.isclose(a.fillna(0), b.fillna(0), atol=0, rtol=0)).all():
            raise ValueError(f"Updated archive changed earlier {field} values")

    matched, audit = cohort(RAW.parent, SCHEDULE.parent, range(2026, 2027))
    found = matched.loc[matched.week_y.eq(7)].copy()
    if len(found) != 57 or not found.week_x.eq(8).all():
        raise ValueError("Expected 57 automatic CFBD Week 7 tracker matches")
    schedule = pd.read_parquet(SCHEDULE).set_index("id")
    aliases = []
    for (home, road), (game_id, canonical_home, canonical_away, home_score, away_score) in ALIASES.items():
        rows = raw.loc[raw.week.eq(8) & raw.Home.eq(home) & raw.Road.eq(road)]
        if len(rows) != 1:
            raise ValueError(f"Missing or duplicate alias: {road} at {home}")
        row = rows.iloc[0].copy()
        game = schedule.loc[game_id]
        if (game.week != 7 or game.home_team != canonical_home or game.away_team != canonical_away
                or game.home_points != home_score or game.away_points != away_score
                or row.hscore != home_score or row.vscore != away_score):
            raise ValueError(f"Alias result or matchup differs: {game_id}")
        row["id"] = game_id
        row["week_y"] = 7
        row["week_x"] = 8
        aliases.append(row)
    fields = ["id", "week_x", "week_y", "hscore", "vscore", *frozen_fields]
    week = pd.concat([found[fields], pd.DataFrame(aliases)[fields]], ignore_index=True)
    if len(week) != 59 or week.id.nunique() != 59 or week[fields].isna().any().any():
        raise ValueError("The 59-game Week 7 source is incomplete")
    prior = pd.read_parquet(PRIOR)
    expected = prior.loc[prior.week.eq(7), ["game_id", "actual_home_margin"]].drop_duplicates()
    if len(expected) != 59 or set(week.id) != set(expected.game_id):
        raise ValueError("F20 Week 7 IDs differ from the 120-cell scientific cohort")
    actual = week.hscore - week.vscore
    if not np.allclose(actual, week.id.map(expected.set_index("game_id").actual_home_margin)):
        raise ValueError("Tracker and scientific Week 7 scores differ")
    return week.sort_values("id").reset_index(drop=True), {
        "archive_rows": len(raw), "earlier_rows_unchanged": len(earlier),
        "week7_tracker_rows": 59, "automatic_matches": len(found),
        "audited_alias_matches": len(aliases), "cfbd_week": 7, "tracker_week": 8,
        "cohort_audit": audit["2026"],
    }


def predict_week7(week: pd.DataFrame) -> pd.DataFrame:
    manifest_path = FROZEN / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("canonical_name") != "F20"
            or tuple(manifest.get("model_ids", [])) != MODEL_IDS
            or manifest.get("feature_columns") != list(FEATURE_COLUMNS)
            or manifest["training_seasons"][-1] != 2024):
        raise ValueError("Frozen F20 scientific model contract changed")
    models, margin_models = {}, {}
    for model in MODEL_IDS:
        probability_path = FROZEN / f"{model}.joblib"
        margin_path = FROZEN / f"{model}_margin.joblib"
        if (sha(probability_path) != manifest["model_sha256"][model]
                or sha(margin_path) != manifest["margin_model_sha256"][model]):
            raise ValueError(f"Frozen F20 {model} model hash changed")
        models[model] = joblib.load(probability_path)
        margin_models[model] = joblib.load(margin_path)
    features = historical_features(week[list(dict.fromkeys(HISTORICAL_COLUMNS.values()))])
    probabilities = predict_panel(models, features.loc[:, list(FEATURE_COLUMNS)])
    margins = predict_margin_panel(margin_models, features.loc[:, list(FEATURE_COLUMNS)])
    rows = []
    for model in MODEL_IDS:
        for index, game in week.iterrows():
            rows.append({"game_id": int(game.id), "week": 7, "fingerprint": "F20", "model_id": model,
                         "pred_home_margin": float(margins[model][index]),
                         "pred_home_win_probability": float(probabilities[model][index]),
                         "actual_home_margin": float(game.hscore - game.vscore)})
    result = pd.DataFrame(rows, columns=MODEL_COLUMNS)
    if (len(result) != 354 or result.duplicated(["game_id", "model_id"]).any()
            or not result.pred_home_win_probability.between(0, 1).all()
            or not np.isfinite(result[["pred_home_margin", "actual_home_margin"]]).all().all()):
        raise ValueError("F20 Week 7 prediction panel is incomplete")
    return result


def cumulative_cards(predictions: pd.DataFrame, games: pd.DataFrame, matched_ids: set[int]) -> pd.DataFrame:
    rows = []
    for week in range(1, 9):
        selected = games.loc[games.game_id.isin(matched_ids) & games.week.le(week)]
        card = build_scorecard(predictions.loc[predictions.game_id.isin(selected.game_id)], selected)
        card.insert(0, "through_week", week)
        rows.append(card)
    result = pd.concat(rows, ignore_index=True)
    if len(result) != 8 * 155 or result.duplicated(["through_week", "row_type", "label"]).any():
        raise ValueError("Cumulative 126-cell scorecard is incomplete")
    return result


def render_cumulative(frame: pd.DataFrame, path: Path) -> None:
    selections = (
        ("roster_consensus", "F0–F20 consensus"),
        ("fingerprint_consensus", "F20 consensus"),
        ("vegas_reference", "Vegas"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#f7f3eb", constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, [
        ("su_accuracy", "Straight-up accuracy"), ("ats_accuracy", "ATS accuracy"),
        ("brier", "Brier · lower is better"), ("margin_mae", "Margin MAE · points")]):
        ax.set_facecolor("#f7f3eb")
        for kind, name in selections:
            if kind == "vegas_reference" and metric == "ats_accuracy":
                continue
            series = frame.loc[frame.row_type.eq(kind)]
            if kind == "fingerprint_consensus":
                series = series.loc[series.fingerprint.eq("F20")]
            ax.plot(series.through_week, series[metric], marker="o", lw=2.5,
                    color=COLORS[kind], label=name)
        ax.set_title(label, fontsize=14)
        ax.set_xticks(range(1, 9))
        ax.set_xlabel("Through CFBD week")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("2026 scientific F0–F20 · matched cumulative results", fontsize=20,
                 weight="bold", color="#20324b")
    fig.get_layout_engine().set(rect=(.02, .085, .98, .91))
    fig.text(.5, .018, "420 shared games, Weeks 1–8. Week 7 F20 is retrospective; source times and market quote times unverified.",
             ha="center", fontsize=9, color="#526173")
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_rotation(card: pd.DataFrame, png: Path, gif: Path) -> None:
    cells = card.loc[card.row_type.eq("model")]
    if len(cells) != 126 or not cells.games.eq(420).all():
        raise ValueError("3D endpoint is not paired across all 126 cells")
    palette = plt.get_cmap("turbo")
    fingers = [*(f"F{i}" for i in range(17)), "F17-market", "F18", "F19", "F20"]
    fig = plt.figure(figsize=(12.8, 9.3), facecolor="#f7f3eb")
    ax = fig.add_axes([.06, .14, .83, .77], projection="3d", facecolor="#f7f3eb")
    for point in cells.itertuples():
        ax.scatter(point.su_accuracy * 100, point.ats_accuracy * 100, point.brier, s=120,
                   color=palette(fingers.index(point.fingerprint) / 20), edgecolors="#f7f3eb",
                   linewidths=.8, depthshade=False)
    for values, setter, floor in [(cells.su_accuracy * 100, ax.set_xlim, 1.0),
                                  (cells.ats_accuracy * 100, ax.set_ylim, 1.0),
                                  (cells.brier, ax.set_zlim, .003)]:
        low, high = values.min(), values.max()
        pad = max((high - low) * .11, floor)
        setter(low - pad, high + pad)
    ax.set(xlabel="SU accuracy · % ↑", ylabel="ATS accuracy · % ↑", zlabel="Brier ↓")
    ax.tick_params(labelsize=11)
    ax.set_box_aspect((1.15, 1.15, .85), zoom=1.12)
    ax.view_init(elev=23, azim=-55)
    fig.suptitle("126 scientific cells · cumulative SU, ATS, Brier", fontsize=20,
                 weight="bold", color="#20324b")
    fig.text(.5, .075, "One marker per F×M pair · 420 shared Weeks 1–8 games · Week 7 F20 retrospective",
             ha="center", fontsize=10, color="#526173")
    fig.savefig(png, dpi=170, facecolor=fig.get_facecolor())
    animation = FuncAnimation(fig, lambda step: ax.view_init(elev=23, azim=-55 + step * 15),
                              frames=24, interval=125, blit=False)
    animation.save(gif, writer=PillowWriter(fps=8), dpi=75,
                   savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)


def main() -> None:
    if PRIVATE.exists() or PUBLIC.exists():
        raise FileExistsError("Week 7 F20 backfill output is immutable; choose a new version")
    week, source_audit = source_week7()
    new_f20 = predict_week7(week)
    previous, games, upstream = load_inputs()
    predictions = pd.concat([previous, new_f20], ignore_index=True)
    if predictions.duplicated(["game_id", "fingerprint", "model_id"]).any():
        raise ValueError("Week 7 F20 backfill duplicates an existing scientific result")
    counts = predictions.groupby("game_id").size().value_counts().to_dict()
    if counts != {126: 420, 120: 13, 6: 8}:
        raise ValueError(f"Unexpected full-season coverage: {counts}")
    week7_ids = set(new_f20.game_id)
    old_ids = set(previous.loc[previous.fingerprint.ne("F20"), "game_id"])
    f20_ids = set(predictions.loc[predictions.fingerprint.eq("F20"), "game_id"])
    matched_ids = f20_ids & old_ids
    if len(week7_ids) != 59 or len(matched_ids) != 420 or set(games.game_id) != set(predictions.game_id):
        raise ValueError("Week 7 or season matched cohort changed")
    week7 = build_scorecard(predictions.loc[predictions.game_id.isin(week7_ids)],
                            games.loc[games.game_id.isin(week7_ids)])
    prior_week7 = pd.read_csv(SAVED_WEEK7_120).loc[
        lambda frame: frame.week.eq(7) & frame.fingerprint.ne("F20")].set_index(["fingerprint", "model_id"])
    current_120 = week7.loc[week7.row_type.eq("model") & week7.fingerprint.ne("F20")].set_index(
        ["fingerprint", "model_id"])
    if len(prior_week7) != 120 or set(prior_week7.index) != set(current_120.index):
        raise ValueError("The saved Week 7 F0–F19 roster differs from the backfill")
    for field in ("games", "su_wins", "su_losses", "ats_wins", "ats_losses", "ats_pushes",
                  "brier", "margin_mae"):
        if not np.allclose(current_120[field], prior_week7[field], atol=1e-8, rtol=0):
            raise ValueError(f"Saved Week 7 F0–F19 {field} results differ")
    season = build_scorecard(predictions, games)
    matched = build_scorecard(predictions.loc[predictions.game_id.isin(matched_ids)],
                              games.loc[games.game_id.isin(matched_ids)])
    probability_ids = matched_ids & set(games.loc[games.market_probability.notna(), "game_id"])
    if len(probability_ids) != 412:
        raise ValueError("Vegas probability paired cohort is not 412 games")
    paired = build_scorecard(predictions.loc[predictions.game_id.isin(probability_ids)],
                             games.loc[games.game_id.isin(probability_ids)])
    cumulative = cumulative_cards(predictions, games, matched_ids)
    for name, card in [("week07", week7), ("season", season), ("matched", matched), ("paired", paired)]:
        if len(card) != 155 or len(card.loc[card.row_type.eq("model")]) != 126:
            raise ValueError(f"{name}: missing scorecard cells")

    PRIVATE.mkdir(parents=True)
    new_f20.to_parquet(PRIVATE / "week07_f20_59_game_model_results.parquet", index=False)
    PUBLIC.mkdir(parents=True)
    files = {
        "week07_matched_59.csv": week7,
        "season_latest_available_441.csv": season,
        "season_matched_420.csv": matched,
        "season_vegas_probability_paired_412.csv": paired,
        "cumulative_matched_420.csv": cumulative,
    }
    for name, card in files.items():
        card.to_csv(PUBLIC / name, index=False, float_format="%.10f")
    render_summary(week7, title="2026 Week 7 · F0–F20 scientific backfill",
                   note="59 shared games · 126 model cells · retrospective F20 replay",
                   path=PUBLIC / "week07_consensus_summary.png")
    render_summary(matched, title="2026 Weeks 1–8 · F0–F20 matched scorecard",
                   note="420 shared games · includes retrospective Week 7 F20 inputs",
                   path=PUBLIC / "season_matched_420_consensus_summary.png")
    render_cumulative(cumulative, PUBLIC / "cumulative_matched_420.png")
    render_rotation(matched, PUBLIC / "cumulative_matched_420_3d.png",
                    PUBLIC / "cumulative_matched_420_3d.gif")
    readme = """# CFBD Week 7 F0–F20 retrospective result backfill

The Prediction Tracker 2026 archive changed on October 25 from 329 to 445 rows. Its tracker Week 8 contains all 59 CFBD Week 7 games. All 59 have a market line and the five required F20 source margins: TeamRankings, Massey, ESPN FPI, FEI, and Sagarin Pred. The prior 329 game keys and frozen input values did not change. Fifty-seven Week 7 games matched CFBD automatically; the App State and Oregon State names required explicit matchup-and-score-checked aliases. The original raw CSV, source receipt, and 354 F20 model/game results are stored in ignored `data/private/f20/` paths and are not part of this public release.

The six frozen F20 M1–M5/M10 probability and margin models were applied to those weekly source values. The models were fitted after these Week 7 games; the archive was obtained after the games; original individual publication times and market quote times are unverified. The owner accepts weekly source values as pregame for a retrospective analysis. These are **not issued pregame F20 forecasts or prospective performance**. F0–F19 Week 7 cells are also retrospective backfills with their own source-timing limits; see `../full_f0_f19/README.md`.

Each scorecard CSV has 155 rows: 126 model cells, 21 fingerprint consensuses, six architecture consensuses, one full-roster consensus, and one Vegas reference. The 59-game Week 7 table has all 126 cells. Through Week 8, the 441-game available-data union contains 420 games shared by all 126 cells, 13 F0–F19-only games, and eight F20-only games. F20 has 428 games; F0–F19 have 433 each. The 412-game table additionally requires a saved Vegas probability for every game. ATS percentages exclude pushes and exact no-picks. Vegas has no ATS record against its own line.

`cumulative_matched_420.csv` and its four-metric PNG show the fully matched cumulative cohort through each of Weeks 1–8. The 3D PNG and GIF show one final cumulative SU/ATS/Brier marker per F×M cell on those same 420 games. The earlier 353/361-game scorecards are preserved and still exclude Week 7 F20.

`manifest.json` records source and output hashes, alias checks, and the source-timing limitations. Game-level predictions and raw provider data remain private.
"""
    (PUBLIC / "README.md").write_text(readme)
    private_path = PRIVATE / "week07_f20_59_game_model_results.parquet"
    receipt = {
        "schema": "tdnet-f20-week07-retrospective-backfill-v1", "season": 2026, "cfbd_week": 7,
        "source_audit": source_audit,
        "raw_sha256": sha(RAW), "frozen_model_manifest_sha256": sha(FROZEN / "manifest.json"),
        "private_model_games_sha256": sha(private_path), "public_path": str(PUBLIC.relative_to(ROOT)),
        "original_expert_publication_times_verified": False,
        "market_quote_times_verified": False, "models_fitted_before_week07": False,
        "source_weekly_pregame_assumption": "owner_accepted_for_retrospective_replay",
    }
    (PRIVATE / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    public_files = sorted(path for path in PUBLIC.iterdir() if path.is_file())
    manifest = {
        "schema": "tdnet-week07-f0-f20-retrospective-backfill-v1", "season": 2026,
        "cfbd_week": 7, "completed_week": 8,
        "cohorts": {"week07_matched": 59, "season_union": 441, "season_matched": 420,
                    "vegas_probability_paired": 412},
        "source_audit": source_audit,
        "limitations": {"F20_Week07_prospective": False,
                        "original_expert_publication_times_verified": False,
                        "market_quote_times_verified": False,
                        "models_fitted_before_week07": False},
        "sources": {"tracker_2026": sha(RAW), "old_tracker_2026": sha(OLD_RAW),
                    "cfbd_schedule": sha(SCHEDULE), "f20_model_manifest": sha(FROZEN / "manifest.json"),
                    "f0_f19_prior": sha(PRIOR), "f0_f19_week07_published": sha(SAVED_WEEK7_120),
                    "prior_market": sha(MARKET),
                    **{name: sha(path) for name, path in upstream.items()}},
        "private_predictions_sha256": sha(private_path),
        "files": {path.name: sha(path) for path in public_files},
    }
    (PUBLIC / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(PUBLIC), "week07_games": 59, "models": 126,
                      "matched_season_games": 420, "paired_vegas_probability_games": 412,
                      "week07_consensus": week7.loc[week7.row_type.eq("roster_consensus")].iloc[0][
                          ["su_wins", "ats_wins", "ats_losses", "ats_pushes", "brier", "margin_mae"]].to_dict()}))


if __name__ == "__main__":
    main()
