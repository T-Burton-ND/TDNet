#!/usr/bin/env python3
"""Build Week 8 and season-to-date scorecards for all 126 scientific cells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
POST = ROOT / "publication/2026/week_08/post_game/scientific/expanded_f0_f20"
PRIVATE = ROOT / "data/private/postgame_2026_week08"
OUTPUT = POST / "full_roster_scorecards"
FINGERPRINTS = [*(f"F{i}" for i in range(17)), "F17-market", "F18", "F19", "F20"]
MODELS = ["M1", "M2", "M3", "M4", "M5", "M10"]
ARCHITECTURES = {"M1": "linear", "M2": "spline", "M3": "tree", "M4": "boosted", "M5": "neural", "M10": "knn"}
MODEL_COLUMNS = ["game_id", "week", "fingerprint", "model_id", "pred_home_margin", "pred_home_win_probability", "actual_home_margin"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    sources = {
        "f0_f19_prior": PRIVATE / "prior_f0_f19_model_games.parquet",
        "f20_prior": ROOT / "data/private/f20/replay_2026_week01_07_v3_margin/f20_replay.parquet",
        "prior_market": PRIVATE / "prior_market_games.csv",
        "f0_f19_week08": POST / "week08_f0_f19_120_model_games.csv",
        "f20_week08": POST / "week08_f0_f20_126_model_games_matched.csv",
        "week08_frozen_market": PRIVATE / "frozen_bundle/public/predictions.csv.gz",
    }
    prior = pd.read_parquet(sources["f0_f19_prior"])
    prior = prior.rename(columns={"margin_error": "source_margin_error"})[MODEL_COLUMNS]
    f20_wide = pd.read_parquet(sources["f20_prior"])
    f20 = []
    for model in MODELS:
        rows = f20_wide[["game_id", "cfbd_week", "actual_home_margin", f"margin_{model}", f"probability_{model}"]].copy()
        rows.columns = ["game_id", "week", "actual_home_margin", "pred_home_margin", "pred_home_win_probability"]
        rows["fingerprint"] = "F20"
        rows["model_id"] = model
        f20.append(rows[MODEL_COLUMNS])
    week = pd.read_csv(sources["f0_f19_week08"])[MODEL_COLUMNS]
    week_f20 = pd.read_csv(sources["f20_week08"])
    week_f20 = week_f20.loc[week_f20.fingerprint.eq("F20"), MODEL_COLUMNS]
    week_f20["week"] = 8
    predictions = pd.concat([prior, *f20, week, week_f20], ignore_index=True)
    predictions["game_id"] = predictions.game_id.astype("int64")
    predictions["week"] = predictions.week.astype("int64")
    if predictions.duplicated(["game_id", "fingerprint", "model_id"]).any():
        raise ValueError("Duplicate game × scientific cell")
    if not np.isfinite(predictions[["pred_home_margin", "pred_home_win_probability", "actual_home_margin"]].to_numpy()).all():
        raise ValueError("Scientific prediction or result is non-finite")
    if not predictions.pred_home_win_probability.between(0, 1).all():
        raise ValueError("Scientific probability outside [0, 1]")
    prior_market = pd.read_csv(sources["prior_market"])[["game_id", "week", "actual_home_margin", "market_home_spread", "market_home_probability"]]
    prior_market.columns = ["game_id", "week", "actual_home_margin", "market_spread", "market_probability"]
    week_market = pd.read_csv(sources["week08_frozen_market"]).drop_duplicates("game_id")
    week_truth = week[["game_id", "actual_home_margin"]].drop_duplicates("game_id")
    week_market = week_market[["game_id", "week", "market_spread_close", "vegas_home_win_probability_as_of_prediction"]]
    week_market = week_market.merge(week_truth, on="game_id", validate="one_to_one")
    week_market.columns = ["game_id", "week", "market_spread", "market_probability", "actual_home_margin"]
    week_market = week_market[["game_id", "week", "actual_home_margin", "market_spread", "market_probability"]]
    games = pd.concat([prior_market, week_market], ignore_index=True)
    f20_only = predictions.loc[
        predictions.fingerprint.eq("F20") & ~predictions.game_id.isin(games.game_id),
        ["game_id", "week", "actual_home_margin"],
    ].drop_duplicates("game_id")
    if len(f20_only) != 8:
        raise ValueError("Expected eight F20-only Week 1 games without archived market rows")
    f20_only["market_spread"] = np.nan
    f20_only["market_probability"] = np.nan
    games = pd.concat([games, f20_only[games.columns]], ignore_index=True)
    games["game_id"] = games.game_id.astype("int64")
    games["week"] = games.week.astype("int64")
    if len(games) != 441 or games.game_id.nunique() != 441:
        raise ValueError("Expected 441 distinct archived 2026 games through Week 8")
    if not set(predictions.game_id).issubset(set(games.game_id)):
        raise ValueError("Prediction without a market/result record")
    check = predictions.merge(games[["game_id", "week", "actual_home_margin"]], on="game_id", validate="many_to_one", suffixes=("", "_market"))
    if not (check.week.eq(check.week_market) & np.isclose(check.actual_home_margin, check.actual_home_margin_market)).all():
        raise ValueError("Prediction and market game/week/result records disagree")
    cell_counts = predictions.groupby(["fingerprint", "model_id"]).game_id.nunique()
    if len(cell_counts) != 126 or set(cell_counts.loc["F20"]) != {369} or set(cell_counts.drop(index="F20")) != {433}:
        raise ValueError("Scientific cell coverage differs from 120×433 plus 6×369")
    return predictions, games, sources


def _summarize(pred: pd.DataFrame, games: pd.DataFrame, *, kind: str, label: str,
               fingerprint: str = "", model_id: str = "") -> dict:
    pred = pred[["game_id", "pred_home_margin", "pred_home_win_probability", "model_count"]]
    if pred.game_id.duplicated().any():
        raise ValueError(f"{label}: more than one forecast per game")
    joined = pred.merge(games, on="game_id", validate="one_to_one")
    n = len(joined)
    if n == 0:
        raise ValueError(f"{label}: empty score cohort")
    actual = joined.actual_home_margin.gt(0)
    probability = joined.pred_home_win_probability
    margin = joined.pred_home_margin
    spread = joined.market_spread
    has_line = spread.notna()
    cover = joined.actual_home_margin + spread
    forecast_cover = margin + spread
    push = has_line & cover.eq(0)
    no_pick = has_line & ~push & forecast_cover.eq(0)
    decision = has_line & ~push & ~no_pick
    win = decision & forecast_cover.gt(0).eq(cover.gt(0))
    loss = decision & ~win
    su_wins = int(probability.ge(.5).eq(actual).sum())
    row = {
        "row_type": kind, "label": label, "fingerprint": fingerprint,
        "model_id": model_id, "architecture": ARCHITECTURES.get(model_id, ""),
        "games": n, "weeks": ",".join(map(str, sorted(joined.week.unique()))),
        "models_per_game_min": int(joined.model_count.min()),
        "models_per_game_max": int(joined.model_count.max()),
        "su_wins": su_wins, "su_losses": n - su_wins, "su_games": n,
        "su_accuracy": su_wins / n,
        "ats_wins": int(win.sum()), "ats_losses": int(loss.sum()),
        "ats_pushes": int(push.sum()), "ats_no_line": int((~has_line).sum()),
        "ats_no_pick": int(no_pick.sum()),
        "ats_decisions": int(decision.sum()),
        "ats_accuracy": float(win.sum() / decision.sum()) if decision.any() else np.nan,
        "brier_games": n, "brier": float(((probability - actual.astype(float)) ** 2).mean()),
        "margin_games": n, "margin_mae": float((margin - joined.actual_home_margin).abs().mean()),
    }
    if sum(row[name] for name in ["ats_wins", "ats_losses", "ats_pushes", "ats_no_line", "ats_no_pick"]) != n:
        raise ValueError(f"{label}: ATS outcome counts do not sum to games")
    return row


def _vegas(games: pd.DataFrame) -> dict:
    line = games.market_spread
    has_line = line.notna()
    actual = games.actual_home_margin.gt(0)
    wins = int(((-line[has_line]).gt(0) == actual[has_line]).sum())
    has_probability = games.market_probability.notna()
    probability = games.loc[has_probability, "market_probability"]
    if not probability.between(0, 1).all():
        raise ValueError("Market probability outside [0, 1]")
    return {
        "row_type": "vegas_reference", "label": "Vegas reference", "fingerprint": "", "model_id": "", "architecture": "",
        "games": len(games), "weeks": ",".join(map(str, sorted(games.week.unique()))),
        "models_per_game_min": 0, "models_per_game_max": 0,
        "su_wins": wins, "su_losses": int(has_line.sum()) - wins, "su_games": int(has_line.sum()),
        "su_accuracy": wins / has_line.sum(),
        "ats_wins": np.nan, "ats_losses": np.nan, "ats_pushes": np.nan,
        "ats_no_line": np.nan, "ats_no_pick": np.nan, "ats_decisions": 0, "ats_accuracy": np.nan,
        "brier_games": int(has_probability.sum()),
        "brier": float(((probability - actual[has_probability].astype(float)) ** 2).mean()),
        "margin_games": int(has_line.sum()),
        "margin_mae": float(((-line[has_line]) - games.loc[has_line, "actual_home_margin"]).abs().mean()),
    }


def build_scorecard(predictions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["cell"] = predictions.fingerprint + "/" + predictions.model_id
    rows = [_vegas(games)]

    def consensus(frame: pd.DataFrame, kind: str, label: str, *, fp: str = "", model: str = "") -> dict:
        grouped = frame.groupby("game_id", as_index=False).agg(
            pred_home_margin=("pred_home_margin", "mean"),
            pred_home_win_probability=("pred_home_win_probability", "mean"),
            model_count=("cell", "nunique"),
        )
        return _summarize(grouped, games, kind=kind, label=label, fingerprint=fp, model_id=model)

    rows.append(consensus(predictions, "roster_consensus", "F0–F20 available-cell consensus"))
    for fp in FINGERPRINTS:
        chunk = predictions.loc[predictions.fingerprint.eq(fp)]
        rows.append(consensus(chunk, "fingerprint_consensus", f"{fp} consensus", fp=fp))
        for model in MODELS:
            cell = chunk.loc[chunk.model_id.eq(model)].copy()
            cell["model_count"] = 1
            rows.append(_summarize(cell, games, kind="model", label=f"{fp}/{model}", fingerprint=fp, model_id=model))
    for model in MODELS:
        rows.append(consensus(predictions.loc[predictions.model_id.eq(model)], "architecture_consensus",
                              f"{model} {ARCHITECTURES[model]} consensus", model=model))
    out = pd.DataFrame(rows)
    if len(out) != 155 or out.loc[out.row_type.eq("model")].shape[0] != 126:
        raise ValueError("Expected Vegas + roster + 21 fingerprints + 126 cells + 6 architectures")
    return out


def verify_against_saved(card: pd.DataFrame, saved_path: Path) -> None:
    saved = pd.read_csv(saved_path).set_index(["fingerprint", "model_id"])
    models = card.loc[card.row_type.eq("model")].set_index(["fingerprint", "model_id"])
    if set(models.index) != set(saved.index):
        raise ValueError(f"Model roster differs from {saved_path}")
    for key in models.index:
        ours, theirs = models.loc[key], saved.loc[key]
        if int(ours.games) != int(theirs.games):
            raise ValueError(f"{saved_path}: {key} game count differs")
        checks = ["brier", "margin_mae", "ats_wins", "ats_losses", "ats_pushes", "ats_no_line", "ats_no_pick", "ats_accuracy"]
        checks.append("su_accuracy")
        for field in checks:
            if not np.isclose(float(ours[field]), float(theirs[field]), rtol=0, atol=2e-6):
                raise ValueError(f"{saved_path}: {key} {field} differs: {ours[field]} versus {theirs[field]}")


def render_summary(card: pd.DataFrame, *, title: str, note: str, path: Path) -> None:
    summary = card.loc[card.row_type.ne("model")].copy()
    def pct(value: float) -> str:
        return "—" if pd.isna(value) else f"{value * 100:.1f}%"
    def record(row: pd.Series, prefix: str) -> str:
        wins = row[f"{prefix}_wins"]
        if pd.isna(wins):
            return "N/A"
        text = f"{int(wins)}–{int(row[f'{prefix}_losses'])}"
        if prefix == "ats" and row.ats_pushes:
            text += f"–{int(row.ats_pushes)}"
        return text
    labels = ["Vegas", "All available", *[f"{fp} consensus" for fp in FINGERPRINTS],
              *[f"{model} · {ARCHITECTURES[model]}" for model in MODELS]]
    values = []
    for label, (_, row) in zip(labels, summary.iterrows()):
        brier = f"{row.brier:.3f}"
        if row.brier_games != row.games:
            brier += f" ({int(row.brier_games)})"
        values.append([label, str(int(row.games)), record(row, "su"), pct(row.su_accuracy),
                       record(row, "ats"), pct(row.ats_accuracy), f"{row.margin_mae:.2f}", brier])
    fig, ax = plt.subplots(figsize=(14, 13.2), facecolor="#f7f3eb")
    ax.axis("off")
    fig.suptitle(title, fontsize=20, fontweight="bold", color="#20324b", y=.985)
    fig.text(.5, .947, note, ha="center", va="center", fontsize=10, color="#4b5867")
    table = ax.table(cellText=values, colLabels=["Row", "Games", "SU", "SU %", "ATS", "ATS %", "Margin MAE · pts", "Brier"],
                     colWidths=[.29, .07, .105, .08, .115, .08, .12, .12],
                     cellLoc="left", colLoc="left", bbox=[.025, .075, .95, .83])
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d4d9de")
        cell.set_linewidth(.45)
        if r == 0:
            cell.set_facecolor("#20324b")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif r in (1, 2):
            cell.set_facecolor("#eee5c9" if r == 1 else "#dce7ef")
            cell.get_text().set_fontweight("bold")
        elif r >= 24:
            cell.set_facecolor("#e8e3f2" if r % 2 == 0 else "#f1edf7")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#edf1f4")
    vegas = summary.iloc[0]
    fig.text(.03, .039,
             f"All 126 F×M cells are in the CSV. Vegas: {int(vegas.su_games)} spread lines, "
             f"{int(vegas.brier_games)} probability quotes; ATS is N/A against its own line.",
             fontsize=9.5, color="#4b5867")
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    predictions, games, sources = load_inputs()
    f20_ids = set(predictions.loc[predictions.fingerprint.eq("F20"), "game_id"])
    other_ids = set(predictions.loc[predictions.fingerprint.ne("F20"), "game_id"])
    week_ids = set(games.loc[games.week.eq(8), "game_id"])
    scopes = {
        "week08_latest_available": week_ids,
        "week08_matched_49": week_ids & f20_ids,
        "season_latest_available": set(games.game_id),
        "season_matched_361": f20_ids & other_ids,
    }
    expected = {"week08_latest_available": 53, "week08_matched_49": 49,
                "season_latest_available": 441, "season_matched_361": 361}
    if {name: len(ids) for name, ids in scopes.items()} != expected:
        raise ValueError("Weekly or season cohort has unexpected game count")
    cell = predictions.groupby("game_id").size()
    if cell.value_counts().to_dict() != {126: 361, 120: 72, 6: 8}:
        raise ValueError("Scientific game-level coverage is not 361 shared, 72 F0–F19-only, and eight F20-only games")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scorecards = {}
    for name, ids in scopes.items():
        game_scope = games.loc[games.game_id.isin(ids)].copy()
        prediction_scope = predictions.loc[predictions.game_id.isin(ids)].copy()
        card = build_scorecard(prediction_scope, game_scope)
        card.to_csv(OUTPUT / f"{name}.csv", index=False, float_format="%.10f")
        scorecards[name] = card
    verify_against_saved(scorecards["week08_matched_49"], POST / "week08_f0_f20_126_model_scorecard_matched.csv")
    verify_against_saved(scorecards["season_latest_available"], POST / "cumulative/latest_available_126_model_scorecard.csv")
    paired_ids = f20_ids & set(games.loc[games.market_probability.notna(), "game_id"])
    if len(paired_ids) != 353:
        raise ValueError("Previously published Vegas-probability paired cohort changed")
    paired_card = build_scorecard(predictions.loc[predictions.game_id.isin(paired_ids)], games.loc[games.game_id.isin(paired_ids)])
    verify_against_saved(paired_card, POST / "cumulative/matched_353_game_126_model_scorecard.csv")
    render_summary(scorecards["week08_latest_available"], title="2026 Week 8 · F0–F20 scientific scorecard",
                   note="F0–F19: 53 games · F20: 49 games · consensus rows average available cells per game",
                   path=OUTPUT / "week08_consensus_summary.png")
    render_summary(scorecards["season_latest_available"], title="2026 through Week 8 · F0–F20 scientific scorecard",
                   note="441-game union · F0–F19: 433 · F20: 369 · Week 7 has no F20 inputs",
                   path=OUTPUT / "season_consensus_summary.png")
    readme = """# 2026 scientific F0–F20 scorecards through Week 8

Each CSV has 155 rows: 126 individual F×M models, 21 fingerprint consensuses, six architecture consensuses, one available-cell roster consensus, and one Vegas reference. M1–M5 and M10 are the six architectures. Fingerprint and architecture consensuses average the constituent home-win probabilities and predicted home margins **per game**, then score those predictions; their metrics are not averages of model metrics.

| Table | Model coverage | Use |
| --- | --- | --- |
| `week08_latest_available.csv` | F0–F19: 53 games; F20: 49 | All saved Week 8 predictions. |
| `week08_matched_49.csv` | Every cell: 49 games | Direct Week 8 model comparison. |
| `season_latest_available.csv` | F0–F19: 433 games; F20: 369; union: 441 | All available 2026 games through Week 8. |
| `season_matched_361.csv` | Every cell: 361 games | Direct 126-model season comparison. |

The latest-available architecture and whole-roster consensuses have 1, 20, or 21 fingerprints (6, 120, or 126 models) per game: 361 shared games, 72 F0–F19-only games, and eight Week 1 F20-only games. F20 is absent in Week 7 and four Week 8 games. The matched tables use all 126 cells on every game. `models_per_game_min` and `models_per_game_max` make that explicit. A separate previously published 353-game comparison additionally requires a Vegas probability for every game; the 361-game matched table keeps eight games without an archived Vegas probability, so its Vegas Brier has 353 observations. The latest-available season Vegas row has 432 spread lines and 425 probability quotes out of the 441-game union. Vegas SU and margin use the archived spread when present; Vegas Brier uses the archived probability. Vegas has no ATS pick against its own line, so its ATS cells are N/A. ATS percentages exclude pushes, missing lines, and exact no-picks; counts retain each category.

The model rows were recomputed from saved game-level forecasts and certified results. All 126 weekly matched and latest-available season model rows reconcile to the previously published scorecards; a 353-game subset also reconciles to the previously published paired scorecard. F20 Weeks 1–6 are retrospective replay on an assumed weekly expert archive. Week 8 F20 probabilities were late pregame research revisions, while its margins were applied after the games using models fitted before kickoff. Many F0–F19 early-week scores are retrospective backfills; treat season rankings as descriptive.

The two PNGs summarize the Vegas, roster, fingerprint, and architecture rows. The CSVs retain every individual model and exact denominator.
"""
    (OUTPUT / "README.md").write_text(readme)
    outputs = sorted(p for p in OUTPUT.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {
        "schema": "tdnet-week08-scientific-full-roster-scorecards-v1",
        "season": 2026, "completed_week": 8,
        "cohorts": {name: len(ids) for name, ids in scopes.items()},
        "rows_per_table": 155,
        "validation": "126 weekly matched and season latest-available model rows plus 126 published 353-game paired model rows reconciled",
        "sources": {name: sha256(path) for name, path in sources.items()},
        "files": {p.name: sha256(p) for p in outputs},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cohorts": manifest["cohorts"], "rows_per_table": 155,
                      "vegas_brier_games": {name: int(card.iloc[0].brier_games) for name, card in scorecards.items()},
                      "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
