#!/usr/bin/env python3
"""Grade the Week 8 F0–F20 research cohort without changing frozen forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridiron_ml.f20.model import FEATURE_COLUMNS
from gridiron_ml.f20.scientific import MODEL_IDS, predict_margin_panel
from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.publication.figure_theme import apply_tdnet_theme


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = Path(
    "/groups/bsavoie2/tburton2/TDNet/f20_2026/prospective/runs/"
    "2026_w08_six_model_week08_capture_20261023T181112743109Z.json"
)


def _load_f20(run_path: Path, models_root: Path) -> tuple[pd.DataFrame, dict]:
    run = json.loads(run_path.read_text())
    if run.get("status") != "captured_with_research_forecasts" or run.get("week") != 8:
        raise ValueError("F20 source is not the recorded Week 8 research capture")
    game_ids = set(map(int, run["research_issuance"]["games"]))
    if len(game_ids) != 49 or run["research_issuance"]["issued"] != 49:
        raise ValueError("Expected exactly 49 archived Week 8 F20 games")
    receipt = json.loads(Path(run["receipt_file"]).read_text())
    snapshot_hash = run["source_raw_sha256"]
    if receipt["raw_sha256"] != snapshot_hash:
        raise ValueError("F20 capture and tracker receipt hashes disagree")
    manifest_path = models_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model_ids") != list(MODEL_IDS):
        raise ValueError("F20 margin model roster differs from M1–M5/M10")
    models = {}
    for model_id in MODEL_IDS:
        path = models_root / f"{model_id}_margin.joblib"
        if sha256_file(path) != manifest["margin_model_sha256"][model_id]:
            raise ValueError(f"F20/{model_id} margin checkpoint hash mismatch")
        models[model_id] = joblib.load(path)
    fit_time = pd.Timestamp(manifest["fit_at_utc"])
    forecast_root = Path(run["receipt_file"]).parents[1] / "forecasts"
    forecast_rows = []
    features = []
    sources = []
    for game_id in sorted(game_ids):
        matches = []
        for path in (forecast_root / str(game_id)).glob("*.json"):
            record = json.loads(path.read_text())
            if record.get("raw_snapshot_sha256") == snapshot_hash:
                matches.append((path, record))
        if len(matches) != 1:
            raise ValueError(f"Expected one captured F20 revision for game {game_id}")
        path, record = matches[0]
        kickoff = pd.Timestamp(record["kickoff_utc"])
        if not (
            pd.Timestamp(record["issued_at_utc"]) < kickoff
            and pd.Timestamp(record["source_site_updated_at_utc"]) < kickoff
            and fit_time < kickoff
        ):
            raise ValueError(f"F20 input or margin model followed kickoff for game {game_id}")
        if record.get("issuance_class") != "late_pregame_revision":
            raise ValueError(f"Unexpected F20 issuance class for game {game_id}")
        if set(record["home_win_probability"]) != set((*MODEL_IDS, "F20")):
            raise ValueError(f"Incomplete F20 probability panel for game {game_id}")
        if set(record["feature_vector"]) != set(FEATURE_COLUMNS):
            raise ValueError(f"F20 feature contract differs for game {game_id}")
        features.append(record["feature_vector"])
        sources.append({"game_id": game_id, "forecast_sha256": sha256_file(path),
                        "issued_at_utc": record["issued_at_utc"],
                        "kickoff_utc": record["kickoff_utc"]})
        forecast_rows.append(record)
    margins = predict_margin_panel(models, pd.DataFrame(features)[list(FEATURE_COLUMNS)])
    rows = []
    for index, record in enumerate(forecast_rows):
        for model_id in MODEL_IDS:
            rows.append({
                "game_id": int(record["game_id"]), "fingerprint": "F20",
                "model_id": model_id, "model_name": f"scientific_F20_{model_id}",
                "home_team": record["home_team"], "away_team": record["away_team"],
                "pred_home_win_probability": float(record["home_win_probability"][model_id]),
                "pred_home_margin": float(margins[model_id][index]),
            })
    return pd.DataFrame(rows), {
        "capture_run": str(run_path), "capture_run_sha256": sha256_file(run_path),
        "tracker_receipt": run["receipt_file"],
        "tracker_receipt_sha256": sha256_file(run["receipt_file"]),
        "tracker_snapshot_sha256": snapshot_hash,
        "margin_model_manifest": str(manifest_path),
        "margin_model_manifest_sha256": sha256_file(manifest_path),
        "margin_fit_at_utc": manifest["fit_at_utc"],
        "probabilities": "issued pregame at archived timestamps",
        "margins": "postgame application of a model frozen before all 49 kickoffs to archived pregame features",
        "forecast_files": sources,
    }


def _score(frame: pd.DataFrame, results: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    if frame.duplicated(["game_id", "fingerprint", "model_id"]).any():
        raise ValueError("Duplicate F×M game predictions")
    merged = frame.merge(results, on="game_id", how="left", validate="many_to_one")
    merged = merged.merge(lines, on="game_id", how="left", validate="many_to_one")
    if merged[["home_points", "away_points"]].isna().any().any():
        raise ValueError("Missing final scores in scientific cohort")
    p = pd.to_numeric(merged["pred_home_win_probability"], errors="coerce")
    margin = pd.to_numeric(merged["pred_home_margin"], errors="coerce")
    if p.isna().any() or margin.isna().any() or not p.between(0, 1).all():
        raise ValueError("Invalid scientific probabilities or margins")
    actual = merged.home_points - merged.away_points
    merged["actual_home_margin"] = actual
    merged["su_correct"] = (p.ge(.5) == actual.gt(0)).astype(int)
    merged["brier"] = (p - actual.gt(0).astype(int)) ** 2
    merged["margin_absolute_error"] = (margin - actual).abs()
    spread = pd.to_numeric(merged["market_spread_close"], errors="coerce")
    cover = actual + spread
    pred_cover = margin + spread
    merged["ats_result"] = np.select(
        [spread.isna(), cover.eq(0), pred_cover.eq(0), pred_cover.gt(0).eq(cover.gt(0))],
        ["No line", "Push", "No pick", "Win"], default="Loss",
    )
    return merged


def _scorecard(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fingerprint, model_id), part in frame.groupby(["fingerprint", "model_id"]):
        counts = part.ats_result.value_counts()
        graded = int(counts.get("Win", 0) + counts.get("Loss", 0))
        rows.append({"fingerprint": fingerprint, "model_id": model_id,
                     "games": int(len(part)), "su_accuracy": float(part.su_correct.mean()),
                     "brier": float(part.brier.mean()),
                     "margin_mae": float(part.margin_absolute_error.mean()),
                     "ats_wins": int(counts.get("Win", 0)),
                     "ats_losses": int(counts.get("Loss", 0)),
                     "ats_pushes": int(counts.get("Push", 0)),
                     "ats_no_line": int(counts.get("No line", 0)),
                     "ats_no_pick": int(counts.get("No pick", 0)),
                     "ats_accuracy": float(counts.get("Win", 0) / graded) if graded else np.nan})
    out = pd.DataFrame(rows)
    out["fingerprint_order"] = out.fingerprint.str.extract(r"^F(\d+)")[0].astype(int)
    return out.sort_values(["fingerprint_order", "model_id"]).drop(columns="fingerprint_order")


def _plot(scorecard: pd.DataFrame, output: Path) -> None:
    apply_tdnet_theme()
    fingerprints = ["F17-market" if i == 17 else f"F{i}" for i in range(21)]
    models = list(MODEL_IDS)
    metrics = [("su_accuracy", "Straight-up accuracy", "viridis", 0, 1),
               ("ats_accuracy", "ATS win rate", "viridis", 0, 1),
               ("brier", "Brier score · lower is better", "magma_r", None, None),
               ("margin_mae", "Margin MAE · lower is better", "magma_r", None, None)]
    fig, axes = plt.subplots(2, 2, figsize=(18, 15), constrained_layout=True)
    for ax, (column, title, cmap, low, high) in zip(axes.flat, metrics):
        matrix = scorecard.pivot(index="fingerprint", columns="model_id", values=column).reindex(index=fingerprints, columns=models)
        im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, vmin=low, vmax=high)
        ax.set_xticks(range(len(models)), models, fontsize=11)
        ax.set_yticks(range(len(fingerprints)), fingerprints, fontsize=10)
        ax.set_title(title, fontsize=16, pad=12)
        fig.colorbar(im, ax=ax, fraction=.045, pad=.02)
    fig.suptitle("2026 Week 8 · F0–F20 × M1–M5/M10 · 49 matched games", fontsize=20)
    fig.savefig(output / "week08_126_model_scorecard.png", dpi=210)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    rng = np.random.default_rng(20)
    for ax, (column, title, _, _, _) in zip(axes.flat, metrics):
        values = [scorecard.loc[scorecard.fingerprint.eq(fp), column].dropna().to_numpy() for fp in fingerprints]
        violin = ax.violinplot(values, positions=np.arange(21), widths=.75, showmedians=True, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#4F78A7")
            body.set_alpha(.35)
        violin["cmedians"].set_color("#173D60")
        for x, y in enumerate(values):
            ax.scatter(np.full(len(y), x) + rng.uniform(-.12, .12, len(y)), y,
                       s=24, color="#173D60", alpha=.8, zorder=3)
        ax.set_xticks(np.arange(21), fingerprints, rotation=60, fontsize=9)
        ax.set_title(title, fontsize=15)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("2026 Week 8 · six models per fingerprint · same 49 games", fontsize=20)
    fig.savefig(output / "week08_126_model_violins.png", dpi=210)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-f19", type=Path, default=ROOT / "data/private/postgame_2026_week08/scientific_frozen/full_f0_f19/scientific_all_game_predictions.csv")
    parser.add_argument("--f0-f6", type=Path, default=ROOT / "data/private/postgame_2026_week08/scientific_frozen/scientific_all_game_predictions.csv")
    parser.add_argument("--results", type=Path, default=ROOT / "data/publication/2026/weekly_operations/week_08/postgame_results.csv")
    parser.add_argument("--f20-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--f20-models", type=Path, default=ROOT / "data/private/f20/models/f20_scientific_2017_2024_v2_margin")
    parser.add_argument("--output", type=Path, default=ROOT / "publication/2026/week_08/post_game/scientific/expanded_f0_f20")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite expanded postgame package: {args.output}")
    frozen = pd.read_csv(args.f0_f19)
    if len(frozen) != 120 * 53 or frozen.game_id.nunique() != 53:
        raise ValueError("F0–F19 freeze must contain 120 cells on 53 games")
    original = pd.read_csv(args.f0_f6)
    lines = original[["game_id", "market_spread_close"]].drop_duplicates("game_id")
    results = pd.read_csv(args.results)
    f20, provenance = _load_f20(args.f20_run, args.f20_models)
    matched_ids = set(f20.game_id)
    if len(matched_ids) != 49 or not matched_ids.issubset(set(frozen.game_id)):
        raise ValueError("F20 capture is not a 49-game subset of the frozen slate")
    full_120 = _score(frozen, results, lines)
    matched = _score(pd.concat([frozen[frozen.game_id.isin(matched_ids)], f20], ignore_index=True), results, lines)
    matched_card = _scorecard(matched)
    if len(matched) != 126 * 49 or len(matched_card) != 126:
        raise ValueError("Matched F0–F20 roster is not complete")
    args.output.mkdir(parents=True, exist_ok=True)
    full_120.to_csv(args.output / "week08_f0_f19_120_model_games.csv", index=False)
    _scorecard(full_120).to_csv(args.output / "week08_f0_f19_120_model_scorecard.csv", index=False)
    matched.to_csv(args.output / "week08_f0_f20_126_model_games_matched.csv", index=False)
    matched_card.to_csv(args.output / "week08_f0_f20_126_model_scorecard_matched.csv", index=False)
    _plot(matched_card, args.output)
    manifest = {
        "schema": "tdnet-week08-expanded-f0-f20-postgame-v1", "season": 2026, "week": 8,
        "roster_cells": 126, "f0_f19_full_games": 53, "f0_f20_matched_games": 49,
        "f20_provenance": provenance,
        "sources": {str(path): sha256_file(path) for path in (args.f0_f19, args.f0_f6, args.results)},
        "files": {path.name: sha256_file(path) for path in args.output.iterdir() if path.is_file()},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output / "README.md").write_text(
        "# Week 8 expanded F0–F20 scientific scorecards\n\n"
        "The 126 F×M cells are compared on the same 49 games. F0–F19 also have a "
        "separate 53-game frozen-slate scorecard. F20 probabilities were issued before "
        "these 49 kickoffs as late pregame research revisions; they were outside the "
        "official weekly cutoff. The six F20 margins are a postgame application of "
        "models frozen before all 49 kickoffs to the archived pregame feature vectors. "
        "The margin and ATS results are therefore research evaluations, not issued picks. "
        "All 21 fingerprints use identical styling in the figures.\n",
        encoding="utf-8",
    )
    print(json.dumps({"matched_games": 49, "cells": len(matched_card),
                      "f20_scorecard": matched_card[matched_card.fingerprint.eq("F20")].to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
