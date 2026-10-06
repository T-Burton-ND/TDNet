#!/usr/bin/env python3
"""Build a separate Week 6 research ballot from all 2026 scientific forecasts.

The ballot fits team strength effects to each model's own predicted margins on
the 271 completed games through Week 5. It does not use realized game results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.publication.scientific_weekly import (
    plot_scientific_all_team_power_ranking,
    plot_scientific_power_top25,
    scientific_consensus_power_rankings,
    validate_scientific_ballots,
)
from gridiron_ml.td_run.poll_viz import plot_ballot_logo_grid

OUTPUT = ROOT / "publication/2026/week_06/pre_game/scientific/full_f0_f17_market_whatif"
FAMILIES = {"M1": "linear", "M2": "spline", "M3": "tree", "M4": "boosted", "M5": "neural", "M10": "knn"}
MODEL_IDS = list(FAMILIES)
STAGES = [*(f"F{i}" for i in range(17)), "F17-market"]


def _load_forecasts() -> pd.DataFrame:
    week_frames = []
    for week in range(6):
        path = ROOT / f"publication/2026/week_{week:02d}/post_game/scientific/full_f0_f8/scientific_model_game_results.csv"
        frame = pd.read_csv(path)
        week_frames.append(frame[["game_id", "week", "home_team", "away_team", "neutral_site", "model_name", "model_family", "fingerprint", "pred_home_margin"]])
    old = pd.concat(week_frames, ignore_index=True)

    backfill_path = ROOT / "data/what_if_2026_fingerprints/f0_f8_week1_backfill/run_all54/tables/all_game_model_predictions.parquet"
    backfill = pd.read_parquet(backfill_path)
    backfill = backfill[["game_id", "week", "home_team", "away_team", "neutral_site", "model_name", "model_family", "fingerprint", "pred_home_margin"]]

    generated = []
    for generation in range(9, 18):
        folder = "f17_market" if generation == 17 else f"f{generation:02d}"
        path = ROOT / f"data/what_if_2026_fingerprints/{folder}_predictions/predictions.parquet"
        frame = pd.read_parquet(path)
        frame["model_family"] = frame.model_name.str.rsplit("_", n=1).str[-1].map(FAMILIES)
        generated.append(frame[["game_id", "week", "home_team", "away_team", "model_family", "fingerprint", "pred_home_margin"]])
    nextgen = pd.concat(generated, ignore_index=True)
    nextgen["model_name"] = nextgen.apply(lambda row: f"scientific_{row.fingerprint}_{next(k for k,v in FAMILIES.items() if v == row.model_family)}", axis=1)
    nextgen["neutral_site"] = False

    forecasts = pd.concat([old, backfill, nextgen], ignore_index=True, sort=False)
    forecasts["game_id"] = pd.to_numeric(forecasts.game_id, errors="coerce").astype("Int64")
    forecasts["pred_home_margin"] = pd.to_numeric(forecasts.pred_home_margin, errors="coerce")
    forecasts["neutral_site"] = forecasts.neutral_site.fillna(False).astype(bool)
    forecasts = forecasts.dropna(subset=["game_id", "home_team", "away_team", "pred_home_margin"])
    forecasts["model_id"] = forecasts.model_name.str.rsplit("_", n=1).str[-1]
    forecasts = forecasts.loc[forecasts.fingerprint.isin(STAGES) & forecasts.model_id.isin(MODEL_IDS)].copy()
    if forecasts.duplicated(["game_id", "fingerprint", "model_id"]).any():
        dupes = forecasts.loc[forecasts.duplicated(["game_id", "fingerprint", "model_id"], keep=False), ["game_id", "fingerprint", "model_id"]]
        raise ValueError(f"Duplicate model/game predictions: {dupes.head().to_dict('records')}")
    counts = forecasts.groupby(["fingerprint", "model_id"]).game_id.nunique()
    if len(counts) != 108 or not counts.eq(271).all():
        raise ValueError(f"Expected 271 forecast games for every one of 108 cells; got {counts.to_dict()}")
    return forecasts


def _model_ballots(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    teams = sorted(set(forecasts.home_team.astype(str)) | set(forecasts.away_team.astype(str)))
    if len(teams) != 138:
        raise ValueError(f"Expected 138 FBS teams; found {len(teams)}")
    team_col = {team: i for i, team in enumerate(teams)}

    for (stage, model_id), games in forecasts.groupby(["fingerprint", "model_id"], sort=False):
        # y = home_team_strength - away_team_strength + home-site effect.
        # Team coefficients are centered after OLS and represent margins versus
        # the all-team average. Forecasts, not target scores, are the response.
        matrix = np.zeros((len(games), len(teams) + 1), dtype=float)
        idx = np.arange(len(games))
        home = games.home_team.astype(str).map(team_col).to_numpy()
        away = games.away_team.astype(str).map(team_col).to_numpy()
        matrix[idx, home] = 1.0
        matrix[idx, away] = -1.0
        matrix[:, -1] = (~games.neutral_site.to_numpy(dtype=bool)).astype(float)
        y = games.pred_home_margin.to_numpy(dtype=float)
        solution, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        ratings = solution[:-1]
        ratings = ratings - ratings.mean()
        ordered = sorted(zip(teams, ratings), key=lambda pair: (-pair[1], pair[0]))
        for rank, (team, score) in enumerate(ordered, start=1):
            rows.append({
                "poll_objective": "predicted_margin_network_fit",
                "keys_team": team,
                "ballot_model": f"scientific_{stage}_{model_id}",
                "power_rating_vs_average": float(score),
                "ballot_rank": rank,
                "poll_points": max(26 - rank, 0),
                "top25_vote": rank <= 25,
                "first_place_vote": rank == 1,
                "model_id": model_id,
                "model_level": "fingerprint_generation",
                "model_family": FAMILIES[model_id],
                "objective": "margin",
                "fingerprint": stage,
                "games_fit": len(games),
                "estimated_home_site_effect": float(solution[-1]),
                "fit_rmse": float(np.sqrt(np.mean((matrix @ solution - y) ** 2))),
                "coverage_status": (
                    "partial_market_timing_week1_backfill" if stage in {"F7", "F8"}
                    else "unverified_market_quote_times" if stage == "F17-market"
                    else "market_only_research_ballot" if stage == "F7"
                    else "cutoff_checked_nonmarket_inputs"
                ),
            })
    ballots = pd.DataFrame(rows)
    validate_scientific_ballots(ballots)
    if ballots.groupby("ballot_model").size().ne(138).any() or len(ballots) != 108 * 138:
        raise ValueError("Ballot must contain 138 unique teams for every model/fingerprint cell.")
    return ballots


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    forecasts = _load_forecasts()
    ballots = _model_ballots(forecasts)
    rankings = scientific_consensus_power_rankings(ballots)
    ballots.to_csv(OUTPUT / "scientific_full_ballots.csv", index=False, float_format="%.6f")
    rankings.to_csv(OUTPUT / "scientific_consensus_power_rankings.csv", index=False, float_format="%.6f")
    rankings.head(25).to_csv(OUTPUT / "scientific_top25_ballot.csv", index=False, float_format="%.6f")
    top_ballots = ballots.loc[ballots.ballot_rank.le(25)].copy()
    logo_dir = ROOT / "data/meta/logos"
    plot_ballot_logo_grid(top_ballots, OUTPUT / "scientific_top25_ballots.png", top_n=25,
                          logo_dir=logo_dir,
                          title="2026 Week 6 · F0–F17-market scientific what-if Top 25")
    plot_scientific_power_top25(rankings, OUTPUT / "scientific_top25.png", season=2026,
                                week=6, logo_dir=logo_dir,
                                display_label="Full F0–F17-market research ballot")
    all_team_power = rankings.copy()
    # Use the same average-rating rank shown in the Top 25 graphic rather than
    # switching to Borda points for the full-team picture.
    all_team_power["poll_points_rank"] = all_team_power["consensus_power_rank"]
    plot_scientific_all_team_power_ranking(
        all_team_power,
        OUTPUT / "scientific_all_fbs_power_rankings.png",
        season=2026,
        week=6,
        bulletin_label="INDEPENDENT FULL-ROSTER RESEARCH BALLOT",
        title="ALL FBS TEAMS · F0–F17-MARKET CONSENSUS",
        footer_label="RANKED BY MEAN POWER RATING ACROSS 108 MODEL × FINGERPRINT BALLOTS",
    )
    manifest = {
        "scope": "Independent 2026 Week 6 research ballot; not an official published prediction",
        "season": 2026,
        "reader_week": 6,
        "team_count": 138,
        "fingerprints": STAGES,
        "models_per_fingerprint": MODEL_IDS,
        "model_fingerprint_cells": 108,
        "forecast_games_per_cell": 271,
        "forecast_games": "Week 0 through Week 5, including four reconstructed Week 1 games",
        "ballot_method": "For each model/fingerprint, least-squares team fixed effects fit to that model's 2026 pregame predicted margins through Week 5. Home-site effect is estimated separately; team ratings are centered to mean zero and ranked as margins versus the average FBS team.",
        "outcomes_used_as_fit_targets": False,
        "market_caveats": {
            "F7_F8": "Four Week 1 forecast rows per cell use line inputs from an October 6 refreshed snapshot; quote times are unavailable. Remaining games use archived weekly lines.",
            "F17-market": "Target-game line quote timestamps are unavailable for all 271 games; exploratory only.",
            "F7": "Market-only generation has no direct team-vs-average matchup representation; its ballot is derived from its predicted game margins as a separate research view.",
        },
        "published_week6_predictions_modified": False,
        "files": ["scientific_full_ballots.csv", "scientific_consensus_power_rankings.csv", "scientific_top25_ballot.csv", "scientific_top25_ballots.png", "scientific_top25.png", "scientific_all_fbs_power_rankings.png"],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "README.md").write_text(
        "# 2026 Week 6 scientific full-roster what-if ballot\n\n"
        "This is an independent research ballot, separate from published and frozen Week 6 predictions. It covers all 18 fingerprint generations (F0–F17-market), all six scientific architectures (M1, M2, M3, M4, M5, M10), and 138 FBS teams. `scientific_top25_ballot.csv` is the consensus Top 25; `scientific_full_ballots.csv` contains all 108 model ballots; `scientific_all_fbs_power_rankings.png` shows the full 138-team power ranking, ordered by the same mean-rating rank as the Top 25.\n\n"
        "Each model's 2026 pregame margin forecasts for the 271 games through Week 5 are fit to team fixed effects plus a home-site effect. The centered team effects are the model's estimated margin versus an average FBS team. Actual game outcomes are not fit targets. This creates comparable full-team ballots even where an architecture has no direct team-vs-average feature transform.\n\n"
        "Market timing caveats: F7/F8 each include four Week 1 forecasts using line inputs from an October 6 refreshed snapshot with unavailable quote times. F17-market uses market inputs with unavailable quote times for all target games. F7 is market-only, so its ballot is derived from game forecasts rather than a direct team-vs-average evaluation. Treat the equal-weight consensus as a research view, not a clean poll replacement.\n"
    )
    print(json.dumps({"output": str(OUTPUT), "ballot_rows": len(ballots),
                      "teams": int(ballots.keys_team.nunique()), "models": int(ballots.ballot_model.nunique()),
                      "top25": rankings.head(25)[["consensus_power_rank", "keys_team", "predicted_margin_vs_average_team"]].to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
