#!/usr/bin/env python3
"""Fit a descriptive 126-cell team margin network on the shared 2026 games."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.publication.scientific_weekly import (
    plot_scientific_all_team_power_ranking,
    plot_scientific_power_top25,
    scientific_consensus_power_rankings,
    validate_scientific_ballots,
)
from gridiron_ml.publication.social_top10 import render_top10_social
from gridiron_ml.td_run.poll_viz import plot_ballot_logo_grid


ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "data/private/postgame_2026_week08"
WEEK = ROOT / "publication/2026/week_08/post_game/scientific/expanded_f0_f20"
OUTPUT = WEEK / "power_rankings_research"
MODELS = ("M1", "M2", "M3", "M4", "M5", "M10")
FINGERPRINTS = [*(f"F{i}" for i in range(17)), "F17-market", "F18", "F19", "F20"]


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite research power package: {OUTPUT}")
    prior = pd.read_parquet(PRIVATE / "prior_f0_f19_model_games.parquet")
    market = pd.read_csv(PRIVATE / "prior_market_games.csv")
    f20 = pd.read_parquet(ROOT / "data/private/f20/replay_2026_week01_07_v3_margin/f20_replay.parquet")
    week = pd.read_csv(WEEK / "week08_f0_f20_126_model_games_matched.csv")
    old_ids = set(f20.game_id) & set(market.loc[market.market_home_probability.notna(), "game_id"]) & set(prior.game_id)
    if len(old_ids) != 304 or week.game_id.nunique() != 49 or old_ids & set(week.game_id):
        raise ValueError("Expected 304 prior and 49 distinct Week 8 matched games")
    old = prior[prior.game_id.isin(old_ids)][["game_id", "fingerprint", "model_id", "pred_home_margin"]].copy()
    f20_rows = []
    for model_id in MODELS:
        chunk = f20[f20.game_id.isin(old_ids)][["game_id", f"margin_{model_id}"]].rename(
            columns={f"margin_{model_id}": "pred_home_margin"})
        chunk["fingerprint"] = "F20"
        chunk["model_id"] = model_id
        f20_rows.append(chunk)
    all_predictions = pd.concat([
        old, *f20_rows,
        week[["game_id", "fingerprint", "model_id", "pred_home_margin"]],
    ], ignore_index=True)
    ids = sorted(old_ids | set(week.game_id))
    cells = [(fp, model) for fp in FINGERPRINTS for model in MODELS]
    if len(ids) != 353 or len(all_predictions) != 126 * 353 or all_predictions.duplicated(["game_id", "fingerprint", "model_id"]).any():
        raise ValueError("Shared 126×353 prediction matrix is incomplete")
    schedule = pd.read_parquet(ROOT / "data/raw/cfbd/v2/games/2026.parquet")
    schedule = schedule[schedule.id.isin(ids)].drop_duplicates("id").set_index("id").loc[ids]
    if len(schedule) != 353 or not (schedule.home_classification.eq("fbs") & schedule.away_classification.eq("fbs")).all():
        raise ValueError("Power network includes a non-FBS or missing game")
    teams = sorted(set(schedule.home_team) | set(schedule.away_team))
    if len(teams) != 136:
        raise ValueError(f"Expected 136 covered FBS teams; got {len(teams)}")
    all_fbs = set(pd.read_parquet(ROOT / "data/raw/cfbd/v2/teams_fbs/2026.parquet").school)
    uncovered = sorted(all_fbs - set(teams))
    if uncovered != ["App State", "Oregon State"]:
        raise ValueError(f"Unexpected teams absent from common cohort: {uncovered}")
    team_col = {team: i for i, team in enumerate(teams)}
    design = np.zeros((len(ids), len(teams) + 1))
    idx = np.arange(len(ids))
    design[idx, schedule.home_team.map(team_col).to_numpy()] = 1
    design[idx, schedule.away_team.map(team_col).to_numpy()] = -1
    design[:, -1] = (~schedule.neutral_site.astype(bool)).astype(float)
    ordered = all_predictions.copy()
    ordered["cell"] = list(zip(ordered.fingerprint, ordered.model_id))
    matrix = ordered.pivot(index="game_id", columns="cell", values="pred_home_margin").reindex(index=ids, columns=cells)
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Power network lacks a model margin")
    solution, *_ = np.linalg.lstsq(design, matrix.to_numpy(), rcond=None)
    ratings = solution[:-1] - solution[:-1].mean(axis=0)
    ballots = []
    for col, (fingerprint, model_id) in enumerate(cells):
        order = sorted(zip(teams, ratings[:, col]), key=lambda pair: (-pair[1], pair[0]))
        fit_rmse = float(np.sqrt(np.mean((design @ solution[:, col] - matrix.iloc[:, col].to_numpy()) ** 2)))
        for rank, (team, rating) in enumerate(order, 1):
            ballots.append({"poll_objective": "shared_prediction_margin_network_fit",
                            "keys_team": team, "ballot_model": f"scientific_{fingerprint}_{model_id}",
                            "fingerprint": fingerprint, "model_id": model_id,
                            "power_rating_vs_average": float(rating),
                            "ballot_rank": rank, "poll_points": max(26 - rank, 0),
                            "top25_vote": rank <= 25, "first_place_vote": rank == 1,
                            "games_fit": len(ids), "fit_rmse": fit_rmse,
                            "estimated_home_site_effect": float(solution[-1, col])})
    ballots = pd.DataFrame(ballots)
    validate_scientific_ballots(ballots)
    power = scientific_consensus_power_rankings(ballots)
    if len(power) != 136 or not power.scientific_models.eq(126).all():
        raise ValueError("Power consensus lacks model votes")
    ap = pd.read_csv(PRIVATE / "ap_top25_week09.csv")
    power["ap_rank_week09"] = power.keys_team.map(ap.set_index("team")["rank"])
    OUTPUT.mkdir(parents=True)
    ballots_path = OUTPUT / "scientific_126_model_ballots.csv"
    power_path = OUTPUT / "scientific_136_team_power_rankings.csv"
    top25_path = OUTPUT / "scientific_top25.csv"
    ballots.to_csv(ballots_path, index=False)
    power.to_csv(power_path, index=False)
    power.head(25).to_csv(top25_path, index=False)
    logos = ROOT / "data/meta/logos/by_team"
    plot_ballot_logo_grid(ballots[ballots.ballot_rank.le(25)], OUTPUT / "scientific_top25_ballots.png",
                          top_n=25, logo_dir=logos,
                          title="2026 Week 8: F0–F20 Shared-Game Research Ballots")
    plot_scientific_power_top25(power, OUTPUT / "scientific_top25.png", season=2026, week=8,
                                logo_dir=logos, display_label="F0–F20 shared-game research")
    plot_scientific_all_team_power_ranking(power, OUTPUT / "scientific_all_team_power_rankings.png",
                                           season=2026, week=8,
                                           bulletin_label="TDNET RESEARCH BULLETIN · 136 COVERED FBS TEAMS",
                                           title="TDNET F0–F20 PREDICTION-NETWORK POWER RANKING")
    social = power.sort_values("poll_points_rank").head(10).copy()
    social["rank"] = social.poll_points_rank
    social["reference_rank"] = social.ap_rank_week09
    social["ballots_seen"] = social.scientific_models
    for variant in ("4x5", "1x1", "16x9"):
        render_top10_social(social, OUTPUT / f"scientific_top10_research_{variant}.png",
                            season=2026, week=8, logo_dir=logos, variant=variant,
                            source_sha256=sha256_file(power_path), reference_label="Week 9 AP",
                            header_title="TDNet Research Ballot",
                            header_subtitle="F0–F20 · shared 353-game prediction network")
    manifest = {"schema": "tdnet-f0-f20-week08-prediction-network-power-v1",
                "team_coverage": 136, "uncovered_fbs_teams": uncovered,
                "models": 126, "shared_games": 353,
                "prior_games": 304, "week08_games": 49,
                "interpretation": "descriptive ratings fitted to saved predicted margins, not live Week 9 model team states",
                "f20_prior_status": "retrospective model on assumed pregame weekly archive",
                "f20_week08_status": "pregame expert features; margin model applied after games",
                "sources": {str(p): sha256_file(p) for p in [PRIVATE / "prior_f0_f19_model_games.parquet",
                           PRIVATE / "prior_market_games.csv",
                           ROOT / "data/private/f20/replay_2026_week01_07_v3_margin/f20_replay.parquet",
                           WEEK / "week08_f0_f20_126_model_games_matched.csv",
                           PRIVATE / "ap_top25_week09.csv"]},
                "files": {p.name: sha256_file(p) for p in OUTPUT.iterdir() if p.is_file()}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUTPUT / "README.md").write_text(
        "# F0–F20 shared-game prediction-network rankings\n\n"
        "These are descriptive research ratings fitted to 126 sets of saved predicted margins "
        "on 353 matched completed games from Weeks 1–6 and 8. They are not the operational "
        "Week 9 TDNet poll or new Week 9 game forecasts. F20's Weeks 1–6 margins were "
        "replayed retrospectively from an assumed weekly expert archive; Week 8 margins "
        "use a model frozen before kickoff on archived pregame expert features. "
        "The shared game network covers 136 FBS teams; App State and Oregon State have "
        "no matched game and are unranked. Top 25 and Top 10 are for this 136-team cohort.\n",
        encoding="utf-8",
    )
    print(json.dumps({"games": 353, "models": 126, "teams": 136,
                      "top5": power[["keys_team", "predicted_margin_vs_average_team"]].head(5).to_dict("records")}))


if __name__ == "__main__":
    main()
