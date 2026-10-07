#!/usr/bin/env python3
"""Build an independent Week 6 ballot from the frozen F0–F19 margin forecasts."""
from __future__ import annotations

import hashlib
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

SOURCE = ROOT / "publication/2026/week_05/post_game/scientific/current_season_f0_f19"
OUTPUT = ROOT / "publication/2026/week_06/pre_game/scientific/full_f0_f19_whatif"
MODEL_IDS = ("M1", "M2", "M3", "M4", "M5", "M10")
STAGES = (*[f"F{i}" for i in range(17)], "F17-market", "F18", "F19")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError("Existing ballot manifest is immutable; choose a new output path")
    source = SOURCE / "scientific_2026_model_game_results.parquet"
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    if digest(source) != source_manifest["outputs"]["model_games_parquet"]["sha256"]:
        raise ValueError("2026 full-roster source differs from its manifest")
    model = pd.read_parquet(source)
    if len(model) != 120 * 271 or model.duplicated(["game_id", "fingerprint", "model_id"]).any():
        raise ValueError("Expected 120 complete model cells on the canonical game cohort")
    if set(model.fingerprint) != set(STAGES) or set(model.model_id) != set(MODEL_IDS):
        raise ValueError("Scientific roster changed")
    common_games = set(model.loc[(model.fingerprint == "F19") &
                                 (model.forecast_status == "available"), "game_id"])
    if len(common_games) != 263:
        raise ValueError("Expected 263 common archived-market games")
    forecasts = model.loc[model.game_id.isin(common_games)].copy()
    if len(forecasts) != 120 * 263 or not forecasts.forecast_status.eq("available").all():
        raise ValueError("Every ballot needs all 263 common forecasts")
    schedule = pd.read_parquet(ROOT / "data/raw/cfbd/v2/games/2026.parquet",
                               columns=["id", "neutral_site", "season_type", "home_classification", "away_classification"])
    schedule = schedule.loc[schedule.id.isin(common_games) &
                            schedule.season_type.astype(str).str.lower().eq("regular") &
                            schedule.home_classification.astype(str).str.lower().eq("fbs") &
                            schedule.away_classification.astype(str).str.lower().eq("fbs")].drop_duplicates("id")
    if set(schedule.id) != common_games or schedule.id.duplicated().any():
        raise ValueError("Forecasts do not map one-to-one to the FBS schedule")
    forecasts["neutral_site"] = forecasts.game_id.map(schedule.set_index("id").neutral_site).fillna(False).astype(bool)
    teams = sorted(set(forecasts.home_team) | set(forecasts.away_team))
    if len(teams) != 138:
        raise ValueError("Expected 138 FBS teams")
    team_col = {team: i for i, team in enumerate(teams)}
    rows: list[dict] = []
    for (fingerprint, model_id), games in forecasts.groupby(["fingerprint", "model_id"], sort=False):
        matrix = np.zeros((263, len(teams) + 1), dtype=float)
        idx = np.arange(263)
        matrix[idx, games.home_team.map(team_col).to_numpy()] = 1
        matrix[idx, games.away_team.map(team_col).to_numpy()] = -1
        matrix[:, -1] = (~games.neutral_site.to_numpy()).astype(float)
        y = games.pred_margin.to_numpy(dtype=float)
        solution, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        ratings = solution[:-1] - solution[:-1].mean()
        ordered = sorted(zip(teams, ratings), key=lambda item: (-item[1], item[0]))
        for rank, (team, rating) in enumerate(ordered, 1):
            rows.append({
                "poll_objective": "predicted_margin_network_fit",
                "keys_team": team,
                "ballot_model": f"scientific_{fingerprint}_{model_id}",
                "power_rating_vs_average": float(rating),
                "ballot_rank": rank,
                "poll_points": max(26 - rank, 0),
                "top25_vote": rank <= 25,
                "first_place_vote": rank == 1,
                "model_id": model_id,
                "model_family": games.model_family.iloc[0],
                "fingerprint": fingerprint,
                "games_fit": 263,
                "estimated_home_site_effect": float(solution[-1]),
                "fit_rmse": float(np.sqrt(np.mean((matrix @ solution - y) ** 2))),
                "coverage_status": "common_263_archived_pregame_games",
            })
    ballots = pd.DataFrame(rows)
    validate_scientific_ballots(ballots)
    if len(ballots) != 120 * 138:
        raise ValueError("Expected 120 complete 138-team ballots")
    rankings = scientific_consensus_power_rankings(ballots)
    if len(rankings) != 138 or not rankings.scientific_models.eq(120).all():
        raise ValueError("Every team needs 120 consensus votes")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "ballots": OUTPUT / "scientific_full_ballots.csv",
        "rankings": OUTPUT / "scientific_consensus_power_rankings.csv",
        "top25": OUTPUT / "scientific_top25_ballot.csv",
    }
    ballots.to_csv(outputs["ballots"], index=False, float_format="%.6f")
    rankings.to_csv(outputs["rankings"], index=False, float_format="%.6f")
    rankings.head(25).to_csv(outputs["top25"], index=False, float_format="%.6f")
    logo_dir = ROOT / "data/meta/logos"
    outputs["ballot_figure"] = OUTPUT / "scientific_top25_ballots.png"
    plot_ballot_logo_grid(ballots.loc[ballots.ballot_rank.le(25)], outputs["ballot_figure"],
                          top_n=25, logo_dir=logo_dir,
                          title="2026 Week 6 · F0–F19 scientific what-if Top 25")
    outputs["top25_figure"] = OUTPUT / "scientific_top25.png"
    plot_scientific_power_top25(rankings, outputs["top25_figure"], season=2026,
                                week=6, logo_dir=logo_dir,
                                display_label="Full F0–F19 research ballot")
    all_team_power = rankings.copy()
    all_team_power["poll_points_rank"] = all_team_power.consensus_power_rank
    outputs["all_team_figure"] = OUTPUT / "scientific_all_fbs_power_rankings.png"
    plot_scientific_all_team_power_ranking(
        all_team_power, outputs["all_team_figure"], season=2026, week=6,
        bulletin_label="INDEPENDENT FULL-ROSTER RESEARCH BALLOT",
        title="ALL FBS TEAMS · F0–F19 CONSENSUS",
        footer_label="MEAN POWER RATING ACROSS 120 BALLOTS · 263 COMMON GAMES",
    )
    readme = OUTPUT / "README.md"
    readme.write_text(
        "# 2026 Week 6 F0–F19 scientific research ballot\n\n"
        "This independent ballot combines 20 fingerprints (F0–F16, F17-market, F18, F19) "
        "with the six scientific models M1, M2, M3, M4, M5, and M10: 120 complete ballots "
        "for 138 FBS teams. The 263 games with all 120 archived pregame forecasts are the "
        "common fitting cohort. Eight opening games without F19 market snapshots are "
        "excluded from every ballot so the schedule is identical.\n\n"
        "For each cell, least squares fits team effects plus a home-site effect to that "
        "cell's predicted game margins; ratings are centered against the average FBS team. "
        "Realized scores are not fitting targets. The consensus mean is descriptive and "
        "was not optimized against 2026 outcomes. Rankings use mean power rating; points "
        "rank is separately available in the table. This output is separate from the frozen "
        "published Week 6 bundle and prior 108-ballot research export. F18/F19 forecasts "
        "were frozen before their evaluation, but broader TDNet work had previously analyzed "
        "2026 outcomes. Historical F19 quote-level timing is unverified.\n"
    )
    manifest = {
        "scope": "Independent 2026 Week 6 full F0-F19 scientific research ballot",
        "season": 2026, "reader_week": 6, "team_count": 138,
        "fingerprints": STAGES, "models_per_fingerprint": MODEL_IDS,
        "model_fingerprint_cells": 120, "forecast_games_per_cell": 263,
        "outcomes_used_as_fit_targets": False,
        "source": {"path": str(source.relative_to(ROOT)), "sha256": digest(source)},
        "script_sha256": digest(Path(__file__)),
        "files": {name: {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
                  for name, path in outputs.items()},
        "readme_sha256": digest(readme),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "ballot_rows": len(ballots),
                      "teams": len(rankings), "models": ballots.ballot_model.nunique(),
                      "top25": rankings.head(25).keys_team.tolist()}, indent=2))


if __name__ == "__main__":
    main()
