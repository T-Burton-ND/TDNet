#!/usr/bin/env python3
"""Generate the publication feature-by-fingerprint inclusion matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gridiron_ml.experiments.publication import expand_feature_registry  # noqa: E402


TIERS = tuple(f"F{i}" for i in range(9))
MANIFEST_ROOT = ROOT / "docs/publication_2026/feature_manifests"
REGISTRY_PATH = ROOT / "configs/features/feature_registry.yaml"
DEFAULT_OUTPUT = ROOT / "docs/publication_2026/FINGERPRINT_FEATURE_MATRIX.csv"


def aggregation_text(base: str, window: str) -> str:
    if window == "mean_to_date":
        return f"cumulative mean through week w: n^(-1) sum_g a_g({base})"
    if window == "last3":
        return f"mean of the most recent min(3,n) adjusted game contributions a_g({base})"
    if window == "ewm":
        return f"EWM of adjusted contributions: e_w = 0.45 a_w + 0.55 e_(w-1) for {base}"
    raise ValueError(window)


def opponent_adjusted_calculation(feature: str) -> str:
    if feature.endswith("_games_played"):
        return "cumulative count of completed team-games in the team-season"
    if feature.endswith("_unique_opponents"):
        return "cumulative number of distinct opponents faced in the team-season"

    match = re.fullmatch(r"opp_adj_v1_4_(.+)_(mean_to_date|last3|ewm)", feature)
    if match is None:
        return "selected v1.4 Elo-context feature computed from games before the target cutoff"
    base, window = match.groups()
    if base == "elo_team_rating":
        contribution = "pregame sequential Elo rating R_t"
    elif base == "elo_opponent_rating":
        contribution = "pregame sequential opponent Elo rating R_o"
    elif base == "elo_rating_edge":
        contribution = "pregame Elo edge R_t - R_o"
    elif base == "target_team_margin":
        contribution = "a_g = margin - [7(R_t-R_o)/sd(R) + 2.5 home]"
    else:
        contribution = (
            f"a_g({base}) = x_g - historical_mean(x) "
            "+ 0.10 [R_o/sd(R)] historical_sd(x)"
        )
    return contribution + "; " + aggregation_text(base, window)


def calculation_for(feature: str, family: str) -> tuple[str, str]:
    builder = "src/gridiron_ml/fingerprints/builders/v0.py"
    cleaners = "src/gridiron_ml/pipeline/pre_processing/cleaners.py"

    if feature == "games_played":
        return (
            "cumulative count of completed team-games in the team-season",
            builder,
        )
    if feature == "roster_talent":
        return (
            "CFBD season talent composite; coerced numeric and forward-filled within team-season",
            builder,
        )
    if feature == "target_points_for_avg":
        return (
            "cumulative mean through week w: n^(-1) sum_g points_for_g",
            builder,
        )
    if feature == "target_points_against_avg":
        return (
            "cumulative mean through week w: n^(-1) sum_g points_against_g",
            builder,
        )
    if feature == "statOff_completion_rate":
        return (
            "per game: completions / pass_attempts; then cumulative team-season mean",
            cleaners + "; " + builder,
        )
    if feature == "statOff_third_down_rate":
        return (
            "per game: third_down_successes / third_down_attempts; then cumulative team-season mean",
            cleaners + "; " + builder,
        )
    if feature == "statOff_fourth_down_rate":
        return (
            "per game: fourth_down_successes / fourth_down_attempts; then cumulative team-season mean",
            cleaners + "; " + builder,
        )
    if feature == "statGen_possession_time":
        return (
            "parse MM:SS as minutes + seconds/60; then cumulative team-season mean",
            cleaners + "; " + builder,
        )
    if family.startswith("box_score"):
        return (
            f"coerce raw per-game {feature} numeric; cumulative team-season mean n^(-1) sum_g x_g",
            builder,
        )
    if family == "efficiency":
        return (
            f"CFBD per-game advanced statistic {feature}; cumulative team-season mean n^(-1) sum_g x_g",
            builder,
        )
    if family == "opponent_adjusted":
        return (
            opponent_adjusted_calculation(feature),
            "src/gridiron_ml/experiments/opponent_adjusted.py",
        )
    if family == "returning_production":
        return (
            "CFBD season returning-production value; coerced numeric and forward-filled within team-season",
            builder,
        )
    if family == "coaching":
        if feature == "coach_career_seasons":
            raw = "number of distinct prior coach-seasons"
        elif feature == "coach_career_mean_sp_offense":
            raw = "mean prior-season coach SP+ offense"
        elif feature == "coach_career_mean_sp_defense":
            raw = "mean prior-season coach SP+ defense"
        else:
            raw = "pre-cutoff coach-career aggregate"
        return (
            raw + "; forward-filled within team-season",
            "src/gridiron_ml/pipeline/pre_processing/parquet_loader.py; " + builder,
        )
    if feature == "travel_distance_diff":
        return (
            "per game: log(1 + haversine_km(team,venue)) - log(1 + haversine_km(opponent,venue)); then cumulative team-season mean",
            cleaners + "; " + builder,
        )
    if feature == "travel_tz_diff":
        return (
            "per game: clip(|team_tz-venue_tz| - |opponent_tz-venue_tz|, -6, 6); then cumulative team-season mean",
            cleaners + "; " + builder,
        )
    if family == "temporal":
        base = feature.split("__", 1)[-1]
        if feature.startswith("time_adj_last1__"):
            calculation = f"latest game contribution reconstructed from season mean: x_g=(n_w*mean_w-n_(w-1)*mean_(w-1))/(n_w-n_(w-1)) for {base}"
        elif feature.startswith("time_adj_last3__"):
            calculation = f"mean of the latest min(3,n) reconstructed game contributions for {base}"
        elif feature.startswith("time_adj_ewm_hl3__"):
            calculation = f"EWM of reconstructed {base} game contributions: e_w=(1-2^(-1/3))*x_w+2^(-1/3)*e_(w-1)"
        elif feature.startswith("time_adj_recent_minus_season__"):
            calculation = f"last-3 reconstructed mean of {base} minus the season-to-date mean"
        else:
            calculation = f"population standard deviation of the latest min(5,n) reconstructed {base} game contributions"
        return (
            calculation,
            "src/gridiron_ml/fingerprints/ladder.py",
        )
    if family == "schedule_graph":
        graph_calculations = {
            "graph_colley_rating": "Colley rating r from C*r=b, with C_ii=2+games_i, C_ij=-meetings_ij, b_i=1+(wins_i-losses_i)/2",
            "graph_pagerank_z": "z-score of PageRank on the directed weighted loser-to-winner graph; damping=0.85 and margin weight=1+min(|margin|,35)/14",
            "graph_schedule_strength": "mean current Colley rating of opponents already faced",
            "graph_win_quality": "mean current Colley rating of opponents already beaten",
            "graph_loss_quality": "mean current Colley rating of opponents already lost to",
            "graph_games_played": "count of graph edges incident on the team through week w",
            "graph_unique_opponents": "count of distinct graph neighbors through week w",
            "graph_colley_edge_next": "team Colley rating minus next opponent Colley rating, both fit using games through week w",
        }
        return graph_calculations[feature], "src/gridiron_ml/fingerprints/ladder.py"
    if family == "market":
        if feature == "market_win_probability":
            calculation = "vig-adjusted home win probability derived from the pre-deadline market snapshot"
        else:
            calculation = "raw market value captured at the declared pregame prediction cutoff"
        return calculation, "src/gridiron_ml/td_run/market.py"
    return (
        "declared source value; see feature registry and recorded implementation",
        "configs/features/feature_registry.yaml",
    )


def main() -> int:
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT
    manifests = {
        tier: json.loads((MANIFEST_ROOT / f"{tier}.json").read_text(encoding="utf-8"))
        for tier in TIERS
    }
    included = {tier: set(manifest["feature_names"]) for tier, manifest in manifests.items()}
    all_features = sorted(set().union(*included.values()))
    definitions = expand_feature_registry(
        all_features,
        registry_path=REGISTRY_PATH,
        strict=True,
    )

    rows = []
    for feature in all_features:
        definition = definitions[feature]
        metadata = definition.metadata
        first_tier = next(tier for tier in TIERS if feature in included[tier])
        calculation, implementation = calculation_for(feature, definition.family)
        row = {
            "feature": feature,
            "family": definition.family,
            "first_included_tier": first_tier,
            "source": metadata["source"],
            "units": metadata["units"],
            "description": metadata["description"],
            "calculation_from_raw_data": calculation,
            "implementation": implementation,
        }
        row.update({tier: "X" if feature in included[tier] else "" for tier in TIERS})
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature",
        "family",
        "first_included_tier",
        "source",
        "units",
        "description",
        "calculation_from_raw_data",
        "implementation",
        *TIERS,
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} features to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
