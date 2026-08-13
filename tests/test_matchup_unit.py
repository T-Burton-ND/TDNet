import pandas as pd
import pytest

from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.td_run.matchups.unit_matchups import (
    PRIMARY_UNIT_MATCHUP_COUNTERPARTS,
    default_unit_pairing_specs,
    feature_direction,
)
from gridiron_ml.models import TDStat
from gridiron_ml.fingerprints.features import split_frame


def test_unit_matchup_pairs_offense_against_opponent_defense_on_strength_scale():
    home = pd.DataFrame(
        {
            "offense_ppa": [0.30],
            "defense_ppa": [0.05],
            "statOff_yards_per_pass": [8.5],
            "defense_passing_plays_ppa": [0.20],
            "statSpe_kicking_points": [7.0],
            "travel_distance_diff": [100.0],
        }
    )
    away = pd.DataFrame(
        {
            "offense_ppa": [0.10],
            "defense_ppa": [0.20],
            "statOff_yards_per_pass": [6.5],
            "defense_passing_plays_ppa": [0.40],
            "statSpe_kicking_points": [4.0],
            "travel_distance_diff": [250.0],
        }
    )

    out = MatchupBuilder(representation="unit_matchup").build_many(home, away)

    assert out.loc[0, "home_offense_ppa_vs_away_defense_ppa"] == pytest.approx(0.50)
    assert out.loc[0, "away_offense_ppa_vs_home_defense_ppa"] == pytest.approx(0.15)
    assert out.loc[0, "net_offense_ppa_vs_defense_ppa"] == pytest.approx(0.35)
    assert out.loc[
        0, "home_statOff_yards_per_pass_vs_away_defense_passing_plays_ppa"
    ] == pytest.approx(8.90)
    assert out.loc[
        0, "home_statSpe_kicking_points_vs_away_statSpe_kicking_points"
    ] == pytest.approx(3.0)
    assert out.loc[
        0, "home_travel_distance_diff_vs_away_travel_distance_diff"
    ] == pytest.approx(150.0)


def test_unit_matchup_uses_csv_pairing_contract_when_supplied(tmp_path):
    pairings_path = tmp_path / "pairings.csv"
    pd.DataFrame(
        [
            {
                "source_feature": "statOff_yards_per_pass",
                "source_direction": "higher_better",
                "primary_opponent_counterpart": "statDef_sacks",
                "primary_counterpart_direction": "higher_better",
            }
        ]
    ).to_csv(pairings_path, index=False)
    home = pd.DataFrame({"statOff_yards_per_pass": [9.0], "statDef_sacks": [2.0]})
    away = pd.DataFrame({"statOff_yards_per_pass": [7.0], "statDef_sacks": [5.0]})

    out = MatchupBuilder(
        representation="unit_matchup", unit_pairings_path=pairings_path
    ).build_many(home, away)

    assert list(out.columns) == [
        "home_statOff_yards_per_pass_vs_away_statDef_sacks",
        "away_statOff_yards_per_pass_vs_home_statDef_sacks",
        "net_statOff_yards_per_pass_vs_statDef_sacks",
    ]
    assert out.loc[
        0, "home_statOff_yards_per_pass_vs_away_statDef_sacks"
    ] == pytest.approx(4.0)


def test_unit_matchup_contract_lives_in_importable_pipeline_code():
    specs = default_unit_pairing_specs(
        ["statOff_yards_per_pass", "defense_passing_plays_ppa"]
    )

    assert (
        PRIMARY_UNIT_MATCHUP_COUNTERPARTS["statOff_yards_per_pass"]
        == "defense_passing_plays_ppa"
    )
    assert feature_direction("defense_passing_plays_ppa") == "lower_better"
    assert specs[0]["primary_opponent_counterpart"] == "defense_passing_plays_ppa"
    assert specs[0]["primary_counterpart_direction"] == "lower_better"


def test_unit_matchup_does_not_positionally_pair_unmatched_stat_columns():
    home = pd.DataFrame({"statOff_yards_per_pass": [9.0], "statDef_sacks": [2.0]})
    away = pd.DataFrame({"statOff_yards_per_pass": [7.0], "statDef_sacks": [5.0]})

    with pytest.raises(
        ValueError, match="could not resolve any usable feature pairings"
    ):
        MatchupBuilder(representation="unit_matchup").build_many(home, away)


def test_unit_matchup_passes_adjusted_fingerprint_columns_without_pairing_contract():
    home = pd.DataFrame(
        {
            "opp_adj_success_rate": [0.18],
            "opponent_adjusted_explosiveness": [1.25],
        }
    )
    away = pd.DataFrame(
        {
            "opp_adj_success_rate": [0.11],
            "opponent_adjusted_explosiveness": [0.95],
        }
    )

    out = MatchupBuilder(representation="unit_matchup").build_many(home, away)

    assert list(out.columns) == [
        "home_opp_adj_success_rate",
        "home_opponent_adjusted_explosiveness",
        "away_opp_adj_success_rate",
        "away_opponent_adjusted_explosiveness",
        "net_opp_adj_success_rate",
        "net_opponent_adjusted_explosiveness",
    ]
    assert out.loc[0, "home_opp_adj_success_rate"] == pytest.approx(0.18)
    assert out.loc[0, "away_opp_adj_success_rate"] == pytest.approx(0.11)
    assert out.loc[0, "net_opp_adj_success_rate"] == pytest.approx(0.07)


def test_unit_matchup_passes_schedule_graph_columns_without_pairing_contract():
    home = pd.DataFrame(
        {
            "graph_colley_rating": [0.62],
            "graph_schedule_strength": [0.55],
        }
    )
    away = pd.DataFrame(
        {
            "graph_colley_rating": [0.48],
            "graph_schedule_strength": [0.51],
        }
    )

    out = MatchupBuilder(representation="unit_matchup").build_many(home, away)

    assert list(out.columns) == [
        "home_graph_colley_rating",
        "home_graph_schedule_strength",
        "away_graph_colley_rating",
        "away_graph_schedule_strength",
        "net_graph_colley_rating",
        "net_graph_schedule_strength",
    ]
    assert out.loc[0, "home_graph_colley_rating"] == pytest.approx(0.62)
    assert out.loc[0, "away_graph_colley_rating"] == pytest.approx(0.48)
    assert out.loc[0, "net_graph_colley_rating"] == pytest.approx(0.14)


def test_fingerprint_split_keeps_explicit_registry_features():
    frame = pd.DataFrame(
        {
            "keys_season": [2025],
            "keys_team": ["A"],
            "offense_ppa": [0.2],
            "graph_colley_rating": [0.62],
            "games_played": [4.0],
            "target_points_for_avg": [28.0],
            "target_points_against_avg": [17.0],
            "y_next_margin": [3.0],
        }
    )

    features, _, _, _ = split_frame(frame)

    assert "offense_ppa" in features
    assert "graph_colley_rating" in features
    assert "games_played" in features
    assert "target_points_for_avg" in features
    assert "target_points_against_avg" in features


def test_unit_matchup_derives_reviewed_rate_inputs_from_raw_fingerprint_columns():
    home = pd.DataFrame(
        {
            "offense_plays": [80.0],
            "defense_plays": [70.0],
            "statOff_pass_attempts": [30.0],
            "statOff_rushing_attempts": [30.0],
            "statDef_sacks": [7.0],
            "statDef_passes_intercepted": [3.0],
            "statDef_tackles": [63.0],
        }
    )
    away = pd.DataFrame(
        {
            "offense_plays": [100.0],
            "defense_plays": [80.0],
            "statOff_pass_attempts": [40.0],
            "statOff_rushing_attempts": [40.0],
            "statDef_sacks": [8.0],
            "statDef_passes_intercepted": [4.0],
            "statDef_tackles": [72.0],
        }
    )

    out = MatchupBuilder(representation="unit_matchup").build_many(home, away)

    assert "home_statOff_pass_rate_vs_away_statDef_sack_rate" in out.columns
    assert "home_statDef_passes_intercepted_rate_vs_away_statOff_pass_rate" in out.columns
    assert "home_statDef_tackle_rate_vs_away_statOff_rush_rate" in out.columns
    assert "home_statDef_sack_rate_vs_away_statOff_pass_rate" in out.columns
    assert out.loc[
        0, "home_statOff_pass_rate_vs_away_statDef_sack_rate"
    ] == pytest.approx(-0.60)
    assert out.loc[
        0, "home_statDef_sack_rate_vs_away_statOff_pass_rate"
    ] == pytest.approx(0.60)


def test_unit_matchup_works_for_team_vs_average():
    features = pd.DataFrame(
        {
            "offense_ppa": [0.2, 0.4],
            "defense_ppa": [0.1, 0.3],
        }
    )
    meta = pd.DataFrame(
        {"keys_team": ["A", "B"], "keys_season": [2026, 2026], "keys_week": [0, 0]}
    )
    average = pd.DataFrame({"offense_ppa": [0.25], "defense_ppa": [0.25]})

    out, out_meta, out_market = MatchupBuilder(
        representation="unit_matchup"
    ).team_vs_average(
        features,
        meta,
        average_team_df=average,
    )

    assert len(out) == 2
    assert len(out_meta) == 2
    assert out_market.empty
    assert "home_offense_ppa_vs_away_defense_ppa" in out.columns


def test_tdstat_treats_unit_matchup_edges_as_engineered_home_margin_features():
    model = TDStat({})

    assert model._feature_direction("home_offense_ppa_vs_away_defense_ppa") == 1.0
    assert model._feature_direction("net_offense_ppa_vs_defense_ppa") == 1.0
    assert model._feature_direction("away_offense_ppa_vs_home_defense_ppa") == -1.0
    assert (
        model._feature_family_key(
            "home_statOff_yards_per_pass_vs_away_defense_passing_plays_ppa"
        )
        == "offense"
    )
    assert (
        model._feature_family_key("net_statDef_sacks_vs_statOff_pass_attempts")
        == "defense"
    )
