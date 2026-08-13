"""Semantic unit matchup pairing rules for TDNet fingerprints.

Usage:
    Import `default_unit_pairing_specs` when a matchup builder needs to turn
    team-level fingerprint columns into offense-vs-defense matchup features.

Logic flow:
    1. Keep the reviewed pairing contract in importable pipeline code.
    2. Use each reviewed mathematical comparison to assign feature directions.
    3. Emit only reviewed comparisons whose source and opponent columns exist.

Responsibility:
    Keep the unit matchup contract in code instead of a planning spreadsheet
    artifact. The source of truth for this table was
    `docs/technical/unit_matchup_feature_pairing_reviewed.xlsx`, sheet `Codex Ready`.
"""

from __future__ import annotations


VALID_DIRECTIONS = {
    "higher_better",
    "lower_better",
    "volume_or_context",
    "direction_needs_review",
}

REVIEWED_UNIT_MATCHUP_ROWS = (
    (
        "games_played",
        "games_played",
        "home.games_played - away.games_played",
        "availability",
        "keep_context_only",
        "sample_size",
    ),
    (
        "offense_drives",
        "defense_drives",
        "home.offense_drives - away.defense_drives",
        "home_offense_vs_away_defense",
        "keep_context_only",
        "pace_volume",
    ),
    (
        "offense_explosiveness",
        "defense_explosiveness",
        "home.offense_explosiveness + away.defense_explosiveness",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_plays",
        "defense_plays",
        "home.offense_plays - away.defense_plays",
        "home_offense_vs_away_defense",
        "keep_context_only",
        "pace_volume",
    ),
    (
        "offense_power_success",
        "defense_power_success",
        "home.offense_power_success + away.defense_power_success",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_ppa",
        "defense_ppa",
        "home.offense_ppa + away.defense_ppa",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_stuff_rate",
        "defense_stuff_rate",
        "-home.offense_stuff_rate - away.defense_stuff_rate",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_success_rate",
        "defense_success_rate",
        "home.offense_success_rate + away.defense_success_rate",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_passing_downs_explosiveness",
        "defense_passing_downs_explosiveness",
        "home.offense_passing_downs_explosiveness + away.defense_passing_downs_explosiveness",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_passing_downs_ppa",
        "defense_passing_downs_ppa",
        "home.offense_passing_downs_ppa + away.defense_passing_downs_ppa",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_passing_downs_success_rate",
        "defense_passing_downs_success_rate",
        "home.offense_passing_downs_success_rate + away.defense_passing_downs_success_rate",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_passing_plays_explosiveness",
        "defense_passing_plays_explosiveness",
        "home.offense_passing_plays_explosiveness + away.defense_passing_plays_explosiveness",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_passing_plays_ppa",
        "defense_passing_plays_ppa",
        "home.offense_passing_plays_ppa + away.defense_passing_plays_ppa",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_passing_plays_success_rate",
        "defense_passing_plays_success_rate",
        "home.offense_passing_plays_success_rate + away.defense_passing_plays_success_rate",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_rushing_plays_explosiveness",
        "defense_rushing_plays_explosiveness",
        "home.offense_rushing_plays_explosiveness + away.defense_rushing_plays_explosiveness",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_rushing_plays_ppa",
        "defense_rushing_plays_ppa",
        "home.offense_rushing_plays_ppa + away.defense_rushing_plays_ppa",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_rushing_plays_success_rate",
        "defense_rushing_plays_success_rate",
        "home.offense_rushing_plays_success_rate + away.defense_rushing_plays_success_rate",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_standard_downs_explosiveness",
        "defense_standard_downs_explosiveness",
        "home.offense_standard_downs_explosiveness + away.defense_standard_downs_explosiveness",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_db_havoc_rate",
        "defense_db_havoc_rate",
        "-home.offense_db_havoc_rate - away.defense_db_havoc_rate",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_front_seven_havoc_rate",
        "defense_front_seven_havoc_rate",
        "-home.offense_front_seven_havoc_rate - away.defense_front_seven_havoc_rate",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "offense_havoc_rate",
        "defense_havoc_rate",
        "-home.offense_havoc_rate - away.defense_havoc_rate",
        "home_offense_vs_away_defense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_drives",
        "offense_drives",
        "home.defense_drives - away.offense_drives",
        "home_defense_vs_away_offense",
        "keep_context_only",
        "pace_volume",
    ),
    (
        "defense_explosiveness",
        "offense_explosiveness",
        "-home.defense_explosiveness - away.offense_explosiveness",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_plays",
        "offense_plays",
        "home.defense_plays - away.offense_plays",
        "home_defense_vs_away_offense",
        "keep_context_only",
        "pace_volume",
    ),
    (
        "defense_power_success",
        "offense_power_success",
        "-home.defense_power_success - away.offense_power_success",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_ppa",
        "offense_ppa",
        "-home.defense_ppa - away.offense_ppa",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_stuff_rate",
        "offense_stuff_rate",
        "home.defense_stuff_rate + away.offense_stuff_rate",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_success_rate",
        "offense_success_rate",
        "-home.defense_success_rate - away.offense_success_rate",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_passing_downs_explosiveness",
        "offense_passing_downs_explosiveness",
        "-home.defense_passing_downs_explosiveness - away.offense_passing_downs_explosiveness",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_passing_downs_ppa",
        "offense_passing_downs_ppa",
        "-home.defense_passing_downs_ppa - away.offense_passing_downs_ppa",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_passing_downs_success_rate",
        "offense_passing_downs_success_rate",
        "-home.defense_passing_downs_success_rate - away.offense_passing_downs_success_rate",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_passing_plays_explosiveness",
        "offense_passing_plays_explosiveness",
        "-home.defense_passing_plays_explosiveness - away.offense_passing_plays_explosiveness",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_passing_plays_ppa",
        "offense_passing_plays_ppa",
        "-home.defense_passing_plays_ppa - away.offense_passing_plays_ppa",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_passing_plays_success_rate",
        "offense_passing_plays_success_rate",
        "-home.defense_passing_plays_success_rate - away.offense_passing_plays_success_rate",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_rushing_plays_explosiveness",
        "offense_rushing_plays_explosiveness",
        "-home.defense_rushing_plays_explosiveness - away.offense_rushing_plays_explosiveness",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_rushing_plays_ppa",
        "offense_rushing_plays_ppa",
        "-home.defense_rushing_plays_ppa - away.offense_rushing_plays_ppa",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_rushing_plays_success_rate",
        "offense_rushing_plays_success_rate",
        "-home.defense_rushing_plays_success_rate - away.offense_rushing_plays_success_rate",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_standard_downs_explosiveness",
        "offense_standard_downs_explosiveness",
        "-home.defense_standard_downs_explosiveness - away.offense_standard_downs_explosiveness",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_db_havoc_rate",
        "offense_db_havoc_rate",
        "home.defense_db_havoc_rate + away.offense_db_havoc_rate",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_front_seven_havoc_rate",
        "offense_front_seven_havoc_rate",
        "home.defense_front_seven_havoc_rate + away.offense_front_seven_havoc_rate",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "defense_havoc_rate",
        "offense_havoc_rate",
        "home.defense_havoc_rate + away.offense_havoc_rate",
        "home_defense_vs_away_offense",
        "keep",
        "advanced_quality",
    ),
    (
        "statOff_first_down_rate",
        "defense_success_rate",
        "home.statOff_first_down_rate + away.defense_success_rate",
        "home_offense_vs_away_defense",
        "derive_or_drop",
        "raw_offense_vs_defense",
    ),
    (
        "statOff_rushing_td_rate",
        "defense_power_success",
        "home.statOff_rushing_td_rate + away.defense_power_success",
        "home_rush_offense_vs_away_redzone_shortyardage_defense",
        "derive_or_drop",
        "raw_offense_vs_defense",
    ),
    (
        "statOff_passing_td_rate",
        "defense_passing_plays_ppa",
        "home.statOff_passing_td_rate + away.defense_passing_plays_ppa",
        "home_pass_offense_vs_away_pass_defense",
        "derive_or_drop",
        "raw_offense_vs_defense",
    ),
    (
        "statOff_yards_per_rush_attempt",
        "defense_rushing_plays_ppa",
        "home.statOff_yards_per_rush_attempt + away.defense_rushing_plays_ppa",
        "home_rush_offense_vs_away_rush_defense",
        "keep_low_confidence",
        "raw_efficiency_vs_advanced",
    ),
    (
        "statOff_rush_rate",
        "defense_rushing_plays_success_rate",
        "home.statOff_rush_rate + away.defense_rushing_plays_success_rate",
        "home_rush_tendency_vs_away_rush_defense",
        "derive_or_drop",
        "tendency_vs_allowed_efficiency",
    ),
    (
        "statOff_yards_per_pass",
        "defense_passing_plays_ppa",
        "home.statOff_yards_per_pass + away.defense_passing_plays_ppa",
        "home_pass_offense_vs_away_pass_defense",
        "keep_low_confidence",
        "raw_efficiency_vs_advanced",
    ),
    (
        "statOff_pass_rate",
        "statDef_sack_rate",
        "-home.statOff_pass_rate - away.statDef_sack_rate",
        "home_pass_protection_risk_vs_away_pass_rush",
        "derive_or_drop",
        "pass_exposure_vs_pressure",
    ),
    (
        "statOff_third_down_rate",
        "defense_passing_downs_success_rate",
        "home.statOff_third_down_rate + away.defense_passing_downs_success_rate",
        "home_conversion_offense_vs_away_passing_downs_defense",
        "keep_low_confidence",
        "conversion_downs",
    ),
    (
        "statOff_fourth_down_rate",
        "defense_power_success",
        "home.statOff_fourth_down_rate + away.defense_power_success",
        "home_conversion_offense_vs_away_shortyardage_defense",
        "keep_low_confidence",
        "conversion_downs",
    ),
    (
        "statOff_completion_rate",
        "statDef_passes_deflected_rate",
        "home.statOff_completion_rate - away.statDef_passes_deflected_rate",
        "home_pass_offense_vs_away_pass_disruption",
        "derive_or_drop",
        "passing_accuracy_vs_disruption",
    ),
    (
        "statDef_interception_td_rate",
        "statOff_pass_rate",
        "home.statDef_interception_td_rate + away.statOff_pass_rate",
        "home_pass_defense_vs_away_pass_exposure",
        "drop_or_rare_event_rate",
        "rare_defensive_scoring",
    ),
    (
        "statDef_interception_rate",
        "statOff_pass_rate",
        "home.statDef_interception_rate + away.statOff_pass_rate",
        "home_pass_defense_vs_away_pass_exposure",
        "derive",
        "takeaway_rate_vs_exposure",
    ),
    (
        "statDef_passes_intercepted_rate",
        "statOff_pass_rate",
        "home.statDef_passes_intercepted_rate + away.statOff_pass_rate",
        "home_pass_defense_vs_away_pass_exposure",
        "derive",
        "passes_intercepted_rate_vs_exposure",
    ),
    (
        "statDef_interception_rate",
        "statOff_pass_rate",
        "home.statDef_interception_rate + away.statOff_pass_rate",
        "home_pass_defense_vs_away_pass_exposure",
        "derive",
        "takeaway_rate_vs_exposure",
    ),
    (
        "statDef_tfl_rate",
        "statOff_yards_per_rush_attempt",
        "home.statDef_tfl_rate - away.statOff_yards_per_rush_attempt",
        "home_run_defense_vs_away_run_offense",
        "derive",
        "run_disruption_vs_run_efficiency",
    ),
    (
        "statDef_tackle_rate",
        "statOff_rush_rate",
        "home.statDef_tackle_rate + away.statOff_rush_rate",
        "home_tackle_volume_vs_away_rush_exposure",
        "derive",
        "tackle_rate_vs_rush_exposure",
    ),
    (
        "statDef_defensive_td_rate",
        "statGen_turnover_rate",
        "home.statDef_defensive_td_rate + away.statGen_turnover_rate",
        "home_defense_vs_away_ball_security",
        "drop_or_rare_event_rate",
        "rare_defensive_scoring",
    ),
    (
        "statDef_sack_rate",
        "statOff_pass_rate",
        "home.statDef_sack_rate + away.statOff_pass_rate",
        "home_pass_rush_vs_away_pass_exposure",
        "derive",
        "pass_rush_vs_pass_exposure",
    ),
    (
        "statDef_qb_hurry_rate",
        "statOff_pass_rate",
        "home.statDef_qb_hurry_rate + away.statOff_pass_rate",
        "home_pass_rush_vs_away_pass_exposure",
        "derive",
        "pressure_rate_vs_pass_exposure",
    ),
    (
        "statDef_passes_deflected_rate",
        "statOff_completion_rate",
        "home.statDef_passes_deflected_rate - away.statOff_completion_rate",
        "home_pass_defense_vs_away_passing_accuracy",
        "derive",
        "passing_disruption_vs_accuracy",
    ),
    (
        "statGen_fumble_recovery_rate",
        "statGen_fumble_lost_rate",
        "home.statGen_fumble_recovery_rate + away.statGen_fumble_lost_rate",
        "home_takeaway_vs_away_ball_security",
        "derive",
        "turnover_luck_ball_security",
    ),
    (
        "statGen_possession_time",
        "statGen_possession_time",
        "home.statGen_possession_time - away.statGen_possession_time",
        "overall",
        "keep_context_only",
        "tempo_control",
    ),
    (
        "statGen_fumble_lost_rate",
        "statGen_fumble_recovery_rate",
        "-home.statGen_fumble_lost_rate - away.statGen_fumble_recovery_rate",
        "home_ball_security_vs_away_takeaway",
        "derive",
        "ball_security_vs_takeaway",
    ),
    (
        "statGen_turnover_rate",
        "statDef_interception_rate",
        "-home.statGen_turnover_rate - away.statDef_interception_rate",
        "home_ball_security_vs_away_takeaway",
        "derive",
        "ball_security_vs_takeaway",
    ),
    (
        "statGen_penalties",
        "statGen_penalties",
        "-home.statGen_penalties + away.statGen_penalties",
        "overall",
        "keep",
        "discipline",
    ),
    (
        "statGen_fumble_rate",
        "statGen_fumble_recovery_rate",
        "-home.statGen_fumble_rate - away.statGen_fumble_recovery_rate",
        "home_ball_security_vs_away_takeaway",
        "derive",
        "ball_security_vs_takeaway",
    ),
    (
        "statSpe_punt_return_tds",
        "statSpe_punt_return_tds",
        "home.statSpe_punt_return_tds - away.statSpe_punt_return_tds",
        "special_teams",
        "keep_low_confidence",
        "special_teams_scoring",
    ),
    (
        "statSpe_punt_returns",
        "statSpe_punt_returns",
        "home.statSpe_punt_returns - away.statSpe_punt_returns",
        "special_teams",
        "keep_context_only",
        "special_teams_volume",
    ),
    (
        "statSpe_kick_return_tds",
        "statSpe_kick_return_tds",
        "home.statSpe_kick_return_tds - away.statSpe_kick_return_tds",
        "special_teams",
        "keep_low_confidence",
        "special_teams_scoring",
    ),
    (
        "statSpe_kick_returns",
        "statSpe_kick_returns",
        "home.statSpe_kick_returns - away.statSpe_kick_returns",
        "special_teams",
        "keep_context_only",
        "special_teams_volume",
    ),
    (
        "statSpe_kicking_points",
        "statSpe_kicking_points",
        "home.statSpe_kicking_points - away.statSpe_kicking_points",
        "special_teams",
        "keep",
        "special_teams_scoring",
    ),
    (
        "roster_return_total_rushing_p_p_a",
        "defense_rushing_plays_ppa",
        "home.roster_return_total_rushing_p_p_a + away.defense_rushing_plays_ppa",
        "home_rush_offense_vs_away_rush_defense",
        "keep_low_confidence",
        "returning_production_vs_defense",
    ),
    (
        "roster_return_percent_p_p_a",
        "defense_ppa",
        "home.roster_return_percent_p_p_a + away.defense_ppa",
        "home_offense_vs_away_defense",
        "keep_low_confidence",
        "returning_production_vs_defense",
    ),
    (
        "roster_return_percent_passing_p_p_a",
        "defense_passing_plays_ppa",
        "home.roster_return_percent_passing_p_p_a + away.defense_passing_plays_ppa",
        "home_pass_offense_vs_away_pass_defense",
        "keep_low_confidence",
        "returning_production_vs_defense",
    ),
    (
        "roster_return_percent_receiving_p_p_a",
        "defense_passing_plays_success_rate",
        "home.roster_return_percent_receiving_p_p_a + away.defense_passing_plays_success_rate",
        "home_receiving_vs_away_pass_defense",
        "keep_low_confidence",
        "returning_production_vs_defense",
    ),
    (
        "roster_return_percent_rushing_p_p_a",
        "defense_rushing_plays_ppa",
        "home.roster_return_percent_rushing_p_p_a + away.defense_rushing_plays_ppa",
        "home_rush_offense_vs_away_rush_defense",
        "keep_low_confidence",
        "returning_production_vs_defense",
    ),
    (
        "roster_return_rushing_usage",
        "defense_stuff_rate",
        "home.roster_return_rushing_usage - away.defense_stuff_rate",
        "home_rush_tendency_vs_away_run_disruption",
        "keep_low_confidence",
        "rushing_identity_vs_run_disruption",
    ),
    (
        "roster_talent",
        "roster_talent",
        "home.roster_talent - away.roster_talent",
        "overall",
        "keep",
        "talent",
    ),
    (
        "coach_career_seasons",
        "coach_career_seasons",
        "home.coach_career_seasons - away.coach_career_seasons",
        "overall",
        "keep_context_only",
        "coach_experience",
    ),
    (
        "coach_career_mean_sp_offense",
        "coach_career_mean_sp_defense",
        "home.coach_career_mean_sp_offense + away.coach_career_mean_sp_defense",
        "home_coach_offense_vs_away_coach_defense",
        "keep",
        "coach_unit_history",
    ),
    (
        "coach_career_mean_sp_defense",
        "coach_career_mean_sp_offense",
        "-home.coach_career_mean_sp_defense - away.coach_career_mean_sp_offense",
        "home_coach_defense_vs_away_coach_offense",
        "keep",
        "coach_unit_history",
    ),
    (
        "coach_career_mean_postseason_rank_points",
        "coach_career_mean_postseason_rank_points",
        "home.coach_career_mean_postseason_rank_points - away.coach_career_mean_postseason_rank_points",
        "overall",
        "keep",
        "coach_track_record",
    ),
    (
        "travel_tz_diff",
        "travel_tz_diff",
        "-home.travel_tz_diff + away.travel_tz_diff",
        "overall",
        "keep",
        "travel",
    ),
    (
        "travel_distance_diff",
        "travel_distance_diff",
        "-home.travel_distance_diff + away.travel_distance_diff",
        "overall",
        "keep",
        "travel",
    ),
    (
        "target_points_for_avg",
        "target_points_against_avg",
        "home.target_points_for_avg + away.target_points_against_avg",
        "home_scoring_vs_away_points_allowed",
        "keep",
        "target_scoring_baseline",
    ),
    (
        "target_points_against_avg",
        "target_points_for_avg",
        "-home.target_points_against_avg - away.target_points_for_avg",
        "home_points_allowed_vs_away_scoring",
        "keep",
        "target_scoring_baseline",
    ),
)


def _sign_for_token(expression, token):
    """Return +1 or -1 for a token in a reviewed comparison expression."""
    compact = str(expression).replace(" ", "")
    idx = compact.find(token)
    if idx < 0:
        raise ValueError(
            f"Reviewed unit matchup expression '{expression}' is missing '{token}'."
        )
    if idx == 0:
        return 1
    sign = compact[idx - 1]
    if sign == "-":
        return -1
    if sign == "+":
        return 1
    raise ValueError(
        f"Reviewed unit matchup expression '{expression}' has an unsupported sign before '{token}'."
    )


def _direction_from_strength_sign(sign, action):
    if sign < 0:
        return "lower_better"
    if action == "keep_context_only":
        return "volume_or_context"
    return "higher_better"


def _directions_from_reviewed_formula(source, counterpart, expression, action):
    home_sign = _sign_for_token(expression, f"home.{source}")
    away_sign = _sign_for_token(expression, f"away.{counterpart}")
    source_strength_sign = home_sign
    counterpart_strength_sign = -away_sign
    return (
        _direction_from_strength_sign(source_strength_sign, action),
        _direction_from_strength_sign(counterpart_strength_sign, action),
    )


def _build_reviewed_specs():
    specs = []
    for (
        source,
        counterpart,
        expression,
        role,
        action,
        family,
    ) in REVIEWED_UNIT_MATCHUP_ROWS:
        source_direction, counterpart_direction = _directions_from_reviewed_formula(
            source,
            counterpart,
            expression,
            action,
        )
        specs.append(
            {
                "source_feature": source,
                "source_direction": source_direction,
                "primary_opponent_counterpart": counterpart,
                "primary_counterpart_direction": counterpart_direction,
                "secondary_opponent_counterparts": "",
                "matchup_role": role,
                "recommended_action": action,
                "comparison_family": family,
                "suggested_mathematical_comparison": expression,
            }
        )
    return tuple(specs)


REVIEWED_UNIT_MATCHUP_SPECS = _build_reviewed_specs()

PRIMARY_UNIT_MATCHUP_COUNTERPARTS = {
    spec["source_feature"]: spec["primary_opponent_counterpart"]
    for spec in REVIEWED_UNIT_MATCHUP_SPECS
}

SECONDARY_UNIT_MATCHUP_COUNTERPARTS = {}

UNIT_MATCHUP_DIRECTION_OVERRIDES = {}
for _spec in REVIEWED_UNIT_MATCHUP_SPECS:
    UNIT_MATCHUP_DIRECTION_OVERRIDES.setdefault(
        _spec["source_feature"], _spec["source_direction"]
    )
    UNIT_MATCHUP_DIRECTION_OVERRIDES.setdefault(
        _spec["primary_opponent_counterpart"],
        _spec["primary_counterpart_direction"],
    )


def normalize_unit_direction(value, fallback="direction_needs_review"):
    """Normalize a direction value from code, YAML, or an external pairing table."""
    if value is None:
        return fallback
    direction = str(value).strip().lower()
    if direction in VALID_DIRECTIONS:
        return direction
    return fallback


def feature_direction(feature):
    """Infer whether larger raw values are better for a reviewed matchup feature."""
    feature = str(feature)
    if feature in UNIT_MATCHUP_DIRECTION_OVERRIDES:
        return UNIT_MATCHUP_DIRECTION_OVERRIDES[feature]
    return "direction_needs_review"


def default_counterpart(feature, available=None):
    """Return the reviewed opponent counterpart for one source feature, if present."""
    feature = str(feature)
    counterpart = PRIMARY_UNIT_MATCHUP_COUNTERPARTS.get(feature)
    if counterpart is None:
        return None
    available_set = set(available or [])
    if available_set and counterpart not in available_set:
        return None
    return counterpart


def default_unit_pairing_specs(columns):
    """Build reviewed unit matchup pairing records for available feature columns."""
    available = set(columns)
    rows = []
    seen = set()
    for spec in REVIEWED_UNIT_MATCHUP_SPECS:
        source = spec["source_feature"]
        counterpart = spec["primary_opponent_counterpart"]
        key = (source, counterpart)
        if key in seen or source not in available or counterpart not in available:
            continue
        seen.add(key)
        rows.append(dict(spec))
    return rows
