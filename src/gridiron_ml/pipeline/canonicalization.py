"""src.gridiron_ml.pipeline.canonicalization.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Transform cached raw data into canonical tables and downstream artifacts.
"""

def canonicalize_team_game_table_columns(df):
    """
    FINAL schema standardization for team-game tables.

    This function should be called ONCE, at the very end of table generation.
    After this point, column names are considered stable and contractual.
    """

    RENAME_MAP = {

        # ======================
        # IDENTIFIERS / KEYS
        # ======================
        "season": "keys_season",
        "week": "keys_week",
        "game_id": "keys_game_id",
        "team": "keys_team",
        "opponent": "keys_opponent",
        "conference": "keys_conference",
        "game_date": "keys_game_date",
        "season_type": "keys_season_type",
        
        # ======================
        # GAME INFO
        # ======================
        "home_away": "game_home_away",
        "is_home": "game_is_home",
        "kickoff_minutes_after_midnight": "game_kickoff_min",
        "is_night_game": "game_is_night",
        "game_day_of_week": "game_day_of_week",

        # ======================
        # TARGETS
        # ======================
        "points_for": "target_points_for",
        "points_against": "target_points_against",
        "team_margin": "target_team_margin",

        # ======================
        # MARKET
        # ======================
        "line_over_under": "market_over_under",
        "line_team_spread": "market_spread_close",
        "pregame_team_spread": "market_spread_open",
        "team_win_probability": "market_win_probability",

        # ======================
        # OFFENSE (advanced)
        # ======================
        "offense_drives": "offense_drives",
        "offense_plays": "offense_plays",
        "offense_explosiveness": "offense_explosiveness",
        "offense_powerSuccess": "offense_power_success",
        "offense_ppa": "offense_ppa",
        "offense_stuffRate": "offense_stuff_rate",
        "offense_successRate": "offense_success_rate",

        "offense_passingDowns_explosiveness": "offense_passing_downs_explosiveness",
        "offense_passingDowns_ppa": "offense_passing_downs_ppa",
        "offense_passingDowns_successRate": "offense_passing_downs_success_rate",

        "offense_passingPlays_explosiveness": "offense_passing_plays_explosiveness",
        "offense_passingPlays_ppa": "offense_passing_plays_ppa",
        "offense_passingPlays_successRate": "offense_passing_plays_success_rate",

        "offense_rushingPlays_explosiveness": "offense_rushing_plays_explosiveness",
        "offense_rushingPlays_ppa": "offense_rushing_plays_ppa",
        "offense_rushingPlays_successRate": "offense_rushing_plays_success_rate",

        "offense_standardDowns_explosiveness": "offense_standard_downs_explosiveness",

        "offense_dbHavocRate": "offense_db_havoc_rate",
        "offense_frontSevenHavocRate": "offense_front_seven_havoc_rate",
        "offense_havocRate": "offense_havoc_rate",

        # ======================
        # DEFENSE (advanced)
        # ======================
        "defense_drives": "defense_drives",
        "defense_plays": "defense_plays",
        "defense_explosiveness": "defense_explosiveness",
        "defense_powerSuccess": "defense_power_success",
        "defense_ppa": "defense_ppa",
        "defense_stuffRate": "defense_stuff_rate",
        "defense_successRate": "defense_success_rate",

        "defense_passingDowns_explosiveness": "defense_passing_downs_explosiveness",
        "defense_passingDowns_ppa": "defense_passing_downs_ppa",
        "defense_passingDowns_successRate": "defense_passing_downs_success_rate",

        "defense_passingPlays_explosiveness": "defense_passing_plays_explosiveness",
        "defense_passingPlays_ppa": "defense_passing_plays_ppa",
        "defense_passingPlays_successRate": "defense_passing_plays_success_rate",

        "defense_rushingPlays_explosiveness": "defense_rushing_plays_explosiveness",
        "defense_rushingPlays_ppa": "defense_rushing_plays_ppa",
        "defense_rushingPlays_successRate": "defense_rushing_plays_success_rate",

        "defense_standardDowns_explosiveness": "defense_standard_downs_explosiveness",

        "defense_dbHavocRate": "defense_db_havoc_rate",
        "defense_frontSevenHavocRate": "defense_front_seven_havoc_rate",
        "defense_havocRate": "defense_havoc_rate",

        # ======================
        # STATS (boxscore)
        # ======================
        # Offense
        "stat_rushingTDs": "statOff_rushing_tds",
        "stat_passingTDs": "statOff_passing_tds",
        "stat_firstDowns": "statOff_first_downs",
        "stat_yardsPerRushAttempt": "statOff_yards_per_rush_attempt",
        "stat_rushingAttempts": "statOff_rushing_attempts",
        "stat_yardsPerPass": "statOff_yards_per_pass",
        "stat_passAttempts": "statOff_pass_attempts",
        "stat_3Down_Rate": "statOff_third_down_rate",
        "stat_4Down_Rate": "statOff_fourth_down_rate",
        "stat_completion_rate": "statOff_completion_rate",
        # Defense
        "stat_interceptionTDs": "statDef_interception_tds",
        "stat_passesIntercepted": "statDef_passes_intercepted",
        "stat_interceptions": "statDef_interceptions",
        "stat_tacklesForLoss": "statDef_tackles_for_loss",
        "stat_defensiveTDs": "statDef_defensive_tds",
        "stat_tackles": "statDef_tackles",
        "stat_sacks": "statDef_sacks",
        "stat_qbHurries": "statDef_qb_hurries",
        "stat_passesDeflected": "statDef_passes_deflected",
        # Special Teams
        "stat_kickReturnTDs": "statSpe_kick_return_tds",
        "stat_kickReturns": "statSpe_kick_returns",
        "stat_kickingPoints": "statSpe_kicking_points",
        "stat_puntReturnTDs": "statSpe_punt_return_tds",
        "stat_puntReturns": "statSpe_punt_returns",
        # General
        "stat_fumblesRecovered": "statGen_fumbles_recovered",
        "stat_totalFumbles": "statGen_total_fumbles",
        "stat_turnovers": "statGen_turnovers",
        "stat_fumblesLost": "statGen_fumbles_lost",
        "stat_penalties": "statGen_penalties",
        "stat_possessionTime": "statGen_possession_time",
    
        # ======================
        # COACH
        # ======================
        "coach_career_seasons": "coach_career_seasons",
        "coach_career_mean_sp_offense": "coach_career_mean_sp_offense",
        "coach_career_mean_sp_defense": "coach_career_mean_sp_defense",
        "coach_career_mean_postseason_rank_points": "coach_career_mean_postseason_rank_points",
        "coach_season_games": "coach_season_games",

        # ======================
        # ROSTER / RETURNING
        # ======================
        "talent": "roster_talent",
        "return_team_total_rushing_p_p_a": "roster_return_total_rushing_p_p_a",
        "return_team_percent_p_p_a": "roster_return_percent_p_p_a",
        "return_team_percent_passing_p_p_a": "roster_return_percent_passing_p_p_a",
        "return_team_percent_receiving_p_p_a": "roster_return_percent_receiving_p_p_a",
        "return_team_percent_rushing_p_p_a": "roster_return_percent_rushing_p_p_a",
        "return_team_rushing_usage": "roster_return_rushing_usage",

        # ======================
        # TRAVEL / CONTEXT
        # ======================
        "travel_diff": "travel_distance_diff",
        "tz_diff": "travel_tz_diff",

    }

    # Apply renaming
    df = df.rename(columns=RENAME_MAP)

    # Safety check: ensure no unexpected columns slipped through
    missing = set(RENAME_MAP) - set(df.columns)
    if missing:
        print("[standardize_team_game_columns] WARNING: expected columns missing:")
        for c in sorted(missing):
            print("  -", c)

    return df
