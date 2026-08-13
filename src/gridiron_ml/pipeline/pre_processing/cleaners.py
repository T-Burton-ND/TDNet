"""src.gridiron_ml.pipeline.pre_processing.cleaners.

Usage:
    Import this module from TDNet code or run it as documented by its public entry points.

Logic flow:
    1. Load or receive the inputs required for this module's responsibility.
    2. Apply the package-specific transformations or orchestration.
    3. Return normalized objects, saved artifacts, or command-line results.

Responsibility:
    Clean and normalize source tables before feature construction.
"""

import numpy as np
import pandas as pd
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:  # older Python
    from backports.zoneinfo import ZoneInfo


def prep_for_fingerprints(
    team_games: pd.DataFrame,
    drop_postseason: bool = True,
) -> pd.DataFrame:
    """
    Prepare team-game table for downstream fingerprint construction.

    Enforces:
    - valid final-score targets for completed games while preserving future schedules
    - normalized season_type
    - optional removal of postseason games
    - consistent home/away encoding

    Parameters
    ----------
    team_games : pd.DataFrame
        Cleaned team-game table (one row per team per game).
    drop_postseason : bool
        If True, remove postseason games (default: True).

    Returns
    -------
    pd.DataFrame
        Filtered, fingerprint-ready dataframe.
    """
    df = team_games.copy()

    # --------------------------------------------------
    # 1) Drop partial target rows, but keep future scheduled games.
    # --------------------------------------------------
    target_cols = ["points_for", "points_against", "team_margin"]
    existing_targets = [col for col in target_cols if col in df.columns]
    if existing_targets:
        target_values = df[existing_targets]
        has_any_target = target_values.notna().any(axis=1)
        has_all_targets = target_values.notna().all(axis=1)
        df = df.loc[has_all_targets | ~has_any_target].copy()

    # --------------------------------------------------
    # 2) Normalize season_type
    # --------------------------------------------------
    if "season_type" in df.columns:
        df["season_type"] = df["season_type"].fillna("regular")

        if drop_postseason:
            before = len(df)
            df = df[df["season_type"] != "postseason"].copy()
            dropped = before - len(df)
            if dropped > 0:
                print(f"[prep_for_fingerprints] Dropped {dropped} postseason rows.")

    # --------------------------------------------------
    # 3) Home / away hygiene
    # --------------------------------------------------
    if "home_away" in df.columns and "is_home" in df.columns:
        mask = df["home_away"].isna()
        df.loc[mask & (df["is_home"] == True), "home_away"] = "home"
        df.loc[mask & (df["is_home"] == False), "home_away"] = "away"

    return df


def clean_fractionals(raw: pd.DataFrame) -> pd.DataFrame:
    """
    One-stop cleaner for the per-team-per-game raw table.
    Inlines all prior cleaning steps (split by comment blocks).
    """
    df = raw.copy()

    # =========================================================
    # 1) Split fractional / packed stat columns
    # =========================================================
    if "stat_totalPenaltiesYards" in df.columns:
        penalties = df["stat_totalPenaltiesYards"].apply(
            lambda x: x.split("-")[0] if isinstance(x, str) and "-" in x else np.nan
        )
        yards = df["stat_totalPenaltiesYards"].apply(
            lambda x: x.split("-")[1] if isinstance(x, str) and "-" in x else np.nan
        )
        df["stat_penalties"] = penalties
        df["stat_penaltyYards"] = yards
        df = df.drop(columns=["stat_totalPenaltiesYards"], errors="ignore")

    if "stat_completionAttempts" in df.columns:
        completions = df["stat_completionAttempts"].apply(
            lambda x: x.split("-")[0] if isinstance(x, str) and "-" in x else np.nan
        )
        attempts = df["stat_completionAttempts"].apply(
            lambda x: x.split("-")[1] if isinstance(x, str) and "-" in x else np.nan
        )
        df["stat_completions"] = completions
        df["stat_passAttempts"] = attempts
        df = df.drop(columns=["stat_completionAttempts"], errors="ignore")

    if "stat_fourthDownEff" in df.columns:
        success = df["stat_fourthDownEff"].apply(
            lambda x: x.split("-")[0] if isinstance(x, str) and "-" in x else np.nan
        )
        attempts = df["stat_fourthDownEff"].apply(
            lambda x: x.split("-")[1] if isinstance(x, str) and "-" in x else np.nan
        )
        df["stat_fourthDown_success"] = success
        df["stat_fourthDown_attempts"] = attempts
        df = df.drop(columns=["stat_fourthDownEff"], errors="ignore")

    if "stat_thirdDownEff" in df.columns:
        success = df["stat_thirdDownEff"].apply(
            lambda x: x.split("-")[0] if isinstance(x, str) and "-" in x else np.nan
        )
        attempts = df["stat_thirdDownEff"].apply(
            lambda x: x.split("-")[1] if isinstance(x, str) and "-" in x else np.nan
        )
        df["stat_thirdDown_success"] = success
        df["stat_thirdDown_attempts"] = attempts
        df = df.drop(columns=["stat_thirdDownEff"], errors="ignore")

    return df


def drop_moneylines(df: pd.DataFrame) -> pd.DataFrame:
    # =========================================================
    # 2) Drop obvious redundant betting columns (moneylines)
    # =========================================================
    """Run the drop_moneylines step and return its normalized result."""
    df = df.drop(
        columns=["line_home_moneyline", "line_away_moneyline"], errors="ignore"
    )
    return df


def clean_coaches(df: pd.DataFrame) -> pd.DataFrame:
    # =========================================================
    # 3) Clean coaching columns (drop extras)
    # =========================================================
    """Run the clean_coaches step and return its normalized result."""
    df = df.drop(
        columns=[
            "coach_hire_date_x",
            "coach_first_name",
            "coach_last_name",
            "coach_hire_date_y",
            "coach_career_mean_preseason_rank",
            "coach_career_mean_postseason_rank",
            "coach_career_mean_postseason_rank_points",
            "coach_season_games",
            "coach_season_wins",
            "coach_season_losses",
            "coach_season_ties",
            "coach_season_mean_preseason_rank",
            "coach_season_mean_postseason_rank",
            "coach_season_mean_preseason_rank_points",
            "coach_season_mean_postseason_rank_points",
            "coach_season_mean_sp_offense",
            "coach_season_mean_sp_defense",
            "coach_season_mean_sp_overall",
            "coach_season_mean_srs",
            "coach_career_total_games",
            "coach_career_total_wins",
            "coach_career_total_losses",
            "coach_career_total_ties",
        ],
        errors="ignore",
    )
    return df


def clean_timezones(df: pd.DataFrame) -> pd.DataFrame:
    # =========================================================
    # 4) Timezone offsets (name -> offset; fallback lon/15; final fallback 0)
    # =========================================================
    """Run the clean_timezones step and return its normalized result."""

    def _tz_name_to_offset_hours(tz_name, reference_dt):
        """Internal helper for the tz_name_to_offset_hours step."""
        if reference_dt is None:
            reference_dt = datetime(2024, 10, 1)
        if pd.isna(tz_name) or tz_name in ("", "UNKNOWN"):
            return np.nan
        try:
            tz = ZoneInfo(str(tz_name))
            offset = tz.utcoffset(reference_dt)
            if offset is None:
                return np.nan
            return offset.total_seconds() / 3600.0
        except Exception:
            return np.nan

    def _approx_offset_from_longitude(lon):
        """Internal helper for the approx_offset_from_longitude step."""
        try:
            if pd.isna(lon):
                return np.nan
            approx = round(float(lon) / 15.0)
            return float(np.clip(approx, -12, 14))
        except Exception:
            return np.nan

    if "team_loc_timezone" in df.columns:
        df["team_tz_offset"] = df["team_loc_timezone"].apply(
            lambda x: _tz_name_to_offset_hours(x, None)
        )
    else:
        df["team_tz_offset"] = np.nan

    if "opp_loc_timezone" in df.columns:
        df["opp_tz_offset"] = df["opp_loc_timezone"].apply(
            lambda x: _tz_name_to_offset_hours(x, None)
        )
    else:
        df["opp_tz_offset"] = np.nan

    if "venue_timezone" in df.columns:
        df["venue_tz_offset"] = df["venue_timezone"].apply(
            lambda x: _tz_name_to_offset_hours(x, None)
        )
    else:
        df["venue_tz_offset"] = np.nan

    if "team_loc_longitude" in df.columns:
        m = df["team_tz_offset"].isna()
        df.loc[m, "team_tz_offset"] = df.loc[m, "team_loc_longitude"].apply(
            _approx_offset_from_longitude
        )

    if "opp_loc_longitude" in df.columns:
        df["opp_loc_longitude"] = pd.to_numeric(
            df["opp_loc_longitude"], errors="coerce"
        )
        m = df["opp_tz_offset"].isna()
        df.loc[m, "opp_tz_offset"] = df.loc[m, "opp_loc_longitude"].apply(
            _approx_offset_from_longitude
        )

    if "venue_longitude" in df.columns:
        m = df["venue_tz_offset"].isna()
        df.loc[m, "venue_longitude"] = pd.to_numeric(
            df.loc[m, "venue_longitude"], errors="coerce"
        )
        df.loc[m, "venue_tz_offset"] = df.loc[m, "venue_longitude"].apply(
            _approx_offset_from_longitude
        )

    for col in ["team_tz_offset", "opp_tz_offset", "venue_tz_offset"]:
        df[col] = df[col].fillna(0.0)
    return df


def clean_geography(df: pd.DataFrame) -> pd.DataFrame:
    # =========================================================
    # 5) Elevations: drop (and stop carrying them)
    # =========================================================
    """Run the clean_geography step and return its normalized result."""
    df = df.drop(
        columns=["venue_elevation", "team_loc_elevation", "opp_loc_elevation"],
        errors="ignore",
    )
    return df


def clean_time(df: pd.DataFrame) -> pd.DataFrame:
    # =========================================================
    # 6) Split start_date -> game_date/game_time/day_of_week
    # =========================================================
    """Run the clean_time step and return its normalized result."""
    if "start_date" in df.columns:
        dt = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
        df["game_date"] = dt.dt.date
        df["game_time"] = dt.dt.time
        df["game_day_of_week"] = dt.dt.dayofweek
        df = df.drop(columns=["start_date"], errors="ignore")
    return df


def clean_travel(df):

    # =========================================================
    # 7) Travel diff (log1p km) and drop raw lat/lon
    # =========================================================
    """Run the clean_travel step and return its normalized result."""
    for c in [
        "team_loc_latitude",
        "team_loc_longitude",
        "opp_loc_latitude",
        "opp_loc_longitude",
        "venue_latitude",
        "venue_longitude",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if all(
        c in df.columns
        for c in [
            "team_loc_latitude",
            "team_loc_longitude",
            "opp_loc_latitude",
            "opp_loc_longitude",
            "venue_latitude",
            "venue_longitude",
        ]
    ):
        R = 6371.0088
        tlat = np.radians(df["team_loc_latitude"].astype(float))
        tlon = np.radians(df["team_loc_longitude"].astype(float))
        olat = np.radians(df["opp_loc_latitude"].astype(float))
        olon = np.radians(df["opp_loc_longitude"].astype(float))
        vlat = np.radians(df["venue_latitude"].astype(float))
        vlon = np.radians(df["venue_longitude"].astype(float))

        dlat_t = vlat - tlat
        dlon_t = vlon - tlon
        a_t = (
            np.sin(dlat_t / 2) ** 2
            + np.cos(tlat) * np.cos(vlat) * np.sin(dlon_t / 2) ** 2
        )
        team_km = 2 * R * np.arcsin(np.sqrt(a_t))

        dlat_o = vlat - olat
        dlon_o = vlon - olon
        a_o = (
            np.sin(dlat_o / 2) ** 2
            + np.cos(olat) * np.cos(vlat) * np.sin(dlon_o / 2) ** 2
        )
        opp_km = 2 * R * np.arcsin(np.sqrt(a_o))

        df["travel_diff"] = np.log1p(team_km) - np.log1p(opp_km)

        df = df.drop(
            columns=[
                "team_loc_latitude",
                "team_loc_longitude",
                "opp_loc_latitude",
                "opp_loc_longitude",
                "venue_latitude",
                "venue_longitude",
            ],
            errors="ignore",
        )
    return df


def clean_tz(df: pd.DataFrame) -> pd.DataFrame:
    # =========================================================
    # 8) TZ diff relative to venue; then drop tz offsets
    # =========================================================
    """Run the clean_tz step and return its normalized result."""
    team_mismatch = (df["team_tz_offset"] - df["venue_tz_offset"]).abs()
    opp_mismatch = (df["opp_tz_offset"] - df["venue_tz_offset"]).abs()
    df["tz_diff"] = (team_mismatch - opp_mismatch).clip(-6, 6)

    df = df.drop(
        columns=["team_tz_offset", "opp_tz_offset", "venue_tz_offset"], errors="ignore"
    )
    df = df.drop(
        columns=["team_loc_timezone", "opp_loc_timezone", "venue_timezone"],
        errors="ignore",
    )
    return df


def clean_kickoff(df: pd.DataFrame) -> pd.DataFrame:
    # =========================================================
    # 9) Drop raw game_time (we'll derive kickoff features, then drop)
    # =========================================================
    """Run the clean_kickoff step and return its normalized result."""
    if "game_time" in df.columns:
        df["kickoff_minutes_after_midnight"] = df["game_time"].apply(
            lambda t: t.hour * 60 + t.minute if hasattr(t, "hour") else np.nan
        )
        df["is_night_game"] = df["kickoff_minutes_after_midnight"].apply(
            lambda m: True if (pd.notna(m) and m >= 18 * 60) else False
        )
    df = df.drop(columns=["game_time"], errors="ignore")
    return df


def clean_score(df):
    # =========================================================
    # 10) Team scoring (home/away -> team-centric)
    # =========================================================
    """Run the clean_score step and return its normalized result."""
    if "is_home" in df.columns and df["is_home"].dtype != bool:
        df["is_home"] = df["is_home"].astype(bool)

    if "home_score" in df.columns and "away_score" in df.columns:
        df["points_for"] = np.where(df["is_home"], df["home_score"], df["away_score"])
        df["points_against"] = np.where(
            df["is_home"], df["away_score"], df["home_score"]
        )
        df["team_margin"] = df["points_for"] - df["points_against"]
        df = df.drop(
            columns=["home_score", "away_score", "home_margin", "margin"],
            errors="ignore",
        )
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # =========================================================
    # 11) Convert numeric columns (broad-strokes)
    # =========================================================
    # (keep this lightweight; anything non-numeric will become NaN)
    """Run the clean_numeric step and return its normalized result."""
    for c in df.columns:
        if (
            c in ("season", "week", "team_id", "opponent_id", "game_day_of_week")
            or c.startswith("stat_")
            or c.startswith("offense_")
            or c.startswith("defense_")
            or c
            in (
                "talent",
                "recruit_rank",
                "recruit_points",
                "line_team_spread",
                "line_over_under",
                "team_win_probability",
            )
        ):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    if "team_id" in df.columns:
        df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    if "opponent_id" in df.columns:
        df["opponent_id"] = pd.to_numeric(df["opponent_id"], errors="coerce").astype(
            "Int64"
        )
    return df


def clean_posession(df: pd.DataFrame) -> pd.DataFrame:

    # =========================================================
    # 12) Possession time -> minutes
    # =========================================================
    """Run the clean_posession step and return its normalized result."""
    if "stat_possessionTime" in df.columns:

        def to_minutes(x):
            """Run the to_minutes step and return its normalized result."""
            if isinstance(x, str) and ":" in x:
                m, s = x.split(":")
                return int(m) + int(s) / 60.0
            return np.nan

        df["stat_possessionTime"] = df["stat_possessionTime"].apply(to_minutes)
    return df


def clean_downs(df: pd.DataFrame):
    # =========================================================
    # 13) Leverage rates (3rd/4th down)
    # =========================================================
    """Run the clean_downs step and return its normalized result."""
    new_columns = {}
    if (
        "stat_thirdDown_success" in df.columns
        and "stat_thirdDown_attempts" in df.columns
    ):
        s = pd.to_numeric(df["stat_thirdDown_success"], errors="coerce")
        a = pd.to_numeric(df["stat_thirdDown_attempts"], errors="coerce")
        new_columns["stat_3Down_Rate"] = (s / a.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )

    if (
        "stat_fourthDown_success" in df.columns
        and "stat_fourthDown_attempts" in df.columns
    ):
        s = pd.to_numeric(df["stat_fourthDown_success"], errors="coerce")
        a = pd.to_numeric(df["stat_fourthDown_attempts"], errors="coerce")
        new_columns["stat_4Down_Rate"] = (s / a.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )

    df = df.drop(
        columns=[
            "stat_thirdDown_attempts",
            "stat_thirdDown_success",
            "stat_fourthDown_attempts",
            "stat_fourthDown_success",
        ],
        errors="ignore",
    )
    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1).copy()
    return df


def clean_redundant(df: pd.DataFrame):
    # =========================================================
    # 14) Drop yardage totals & havoc event counts; drop redundant plays
    # =========================================================
    # yardage totals (stat/offense/defense) ending in Yards but not Per*
    """Run the clean_redundant step and return its normalized result."""
    drop_yards = [
        c
        for c in df.columns
        if (
            c.startswith("stat_")
            or c.startswith("offense_")
            or c.startswith("defense_")
        )
        and c.endswith("Yards")
        and "Per" not in c
    ]
    df = df.drop(columns=drop_yards, errors="ignore")

    # havoc event counts (keep Rate columns)
    drop_havoc = [
        c
        for c in df.columns
        if ("havoc" in c.lower())
        and ("event" in c.lower())
        and (not c.lower().endswith("rate"))
    ]
    df = df.drop(columns=drop_havoc, errors="ignore")

    # redundant plays
    # df = df.drop(columns=["offense_plays", "defense_plays"], errors="ignore")
    return df


def clean_pass_rate(df):
    # =========================================================
    # 15) Completion rate + drop stat_completions
    # =========================================================
    """Run the clean_pass_rate step and return its normalized result."""
    if "stat_completions" in df.columns and "stat_passAttempts" in df.columns:
        completion_rate = (df["stat_completions"] / df["stat_passAttempts"]).replace(
            [np.inf, -np.inf], np.nan
        )
        df = df.drop(columns=["stat_completions"], errors="ignore")
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {"stat_completion_rate": completion_rate},
                    index=df.index,
                ),
            ],
            axis=1,
        ).copy()
    return df


def drop_summary_buckets(df: pd.DataFrame):
    # =========================================================
    # 16) Drop summary buckets + advanced totals (your final pruning)
    # =========================================================
    """Run the drop_summary_buckets step and return its normalized result."""
    df = df.drop(
        columns=[
            "offense_firstDown",
            "offense_overall",
            "offense_passing",
            "offense_rushing",
            "offense_secondDown",
            "offense_thirdDown",
            "defense_firstDown",
            "defense_overall",
            "defense_passing",
            "defense_rushing",
            "defense_secondDown",
            "defense_thirdDown",
            "offense_totalPlays",
            "defense_totalPlays",
            "offense_lineYardsTotal",
            "offense_openFieldYardsTotal",
            "offense_secondLevelYardsTotal",
            "defense_lineYardsTotal",
            "defense_openFieldYardsTotal",
            "defense_secondLevelYardsTotal",
            "offense_totalPPA",
            "defense_totalPPA",
            "offense_passingPlays_totalPPA",
            "offense_rushingPlays_totalPPA",
            "defense_passingPlays_totalPPA",
            "defense_rushingPlays_totalPPA",
        ],
        errors="ignore",
    )
    return df


def fill_zeros(df: pd.DataFrame):
    # =========================================================
    # 17) Fill zeros for a few count stats
    # =========================================================
    """Run the fill_zeros step and return its normalized result."""
    zero_fill = [
        "stat_interceptionTDs",
        "stat_passesIntercepted",
        "stat_puntReturns",
        "stat_puntReturnTDs",
        "stat_kickReturnTDs",
        "stat_kickReturns",
        "stat_totalFumbles",
    ]
    for c in zero_fill:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def clean_dtypes(df: pd.DataFrame):
    # =========================================================
    # 18) Category dtypes
    # =========================================================
    """Run the clean_dtypes step and return its normalized result."""
    for c in ["team", "opponent", "conference", "season_type"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Ensure game_date is datetime if present
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    df = df.copy()

    return df


def clean_metatypes(df):
    # =========================================================
    # 19) Metatypes
    # =========================================================
    """Run the clean_metatypes step and return its normalized result."""
    drop_cols = [
        # -------------------------------------------------
        # Merge-artifact duplicates (keep canonical versions)
        # -------------------------------------------------
        "season_type_havoc",
        "season_type_lines",
        "season_type_wp",
        "season_type_ppa",
        "conference_havoc",
        "conference_ppa",
        "opponent_adv",
        "opponent_havoc",
        "opponent_conference",
        "opponent_lines",
        "opponent_wp",
        "opponent_ppa",
        "is_home_lines",
        "is_home_wp",
        # -------------------------------------------------
        # Raw IDs (no modeling or aggregation value)
        # -------------------------------------------------
        "venue_id",
        "home_id",
        "away_id",
        "team_id",
        "opponent_id",
        "team_id_lines",
        # -------------------------------------------------
        # Redundant scoring columns
        # -------------------------------------------------
        "points",  # keep points_for / points_against / team_margin
        # -------------------------------------------------
        # Raw location text (already encoded via travel / tz)
        # -------------------------------------------------
        "venue_name",
        "venue_city",
        "venue_state",
        "team_loc_city",
        "team_loc_state",
        "opp_loc_city",
        "opp_loc_state",
        # -------------------------------------------------
        # Raw betting fields replaced by team-centric versions
        # -------------------------------------------------
        "line_spread_raw",
        "pregame_spread_raw",
        "home_win_probability",  # keep team_win_probability
        # -------------------------------------------------
        # Coach identifiers already summarized
        # -------------------------------------------------
        # (full name is not useful once career means exist)
        "coach_full_name",
        "coach_career_mean_srs",
        # -------------------------------------------------
        # Unhelpful Rankings
        # -------------------------------------------------
        "recruit_rank",
        "recruit_points",
        # -------------------------------------------------
        # Highly Correlated
        # -------------------------------------------------
        "return_team_usage",
        "return_team_receiving_usage",
        "return_team_receiving_usage",
        "offense_standardDowns_successRate",
        "defense_standardDowns_successRate",
        "offense_standardDowns_successRate",
        "coach_career_mean_sp_overall",
        "coach_career_mean_preseason_rank_points",
        "coach_career_mean_postseason_rank_points",
        "return_team_total_passing_p_p_a",
        "return_team_total_p_p_a",
        "return_team_passing_usage",
        "return_team_total_receiving_p_p_a",
        "defense_standardDowns_ppa",
        "offense_standardDowns_ppa",
    ]

    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c], errors="ignore")

    return df


CLEANING_STEPS = (
    clean_fractionals,
    drop_moneylines,
    clean_coaches,
    clean_timezones,
    clean_geography,
    clean_time,
    clean_travel,
    clean_tz,
    clean_kickoff,
    clean_score,
    clean_posession,
    clean_numeric,
    clean_downs,
    clean_redundant,
    clean_pass_rate,
    drop_summary_buckets,
    fill_zeros,
    clean_dtypes,
    clean_metatypes,
)


def main_clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Run the main_clean step and return its normalized result."""
    df = raw.copy()
    for step in CLEANING_STEPS:
        df = step(df)
    return df
