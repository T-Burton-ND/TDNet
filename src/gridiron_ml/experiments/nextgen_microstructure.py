"""Structured F09 play primitives; never parse descriptions or use WP.

Raw CFBD score fields are post-play in the observed source. Reconstruct the
pre-play scoreboard in chronological game order before situational filtering.
These are past-game sufficient statistics, not model-ready feature artifacts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RUSH_TYPES = {"Rush", "Rushing Touchdown"}
# Both legacy labels remain in the archived /plays/types catalog. Completion
# occurs primarily before 2014, but also in 2025; do not gate aliases by year.
PASS_TYPES = {"Pass Reception", "Pass Completion", "Pass Incompletion", "Passing Touchdown", "Sack",
              "Interception", "Pass Interception", "Pass Interception Return", "Interception Return Touchdown"}


def play_flags(plays: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "id", "drive_number", "play_number", "offense", "defense",
                "home", "away", "offense_score", "defense_score", "period", "down",
                "distance", "yards_gained", "yards_to_goal", "play_type", "clock.minutes", "clock.seconds"}
    if not required <= set(plays):
        raise ValueError(f"Missing structured play fields: {sorted(required-set(plays))}")
    # Explicit allowlist prevents descriptions and probabilities entering logic.
    p = plays.loc[:, sorted(required | ({"drive_id"} if "drive_id" in plays else set()))].copy()
    if p.duplicated(["game_id", "id"]).any():
        raise ValueError("Duplicate play identity; deduplicate verified partitions first")
    for c in ("offense_score", "defense_score", "period", "down", "distance", "yards_gained",
              "yards_to_goal", "clock.minutes", "clock.seconds", "drive_number", "play_number"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    # Some archived overtime events reuse regulation drive/play positions.
    # Period precedes those positions; arbitrary IDs cannot resolve ties within
    # a period. Unknown order invalidates the affected pre-play scoreboard and
    # its immediate successor, without inventing an ordering from descriptions.
    order = ["game_id", "period", "drive_number", "play_number"]
    p = p.sort_values(order + ["id"], kind="stable").reset_index(drop=True)
    ambiguous = p.duplicated(order, keep=False) | p[order].isna().any(axis=1)
    unknown_pre = ambiguous | ambiguous.groupby(p.game_id).shift(fill_value=False)
    home_offense = p.offense.eq(p.home)
    known_offense = home_offense | p.offense.eq(p.away)
    home_post = p.offense_score.where(home_offense, p.defense_score).where(known_offense)
    away_post = p.defense_score.where(home_offense, p.offense_score).where(known_offense)
    home_pre = home_post.groupby(p.game_id).shift()
    away_pre = away_post.groupby(p.game_id).shift()
    first = ~p.game_id.duplicated()
    # Only a recognizable scoreless opening can establish the initial score.
    opening = first & p.period.eq(1) & home_post.eq(0) & away_post.eq(0)
    home_pre.loc[opening] = 0
    away_pre.loc[opening] = 0
    home_pre = home_pre.mask(unknown_pre)
    away_pre = away_pre.mask(unknown_pre)
    p["preplay_absolute_lead"] = (home_pre-away_pre).abs()
    p["rush"] = p.play_type.isin(RUSH_TYPES)
    p["dropback"] = p.play_type.isin(PASS_TYPES)
    p["qualifying"] = (p.rush | p.dropback) & p.down.between(1, 4) & p.yards_gained.notna()
    q4 = p.period.eq(4) & p.qualifying
    q4_known = p.loc[q4].groupby("game_id").preplay_absolute_lead.agg(lambda x: x.notna().all())
    q4_min = p.loc[q4].groupby("game_id").preplay_absolute_lead.min()
    q4_garbage = q4_known & q4_min.gt(16)
    thresholds = p.period.map({1: 28, 2: 24, 3: 21})
    p["garbage"] = p.preplay_absolute_lead.gt(thresholds)
    p.loc[p.period.eq(4), "garbage"] = p.loc[p.period.eq(4), "game_id"].map(q4_garbage).fillna(False).astype(bool)
    q4_coverage = p.game_id.map(q4_known).fillna(False).astype(bool)
    p["time_eligible"] = (p.qualifying & p.period.between(1, 4) & ~p.garbage
                          & p.preplay_absolute_lead.notna() & (~p.period.eq(4) | q4_coverage))
    clock = 60*p["clock.minutes"] + p["clock.seconds"]
    p["middle_eight"] = p.time_eligible & ((p.period.eq(2) & clock.le(240)) | (p.period.eq(3) & clock.gt(660)))
    p["two_minute"] = p.time_eligible & p.period.isin([2, 4]) & clock.le(120)
    p["one_score"] = p.time_eligible & p.preplay_absolute_lead.le(8)
    p["red_zone"] = p.qualifying & p.yards_to_goal.le(20) & p.yards_to_goal.gt(0)
    p["goal_to_go"] = p.qualifying & p.distance.ge(p.yards_to_goal) & p.yards_to_goal.le(10) & p.yards_to_goal.gt(0)
    p["backed_up"] = p.time_eligible & p.yards_to_goal.ge(90)
    threshold = p.down.map({1: 0.5, 2: 0.7, 3: 1.0, 4: 1.0}) * p.distance
    p["success"] = p.yards_gained.ge(threshold) & p.distance.gt(0)
    p["explosive"] = (p.rush & p.yards_gained.ge(10)) | (p.dropback & p.yards_gained.ge(20))
    # Yardage-credit version of line yards, with loss multiplier and 10-yard cap.
    y = p.yards_gained
    p["line_yards"] = np.where(y.lt(0), 1.2*y, np.minimum(y, 4) + 0.5*np.clip(y-4, 0, 6))
    return p


def game_sufficient_statistics(plays: pd.DataFrame) -> pd.DataFrame:
    """Return additive numerators/denominators for later pre-target aggregation.

    Rushes exclude sacks and unclassified fumbles; sacks are dropbacks. No
    fictional attempts are attributed from ambiguous play types. Rates are
    computed only after aggregating eligible prior games, avoiding averages
    of unstable per-game ratios.
    """
    p = play_flags(plays)
    eligible = p.time_eligible
    metrics = {
        "rush_ypa": (eligible & p.rush, p.yards_gained),
        "dropback_ypa": (eligible & p.dropback, p.yards_gained),
        "success_rate": (eligible & p.distance.gt(0), p.success.astype(float)),
        "explosive_rate": (eligible, p.explosive.astype(float)),
        "rush_line_yards": (eligible & p.rush, p.line_yards),
        "rush_stuff_rate": (eligible & p.rush, p.yards_gained.le(0).astype(float)),
        "rush_opportunity_rate": (eligible & p.rush, p.yards_gained.ge(4).astype(float)),
        "rush_second_level_yards": (eligible & p.rush, (p.yards_gained-4).clip(0, 6)),
        "rush_open_field_yards": (eligible & p.rush, (p.yards_gained-10).clip(lower=0)),
        "power_success": (eligible & p.rush & p.down.isin([3, 4]) & p.distance.between(1, 2), p.success.astype(float)),
        "sack_rate": (eligible & p.dropback, p.play_type.eq("Sack").astype(float)),
    }
    for q in range(1, 5):
        metrics[f"rush_ypa_q{q}"] = (eligible & p.rush & p.period.eq(q), p.yards_gained)
    for label, mask in {"first_half": p.period.le(2), "second_half": p.period.isin([3, 4]),
                        "middle_eight": p.middle_eight, "one_score": p.one_score,
                        "two_minute": p.two_minute, "backed_up": p.backed_up,
                        "early_down": p.down.le(2), "third_down": p.down.eq(3),
                        "fourth_down": p.down.eq(4)}.items():
        metrics[f"{label}_success_rate"] = (eligible & mask & p.distance.gt(0), p.success.astype(float))
    # Location concepts deliberately retain garbage-time plays.
    for label in ("red_zone", "goal_to_go"):
        metrics[f"{label}_success_rate"] = (p[label] & p.distance.gt(0), p.success.astype(float))
    blocks = []
    for side in ("offense", "defense"):
        values = {"game_id": p.game_id, "team": p[side]}
        for name, (mask, value) in metrics.items():
            good = mask & value.notna()
            values[f"{side}_{name}__sum"] = value.where(good, 0).astype(float)
            values[f"{side}_{name}__n"] = good.astype(int)
        blocks.append(pd.DataFrame(values).groupby(["game_id", "team"]).sum())
    return pd.concat(blocks, axis=1).reset_index()


def drive_sufficient_statistics(drives: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """Competitive whole-drive pace, field position and scoring opportunities.

    A drive must contain a qualifying scrimmage play and every such play must
    pass the common time filter. Opportunity means a qualifying snap at or
    inside the opposing 40; quality means reaching that area or gaining 40
    drive yards. These are explicit TDNet definitions, not inferred WP states.
    """
    p = play_flags(plays)
    if "drive_id" not in p:
        raise ValueError("Drive analysis requires structured play drive IDs")
    p["drive_key"] = p.drive_id.astype(str)
    q = p.loc[p.qualifying].copy()
    q["opportunity"] = q.yards_to_goal.le(40)
    flags = q.groupby(["game_id", "drive_key"]).agg(
        eligible=("time_eligible", "all"), qualifying_plays=("id", "size"),
        opportunity=("opportunity", "any"))
    d = drives.copy()
    d["drive_key"] = d.id.astype(str)
    if d.duplicated(["game_id", "drive_key"]).any():
        raise ValueError("Duplicate drive identity")
    d = d.merge(flags.reset_index(), on=["game_id", "drive_key"], validate="one_to_one")
    d = d.loc[d.eligible & d.start_period.between(1, 4) & d.end_period.between(1, 4)].copy()
    for col in ("start_offense_score", "end_offense_score", "start_defense_score", "end_defense_score",
                "start_yards_to_goal", "yards", "elapsed.minutes", "elapsed.seconds"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    points = d.end_offense_score-d.start_offense_score
    conceded = d.end_defense_score-d.start_defense_score
    valid_points = points.between(0, 8) & conceded.between(0, 8)
    elapsed = 60*d["elapsed.minutes"]+d["elapsed.seconds"]
    metrics = {
        "drive_start_field_position": (100-d.start_yards_to_goal, d.start_yards_to_goal.between(0, 100)),
        "net_points_per_drive": ((points-conceded).where(valid_points), valid_points),
        "points_per_opportunity": (points.where(valid_points), valid_points & d.opportunity),
        "scoring_opportunity_rate": (d.opportunity.astype(float), pd.Series(True, index=d.index)),
        "quality_drive_rate": ((d.opportunity | d.yards.ge(40)).astype(float), d.yards.notna()),
        "drive_seconds_per_play": (elapsed, elapsed.between(1, 900)),
    }
    blocks = []
    for side in ("offense", "defense"):
        values = {"game_id": d.game_id, "team": d[side]}
        for name, (value, mask) in metrics.items():
            good = mask & value.notna()
            values[f"{side}_{name}__sum"] = value.where(good, 0).astype(float)
            denom = d.qualifying_plays if name == "drive_seconds_per_play" else pd.Series(1, index=d.index)
            values[f"{side}_{name}__n"] = denom.where(good, 0).astype(float)
        blocks.append(pd.DataFrame(values).groupby(["game_id", "team"]).sum())
    return pd.concat(blocks, axis=1).reset_index()
