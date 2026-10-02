"""Historical, cutoff-safe candidate states for the F13–F16 A research rounds.

This is a separate experiment path. Saved F09–F12 artifacts and results remain
immutable; the repaired F09 classifier is first measured as its own reference.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .nextgen_microstructure import play_flags


F13_METRICS = ("resid_yards", "resid_success", "lower_tail", "upper_tail")
F14_METRICS = ("failure_recovery", "success_pair", "third_after_failure", "drive_two_success")
F15_METRICS = ("early_rush_hhi", "early_rush_top_share", "short_rush_hhi",
               "short_rush_top_share", "standard_reception_hhi",
               "standard_reception_top_share", "redzone_reception_hhi",
               "redzone_reception_top_share")


def context_plays(flagged: pd.DataFrame) -> pd.DataFrame:
    """Keep supported, competitive regulation scrimmage plays and fixed bins."""
    p = flagged.loc[
        flagged.time_eligible & flagged.distance.gt(0)
        & flagged.yards_to_goal.between(1, 99)
        & flagged.down.between(1, 4)
    ].copy()
    if p.empty:
        p["context_id"] = pd.Series(dtype="int16")
        return p
    kind = p.dropback.astype(int)
    down = p.down.gt(2).astype(int)
    distance = np.select([p.distance.le(3), p.distance.le(7)], [0, 1], default=2)
    field = np.select([p.yards_to_goal.le(20), p.yards_to_goal.lt(80)], [0, 1], default=2)
    close = p.preplay_absolute_lead.le(8).astype(int)
    p["context_id"] = ((((kind * 2 + down) * 3 + distance) * 3 + field) * 2 + close).astype("int16")
    return p


def context_totals(p: pd.DataFrame, season: int) -> pd.DataFrame:
    if p.empty:
        return pd.DataFrame(columns=["season", "context_id", "n", "sum_yards", "sum_success"])
    q = p.assign(season=int(season), success_value=p.success.astype(float))
    return q.groupby(["season", "context_id"], as_index=False).agg(
        n=("id", "size"), sum_yards=("yards_gained", "sum"),
        sum_success=("success_value", "sum"))


def past_only_context_baselines(totals: pd.DataFrame, years=range(2010, 2026),
                                *, shrink_count: int = 50) -> dict[int, pd.DataFrame]:
    """A season's expected outcomes use earlier seasons only.

    Small context cells shrink to their prior-year play-class mean. There is
    no current-season observation, target outcome, external rating or market
    input in this baseline.
    """
    by_year = {int(y): x.groupby("context_id")[["n", "sum_yards", "sum_success"]].sum()
               for y, x in totals.groupby("season")}
    cumulative = pd.DataFrame(columns=["n", "sum_yards", "sum_success"], dtype=float)
    outputs = {}
    for year in years:
        prior = cumulative.copy()
        if not prior.empty:
            prior["class"] = (prior.index.to_numpy(dtype=int) // 36).astype(int)
            classes = prior.groupby("class")[["n", "sum_yards", "sum_success"]].sum()
            class_y = classes.sum_yards / classes.n
            class_s = classes.sum_success / classes.n
            prior["expected_yards"] = (prior.sum_yards + shrink_count *
                                       prior["class"].map(class_y)) / (prior.n + shrink_count)
            prior["expected_success"] = (prior.sum_success + shrink_count *
                                         prior["class"].map(class_s)) / (prior.n + shrink_count)
            # A missing fine cell still has a documented, past-only class fallback.
            missing = sorted(set(range(72)) - set(prior.index.astype(int)))
            if missing:
                extra = pd.DataFrame(index=missing)
                extra["class"] = (extra.index.to_numpy(dtype=int) // 36).astype(int)
                extra["expected_yards"] = extra["class"].map(class_y)
                extra["expected_success"] = extra["class"].map(class_s)
                prior = pd.concat([prior, extra], axis=0)
            outputs[int(year)] = prior[["expected_yards", "expected_success"]].rename_axis("context_id").reset_index()
        else:
            outputs[int(year)] = pd.DataFrame(columns=["context_id", "expected_yards", "expected_success"])
        current = by_year.get(int(year))
        if current is not None:
            cumulative = cumulative.add(current, fill_value=0)
    return outputs


def _two_sided_sums(p: pd.DataFrame, metrics: dict[str, tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    blocks = []
    for side in ("offense", "defense"):
        data = {"game_id": p.game_id, "team": p[side]}
        for name, (value, eligible) in metrics.items():
            good = eligible & value.notna()
            data[f"{side}_{name}__sum"] = value.where(good, 0).astype(float)
            data[f"{side}_{name}__n"] = good.astype(int)
        blocks.append(pd.DataFrame(data).groupby(["game_id", "team"]).sum())
    return pd.concat(blocks, axis=1).reset_index()


def residual_game_statistics(flagged: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    p = context_plays(flagged)
    if p.empty or baseline.empty:
        return pd.DataFrame(columns=["game_id", "team"])
    p = p.merge(baseline, on="context_id", how="left", validate="many_to_one")
    valid = p.expected_yards.notna() & p.expected_success.notna()
    ry = p.yards_gained - p.expected_yards
    rs = p.success.astype(float) - p.expected_success
    return _two_sided_sums(p, {
        "resid_yards": (ry, valid),
        "resid_success": (rs, valid),
        "lower_tail": (ry.lt(-5).astype(float), valid),
        "upper_tail": (ry.gt(10).astype(float), valid),
    })


def sequence_game_statistics(flagged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Count observed transitions only when a drive's order is unambiguous."""
    p = flagged.copy()
    order = ["game_id", "period", "drive_number", "play_number"]
    ambiguous = p.duplicated(order, keep=False) | p[order].isna().any(axis=1)
    p["bad_drive"] = ambiguous.groupby([p.game_id, p.drive_id]).transform("any")
    excluded = int(p.loc[p.bad_drive, ["game_id", "drive_id"]].drop_duplicates().shape[0])
    p = p.sort_values(order, kind="stable")
    p["event_position"] = p.groupby(["game_id", "drive_id"], sort=False).cumcount()
    p = p.loc[p.time_eligible & ~p.bad_drive & p.drive_id.notna()].copy()
    if p.empty:
        return pd.DataFrame(columns=["game_id", "team"]), {"excluded_ambiguous_drives": excluded}
    group = p.groupby(["game_id", "drive_id"], sort=False)
    prev_success = group.success.shift()
    prev_down = group.down.shift()
    prev_position = group.event_position.shift()
    # A skipped penalty, no-play, administrative event or unavailable snap
    # breaks the observed transition; it cannot be silently bridged.
    previous = prev_success.notna() & p.event_position.eq(prev_position + 1)
    failure = previous & ~prev_success.fillna(False).astype(bool)
    success_pair = previous & prev_success.fillna(False).astype(bool)
    third = failure & prev_down.eq(2) & p.down.eq(3)
    metrics = {
        "failure_recovery": (p.success.astype(float), failure),
        "success_pair": (p.success.astype(float), success_pair),
        "third_after_failure": (p.success.astype(float), third),
    }
    result = _two_sided_sums(p, metrics)
    drive = p.assign(pair=(success_pair & p.success).astype(int)).groupby(
        ["game_id", "drive_id", "offense", "defense"], as_index=False).agg(
            pair=("pair", "max"), plays=("id", "size"))
    drive = drive.loc[drive.plays.ge(2)]
    if not drive.empty:
        streak = _two_sided_sums(drive.rename(columns={"pair": "success"}), {
            "drive_two_success": (drive.pair.astype(float), pd.Series(True, index=drive.index))})
        result = result.merge(streak, on=["game_id", "team"], how="outer", validate="one_to_one")
    else:
        for side in ("offense", "defense"):
            result[f"{side}_drive_two_success__sum"] = 0.0
            result[f"{side}_drive_two_success__n"] = 0
    return result, {"excluded_ambiguous_drives": excluded,
                    "eligible_sequence_drives": int(p[["game_id", "drive_id"]].drop_duplicates().shape[0])}


def play_actor_lookup(flagged: pd.DataFrame) -> pd.DataFrame:
    columns = ["game_id", "id", "offense", "yards_gained", "down", "distance",
               "yards_to_goal", "play_type", "rush", "dropback", "time_eligible"]
    return flagged.loc[:, columns].rename(columns={"id": "play_id"})


def actor_game_statistics(actors: pd.DataFrame, play_lookup: pd.DataFrame,
                          *, minimum_role_events: int = 2,
                          minimum_role_coverage: float = 0.8) -> tuple[pd.DataFrame, dict]:
    """Documented rush/reception role concentration, with conflicts excluded."""
    expected = play_lookup.copy()
    reception = expected.play_type.isin(["Pass Reception", "Pass Completion", "Passing Touchdown"])
    expected["role"] = np.select([
        expected.rush & expected.down.isin([3, 4]) & expected.distance.between(1, 3),
        reception & expected.yards_to_goal.between(1, 20),
        expected.rush & expected.down.isin([1, 2]),
        reception & expected.down.isin([1, 2]),
    ], ["short_rush", "redzone_reception", "early_rush", "standard_reception"], default="")
    expected = expected.loc[expected.time_eligible & expected.role.ne("")]
    counts_expected = expected.groupby(["game_id", "offense", "role"]).size().rename(
        "expected").reset_index().rename(columns={"offense": "team"})
    a = actors.loc[actors.stat_type.isin(["Rush", "Reception"]),
                   ["game_id", "team", "play_id", "athlete_id", "stat_type", "stat"]].copy()
    a = a.drop_duplicates(["game_id", "team", "play_id", "athlete_id", "stat_type"])
    a = a.merge(play_lookup, on=["game_id", "play_id"], how="left", validate="many_to_one")
    role_ok = ((a.stat_type.eq("Rush") & a.rush.fillna(False))
               | (a.stat_type.eq("Reception") & a.dropback.fillna(False)))
    yards = pd.to_numeric(a.stat, errors="coerce")
    # The actor endpoint stores negative rushing-yard magnitudes as positive
    # values in some eras. Play outcome and sign come from /plays; this match
    # checks actor association without rewriting canonical yardage.
    same_yards = yards.eq(a.yards_gained) | (a.yards_gained.lt(0) & yards.eq(-a.yards_gained))
    provenance_ok = role_ok & a.team.eq(a.offense) & a.athlete_id.notna() & same_yards
    conflicts = int((~provenance_ok).sum())
    good = provenance_ok & a.time_eligible.fillna(False)
    a = a.loc[good].copy()
    # A play with multiple purported actors is ambiguous for role shares.
    a = a.loc[~a.duplicated(["game_id", "play_id", "stat_type"], keep=False)].copy()
    role = np.select([
        a.stat_type.eq("Rush") & a.down.isin([3, 4]) & a.distance.between(1, 3),
        a.stat_type.eq("Reception") & a.yards_to_goal.between(1, 20),
        a.stat_type.eq("Rush") & a.down.isin([1, 2]),
        a.stat_type.eq("Reception") & a.down.isin([1, 2]),
    ], ["short_rush", "redzone_reception", "early_rush", "standard_reception"], default="")
    a["role"] = role
    a = a.loc[a.role.ne("")]
    if a.empty:
        return pd.DataFrame(columns=["game_id", "team"]), {
            "conflicting_or_unverified_rows": conflicts,
            "eligible_role_events": int(len(expected)), "observed_role_events": 0,
            "supported_game_roles": 0, "excluded_low_coverage_roles": len(counts_expected)}
    counts = a.groupby(["game_id", "team", "role", "athlete_id"], as_index=False).size()
    totals = counts.groupby(["game_id", "team", "role"])["size"].transform("sum")
    counts["share"] = counts["size"] / totals
    summaries = counts.groupby(["game_id", "team", "role"]).agg(
        total=("size", "sum"), hhi=("share", lambda s: float(np.square(s).sum())),
        top_share=("share", "max")).reset_index()
    observed_events = int(summaries.total.sum())
    summaries = summaries.merge(counts_expected, on=["game_id", "team", "role"],
                                how="inner", validate="one_to_one")
    coverage = summaries.total / summaries.expected
    excluded_coverage = int((~coverage.between(minimum_role_coverage, 1.0)).sum())
    summaries = summaries.loc[summaries.total.ge(minimum_role_events)
                              & coverage.between(minimum_role_coverage, 1.0)]
    rows = []
    for row in summaries.itertuples(index=False):
        base = {"game_id": row.game_id, "team": row.team}
        for metric in ("hhi", "top_share"):
            base[f"{row.role}_{metric}__sum"] = getattr(row, metric) * row.total
            base[f"{row.role}_{metric}__n"] = row.total
        rows.append(base)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.groupby(["game_id", "team"], as_index=False).sum(min_count=1)
    return result, {"conflicting_or_unverified_rows": conflicts,
                    "eligible_role_events": int(len(expected)),
                    "observed_role_events": observed_events,
                    "supported_game_roles": len(summaries),
                    "excluded_low_coverage_roles": excluded_coverage}


def trailing_research_state(game_stats: pd.DataFrame, schedule: pd.DataFrame,
                            targets: pd.DataFrame, *, lag_hours: int = 48,
                            window: int = 12) -> pd.DataFrame:
    """Aggregate prior available games for each target team; no future rows."""
    names = sorted({c.removesuffix("__sum") for c in game_stats if c.endswith("__sum")})
    by_game = schedule.set_index("id")
    s = game_stats.loc[game_stats.game_id.isin(by_game.index)].copy()
    s["kickoff"] = pd.to_datetime(s.game_id.map(by_game.start_date), utc=True)
    s["available"] = s.kickoff + pd.Timedelta(hours=lag_hours)
    history = {str(team): block.sort_values(["available", "game_id"])
               for team, block in s.groupby("team")}
    rows = []
    for row in targets.itertuples(index=False):
        h = history.get(str(row.team))
        if h is None:
            continue
        start = pd.Timestamp(row.target_start_utc)
        prior = h.loc[h.available.lt(start)].tail(window)
        if prior.empty:
            continue
        values = {"target_game_id": int(row.target_game_id), "team": str(row.team),
                  "latest_source_game_id": int(prior.iloc[-1].game_id),
                  "latest_source_game_utc": prior.iloc[-1].kickoff,
                  "feature_available_utc": prior.iloc[-1].available,
                  "source_game_count": len(prior)}
        for name in names:
            numerator = prior[name + "__sum"].sum(min_count=1)
            denominator = prior[name + "__n"].sum(min_count=1)
            minimum = 50 if name.split("_", 1)[-1] in F13_METRICS else 10
            values[name] = (float(numerator / denominator)
                            if pd.notna(denominator) and denominator >= minimum else np.nan)
        # F16 compares the same adjusted evidence on short and long windows.
        recent = prior.tail(3)
        for side in ("offense", "defense"):
            for metric in ("resid_yards", "resid_success"):
                name = f"{side}_{metric}"
                short_n = recent.get(name + "__n", pd.Series(dtype=float)).sum()
                short_sum = recent.get(name + "__sum", pd.Series(dtype=float)).sum()
                long_n = prior.get(name + "__n", pd.Series(dtype=float)).sum()
                values[f"{side}_{metric}_recent_change"] = (
                    float(short_sum / short_n - values[name])
                    if short_n >= 30 and long_n >= 100 and np.isfinite(values.get(name, np.nan))
                    else np.nan)
            name = f"{side}_failure_recovery"
            per_game = prior[name + "__sum"] / prior[name + "__n"].replace(0, np.nan)
            values[f"{side}_recovery_game_sd"] = (
                float(per_game.dropna().std(ddof=0)) if per_game.notna().sum() >= 4 else np.nan)
        rows.append(values)
    result = pd.DataFrame(rows)
    if result.duplicated(["target_game_id", "team"]).any():
        raise ValueError("Duplicate research state for a target team")
    return result
