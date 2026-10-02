"""Usage-weighted observed experience, independent of ambiguous roster class year.

Counts mean games with a finite player box-score record in acquired regular
history. They are neither career games played nor proof of participation/snaps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .nextgen_players import USAGE_METRICS, stable_id


def appearance_index(checked_history):
    """Index unique stable-ID game records by their checked availability time."""
    required={'athlete_id','game_id','season','available','value'}
    if not required<=set(checked_history):
        raise ValueError('Expected schedule-checked player history')
    if checked_history.season.gt(2025).any():
        raise ValueError('Quarantined experience source')
    history=checked_history.copy()
    history['athlete_id']=history.athlete_id.map(stable_id)
    history=history.loc[history.athlete_id.notna() & np.isfinite(history.value)]
    if history.available.isna().any():
        raise ValueError('Missing appearance availability')
    observations=history[['athlete_id','game_id','available']].drop_duplicates()
    if observations.duplicated(['athlete_id','game_id']).any():
        raise ValueError('Conflicting availability for a player-game record')
    # Pandas may store datetimes in us or ns. Timestamp.value is always ns,
    # matching the cutoff scalar used by searchsorted below.
    return {pid:group.available.sort_values().map(lambda t: pd.Timestamp(t).value).to_numpy(dtype=np.int64)
            for pid,group in observations.groupby('athlete_id')}


def usage_weighted_experience(prior_usage, appearances, *, cutoff):
    """Weight earlier observed-game counts by pre-target actual count shares.

    `prior_usage` is the selected team's trailing available regular-game window.
    Unknown IDs/counts leave the metric missing, never a fictional freshman zero.
    The returned latest availability includes every contributing player's past
    record, even when that record was for a different team before a transfer.
    """
    when=pd.Timestamp(cutoff)
    if when.tzinfo is None or when.year>2025:
        raise ValueError('Timezone-aware pre-2026 target cutoff required')
    if prior_usage.available.isna().any() or not prior_usage.available.lt(when).all():
        raise ValueError('Usage includes target or unavailable game')
    usage=prior_usage.copy();usage['athlete_id']=usage.athlete_id.map(stable_id)
    output={};latest=None
    for metric in USAGE_METRICS:
        rows=usage.loc[usage.metric.eq(metric)]
        valid=(not rows.empty and rows.athlete_id.notna().all() and rows.value.notna().all()
               and np.isfinite(rows.value).all() and rows.value.ge(0).all() and rows.value.sum()>0)
        key=metric+'_weighted_observed_games'
        if not valid:
            output[key]=np.nan;continue
        counts=rows.groupby('athlete_id').value.sum()
        weighted=0.;metric_latest=None
        for pid,weight in counts.items():
            if weight==0:continue
            dates=appearances.get(pid)
            n=int(np.searchsorted(dates,when.value,side='left')) if dates is not None else 0
            if not n:
                valid=False;break
            weighted+=float(weight)*n
            stamp=pd.Timestamp(int(dates[n-1]),tz='UTC')
            metric_latest=max(metric_latest,stamp) if metric_latest is not None else stamp
        output[key]=weighted/float(counts.sum()) if valid else np.nan
        if valid and metric_latest is not None:
            latest=max(latest,metric_latest) if latest is not None else metric_latest
    return output,latest


def observed_experience_state(observations, schedule, *, window=12):
    """Freeze observed history at the team's latest available game, then weight it.

    Player records from prior teams count, but records later than the team's
    latest source game do not. This keeps the entire state at one explicit cutoff.
    """
    from .nextgen_players import checked_player_history
    if window < 1:
        raise ValueError('Positive history window required')
    history = checked_player_history(observations, schedule)
    appearances = appearance_index(history)
    teams = {t: h for t, h in history.groupby('team')}
    rows, excluded = [], []
    for game in schedule.itertuples(index=False):
        if str(game.home_classification).lower() != 'fbs' or str(game.away_classification).lower() != 'fbs':
            continue
        target = pd.Timestamp(game.start_date)
        pair = []
        for team in (game.home_team, game.away_team):
            h = teams.get(team)
            if h is None:
                break
            prior = h.loc[h.available.lt(target)]
            games = prior[['game_id', 'kickoff', 'available']].drop_duplicates().sort_values(['available', 'game_id']).tail(window)
            if games.empty:
                break
            latest = games.iloc[-1]
            prior = prior.loc[prior.game_id.isin(games.game_id)]
            values, _ = usage_weighted_experience(prior, appearances, cutoff=latest.available + pd.Timedelta(nanoseconds=1))
            values.update(season=int(game.season), season_type='regular', team=team,
                target_game_id=int(game.id), target_start_utc=target, feature_kind='dynamic',
                latest_source_game_id=int(latest.game_id), latest_source_game_utc=latest.kickoff,
                latest_source_season_type='regular', feature_available_utc=latest.available,
                static_availability_documentation=None)
            pair.append(values)
        if len(pair) == 2:
            rows.extend(pair)
        else:
            excluded.append(int(game.id))
    return pd.DataFrame(rows), {'excluded_target_games': excluded, 'window_games': window,
        'experience_scope': 'acquired finite player-game records, not career participation or class year'}
