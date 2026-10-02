"""Observed usage continuity, without treating event absence as roster status."""
import numpy as np
import pandas as pd
from .nextgen_players import USAGE_METRICS, stable_id


def observed_usage_continuity(history, *, team, season, cutoff):
    """Compare current observed users with the same team's prior-season users.

    Before current-season usage exists, values remain missing. A past player
    absent from current observations is not labeled departed, injured, or out.
    This is a dynamic observed-use overlap, not frozen-roster retention.
    """
    when=pd.Timestamp(cutoff)
    if season>2025 or when.tzinfo is None or history.season.gt(2025).any():
        raise ValueError('Timezone-aware pre-2026 continuity source required')
    source=history.loc[history.team.eq(team) & history.available.lt(when)].copy()
    source['athlete_id']=source.athlete_id.map(stable_id)
    current=source.loc[source.season.eq(season)]
    previous=source.loc[source.season.eq(season-1)]
    result={};support={}
    for metric in USAGE_METRICS:
        now=current.loc[current.metric.eq(metric)]
        before=previous.loc[previous.metric.eq(metric)]
        valid=all(not x.empty and x.athlete_id.notna().all() and x.value.notna().all()
                  and np.isfinite(x.value).all() and x.value.ge(0).all() and x.value.sum()>0
                  for x in (now,before))
        returning=retained=np.nan
        if valid:
            n=now.groupby('athlete_id').value.sum()
            p=before.groupby('athlete_id').value.sum()
            current_ids=set(n.index[n.gt(0)])
            prior_ids=set(p.index[p.gt(0)])
            overlap=current_ids&prior_ids
            returning=float(n.reindex(list(overlap)).sum()/n.sum())
            retained=float(p.reindex(list(overlap)).sum()/p.sum())
            support[metric]={'current_observed_users':len(current_ids),'previous_observed_users':len(prior_ids),
                             'overlapping_users':len(overlap)}
        result[metric+'_observed_returning_usage_share']=returning
        result[metric+'_prior_production_share_of_observed_returners']=retained
    return result,support


def continuity_state(observations, schedule):
    from .nextgen_players import checked_player_history
    history=checked_player_history(observations,schedule)
    histories={t:g for t,g in history.groupby('team')}
    rows=[];excluded=[]
    for g in schedule.itertuples(index=False):
        if str(g.home_classification).lower()!='fbs' or str(g.away_classification).lower()!='fbs':continue
        target=pd.Timestamp(g.start_date);pair=[]
        for team in (g.home_team,g.away_team):
            h=histories.get(team)
            if h is None:break
            h=h.loc[h.season.between(g.season-1,g.season)&h.available.lt(target)]
            if h.empty:break
            values,_=observed_usage_continuity(h,team=team,season=g.season,cutoff=target)
            latest=h.sort_values(['available','game_id']).iloc[-1]
            values.update(season=int(g.season),season_type='regular',team=team,target_game_id=int(g.id),
                target_start_utc=target,feature_kind='dynamic',latest_source_game_id=int(latest.game_id),
                latest_source_game_utc=latest.kickoff,latest_source_season_type='regular',
                feature_available_utc=latest.available,static_availability_documentation=None)
            pair.append(values)
        if len(pair)==2:rows.extend(pair)
        else:excluded.append(int(g.id))
    return pd.DataFrame(rows),{'excluded_target_games':excluded,
        'scope':'observed current/prior-season workload overlap; no frozen-roster retention inference'}
