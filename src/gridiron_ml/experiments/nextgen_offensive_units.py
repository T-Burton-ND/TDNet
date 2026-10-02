"""Observed offensive-room PPA states, without asserting current membership."""
import numpy as np
import pandas as pd

ROOMS={'qb': 'average_p_p_a.pass', 'rb': 'average_p_p_a.rush', 'wrte': 'average_p_p_a.pass'}


def offensive_unit_state(bound, schedule, *, window=12, minimum_games=3):
    """Equal-player/game PPA mean and dispersion in each observed source room.

    The provider supplies per-player averages without attempt counts, so these
    are explicitly NOT pooled per-play efficiency or participation weights.
    QB passing, RB rushing, and WR/TE receiving outcomes have shared causation;
    room labels do not identify isolated individual causal contributions.
    """
    if window<1 or minimum_games<1:
        raise ValueError('Positive window and support required')
    if (schedule.season.gt(2025).any() or bound.season.gt(2025).any()
            or not schedule.season_type.eq('regular').all() or not schedule.completed.eq(True).all()):
        raise ValueError('Pre-2026 completed regular history required')
    if bound.duplicated(['game_id','team','athlete_id']).any():
        raise ValueError('Duplicate observed player-game')
    # Validate the supplied intermediate against schedule again, not just metadata.
    games=schedule.set_index('id')
    for row in bound[['game_id','team','kickoff','available']].drop_duplicates().itertuples(index=False):
        if row.game_id not in games.index:
            raise ValueError('Unscheduled offensive-unit source')
        g=games.loc[row.game_id]
        kickoff=pd.Timestamp(g.start_date)
        if (row.team not in (g.home_team,g.away_team) or row.kickoff!=kickoff
                or row.available!=kickoff+pd.Timedelta(hours=48)):
            raise ValueError('Offensive-unit source timing or participant mismatch')
    histories={team:part for team,part in bound.groupby('team')}
    rows=[];excluded=[]
    for g in schedule.itertuples(index=False):
        if str(g.home_classification).lower()!='fbs' or str(g.away_classification).lower()!='fbs':
            continue
        target=pd.Timestamp(g.start_date);pair=[]
        for team in (g.home_team,g.away_team):
            history=histories.get(team)
            if history is None:
                break
            prior=history.loc[history.available.lt(target)]
            selected=prior[['game_id','kickoff','available']].drop_duplicates().sort_values(['available','game_id']).tail(window)
            if selected.empty:
                break
            prior=prior.loc[prior.game_id.isin(selected.game_id)]
            values={}
            for room,column in ROOMS.items():
                observed=prior.loc[prior.unit.eq(room),['game_id',column]].replace([np.inf,-np.inf],np.nan).dropna()
                enough=observed.game_id.nunique()>=minimum_games
                ppa=observed[column]
                values[room+'_observed_player_ppa_mean']=float(ppa.mean()) if enough else np.nan
                values[room+'_observed_player_ppa_dispersion']=float(ppa.std(ddof=0)) if enough else np.nan
            latest=selected.iloc[-1]
            values.update(season=int(g.season),season_type='regular',team=team,target_game_id=int(g.id),
                target_start_utc=target,feature_kind='dynamic',latest_source_game_id=int(latest.game_id),
                latest_source_game_utc=latest.kickoff,latest_source_season_type='regular',
                feature_available_utc=latest.available,static_availability_documentation=None)
            pair.append(values)
        if len(pair)==2:
            rows.extend(pair)
        else:
            excluded.append(int(g.id))
    return pd.DataFrame(rows),{'excluded_target_games':excluded,'window_games':window,'minimum_games':minimum_games,
        'weighting':'equal observed player-game averages, not equal plays',
        'position_semantics':'historical source-game role, not target roster membership'}
