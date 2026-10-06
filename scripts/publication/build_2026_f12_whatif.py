#!/usr/bin/env python3
"""Build F12 corrected-A states and fit 2026 scientific what-if forecasts."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'scripts/publication')]
from build_2026_f09_whatif import DATA,ROUND_DATA,read_schedule,target_matrix
from gridiron_ml.experiments.nextgen_players import flatten_player_boxes
from gridiron_ml.experiments.nextgen_f12 import RATIOS,special_team_games
from gridiron_ml.experiments.nextgen_units import POSITION_UNIT
from nextgen_rounds_train import load_stage_matrix
from nextgen_rounds_scientific_train import make_model
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import build_estimator,residual_probability

ARCHIVE=Path('/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/canonical')
OUT=DATA/'f12_features'
PRED=DATA/'f12_predictions'
FAMILY={'M1':'linear','M2':'spline','M3':'tree','M4':'boosted','M5':'neural','M10':'knn'}

def _fbs_names():
    t=pd.read_parquet(ROOT/'data/raw/cfbd/v2/teams_fbs/2026.parquet')
    return set(t.loc[t.classification.astype(str).str.lower().eq('fbs'),'school'].astype(str))

def observations(schedule):
    files=sorted((ARCHIVE/'player_observations').glob('*.parquet'))
    old=pd.concat([pd.read_parquet(p,columns=['game_id','team','athlete_id','metric','value']) for p in files],ignore_index=True)
    current=[]
    for p in (DATA/'raw_cache/games_players').glob('*.parquet'):
        frame=pd.read_parquet(p); obs,_=flatten_player_boxes(frame)
        if not obs.empty: current.append(obs)
    source=pd.concat([old,*current],ignore_index=True)
    games=schedule.loc[schedule.season_type.astype(str).str.lower().eq('regular')&schedule.completed.fillna(False),
                       ['id','season','week','kickoff','home_team','away_team']]
    source=source.merge(games,left_on='game_id',right_on='id',how='left',validate='many_to_one')
    if source.season.isna().any(): raise ValueError('Player box source has no completed-game schedule match')
    aliases={'Savannah State':'Savannah St','Saint Francis':'St. Francis (PA)'}
    source['team']=source.team.replace(aliases)
    ok=source.team.eq(source.home_team)|source.team.eq(source.away_team)
    if ((~ok)&source.team.isin(_fbs_names())).any(): raise ValueError('Unmatched FBS player team name')
    source=source.loc[ok].copy()
    source['available']=source.kickoff+pd.Timedelta(hours=48)
    return source[['game_id','team','athlete_id','metric','value','season','kickoff','available']]

def _special_state(obs,schedule,targets):
    wanted={m for pair in RATIOS.values() for m in pair[:2]}
    stats=special_team_games(obs[['game_id','team','metric','value']])
    joined=stats.merge(schedule[['id','kickoff','home_team','away_team']],left_on='game_id',right_on='id',how='left',validate='many_to_one')
    if joined.id.isna().any() or not (joined.team.eq(joined.home_team)|joined.team.eq(joined.away_team)).all():
        raise ValueError('Unscheduled special-team game source')
    joined['available']=joined.kickoff+pd.Timedelta(hours=48)
    histories={t:g.sort_values(['available','game_id']) for t,g in joined.groupby('team')}
    rows=[]
    for g in targets.itertuples(index=False):
        for team in (g.home_team,g.away_team):
            h=histories.get(team)
            prior=h.loc[h.available.lt(g.kickoff)].tail(12) if h is not None else pd.DataFrame()
            values={}
            for name,(num,den,_) in RATIOS.items():
                valid=prior[[num,den]].replace([np.inf,-np.inf],np.nan).dropna() if not prior.empty else pd.DataFrame()
                valid=valid.loc[valid[den].ge(0)] if not valid.empty else valid
                count=valid[den].sum() if not valid.empty else 0
                values[name]=float(valid[num].sum()/count) if count>=10 else np.nan
            latest=prior.iloc[-1] if not prior.empty else None
            values.update(target_game_id=int(g.id),team=team,
                          latest_source_available_utc=latest.available if latest is not None else pd.NaT,
                          latest_source_game_id=int(latest.game_id) if latest is not None else None,
                          target_start_utc=g.kickoff)
            rows.append(values)
    return pd.DataFrame(rows)

def _bind_current_ppa(schedule):
    raw=[]
    for p in (DATA/'raw_cache/ppa_players_games').glob('*.parquet'):
        d=pd.read_parquet(p).rename(columns={'seasonType':'season_type','averagePPA.all':'average_p_p_a.all',
            'averagePPA.pass':'average_p_p_a.pass','averagePPA.rush':'average_p_p_a.rush'})
        raw.append(d)
    source=pd.concat(raw,ignore_index=True).drop_duplicates()
    mapping=[]
    games=schedule.loc[schedule.season_type.astype(str).str.lower().eq('regular')&schedule.completed.fillna(False)]
    for g in games.itertuples(index=False):
        mapping.extend([{'season':int(g.season),'week':int(g.week),'team':g.home_team,'opponent':g.away_team,'game_id':int(g.id),'kickoff':g.kickoff},
                        {'season':int(g.season),'week':int(g.week),'team':g.away_team,'opponent':g.home_team,'game_id':int(g.id),'kickoff':g.kickoff}])
    m=pd.DataFrame(mapping);keys=['season','week','team','opponent']; unique=m.loc[~m.duplicated(keys,keep=False)]
    source['season']=pd.to_numeric(source.season,errors='coerce'); source['week']=pd.to_numeric(source.week,errors='coerce')
    source['athlete_id']=source.id.astype(str).replace({'nan':np.nan,'None':np.nan})
    bound=source.merge(unique,on=keys,how='inner',validate='many_to_one')
    bound=bound.loc[bound.athlete_id.notna()].copy()
    for c in ('average_p_p_a.all','average_p_p_a.pass','average_p_p_a.rush'):
        bound[c]=pd.to_numeric(bound[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    ident=['game_id','team','athlete_id']; conflicts=bound.duplicated(ident,keep=False)
    bound=bound.loc[~conflicts].copy()
    bound['unit']=bound.position.astype(str).str.strip().str.upper().map(POSITION_UNIT)
    bound['available']=bound.kickoff+pd.Timedelta(hours=48)
    historic=pd.read_parquet(ARCHIVE/'player_ppa_games.parquet')
    return pd.concat([historic,bound],ignore_index=True,sort=False),{'raw_2026_rows':len(source),'bound_2026_rows':len(bound),'conflicting_rows_excluded':int(conflicts.sum())}

def _ppa_state(bound,targets):
    histories={t:g for t,g in bound.groupby('team')}; rooms={'qb':'average_p_p_a.pass','rb':'average_p_p_a.rush','wrte':'average_p_p_a.pass'}; rows=[]
    for g in targets.itertuples(index=False):
        for team in (g.home_team,g.away_team):
            h=histories.get(team); prior=h.loc[h.available.lt(g.kickoff)] if h is not None else pd.DataFrame()
            selected=(prior[['game_id','available']].drop_duplicates().sort_values(['available','game_id']).tail(12)
                      if not prior.empty else pd.DataFrame())
            use=prior.loc[prior.game_id.isin(selected.game_id)] if not selected.empty else pd.DataFrame()
            v={}
            for room,col in rooms.items():
                q=use.loc[use.unit.eq(room),['game_id',col]].replace([np.inf,-np.inf],np.nan).dropna() if not use.empty else pd.DataFrame()
                enough=q.game_id.nunique()>=3
                v[room+'_observed_player_ppa_mean']=float(q[col].mean()) if enough else np.nan
                v[room+'_observed_player_ppa_dispersion']=float(q[col].std(ddof=0)) if enough else np.nan
            latest=selected.iloc[-1] if not selected.empty else None
            v.update(target_game_id=int(g.id),team=team,
                     latest_source_available_utc=latest.available if latest is not None else pd.NaT,
                     latest_source_game_id=int(latest.game_id) if latest is not None else None,
                     target_start_utc=g.kickoff)
            rows.append(v)
    return pd.DataFrame(rows)

def build_features():
    schedule=__import__('build_2026_f09_whatif').read_schedule()
    schedule=schedule.loc[schedule.season_type.astype(str).str.lower().eq('regular')&schedule.completed.fillna(False)].copy()
    schedule['kickoff']=pd.to_datetime(schedule.start_date,utc=True)
    targets=pd.read_parquet(ROOT/'data/raw/cfbd/v2/games/2026.parquet')
    targets=targets.loc[targets.season_type.astype(str).str.lower().eq('regular')&targets.completed.fillna(False)
                        &targets.home_classification.astype(str).str.lower().eq('fbs')
                        &targets.away_classification.astype(str).str.lower().eq('fbs')].copy()
    targets['kickoff']=pd.to_datetime(targets.start_date,utc=True)
    obs=observations(schedule)
    st=_special_state(obs,schedule,targets)
    ppa,audit=_bind_current_ppa(schedule)
    units=_ppa_state(ppa,targets)
    state=st.merge(units,on=['target_game_id','team','target_start_utc'],how='inner',validate='one_to_one',suffixes=('_st','_ppa'))
    feature_names=[r['name'] for r in json.loads(Path('/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/fingerprints/F12_F_a/feature_manifest.json').read_text()) if r['generation']=='F12']
    if len(feature_names)!=13 or set(feature_names)-set(state): raise ValueError('F12 corrected A source features incomplete')
    if len(state)!=542 or state.duplicated(['target_game_id','team']).any(): raise ValueError('F12 target states incomplete')
    state.to_parquet(OUT/'f12_target_state.parquet',index=False,compression='zstd')
    receipt={'status':'features_ready','generation':'F12 corrected A','target_games':271,'target_team_rows':len(state),
        'feature_count':len(feature_names),'reporting_lag_hours':48,'target_game_excluded':True,'market_features_used':False,
        'ppa_source_audit':audit,'special_teams_available_rows':int(state.latest_source_available_utc_st.notna().sum()),
        'ppa_available_rows':int(state.latest_source_available_utc_ppa.notna().sum())}
    (OUT/'features_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return state,feature_names

def main():
    OUT.mkdir(parents=True,exist_ok=True);PRED.mkdir(parents=True,exist_ok=True)
    state,feature_names=build_features()
    states=[pd.read_parquet(DATA/'f09_predictions/f09_target_state.parquet'),
            pd.read_parquet(DATA/'f10_features/f10_target_state.parquet'),
            pd.read_parquet(DATA/'f11_predictions/f11_target_state.parquet')]
    f10_names=[r['name'] for r in json.loads(Path('/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/fingerprints/F12_F_a/feature_manifest.json').read_text()) if r['generation']=='F10']
    f11_names=[c for c in states[2] if c.startswith('prior_staff_')]
    state=states[0].merge(states[1][['target_game_id','team',*f10_names]],on=['target_game_id','team'],validate='one_to_one')
    state=state.merge(states[2][['target_game_id','team',*f11_names]],on=['target_game_id','team'],validate='one_to_one')
    state=state.merge(pd.read_parquet(OUT/'f12_target_state.parquet')[['target_game_id','team',*feature_names]],on=['target_game_id','team'],validate='one_to_one')
    xh,meta,_,evidence=load_stage_matrix(ROUND_DATA,'F12_corrected');xt,tm,sp=target_matrix(state,evidence)
    cols=[f'matchup__{n}' for n in evidence['source_features']];X=pd.DataFrame(xh,columns=cols);T=pd.DataFrame(xt,columns=cols);y=meta.next_game_margin.to_numpy(float)
    cfg=json.loads((ROOT/'configs/experiments/nextgen_rounds_scientific_models_v1.json').read_text());points=json.loads((ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    path=PRED/'predictions.parquet';rows=pd.read_parquet(path).to_dict('records') if path.exists() else [];done={r['model_name'].rsplit('_',1)[-1] for r in rows}
    for r in rows:r['model_family']=FAMILY.get(r['model_name'].rsplit('_',1)[-1],r['model_family'])
    for arch in ('M1','M2','M3','M4','M5','M10'):
        if arch in done:continue
        print('F12 corrected refit',arch,flush=True)
        specs=[('setpoint',p) for p in points[arch]] if arch in ('M2','M4') else [('seed',s) for s in ([1701] if arch=='M3' else cfg['seeds'])]
        predprob=[]
        for kind,spec in specs:
            if kind=='setpoint':
                saved=pd.read_parquet(ROUND_DATA/'experiments/F12_corrected'/arch/spec['id']/'predictions.parquet');res=(saved.actual_margin-saved.predicted_margin).to_numpy(float)
                model=build_estimator(arch,spec);model.fit(X,y);pred=np.asarray(model.predict(T),float)
            else:
                seed=int(spec);saved=pd.read_parquet(ROUND_DATA/f'scientific_model_runs/experiments/F12_corrected/{arch}/seed_{seed}/predictions.parquet');res=(saved.actual_margin-saved.predicted_margin).to_numpy(float)
                model,_=make_model(arch,seed,'F12_corrected');model.train(X,y);pred=np.asarray(model.predict_margin(T),float).reshape(-1)
            predprob.append((pred,residual_probability(pred,res)))
        pred=np.mean([z[0] for z in predprob],axis=0);prob=np.mean([z[1] for z in predprob],axis=0)
        for i,g in tm.reset_index(drop=True).iterrows():
            rows.append({'game_id':int(g.target_game_id),'season':2026,'week':int(g.week),'home_team':g.home_team,'away_team':g.away_team,
                'model_name':f'scientific_F12_{arch}','model_family':FAMILY[arch],'fingerprint':'F12','pred_home_margin':float(pred[i]),
                'pred_home_win_probability':float(prob[i]),'market_spread_close':float(sp[i]) if np.isfinite(sp[i]) else np.nan,'market_win_probability':np.nan})
        pd.DataFrame(rows).to_parquet(path,index=False,compression='zstd');done.add(arch);print('F12 saved',arch,flush=True)
    receipt={'status':'success' if len(done)==6 else 'partial','generation':'F12 corrected A','models':sorted(done),'games':int(tm.target_game_id.nunique()),
        'features':len(feature_names),'training_through_season':2025,'calibration_years':[2024,2025],'historical_calibration_models_trained_through':2023,
        'market_features_used':False,'M3_replicates':1,'other_seed_replicates':3,'M2_M4_setpoints':10}
    (PRED/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
