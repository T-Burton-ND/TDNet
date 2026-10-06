#!/usr/bin/env python3
"""Build 2026 F11 prior-staff states and train/predict the F11-A roster."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'scripts/publication')]
from build_2026_f09_whatif import DATA,ROUND_DATA,read_schedule,target_matrix
from nextgen_rounds_train import load_stage_matrix
from nextgen_rounds_scientific_train import make_model
from gridiron_ml.experiments.nextgen_screening_reduced_parallel import build_estimator,residual_probability

ARCHIVE=Path('/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/canonical/coach_regular_history.parquet')
OUT=DATA/'f11_predictions'
FAMILY={'M1':'linear','M2':'spline','M3':'tree','M4':'boosted','M5':'neural','M10':'knn'}

def f11_state():
    h=pd.read_parquet(ARCHIVE)
    if h.duplicated(['season','team']).any() or h.season.max()!=2025:
        raise ValueError('Prior-staff archive is not a unique through-2025 source')
    games=pd.read_parquet(ROOT/'data/raw/cfbd/v2/games/2026.parquet')
    games=games.loc[games.season_type.astype(str).str.lower().eq('regular') & games.completed.fillna(False).astype(bool)
                    & games.home_classification.astype(str).str.lower().eq('fbs')
                    & games.away_classification.astype(str).str.lower().eq('fbs')].copy()
    games['kickoff']=pd.to_datetime(games.start_date,utc=True)
    starts={}
    for g in games.itertuples(index=False):
        for team in (g.home_team,g.away_team): starts[team]=min(starts.get(team,g.kickoff),g.kickoff)
    frozen={}
    for team,cutoff in starts.items():
        prev=h.loc[h.season.eq(2025)&h.team.eq(team)]
        prior=pd.DataFrame()
        if len(prev)==1:
            coach=prev.iloc[0].coach_id
            prior=h.loc[h.coach_id.eq(coach)&h.season.lt(2026)&h.available.lt(cutoff)].sort_values('season')
            if not prev.iloc[0].available<cutoff: prior=pd.DataFrame()
        if prior.empty:
            frozen[team]={**{n:np.nan for n in (
                'prior_staff_regular_win_fraction','prior_staff_regular_points_for_per_game',
                'prior_staff_regular_points_against_per_game','prior_staff_regular_close_win_fraction',
                'prior_staff_observed_regular_games','prior_staff_prior_program_count','prior_staff_regular_margin_trend')},
                'staff_available_utc':pd.NaT}
            continue
        gp=prior.games.sum(); close=prior.close_games.sum()
        margins=(prior.points_for-prior.points_against)/prior.games
        years=prior.season.to_numpy(float)
        frozen[team]={
            'prior_staff_regular_win_fraction':float((prior.wins.sum()+.5*prior.ties.sum())/gp),
            'prior_staff_regular_points_for_per_game':float(prior.points_for.sum()/gp),
            'prior_staff_regular_points_against_per_game':float(prior.points_against.sum()/gp),
            'prior_staff_regular_close_win_fraction':float(prior.close_wins.sum()/close) if close else np.nan,
            'prior_staff_observed_regular_games':int(gp),
            'prior_staff_prior_program_count':int(prior.team.nunique()),
            'prior_staff_regular_margin_trend':float(np.polyfit(years-years.min(),margins,1)[0]) if len(np.unique(years))>=3 else np.nan,
            'staff_available_utc':prior.available.max(),
        }
    rows=[]
    for g in games.itertuples(index=False):
        for team in (g.home_team,g.away_team):
            rows.append({**frozen[team],'target_game_id':int(g.id),'team':team,'target_start_utc':g.kickoff})
    state=pd.DataFrame(rows)
    if state.duplicated(['target_game_id','team']).any() or state.target_game_id.nunique()!=271 or len(state)!=542:
        raise ValueError('F11 target feature keys incomplete')
    available=state.staff_available_utc.notna()
    if not (pd.to_datetime(state.loc[available,'staff_available_utc'],utc=True)
            < pd.to_datetime(state.loc[available,'target_start_utc'],utc=True)).all():
        raise ValueError('F11 staff source is not available before target kickoff')
    return state

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'f11_target_state.parquet'
    state=pd.read_parquet(p) if p.exists() else f11_state()
    if not p.exists(): state.to_parquet(p,index=False,compression='zstd')
    s9=pd.read_parquet(DATA/'f09_predictions/f09_target_state.parquet')
    s10=pd.read_parquet(DATA/'f10_features/f10_target_state.parquet')
    names=[c for c in state if c.startswith('prior_staff_')]
    state=s9.merge(s10[['target_game_id','team',*[c for c in s10 if c.startswith(('pass_attempts_','rush_attempts_','receptions_','tackles_','field_goal_attempts_','punts_','kick_returns_','punt_returns_','recruit_history_'))]]],on=['target_game_id','team'],validate='one_to_one')
    state=state.merge(pd.read_parquet(p)[['target_game_id','team',*names]],on=['target_game_id','team'],validate='one_to_one')
    xh,meta,_,evidence=load_stage_matrix(ROUND_DATA,'F11')
    xt,tm,sp=target_matrix(state,evidence)
    cols=[f'matchup__{n}' for n in evidence['source_features']]
    X=pd.DataFrame(xh,columns=cols); T=pd.DataFrame(xt,columns=cols); y=meta.next_game_margin.to_numpy(float)
    cfg=json.loads((ROOT/'configs/experiments/nextgen_rounds_scientific_models_v1.json').read_text())
    points=json.loads((ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    path=OUT/'predictions.parquet'; rows=pd.read_parquet(path).to_dict('records') if path.exists() else []
    done={r['model_name'].rsplit('_',1)[-1] for r in rows}
    family_to_correct={a:FAMILY[a] for a in FAMILY}
    for r in rows:r['model_family']=family_to_correct.get(r['model_name'].rsplit('_',1)[-1],r['model_family'])
    for arch in ('M1','M2','M3','M4','M5','M10'):
        if arch in done: continue
        specs=[('setpoint',z) for z in points[arch]] if arch in ('M2','M4') else [('seed',z) for z in ([1701] if arch=='M3' else cfg['seeds'])]
        individual=[]
        for kind,spec in specs:
            if kind=='setpoint':
                saved=pd.read_parquet(ROUND_DATA/'experiments/F11'/arch/spec['id']/'predictions.parquet')
                residuals=(saved.actual_margin-saved.predicted_margin).to_numpy(float)
                model=build_estimator(arch,spec); model.fit(X,y); pred=np.asarray(model.predict(T),float)
            else:
                seed=int(spec); saved=pd.read_parquet(ROUND_DATA/f'scientific_model_runs/experiments/F11/{arch}/seed_{seed}/predictions.parquet')
                residuals=(saved.actual_margin-saved.predicted_margin).to_numpy(float)
                model,_=make_model(arch,seed,'F11'); model.train(X,y); pred=np.asarray(model.predict_margin(T),float).reshape(-1)
            individual.append((pred,residual_probability(pred,residuals)))
        pred=np.mean([z[0] for z in individual],axis=0); prob=np.mean([z[1] for z in individual],axis=0)
        for i,g in tm.reset_index(drop=True).iterrows():
            rows.append({'game_id':int(g.target_game_id),'season':2026,'week':int(g.week),'home_team':g.home_team,'away_team':g.away_team,
                         'model_name':f'scientific_F11_{arch}','model_family':FAMILY[arch],'fingerprint':'F11',
                         'pred_home_margin':float(pred[i]),'pred_home_win_probability':float(prob[i]),
                         'market_spread_close':float(sp[i]) if np.isfinite(sp[i]) else np.nan,'market_win_probability':np.nan})
        pd.DataFrame(rows).to_parquet(path,index=False,compression='zstd');done.add(arch);print('F11 saved',arch,flush=True)
    receipt={'status':'success' if len(done)==6 else 'partial','generation':'F11 prior staff A','models':sorted(done),
             'games':int(tm.target_game_id.nunique()),'features':len(names),'training_through_season':2025,
             'calibration_years':[2024,2025],'market_features_used':False,'no_current_2026_staff_assignment_used':True}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
