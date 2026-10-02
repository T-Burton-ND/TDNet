"""F11 prior-staff history, distinct from unverified current-coach assignment."""
import json
from pathlib import Path
import pandas as pd
from .nextgen_artifacts import NextgenFeatureBuilder
from .nextgen_coaching import coach_history_features
from .nextgen_designs import Formula, feature_record, reciprocal_counterparts
from gridiron_ml.pipeline.fetch.nextgen_acquisition import load_authoritative_schedule,sha256_file,atomic_json
ROOT=Path(__file__).resolve().parents[3]


def prior_staff_state(history,schedule):
    if history.season.gt(2025).any() or schedule.season.gt(2025).any():
        raise ValueError('Quarantined coaching state')
    if history.duplicated(['season','team']).any():
        raise ValueError('Ambiguous previous-season staff')
    starts={}
    for g in schedule.itertuples(index=False):
        for team in (g.home_team,g.away_team):
            key=(int(g.season),team);time=pd.Timestamp(g.start_date)
            starts[key]=min(starts.get(key,time),time)
    frozen={}
    for (year,team),cutoff in starts.items():
        previous=history.loc[history.season.eq(year-1)&history.team.eq(team)]
        if len(previous)!=1:continue
        coach=previous.iloc[0].coach_id
        eligible=history.loc[history.coach_id.eq(coach)&history.season.lt(year)&history.available.lt(cutoff)]
        if eligible.empty or not previous.iloc[0].available<cutoff:continue
        values=coach_history_features(history,coach,target_season=year,cutoff=cutoff)
        frozen[(year,team)]={**{'prior_staff_'+k:v for k,v in values.items()},
            'feature_available_utc':eligible.available.max(),
            'static_availability_documentation':'Previous-season unambiguous staff only; regular-game score history through prior season with 48h reconstructed reporting lag. No current-season coach assignment or season ratings used.'}
    rows=[];excluded=[]
    for g in schedule.itertuples(index=False):
        if str(g.home_classification).lower()!='fbs' or str(g.away_classification).lower()!='fbs':continue
        if not all((int(g.season),t) in frozen for t in (g.home_team,g.away_team)):
            excluded.append(int(g.id));continue
        for team in (g.home_team,g.away_team):
            rows.append({**frozen[(int(g.season),team)],'season':int(g.season),'season_type':'regular',
                         'team':team,'target_game_id':int(g.id),'target_start_utc':pd.Timestamp(g.start_date),
                         'feature_kind':'static_week0','latest_source_game_id':None,
                         'latest_source_game_utc':None,'latest_source_season_type':None})
    return pd.DataFrame(rows),excluded


class PriorStaffBuilder(NextgenFeatureBuilder):
    generation='F11'
    def __init__(self,root,state,design):
        self.family='prior_staff_'+design;self.state=state
        units={'regular_win_fraction':'fraction','regular_points_for_per_game':'points/game',
               'regular_points_against_per_game':'points/game','regular_close_win_fraction':'fraction',
               'observed_regular_games':'games','prior_program_count':'programs','regular_margin_trend':'points/game/year'}
        self.formulas=[Formula('prior_staff_'+n,('prior_staff_'+n,),'identity',
                              'Previous-season staff observed regular-season history, not a claim about current coaching personnel.',u) for n,u in units.items()]
        if design=='b':
            self.formulas.append(Formula('prior_staff_regular_net_points',('prior_staff_regular_points_for_per_game','prior_staff_regular_points_against_per_game'),'difference','Observed regular scoring balance of the previous-season staff.','points/game'))
        if design=='c':
            remove={'prior_staff_regular_win_fraction','prior_staff_regular_close_win_fraction','prior_staff_regular_points_for_per_game','prior_staff_regular_points_against_per_game'}
            self.formulas=[f for f in self.formulas if f.name not in remove]+[
                Formula('prior_staff_win_consensus',('prior_staff_regular_win_fraction','prior_staff_regular_close_win_fraction'),'mean','Equal mean of overall and close-game prior-staff regular win fractions.','fraction'),
                Formula('prior_staff_regular_net_points',('prior_staff_regular_points_for_per_game','prior_staff_regular_points_against_per_game'),'difference','Observed regular scoring balance of the previous-season staff.','points/game')]
        self.feature_columns=tuple(f.name for f in self.formulas)
        counterparts=reciprocal_counterparts(list(self.feature_columns))
        off='prior_staff_regular_points_for_per_game';defense='prior_staff_regular_points_against_per_game'
        if off in counterparts:counterparts[off]=defense;counterparts[defense]=off
        equations={'regular_win_fraction':'=(SUM(Wins)+0.5*SUM(Ties))/SUM(Games)',
                   'regular_points_for_per_game':'=SUM(PointsFor)/SUM(Games)',
                   'regular_points_against_per_game':'=SUM(PointsAgainst)/SUM(Games)',
                   'regular_close_win_fraction':'=IF(SUM(CloseGames)>0,SUM(CloseWins)/SUM(CloseGames),NA())',
                   'observed_regular_games':'=SUM(Games)','prior_program_count':'=COUNTA(UNIQUE(Programs))',
                   'regular_margin_trend':'=IF(COUNTA(UNIQUE(Years))>=3,SLOPE(AnnualRegularMargin,Years),NA())'}
        records=[]
        for f in self.formulas:
            r=feature_record(f,'F11',design,counterparts[f.name],endpoints=['/coaches/seasons','/games'],kind='static_preseason')
            r.update(availability_rule='Previous-season staff and strictly earlier completed regular scores; latest contributing kickoff plus 48h before week0',
                     aggregation_window='all acquired prior regular seasons since 2010',minimum_sample_rule='one unambiguous previous staff assignment; positive game count; trend needs three distinct years',
                     garbage_time_handling='Full regular-game scores; no postseason',source_inspiration='TDNet-derived regular-score reconstruction',
                     code_path='src/gridiron_ml/experiments/nextgen_f11.py',raw_columns=['coach.id','team.school','year','attribution_complete','home_points','away_points','start_date'],
                     provenance='TDNet-derived',input_equations={'prior_staff_'+n:e for n,e in equations.items() if 'prior_staff_'+n in f.inputs})
            if f.operation=='identity':r['equation_excel']=equations[f.name.removeprefix('prior_staff_')]
            records.append(r)
        self.manifest_path=root/'feature_families/F11'/self.family/'feature_manifest.json'
        self.manifest_path.parent.mkdir(parents=True,exist_ok=True)
        content=json.dumps(records,indent=2)+'\n'
        if self.manifest_path.exists() and self.manifest_path.read_text()!=content:raise ValueError('Version changed staff manifest')
        self.manifest_path.write_text(content)
    def build_frame(self):
        metadata=[c for c in self.state if not c.startswith('prior_staff_')]
        return pd.concat([self.state[metadata].reset_index(drop=True),pd.DataFrame({f.name:f.evaluate(self.state) for f in self.formulas}).reset_index(drop=True)],axis=1)


def main():
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    audit=json.loads((root/'results/coach_regular_history_preparation.json').read_text());path=root/'canonical/coach_regular_history.parquet'
    if sha256_file(path)!=audit['data_sha256']:raise ValueError('Coach history hash changed')
    inventory=json.loads((ROOT/'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text())
    schedule,digest=load_authoritative_schedule(root,inventory)
    if digest!=audit['schedule_sha256']:raise ValueError('Coach schedule changed')
    state,excluded=prior_staff_state(pd.read_parquet(path),schedule)
    report={'scope':'prior-staff family only; not complete F11','excluded_target_games':excluded,'designs':{}}
    for design in 'abc':
        b=PriorStaffBuilder(root,state,design);output=b.materialize(root)
        report['designs'][design]={'path':str(output),'rows':len(state),'features':len(b.feature_columns)}
    atomic_json(root/'results/f11_prior_staff_materialization.json',report)
    print(json.dumps(report['designs'],indent=2))


if __name__=='__main__':main()
