"""Lagged recruiting cohorts, explicitly distinct from current-roster talent."""
import numpy as np
import pandas as pd
from .nextgen_units import POSITION_UNIT

# Recruiting-specific labels from structured position codes.
RECRUIT_UNITS={**POSITION_UNIT,'PRO':'qb','DUAL':'qb','APB':'rb','ATH':None,
               'SDE':'front','WDE':'front','OC':'ol'}


def recruiting_cohort_state(recruits,schedule,*,window=4):
    """Freeze preceding recruiting classes; do not infer enrollment or retention.

    Availability is reconstructed at Jan 1 after the newest contributing class,
    not an assertion of an archived provider snapshot. Current-class records are
    excluded even if they would usually have been signed before the season.
    """
    if recruits.year.gt(2025).any() or schedule.season.gt(2025).any():
        raise ValueError('Quarantined recruiting source')
    if window<1:raise ValueError('Positive cohort window required')
    source=recruits.loc[recruits.recruit_type.eq('HighSchool')].copy()
    if source.duplicated(['year','id']).any():raise ValueError('Duplicate recruiting record')
    source['unit']=source.position.astype(str).str.upper().map(RECRUIT_UNITS)
    source['rating']=pd.to_numeric(source.rating,errors='coerce')
    source['stars']=pd.to_numeric(source.stars,errors='coerce')
    source.loc[~source.rating.between(0,1),'rating']=np.nan
    source.loc[~source.stars.between(0,5),'stars']=np.nan
    starts={}
    for game in schedule.itertuples(index=False):
        for team in (game.home_team,game.away_team):
            key=(int(game.season),team);when=pd.Timestamp(game.start_date)
            starts[key]=min(starts.get(key,when),when)
    frozen={};audit=[]
    units=('ol','qb','rb','wrte','front','lb','secondary','special_teams')
    for (year,team),cutoff in starts.items():
        cohort=source.loc[source.year.between(year-window,year-1)&source.committed_to.eq(team)]
        if cohort.empty:continue
        available=pd.Timestamp(year=int(cohort.year.max())+1,month=1,day=1,tz='UTC')
        if available>=cutoff:continue
        values={}
        for unit in units:
            group=cohort.loc[cohort.unit.eq(unit)]
            ratings=group.rating.dropna();stars=group.stars.dropna()
            values['recruit_history_'+unit+'_rating']=float(ratings.mean()) if len(ratings)>=3 else np.nan
            values['recruit_history_'+unit+'_bluechip_share']=float(stars.ge(4).mean()) if len(stars)>=3 else np.nan
        values.update(feature_available_utc=available,
            static_availability_documentation='Prior recruiting classes only, frozen Jan 1 after newest contributing class; reconstructed historical availability, not archived provider publication. Commitments are not current roster membership.')
        frozen[(year,team)]=values
        audit.append({'season':year,'team':team,'records':len(cohort),'class_min':int(cohort.year.min()),
                      'class_max':int(cohort.year.max()),'unassigned_position_records':int(cohort.unit.isna().sum())})
    rows=[];excluded=[]
    for g in schedule.itertuples(index=False):
        if str(g.home_classification).lower()!='fbs' or str(g.away_classification).lower()!='fbs':continue
        if not all((int(g.season),t) in frozen for t in (g.home_team,g.away_team)):
            excluded.append(int(g.id));continue
        for team in (g.home_team,g.away_team):
            rows.append({**frozen[(int(g.season),team)],'season':int(g.season),'season_type':'regular','team':team,
                'target_game_id':int(g.id),'target_start_utc':pd.Timestamp(g.start_date),'feature_kind':'static_week0',
                'latest_source_game_id':None,'latest_source_game_utc':None,'latest_source_season_type':None})
    return pd.DataFrame(rows),{'cohorts':audit,'excluded_target_games':excluded,'window_classes':window}


from pathlib import Path
import json
from .nextgen_artifacts import NextgenFeatureBuilder
from .nextgen_designs import Formula,feature_record
from .nextgen_f09 import verified_endpoint_records
from gridiron_ml.pipeline.fetch.nextgen_acquisition import load_authoritative_schedule,atomic_json
ROOT=Path(__file__).resolve().parents[3]


class RecruitingHistoryBuilder(NextgenFeatureBuilder):
    generation='F10'
    def __init__(self,root,state,design):
        self.family='recruit_history_'+design;self.state=state
        names=[c for c in state if c.startswith('recruit_history_')]
        self.formulas=[]
        if design=='c':
            for unit in sorted({n.removeprefix('recruit_history_').removesuffix('_rating') for n in names if n.endswith('_rating')}):
                inputs=(f'recruit_history_{unit}_rating',f'recruit_history_{unit}_bluechip_share')
                self.formulas.append(Formula(f'recruit_history_{unit}_consensus',inputs,'mean','Equal-weight past-cohort average rating and four/five-star share; not current-roster talent.','fraction'))
        else:
            self.formulas=[Formula(n,(n,),'identity','Prior high-school recruiting cohort rating or blue-chip share, without assuming enrollment or retention.','fraction') for n in names]
            if design=='b':
                self.formulas.append(Formula('recruit_history_ol_qb_interaction',('recruit_history_ol_rating','recruit_history_qb_rating'),'product','Past offensive-line and quarterback cohort rating interaction; not a current-lineup assertion.','fraction'))
        self.feature_columns=tuple(f.name for f in self.formulas)
        records=[]
        for f in self.formulas:
            equations={n:('=IF(COUNT(Ratings)>=3,AVERAGE(Ratings),NA())' if n.endswith('_rating') else '=IF(COUNT(Stars)>=3,COUNTIF(Stars,">=4")/COUNT(Stars),NA())') for n in f.inputs}
            r=feature_record(f,'F10',design,f.name,endpoints=['/recruiting/players'],kind='static_preseason')
            r.update(raw_columns=['id','year','recruit_type','committed_to','position','rating','stars'],
                availability_rule='Prior classes only; reconstructed Jan 1 after newest class before week0; no archived publication claim',
                aggregation_window='up to four preceding recruiting classes, excluding target-season class',
                minimum_sample_rule='at least three nonmissing valid ratings or star counts in each unit cohort',
                garbage_time_handling='Not applicable: recruiting commitments, no game performance',
                source_inspiration='Structured CFBD high-school recruiting history',provenance='TDNet-derived',
                code_path='src/gridiron_ml/experiments/nextgen_recruiting.py',input_equations=equations,
                missingness_policy='Unknown positions unassigned; unsupported cohorts missing; no enrollment or roster retention imputation')
            if f.operation=='identity':r['equation_excel']=equations[f.name]
            records.append(r)
        self.manifest_path=root/'feature_families/F10'/self.family/'feature_manifest.json'
        self.manifest_path.parent.mkdir(parents=True,exist_ok=True)
        content=json.dumps(records,indent=2)+'\n'
        if self.manifest_path.exists() and self.manifest_path.read_text()!=content:raise ValueError('Version changed recruiting manifest')
        self.manifest_path.write_text(content)
    def build_frame(self):
        metadata=[c for c in self.state if not c.startswith('recruit_history_')]
        return pd.concat([self.state[metadata].reset_index(drop=True),pd.DataFrame({f.name:f.evaluate(self.state) for f in self.formulas}).reset_index(drop=True)],axis=1)


def main():
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    inventory=json.loads((ROOT/'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text())
    schedule,digest=load_authoritative_schedule(root,inventory)
    records=verified_endpoint_records(root,'/recruiting/players')
    recruits=pd.concat([pd.read_parquet(r['cache_path']) for r in records],ignore_index=True)
    state,audit=recruiting_cohort_state(recruits,schedule)
    report={'scope':'lagged recruiting history only, not complete F10','schedule_sha256':digest,
            'source_sha256':{r['request_id']:r['sha256'] for r in records},'coverage':audit,'designs':{}}
    for design in 'abc':
        b=RecruitingHistoryBuilder(root,state,design);path=b.materialize(root)
        report['designs'][design]={'path':str(path),'rows':len(state),'features':len(b.feature_columns)}
    atomic_json(root/'results/f10_recruit_history_materialization.json',report)
    print(json.dumps(report['designs'],indent=2))


if __name__=='__main__':main()
