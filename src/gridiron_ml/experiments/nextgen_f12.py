"""F12 special-teams states from explicit regular-game player box categories."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .nextgen_artifacts import NextgenFeatureBuilder
from .nextgen_designs import Formula,feature_record
from gridiron_ml.pipeline.fetch.nextgen_acquisition import load_authoritative_schedule,sha256_file,atomic_json
ROOT=Path(__file__).resolve().parents[3]
RATIOS={
 'st_field_goal_rate':('field_goals','field_goal_attempts','fraction'),
 'st_extra_point_rate':('extra_points','extra_point_attempts','fraction'),
 'st_punt_yards_per_punt':('punt_yards','punts','yards/punt'),
 'st_punt_inside20_rate':('punts_inside20','punts','fraction'),
 'st_punt_touchback_rate':('punt_touchbacks','punts','fraction'),
 'st_kick_return_yards_per_return':('kick_return_yards','kick_returns','yards/return'),
 'st_punt_return_yards_per_return':('punt_return_yards','punt_returns','yards/return'),
}


def special_team_games(observations):
    wanted={m for pair in RATIOS.values() for m in pair[:2]}
    source=observations.loc[observations.metric.isin(wanted)]
    totals=source.groupby(['game_id','team','metric']).value.agg(lambda x:x.sum(min_count=len(x)))
    return totals.unstack('metric').reindex(columns=sorted(wanted)).reset_index()


def special_team_state(stats,schedule,window=12):
    if (schedule.season.gt(2025).any() or not schedule.season_type.eq('regular').all()
            or not schedule.completed.eq(True).all()):
        raise ValueError('Special-teams source must be pre-2026 regular season')
    if stats.duplicated(['game_id','team']).any():
        raise ValueError('Duplicate special-teams game observation')
    source=stats.merge(schedule[['id','start_date','home_team','away_team']],left_on='game_id',right_on='id',how='left',validate='many_to_one')
    if source.id.isna().any() or not (source.team.eq(source.home_team)|source.team.eq(source.away_team)).all():
        raise ValueError('Unscheduled special-teams source')
    source['kickoff']=pd.to_datetime(source.start_date,utc=True)
    source['available']=source.kickoff+pd.Timedelta(hours=48)
    histories={t:g.sort_values(['available','game_id']) for t,g in source.groupby('team')}
    rows=[];excluded=[]
    for g in schedule.itertuples(index=False):
        if str(g.home_classification).lower()!='fbs' or str(g.away_classification).lower()!='fbs':continue
        pair=[];target=pd.Timestamp(g.start_date)
        for team in (g.home_team,g.away_team):
            h=histories.get(team)
            if h is None:break
            prior=h.loc[h.available.lt(target)].tail(window)
            if prior.empty:break
            values={}
            for name,(numerator,denominator,_) in RATIOS.items():
                valid=prior[[numerator,denominator]].replace([np.inf,-np.inf],np.nan).dropna()
                valid=valid.loc[valid[denominator].ge(0)]
                # Both numerator and denominator must be observed in the same games.
                count=valid[denominator].sum()
                values[name]=float(valid[numerator].sum()/count) if count>=10 else np.nan
            latest=prior.iloc[-1]
            values.update(season=int(g.season),season_type='regular',team=team,target_game_id=int(g.id),
                target_start_utc=target,feature_kind='dynamic',latest_source_game_id=int(latest.game_id),
                latest_source_game_utc=latest.kickoff,latest_source_season_type='regular',
                feature_available_utc=latest.available,static_availability_documentation=None)
            pair.append(values)
        if len(pair)==2:rows.extend(pair)
        else:excluded.append(int(g.id))
    return pd.DataFrame(rows),excluded


class SpecialTeamsBuilder(NextgenFeatureBuilder):
    generation='F12'
    def __init__(self,root,state,design):
        self.family='special_teams_'+design;self.state=state
        self.formulas=[Formula(n,(n,),'identity','Prior regular-game special-teams execution from explicit box-score categories.',u) for n,(_,_,u) in RATIOS.items()]
        if design=='b':
            self.formulas.append(Formula('st_punt_placement_balance',('st_punt_inside20_rate','st_punt_touchback_rate'),'difference','Inside-20 placement fraction minus touchback fraction; punt field-position control.','fraction'))
        if design=='c':
            self.formulas=[f for f in self.formulas if f.name not in {'st_field_goal_rate','st_extra_point_rate'}]+[
                Formula('st_placekick_consistency',('st_field_goal_rate','st_extra_point_rate'),'mean','Equal mean of observed field-goal and extra-point make rates, not a distance-adjusted kicker rating.','fraction')]
        self.feature_columns=tuple(f.name for f in self.formulas)
        records=[]
        for f in self.formulas:
            r=feature_record(f,'F12',design,f.name,endpoints=['/games/players','/games'])
            equations={n:f'=IF(SUM({RATIOS[n][1]})>=10,SUM({RATIOS[n][0]})/SUM({RATIOS[n][1]}),NA())' for n in f.inputs}
            r.update(raw_columns=['teams.categories.types.athletes.stat'],input_equations=equations,
                availability_rule='Latest contributing regular-game kickoff plus 48h strictly before target; reconstructed reporting lag',
                aggregation_window='latest 12 available regular games with explicit special-teams observations, crossing seasons',
                minimum_sample_rule='ten denominator events; numerator and denominator observed in the same contributing games',
                garbage_time_handling='Full regular-game special teams, no postseason; box-score garbage filtering unavailable',
                source_inspiration='TDNet-derived from structured CFBD special-teams box categories',provenance='TDNet-derived',
                code_path='src/gridiron_ml/experiments/nextgen_f12.py')
            if f.operation=='identity':r['equation_excel']=equations[f.name]
            records.append(r)
        self.manifest_path=root/'feature_families/F12'/self.family/'feature_manifest.json'
        self.manifest_path.parent.mkdir(parents=True,exist_ok=True)
        content=json.dumps(records,indent=2)+'\n'
        if self.manifest_path.exists() and self.manifest_path.read_text()!=content:raise ValueError('Version changed special-teams manifest')
        self.manifest_path.write_text(content)
    def build_frame(self):
        metadata=[c for c in self.state if c not in RATIOS]
        return pd.concat([self.state[metadata].reset_index(drop=True),pd.DataFrame({f.name:f.evaluate(self.state) for f in self.formulas}).reset_index(drop=True)],axis=1)


def main():
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    prep=json.loads((root/'results/player_observation_preparation.json').read_text())
    inventory=json.loads((ROOT/'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text())
    schedule,digest=load_authoritative_schedule(root,inventory);parts=[]
    for item in prep['partitions']:
        path=root/'canonical/player_observations'/(item['request_id']+'.parquet')
        if sha256_file(path)!=item['data_sha256'] or digest!=item['schedule_sha256']:raise ValueError('Player observation provenance mismatch')
        parts.append(special_team_games(pd.read_parquet(path,columns=['game_id','team','metric','value'])))
    state,excluded=special_team_state(pd.concat(parts,ignore_index=True),schedule)
    report={'scope':'special-teams family only, not complete F12','excluded_target_games':excluded,'designs':{}}
    for design in 'abc':
        b=SpecialTeamsBuilder(root,state,design);out=b.materialize(root)
        report['designs'][design]={'path':str(out),'rows':len(state),'features':len(b.feature_columns)}
    atomic_json(root/'results/f12_special_teams_materialization.json',report)
    print(json.dumps(report['designs'],indent=2))


if __name__=='__main__':main()
