"""Canonical observed-use continuity; a separate dynamic F10 family."""
import json
from pathlib import Path
import pandas as pd
from .nextgen_artifacts import NextgenFeatureBuilder
from .nextgen_designs import Formula,feature_record
from .nextgen_players import USAGE_METRICS
from .nextgen_continuity import continuity_state
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json,sha256_file,load_authoritative_schedule
ROOT=Path(__file__).resolve().parents[3]


class ObservedContinuityBuilder(NextgenFeatureBuilder):
    generation='F10'
    def __init__(self,root,state,design):
        if design not in ('a','b','c'):raise ValueError('Unknown design')
        self.family='observed_continuity_'+design;self.state=state;self.formulas=[]
        for metric in USAGE_METRICS:
            names=(metric+'_observed_returning_usage_share',metric+'_prior_production_share_of_observed_returners')
            if design=='c':
                self.formulas.append(Formula(metric+'_observed_continuity_balance',names,'mean',
                    'Mean of current workload by observed prior-season users and prior workload of those overlapping users; not roster retention.','fraction'))
            else:
                self.formulas.extend(Formula(n,(n,),'identity','Observed same-team current/prior-season usage overlap, without assigning departure or injury status.','fraction') for n in names)
        if design=='b':
            self.formulas.append(Formula('pass_rush_observed_returner_balance',
                ('pass_attempts_observed_returning_usage_share','rush_attempts_observed_returning_usage_share'),'product',
                'Joint passing/rushing workload continuity among observed users; missing until both workloads have prior/current support.','fraction'))
        self.feature_columns=tuple(f.name for f in self.formulas)
        records=[]
        for f in self.formulas:
            equations={n:('=SUM(CurrentWorkloadOfOverlappingUsers)/SUM(CurrentWorkload)' if n.endswith('_observed_returning_usage_share') else '=SUM(PriorWorkloadOfOverlappingUsers)/SUM(PriorWorkload)') for n in f.inputs}
            r=feature_record(f,'F10',design,f.name,endpoints=['/games/players','/games'])
            r.update(raw_columns=['teams.categories.types.athletes.id','teams.categories.types.athletes.stat'],input_equations=equations,
                availability_rule='Completed regular-game kickoff plus reconstructed 48h lag strictly before target',
                aggregation_window='all available current-season usage and immediately prior-season usage for the same team',
                minimum_sample_rule='positive totals in both seasons; finite nonnegative counts and stable IDs throughout each metric',
                garbage_time_handling='Regular-game box scores; no play-level garbage adjustment available',
                missingness_policy='Before current usage exists or with unknown identity, stay missing; absence never implies nonparticipation, injury, or departure',
                source_inspiration='TDNet-derived observed identity continuity from structured player box records',
                provenance='TDNet-derived',code_path='src/gridiron_ml/experiments/nextgen_f10_continuity.py')
            if f.operation=='identity':r['equation_excel']=equations[f.name]
            records.append(r)
        self.manifest_path=root/'feature_families/F10'/self.family/'feature_manifest.json'
        self.manifest_path.parent.mkdir(parents=True,exist_ok=True)
        content=json.dumps(records,indent=2)+'\n'
        if self.manifest_path.exists() and self.manifest_path.read_text()!=content:raise ValueError('Version changed continuity manifest')
        self.manifest_path.write_text(content)
    def build_frame(self):
        metadata=[c for c in self.state if not c.endswith(('_observed_returning_usage_share','_prior_production_share_of_observed_returners'))]
        return pd.concat([self.state[metadata].reset_index(drop=True),pd.DataFrame({f.name:f.evaluate(self.state) for f in self.formulas}).reset_index(drop=True)],axis=1)


def main():
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    preparation=json.loads((root/'results/player_observation_preparation.json').read_text())
    schedule,digest=load_authoritative_schedule(root,json.loads((ROOT/'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text()))
    parser_hash=sha256_file(ROOT/'src/gridiron_ml/experiments/nextgen_players.py')
    parts=[]
    for item in preparation['partitions']:
        path=root/'canonical/player_observations'/(item['request_id']+'.parquet')
        if sha256_file(path)!=item['data_sha256'] or item['schedule_sha256']!=digest or item['parser_sha256']!=parser_hash:
            raise ValueError('Player observation provenance mismatch')
        parts.append(pd.read_parquet(path,columns=['game_id','team','athlete_id','metric','value']))
    state,coverage=continuity_state(pd.concat(parts,ignore_index=True),schedule)
    report={'scope':'observed-use continuity only, not complete F10','schedule_sha256':digest,'coverage':coverage,'designs':{}}
    for d in 'abc':
        b=ObservedContinuityBuilder(root,state,d);path=b.materialize(root)
        report['designs'][d]={'path':str(path),'rows':len(state),'features':len(b.feature_columns)}
    atomic_json(root/'results/f10_continuity_materialization.json',report)
    print(json.dumps(report['designs'],indent=2))


if __name__=='__main__':main()
