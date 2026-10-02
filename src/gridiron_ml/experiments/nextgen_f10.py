"""F10 observed-player usage family; roster/recruiting families are separate."""
import json
from pathlib import Path
import pandas as pd

from .nextgen_artifacts import NextgenFeatureBuilder
from .nextgen_designs import Formula, feature_record
from .nextgen_players import USAGE_METRICS, player_usage_state
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file, load_authoritative_schedule

ROOT=Path(__file__).resolve().parents[3]


class PlayerUsageBuilder(NextgenFeatureBuilder):
    generation='F10'

    def __init__(self,root,state,design):
        if design not in 'abc' or len(design)!=1:
            raise ValueError('Unknown design')
        self.family='player_usage_'+design
        self.state=state
        self.formulas=[]
        for metric in USAGE_METRICS:
            inputs=(metric+'_top_share',metric+'_hhi')
            if design=='c':
                self.formulas.append(Formula(metric+'_concentration',inputs,'mean',
                    'Equal-weight leading-player share and sum of squared player shares; observed workload concentration, not depth-chart availability.','fraction'))
            else:
                for name in inputs:
                    self.formulas.append(Formula(name,(name,),'identity',
                        'Observed prior-game workload concentration; high values indicate dependence on fewer players, without inferring future availability.','fraction'))
        if design=='b':
            self.formulas.append(Formula('pass_rush_leader_dependence',('pass_attempts_top_share','rush_attempts_top_share'),'product',
                'Joint dependence on leading pass and rush users; does not assume they are different athletes.','fraction'))
        self.feature_columns=tuple(f.name for f in self.formulas)
        records=[]
        for formula in self.formulas:
            r=feature_record(formula,'F10',design,formula.name,endpoints=['/games/players','/games'])
            r.update(raw_columns=['teams.categories.types.athletes.id','teams.categories.types.athletes.stat'],
                availability_rule='Every contributing regular-game kickoff plus 48 hours strictly precedes target; reconstructed reporting lag, not an archived publication timestamp',
                aggregation_window='latest 12 available regular games with player box observations, crossing seasons',
                minimum_sample_rule='positive observed total for each metric, all identities known, all counts finite and nonnegative',
                garbage_time_handling='Full regular-game box scores; no play-level garbage adjustment possible',
                source_inspiration='TDNet-derived player concentration from structured CFBD box scores',
                code_path='src/gridiron_ml/experiments/nextgen_f10.py',provenance='TDNet-derived',
                rate_definition='player share = sum prior observed player count / sum all prior observed player counts; top share = MAX(shares); HHI = SUMSQ(shares)',
                input_equations={n:('=MAX(PlayerShares)' if n.endswith('top_share') else '=SUMSQ(PlayerShares)') for n in formula.inputs})
            if formula.operation=='identity':
                r['equation_excel']=r['input_equations'][formula.name]
            records.append(r)
        self.manifest_path=root/'feature_families/F10'/self.family/'feature_manifest.json'
        content=json.dumps(records,indent=2)+'\n'
        if self.manifest_path.exists() and self.manifest_path.read_text()!=content:
            raise ValueError('Version the usage family before changing its manifest')
        self.manifest_path.parent.mkdir(parents=True,exist_ok=True)
        self.manifest_path.write_text(content)

    def build_frame(self):
        metadata=['season','season_type','team','target_game_id','target_start_utc','feature_kind',
                  'latest_source_game_id','latest_source_game_utc','latest_source_season_type',
                  'feature_available_utc','static_availability_documentation']
        return pd.concat([self.state[metadata].reset_index(drop=True),
                          pd.DataFrame({f.name:f.evaluate(self.state) for f in self.formulas}).reset_index(drop=True)],axis=1)


def materialize_usage(root):
    preparation=json.loads((root/'results/player_observation_preparation.json').read_text())
    inventory=json.loads((ROOT/'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text())
    schedule,schedule_hash=load_authoritative_schedule(root,inventory)
    parts=[]
    for item in preparation['partitions']:
        path=root/'canonical/player_observations'/(item['request_id']+'.parquet')
        if sha256_file(path)!=item['data_sha256'] or item['schedule_sha256']!=schedule_hash:
            raise ValueError('Player observation provenance mismatch')
        if item['parser_sha256']!=sha256_file(ROOT/'src/gridiron_ml/experiments/nextgen_players.py'):
            raise ValueError('Player observation parser changed; rerun preparation')
        parts.append(pd.read_parquet(path,columns=['game_id','team','athlete_id','athlete_name','metric','value']))
    state,coverage=player_usage_state(pd.concat(parts,ignore_index=True),schedule)
    report={'family_scope':'observed player usage only; complete F10 includes additional families',
            'schedule_sha256':schedule_hash,'coverage':coverage,'designs':{}}
    for design in 'abc':
        builder=PlayerUsageBuilder(root,state,design)
        path=builder.materialize(root)
        report['designs'][design]={'path':str(path),'rows':len(state),'features':len(builder.feature_columns)}
    atomic_json(root/'results/f10_usage_materialization.json',report)
    return report


if __name__=='__main__':
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    print(json.dumps(materialize_usage(root),indent=2))
