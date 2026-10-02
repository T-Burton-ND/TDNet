"""Canonical observed offensive-room families for F12."""
import json
from pathlib import Path
import pandas as pd
from .nextgen_artifacts import NextgenFeatureBuilder
from .nextgen_designs import Formula,feature_record
from .nextgen_offensive_units import ROOMS,offensive_unit_state
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json,load_authoritative_schedule,sha256_file
ROOT=Path(__file__).resolve().parents[3]


class OffensiveRoomBuilder(NextgenFeatureBuilder):
    generation='F12'

    def __init__(self,root,state,design):
        if design not in ('a','b','c'):
            raise ValueError('Unknown design')
        self.family='offensive_rooms_'+design
        self.state=state
        self.formulas=[]
        for room in ROOMS:
            names=tuple(room+'_observed_player_ppa_'+stat for stat in ('mean','dispersion'))
            if design=='c':
                self.formulas.append(Formula(room+'_observed_ppa_consistency',names,'difference',
                    'Mean observed player-game PPA minus one population standard deviation; descriptive consistency score, not a confidence bound.','points/play'))
            else:
                self.formulas.extend(Formula(n,(n,),'identity',
                    'Mean or population dispersion across historical player-game PPA averages in this observed room; equal records, not equal plays.','points/play') for n in names)
        if design=='b':
            self.formulas.extend([
                Formula('qb_receiver_observed_ppa_balance',('qb_observed_player_ppa_mean','wrte_observed_player_ppa_mean'),'mean',
                    'Equal-weight passing and receiving room averages; shared play outcomes, not independent causal effects.','points/play'),
                Formula('pass_rush_room_observed_ppa_gap',('qb_observed_player_ppa_mean','rb_observed_player_ppa_mean'),'difference',
                    'Observed quarterback passing minus running-back rushing PPA means; descriptive phase balance.','points/play')])
        self.feature_columns=tuple(f.name for f in self.formulas)
        records=[]
        for f in self.formulas:
            equations={n:('=AVERAGE(ObservedPlayerGamePPA)' if n.endswith('_mean') else '=STDEV.P(ObservedPlayerGamePPA)') for n in f.inputs}
            r=feature_record(f,'F12',design,f.name,endpoints=['/ppa/players/games','/games'])
            r.update(raw_columns=['season','week','team','opponent','id','position','average_p_p_a.pass','average_p_p_a.rush'],
                input_equations=equations,
                availability_rule='Every contributing regular-game kickoff plus reconstructed 48h reporting lag strictly before target',
                aggregation_window='last 12 available team games with observed player PPA, crossing seasons; equal player-game observations within each room',
                minimum_sample_rule='at least three distinct source games with finite room-specific PPA; otherwise missing',
                garbage_time_handling='Provider default PPA; excludeGarbageTime was not requested; no additional text-derived filtering',
                source_inspiration='TDNet-derived observed QB, RB/FB and WR/TE room summaries from CFBD player-game PPA',
                provenance='TDNet-derived',code_path='src/gridiron_ml/experiments/nextgen_f12_offense.py',
                missingness_policy='Unmatched or ambiguous game keys and conflicting player-game records excluded; missing rooms never imputed; historical roles do not establish target roster membership')
            if f.operation=='identity':r['equation_excel']=equations[f.name]
            records.append(r)
        self.manifest_path=root/'feature_families/F12'/self.family/'feature_manifest.json'
        self.manifest_path.parent.mkdir(parents=True,exist_ok=True)
        content=json.dumps(records,indent=2)+'\n'
        if self.manifest_path.exists() and self.manifest_path.read_text()!=content:
            raise ValueError('Version offensive-room manifest before changing it')
        self.manifest_path.write_text(content)

    def build_frame(self):
        metadata=[c for c in self.state if '_observed_player_ppa_' not in c]
        return pd.concat([self.state[metadata].reset_index(drop=True),
            pd.DataFrame({f.name:f.evaluate(self.state) for f in self.formulas}).reset_index(drop=True)],axis=1)


def main():
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    source=root/'canonical/player_ppa_games.parquet'
    provenance=json.loads(source.with_suffix('.provenance.json').read_text())
    schedule,digest=load_authoritative_schedule(root,json.loads((ROOT/'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text()))
    if provenance['data_sha256']!=sha256_file(source) or provenance['schedule_sha256']!=digest:
        raise ValueError('Player PPA source provenance mismatch')
    if provenance['code_sha256']!=sha256_file(ROOT/'src/gridiron_ml/experiments/nextgen_player_ppa.py'):
        raise ValueError('Source binding changed; rebuild player PPA observations')
    state,coverage=offensive_unit_state(pd.read_parquet(source),schedule)
    report={'scope':'observed offensive rooms only, not complete F12','coverage':coverage,
            'source_sha256':provenance['data_sha256'],'schedule_sha256':digest,'designs':{}}
    for design in 'abc':
        b=OffensiveRoomBuilder(root,state,design)
        path=b.materialize(root)
        report['designs'][design]={'path':str(path),'rows':len(state),'features':len(b.feature_columns)}
    atomic_json(root/'results/f12_offensive_rooms_materialization.json',report)
    print(json.dumps(report['designs'],indent=2))


if __name__=='__main__':main()
