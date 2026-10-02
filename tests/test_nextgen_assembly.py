import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_assembly import join_family,assert_family_values


def test_assembly_preserves_parent_and_excludes_whole_unpaired_game():
    parent=pd.DataFrame({'target_game_id':[1,1,2,2],'team':['A','B','A','B'],'parent_feature':[1.,2.,3.,4.]})
    family=pd.DataFrame({'target_game_id':[1,1,2],'team':['A','B','A'],'new_feature':[8.,float('nan'),9.]}).set_index(['target_game_id','team'])
    joined,excluded=join_family(parent,family)
    assert joined.parent_feature.tolist()==[1.,2.]
    assert excluded==[2]
    assert_family_values(joined,family)
    joined.loc[0,'new_feature']=99
    with pytest.raises(ValueError,match='differs'):
        assert_family_values(joined,family)
    with pytest.raises(ValueError,match='overwrite'):
        join_family(parent,family.rename(columns={'new_feature':'parent_feature'}))


def test_full_assembly_keeps_checked_ancestry_and_rejects_parent_value_drift(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from gridiron_ml.experiments.nextgen_assembly import assemble_full, NextgenModelBoundary
    from gridiron_ml.experiments.nextgen_designs import Formula, feature_record
    from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file
    parent=tmp_path/'fingerprints/F09_F_a'
    parent.mkdir(parents=True)
    frame=pd.DataFrame({'target_game_id':[1,1], 'team':['A','B'], 'season':[2020,2020], 'x':[1.,2.]})
    data=parent/'values.parquet'; frame.to_parquet(data,index=False)
    def manifest(name,generation):
        return [feature_record(Formula(name,(name,),'identity','Observed test feature','fraction'),generation,'a',name,endpoints=['/games'])]
    pm=parent/'feature_manifest.json';pm.write_text(json.dumps(manifest('x','F09')))
    new=tmp_path/'feature_families/F10/test_a';new.mkdir(parents=True)
    (new/'feature_manifest.json').write_text(json.dumps(manifest('y','F10')))
    inherited={'generation':'F09','family':'prior_a','manifest_path':'unused-test-path'}
    provenance={'data_sha256':sha256_file(data),'manifest_sha256':sha256_file(pm),
                'source_sha256':'source','schedule_sha256':'schedule','canonical_families':[inherited]}
    (parent/'provenance.json').write_text(json.dumps(provenance))
    x=frame.set_index(['target_game_id','team'])[['x']]
    y=x.rename(columns={'x':'y'})*10
    monkeypatch.setattr(NextgenModelBoundary,'from_canonical',staticmethod(
        lambda root,generation,family,path: SimpleNamespace(for_fit=lambda: x if generation=='F09' else y)))
    result=assemble_full(tmp_path,'F10','a',['test_a'])
    assert result['features']==2
    child=tmp_path/'fingerprints/F10_F_a'
    saved=pd.read_parquet(child/'values.parquet')
    assert saved.x.tolist()==[1.,2.] and saved.y.tolist()==[10.,20.]
    cp=json.loads((child/'provenance.json').read_text())
    assert cp['ancestry']==['F09_F_a'] and cp['canonical_families'][0]==inherited
    frame.loc[0,'x']=99.
    frame.to_parquet(data,index=False)
    provenance['data_sha256']=sha256_file(data)
    (parent/'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError,match='differs'):
        assemble_full(tmp_path,'F10','a',['test_a'])
