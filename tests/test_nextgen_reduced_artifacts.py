import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_reduced_artifacts import select_trial_frame


def test_reduction_drops_columns_without_losing_keys_and_enforces_pair_floor():
    records=[{'name':f'x{i}', 'generation':'F06', 'matchup_counterpart':f'x{i^1}'} for i in range(64)]
    frame=pd.DataFrame({'target_game_id':[7,7], 'team':['A','B'],
                        **{f'x{i}':[float(i),float('nan')] for i in range(64)}})
    names=[f'x{i}' for i in range(60)]
    reduced, kept, counts=select_trial_frame(frame,records,names)
    assert counts=={'F06':60}
    pd.testing.assert_frame_equal(reduced,frame[['target_game_id','team',*names]])
    assert len(kept)==60 and 'x63' not in reduced
    with pytest.raises(ValueError,match='split'):
        select_trial_frame(frame,records,names+['x60'])
    with pytest.raises(ValueError,match='floor'):
        select_trial_frame(frame,records,names[:58])
    with pytest.raises(ValueError,match='remove at least'):
        select_trial_frame(frame,records,[r['name'] for r in records])
    pool = frame[['target_game_id', 'team', *names]]
    retained, _, counts = select_trial_frame(
        pool, records[:60], names, full_reference_names=[r['name'] for r in records])
    pd.testing.assert_frame_equal(retained, pool)
    assert counts == {'F06': 60}
    with pytest.raises(ValueError, match='remove at least'):
        select_trial_frame(pool, records[:60], names, full_reference_names=names)


def test_trial_waits_for_full_evidence_and_is_not_marked_accepted(tmp_path, monkeypatch):
    import json
    import gridiron_ml.experiments.nextgen_reduced_artifacts as module
    from gridiron_ml.experiments.nextgen_designs import Formula, feature_record
    from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file
    names=[f'x{i}' for i in range(62)]
    records=[feature_record(Formula(n,(n,),'identity','Fixture','fraction'),'F06','a',n,
                            endpoints=['fixture']) for n in names]
    source=tmp_path/'fingerprints/F06_F_a';source.mkdir(parents=True)
    frame=pd.DataFrame({'target_game_id':[7,7],'team':['A','B'],'season':[2020,2020],
                        **{n:[1.,2.] for n in names}})
    frame.to_parquet(source/'values.parquet',index=False)
    provenance={'data_sha256':sha256_file(source/'values.parquet'),'schedule_sha256':'fixture',
                'source_sha256':'fixture','manifest_sha256':'fixture','canonical_families':[]}
    monkeypatch.setattr(module,'checked_fingerprint',lambda *args:(None,None,records,provenance))
    monkeypatch.setattr(module,'verify_successful_run',lambda *args:None)
    points=json.loads((module.ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    with pytest.raises(ValueError,match='not finished'):
        module.materialize_late_trial(tmp_path,'F06_F_a',names[:60],rationale='Fixture proposal')
    assert not (tmp_path/'fingerprints/F06_R_a').exists()
    for model in ['M2','M4']:
        for i,point in enumerate(points[model]):
            run_id=f"F06_F_a__{model}__{point['id']}"
            folder=tmp_path/'experiments/F06'/run_id;folder.mkdir(parents=True)
            row={'fingerprint_id':'F06_F_a','model':model,'hyperparameter_setpoint':point['id'],
                 'status':'success' if i<3 else 'failed','mae_2024':10.,'mae_2025':11.}
            (folder/'result.json').write_text(json.dumps(row))
            if i<3:
                pd.DataFrame({'source_feature':names,'normalized_importance':[1/62]*62}).to_parquet(folder/'source_shap.parquet')
    result=module.materialize_late_trial(tmp_path,'F06_F_a',names[:60],rationale='Fixture proposal')
    assert result['features']==60 and result['accepted'] is False
    child=tmp_path/'fingerprints/F06_R_a'
    actual=pd.read_parquet(child/'values.parquet')
    assert 'x60' not in actual and 'x61' not in actual
    saved=json.loads((child/'provenance.json').read_text())
    assert saved['ancestry']==['F06_F_a'] and saved['reduction_accepted'] is False
    with pytest.raises(ValueError,match='already exists'):
        module.materialize_late_trial(tmp_path,'F06_F_a',names[:60],rationale='Fixture proposal')
