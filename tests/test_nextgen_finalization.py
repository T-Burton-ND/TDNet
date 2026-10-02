import json
from pathlib import Path
import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_results import verified_recommendations, architecture_summaries
from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file


def test_persisted_recommendation_rejects_changed_evidence(tmp_path):
    source=tmp_path/'actual-result.json';source.write_text('{"status":"success"}')
    dest=tmp_path/'results/F06';dest.mkdir(parents=True)
    row={'row_type':'fingerprint_recommendation','fingerprint_id':'F06_R_a'}
    report={'evidence':[{'path':str(source),'sha256':sha256_file(source)}],'rows':[row]}
    (dest/'recommendations.json').write_text(json.dumps(report))
    assert verified_recommendations(tmp_path)==[row]
    source.write_text('{"status":"failed"}')
    with pytest.raises(ValueError,match='evidence changed'):
        verified_recommendations(tmp_path)


def test_f06_finalization_requires_measured_acceptable_reduction(tmp_path,monkeypatch):
    import gridiron_ml.experiments.nextgen_finalize_f06 as module
    points=json.loads((module.ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    all_records=[{'name':f'x{i}','generation':'F06','matchup_counterpart':f'x{i}'} for i in range(62)]
    records={};rows=[]
    audit=tmp_path/'results/source_semantics_audit.json';audit.parent.mkdir();audit.write_text('{}')
    for design in 'abc':
        for variant,n in [('F',62),('R',60)]:
            name=f'F06_{variant}_{design}';records[name]=all_records[:n]
            folder=tmp_path/'fingerprints'/name;folder.mkdir(parents=True)
            frame=pd.DataFrame({'target_game_id':[1,1],'team':['A','B'],
                                **{f'x{i}':[1.,2.] for i in range(n)}})
            frame.to_parquet(folder/'values.parquet',index=False)
            (folder/'feature_manifest.json').write_text(json.dumps(records[name]))
            (folder/'provenance.json').write_text('{}')
            if variant=='F':
                diagnostic=tmp_path/'results/F06'/name;diagnostic.mkdir(parents=True)
                (diagnostic/'redundancy.parquet').write_bytes(b'fixture diagnostic')
                (diagnostic/'redundancy.provenance.json').write_text(json.dumps({
                    'source_data_sha256':sha256_file(folder/'values.parquet'),
                    'source_manifest_sha256':sha256_file(folder/'feature_manifest.json'),
                    'output_sha256':sha256_file(diagnostic/'redundancy.parquet')}))
            for model in ['M2','M4']:
                for point in points[model]:
                    run_id=f"{name}__{model}__{point['id']}"
                    d=tmp_path/'experiments/F06'/run_id;d.mkdir(parents=True)
                    row={'row_type':'run','run_id':run_id,'fingerprint_id':name,'model':model,
                         'hyperparameter_setpoint':point['id'],'status':'success','feature_count':n,
                         'max_design_year_used':2025,'mae_2024':10.,'mae_2025':10.}
                    rows.append(row);(d/'result.json').write_text(json.dumps(row))
                    (d/'predictions.parquet').write_bytes(b'fixture prediction')
                    pd.DataFrame({'source_feature':[r['name'] for r in records[name]],
                                  'normalized_importance':[1/n]*n}).to_parquet(d/'source_shap.parquet')
    runs=pd.DataFrame(rows)
    monkeypatch.setattr(module,'checked_fingerprint',lambda root,name:(None,None,records[name],{}))
    def collected(root):
        return pd.concat([runs,pd.DataFrame(architecture_summaries(runs,points))],ignore_index=True)
    monkeypatch.setattr(module,'collect_results',collected)
    # Excess degradation must not create a finalization marker.
    runs.loc[runs.fingerprint_id.eq('F06_R_a'),'mae_2024']=11.
    with pytest.raises(ValueError,match='fails measured'):
        module.finalize_f06(tmp_path)
    assert not (tmp_path/'results/F06/finalization.json').exists()
    runs.loc[runs.fingerprint_id.eq('F06_R_a'),'mae_2024']=10.
    result=module.finalize_f06(tmp_path)
    assert result['training_finalized'] and result['reduction_finalized']
    report=json.loads((tmp_path/'results/F06/recommendations.json').read_text())
    assert len(report['rows'])==6
    assert sum(r['recommendation_selected'] for r in report['rows'])==3
    assert len(verified_recommendations(tmp_path))==6
