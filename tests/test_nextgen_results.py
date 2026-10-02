import pandas as pd
from gridiron_ml.experiments.nextgen_results import architecture_summaries


def test_summary_does_not_hide_failed_or_missing_setpoints():
    points={'M2':[{'id':str(i)} for i in range(10)]}
    runs=pd.DataFrame([dict(fingerprint_id='F06_F_a',model='M2',hyperparameter_setpoint=str(i),status='success',mae_2024=float(i),mae_2025=float(i+1)) for i in range(10)])
    summary=architecture_summaries(runs,points)[0]
    assert summary['mae_2024']==4.5
    assert summary['status']=='success'
    assert not summary['recommendation_eligible']
    runs.loc[0,'status']='failed'
    summary=architecture_summaries(runs,points)[0]
    assert summary['status']=='incomplete'
    assert summary['mae_2024']==5.
    assert summary['screening_usable'] and summary['incomplete_coverage']
    assert architecture_summaries(runs.iloc[1:],points)[0]['status']=='incomplete'


def test_successful_collection_rechecks_input_output_and_execution(tmp_path, monkeypatch):
    import pytest
    from gridiron_ml.experiments import nextgen_screening
    from gridiron_ml.experiments.nextgen_results import verify_successful_run
    from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file

    name='F06_F_a';run_id=name+'__M2__p1'
    fingerprint=tmp_path/'fingerprints'/name;fingerprint.mkdir(parents=True)
    data=fingerprint/'values.parquet';data.write_bytes(b'original input')
    directory=tmp_path/'experiments'/'F06'/run_id;directory.mkdir(parents=True)
    for file in ('predictions.parquet','source_shap.parquet'):
        (directory/file).write_bytes(b'original output')
    binding={'computation':'current'}
    monkeypatch.setattr(nextgen_screening,'execution_binding',lambda *args:binding)
    provenance={'data_sha256':sha256_file(data)}
    row=dict(model='M2',fingerprint_id=name,hyperparameter_setpoint='p1',run_id=run_id,
             data_sha256=provenance['data_sha256'],execution_binding=dict(binding),
             output_sha256={p.name:sha256_file(p) for p in directory.iterdir()})
    points={'M2':[{'id':'p1'}]}
    verify_successful_run(tmp_path,directory,row,provenance,points)
    output=directory/'source_shap.parquet';output.write_bytes(b'changed output')
    with pytest.raises(ValueError,match='output missing or changed'):
        verify_successful_run(tmp_path,directory,row,provenance,points)
    output.write_bytes(b'original output')
    data.write_bytes(b'changed input')
    with pytest.raises(ValueError,match='input data changed'):
        verify_successful_run(tmp_path,directory,row,provenance,points)
    data.write_bytes(b'original input')
    binding['computation']='updated'
    with pytest.raises(ValueError,match='different inputs or computation'):
        verify_successful_run(tmp_path,directory,row,provenance,points)
