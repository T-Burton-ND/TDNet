import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('screening_array',Path(__file__).resolve().parents[1]/'scripts/nextgen_screening_array.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(root,successes,attempt=1):
    jobs=[]
    for i in range(10):
        job={'task_id':i+1,'fingerprint':'F06_F_a','model':'M2','setpoint':str(i)};jobs.append(job)
        path=root/'experiments/F06'/f'F06_F_a__M2__{i}'/'result.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'status':'success' if i<successes else 'failed','attempt_number':attempt}))
    return {'generation':'F06','jobs':jobs}


def test_retry_only_below_three_successes_and_within_cap(tmp_path):
    manifest=fixture(tmp_path,3)
    jobs,decisions=module.retry_jobs(tmp_path,manifest)
    assert jobs==[] and decisions[0]['proceed'] and decisions[0]['incomplete_coverage']
    manifest=fixture(tmp_path,2)
    jobs,decisions=module.retry_jobs(tmp_path,manifest)
    assert len(jobs)==8 and not decisions[0]['proceed']
    assert [j['task_id'] for j in jobs]==list(range(1,9))
    manifest=fixture(tmp_path,2,4)
    assert module.retry_jobs(tmp_path,manifest)[0]==[]


def test_missing_result_is_not_assumed_to_be_a_failed_job(tmp_path):
    manifest=fixture(tmp_path,2)
    (tmp_path/'experiments/F06/F06_F_a__M2__9/result.json').unlink()
    with pytest.raises(ValueError,match='scheduler completion'):
        module.retry_jobs(tmp_path,manifest)
