"""Run one frozen full-screening cell under the parallel authorization."""
import json
import os
from pathlib import Path

from gridiron_ml.experiments.nextgen_screening_parallel import run_task, ROOT
from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file

manifest = json.loads(Path(os.environ['NEXTGEN_JOB_MANIFEST']).read_text())
points = ROOT / 'configs/experiments/nextgen_screening_setpoints_v1.json'
if manifest['setpoints_sha256'] != sha256_file(points):
    raise ValueError('Frozen setpoints changed since planning')
selected = [j for j in manifest['jobs'] if j['task_id'] == int(os.environ['SGE_TASK_ID'])]
if len(selected) != 1:
    raise ValueError('Unknown task index')
job = selected[0]
config = json.loads((ROOT / 'configs/experiments/nextgen_fingerprints_v1.json').read_text())
result = run_task(Path(config['artifact_root']), job['fingerprint'], job['model'], job['setpoint'])
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['status'] == 'success' else 1)
