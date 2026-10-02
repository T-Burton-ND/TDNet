import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from gridiron_ml.experiments import nextgen_screening_parallel, nextgen_screening_reduced_parallel
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file


def reduction_fixture(root, monkeypatch, variant='LR'):
    import gridiron_ml.experiments.nextgen_reduced_artifacts as reduced
    monkeypatch.setattr(reduced, 'verify_successful_run', lambda *args: None)
    full = root/'fingerprints/F09_F_a'
    full.mkdir(parents=True)
    (full/'values.parquet').write_text('fixture data')
    atomic_json(full/'feature_manifest.json', [{'name': 'x', 'matchup_counterpart': 'x'}])
    atomic_json(full/'provenance.json', {'data_sha256': sha256_file(full/'values.parquet'),
                                        'manifest_sha256': sha256_file(full/'feature_manifest.json')})
    points = json.loads((reduced.ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    evidence = []
    for model in ('M2', 'M4'):
        for i, point in enumerate(points[model]):
            folder = root/'experiments/F09'/f"F09_F_a__{model}__{point['id']}"
            folder.mkdir(parents=True)
            path = folder/'result.json'
            atomic_json(path, {'fingerprint_id': 'F09_F_a', 'model': model,
                               'hyperparameter_setpoint': point['id'],
                               'status': 'success' if i < 3 else 'failed',
                               'mae_2024': 10., 'mae_2025': 10.})
            if i < 3:
                pd.DataFrame({'source_feature': ['x'], 'normalized_importance': [1.]}).to_parquet(folder/'source_shap.parquet')
            evidence.append({'path': str(path), 'sha256': sha256_file(path)})
    candidate = root/'fingerprints'/f'F09_{variant}_a'
    candidate.mkdir(parents=True)
    proposal = {'full_reference': 'F09_F_a', 'full_screening_evidence': evidence,
                'progressive_parent': 'F06_R_a', 'parent_authorization': {'fixture': True}}
    atomic_json(candidate/'reduction_proposal.json', proposal)
    atomic_json(candidate/'provenance.json', {
        'reduction_proposal_sha256': sha256_file(candidate/'reduction_proposal.json'),
        'ancestry': ['F06_R_a' if variant == 'PR' else 'F09_F_a']})
    return candidate, evidence


def test_late_reduction_can_run_without_unrelated_generation_finalization(tmp_path, monkeypatch):
    _, evidence = reduction_fixture(tmp_path, monkeypatch)
    nextgen_screening_reduced_parallel.generation_barrier(tmp_path, 'F09_LR_a')
    Path(evidence[-1]['path']).unlink()
    with pytest.raises(ValueError, match='not finished'):
        nextgen_screening_reduced_parallel.generation_barrier(tmp_path, 'F09_LR_a')


def test_progressive_reduction_still_requires_accepted_parent(tmp_path, monkeypatch):
    import gridiron_ml.experiments.nextgen_progressive_artifacts as progressive
    reduction_fixture(tmp_path, monkeypatch, 'PR')
    with pytest.raises(FileNotFoundError):
        nextgen_screening_reduced_parallel.generation_barrier(tmp_path, 'F09_PR_a')
    monkeypatch.setattr(progressive, 'verified_parent', lambda *args: {'fixture': True})
    nextgen_screening_reduced_parallel.generation_barrier(tmp_path, 'F09_PR_a')


def test_proposal_and_terminal_reference_must_remain_bound(tmp_path, monkeypatch):
    candidate, evidence = reduction_fixture(tmp_path, monkeypatch)
    path = Path(evidence[-1]['path'])
    row = json.loads(path.read_text())
    row['mae_2024'] = 11.
    atomic_json(path, row)
    with pytest.raises(ValueError, match='evidence differs'):
        nextgen_screening_reduced_parallel.generation_barrier(tmp_path, 'F09_LR_a')
    (candidate/'reduction_proposal.json').write_text('{}')
    with pytest.raises(ValueError, match='proposal binding'):
        nextgen_screening_reduced_parallel.generation_barrier(tmp_path, 'F09_LR_a')


def test_new_runner_preserves_full_runner_scientific_computation(tmp_path):
    with pytest.raises(ValueError, match='only supports reduction'):
        nextgen_screening_reduced_parallel.generation_barrier(tmp_path, 'F09_F_a')
    old = ast.parse(Path(nextgen_screening_parallel.__file__).read_text())
    new = ast.parse(Path(nextgen_screening_reduced_parallel.__file__).read_text())
    old_functions = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
    for name, node in old_functions.items():
        if name not in ('execution_binding', 'generation_barrier'):
            assert ast.dump(node) == ast.dump(new_functions[name]), name


def test_result_verification_dispatches_to_reduced_runner_binding(tmp_path):
    from gridiron_ml.experiments.nextgen_results import verify_successful_run
    audit = tmp_path/'results/source_semantics_audit.json'
    audit.parent.mkdir(parents=True)
    audit.write_text('{}')
    folder = tmp_path/'fingerprints/F09_LR_a'
    folder.mkdir(parents=True)
    (folder/'values.parquet').write_text('fixture')
    provenance = {'data_sha256': sha256_file(folder/'values.parquet')}
    points = json.loads((nextgen_screening_reduced_parallel.ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    point = points['M2'][0]
    run_id = f"F09_LR_a__M2__{point['id']}"
    out = tmp_path/'experiments/F09'/run_id
    out.mkdir(parents=True)
    hashes = {}
    for name in ('predictions.parquet', 'source_shap.parquet'):
        (out/name).write_text('fixture output')
        hashes[name] = sha256_file(out/name)
    binding = nextgen_screening_reduced_parallel.execution_binding(tmp_path, provenance, 'M2', point)
    row = {'run_id': run_id, 'fingerprint_id': 'F09_LR_a', 'model': 'M2',
           'hyperparameter_setpoint': point['id'], 'data_sha256': provenance['data_sha256'],
           'execution_binding': binding, 'output_sha256': hashes}
    verify_successful_run(tmp_path, out, row, provenance, points)
    assert any(p.endswith('nextgen_progressive_artifacts.py') for p in binding['files'])
    row['execution_binding']['scheduling_policy'] = 'parallel_full_v1'
    with pytest.raises(ValueError, match='different inputs or computation'):
        verify_successful_run(tmp_path, out, row, provenance, points)
