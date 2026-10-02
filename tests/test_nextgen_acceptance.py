import json
from pathlib import Path

import pandas as pd
import pytest

import gridiron_ml.experiments.nextgen_acceptance as module
from gridiron_ml.experiments.nextgen_progressive_artifacts import verified_parent
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file


def fixture(root, monkeypatch, *, kept=60, candidate_mae=10.1):
    points = json.loads((module.ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    code = root/'fixture_code'
    code.write_text('fixture implementation')
    rows, reference_evidence = [], []
    for fingerprint, count in [('F06_F_c', 62), ('F06_R_c', kept)]:
        folder = root/'fingerprints'/fingerprint
        folder.mkdir(parents=True)
        records = [{'name': f'x{i}', 'generation': 'F06', 'matchup_counterpart': f'x{i}'} for i in range(count)]
        frame = pd.DataFrame({'target_game_id': [1, 1], 'team': ['A', 'B'], 'season': [2020, 2020],
                              **{r['name']: [1., 2.] for r in records}})
        frame.to_parquet(folder/'values.parquet', index=False)
        atomic_json(folder/'feature_manifest.json', records)
        atomic_json(folder/'provenance.json', {'data_sha256': sha256_file(folder/'values.parquet'),
                                               'manifest_sha256': sha256_file(folder/'feature_manifest.json'),
                                               'ancestry': [] if fingerprint == 'F06_F_c' else ['F06_F_c']})
        for model in ('M2', 'M4'):
            for point in points[model]:
                run_id = f"{fingerprint}__{model}__{point['id']}"
                directory = root/'experiments/F06'/run_id
                directory.mkdir(parents=True)
                pd.DataFrame({'source_feature': [r['name'] for r in records],
                              'normalized_importance': [1/count]*count}).to_parquet(directory/'source_shap.parquet')
                pd.DataFrame({'fixture_prediction': [1.]}).to_parquet(directory/'predictions.parquet')
                mae = 10. if fingerprint == 'F06_F_c' else candidate_mae
                row = {'row_type': 'run', 'run_id': run_id, 'fingerprint_id': fingerprint,
                       'model': model, 'hyperparameter_setpoint': point['id'], 'status': 'success',
                       'mae_2024': mae, 'mae_2025': mae,
                       'execution_binding': {'files': {str(code): sha256_file(code)}},
                       'output_sha256': {n: sha256_file(directory/n) for n in ('source_shap.parquet', 'predictions.parquet')}}
                atomic_json(directory/'result.json', row)
                rows.append(row)
                if fingerprint == 'F06_F_c':
                    reference_evidence.append({'path': str(directory/'result.json'),
                                               'sha256': sha256_file(directory/'result.json')})
    candidate = root/'fingerprints/F06_R_c'
    atomic_json(candidate/'reduction_proposal.json', {'full_reference': 'F06_F_c',
                                                     'full_screening_evidence': reference_evidence})
    provenance = json.loads((candidate/'provenance.json').read_text())
    provenance['reduction_proposal_sha256'] = sha256_file(candidate/'reduction_proposal.json')
    atomic_json(candidate/'provenance.json', provenance)

    def checked(root, fingerprint):
        folder = root/'fingerprints'/fingerprint
        return None, None, json.loads((folder/'feature_manifest.json').read_text()), json.loads((folder/'provenance.json').read_text())

    monkeypatch.setattr(module, 'checked_fingerprint', checked)
    monkeypatch.setattr(module, 'collect_results', lambda root: pd.DataFrame(rows))
    return candidate


def test_floor_acceptance_authorizes_only_that_parent_without_generation_finalization(tmp_path, monkeypatch):
    candidate = fixture(tmp_path, monkeypatch)
    before = sha256_file(candidate/'provenance.json')
    receipt = module.accept_floor_reduction(tmp_path, 'F06_R_c')
    assert receipt['accepted'] is True and receipt['floor_counts'] == {'F06': 60}
    assert sha256_file(candidate/'provenance.json') == before
    assert not (tmp_path/'results/F06/finalization.json').exists()
    authorization = verified_parent(tmp_path, 'F06_R_c')
    assert authorization['authorization_kind'] == 'floor_acceptance'
    assert module.accept_floor_reduction(tmp_path, 'F06_R_c') == receipt
    (candidate/'values.parquet').write_text('changed')
    with pytest.raises(ValueError, match='evidence changed'):
        verified_parent(tmp_path, 'F06_R_c')


@pytest.mark.parametrize('kept,mae,message', [(61, 10.1, 'generation floors'), (60, 11., 'fails measured')])
def test_passing_nonminimal_or_inaccurate_trial_cannot_authorize_parent(tmp_path, monkeypatch, kept, mae, message):
    fixture(tmp_path, monkeypatch, kept=kept, candidate_mae=mae)
    with pytest.raises(ValueError, match=message):
        module.accept_floor_reduction(tmp_path, 'F06_R_c')
    assert not (tmp_path/'results/F06/F06_R_c/acceptance.json').exists()


def test_parent_receipt_metrics_are_recomputed_not_trusted(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    module.accept_floor_reduction(tmp_path, 'F06_R_c')
    path = tmp_path/'results/F06/F06_R_c/acceptance.json'
    receipt = json.loads(path.read_text())
    receipt['acceptance']['joint_mean_delta'] = 0.
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match='measured acceptance changed'):
        verified_parent(tmp_path, 'F06_R_c')


def test_parent_receipt_cannot_omit_successful_shap_evidence(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    module.accept_floor_reduction(tmp_path, 'F06_R_c')
    path = tmp_path/'results/F06/F06_R_c/acceptance.json'
    receipt = json.loads(path.read_text())
    receipt['evidence'] = [e for e in receipt['evidence'] if not e['path'].endswith('source_shap.parquet')]
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match='successful output evidence'):
        verified_parent(tmp_path, 'F06_R_c')
