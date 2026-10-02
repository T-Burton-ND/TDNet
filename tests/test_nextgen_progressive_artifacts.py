import json
from pathlib import Path

import pandas as pd
import pytest

from gridiron_ml.experiments.nextgen_progressive_artifacts import verified_parent
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file


def finalized_parent(root):
    folder = root/'fingerprints/F06_R_a'
    folder.mkdir(parents=True)
    evidence = []
    for name in ('values.parquet', 'feature_manifest.json', 'provenance.json'):
        path = folder/name
        path.write_text('fixture')
        evidence.append({'path': str(path), 'sha256': sha256_file(path)})
    output = root/'results/F06'
    output.mkdir(parents=True)
    report = {'generation': 'F06', 'evidence': evidence,
              'reduction_acceptance': {'F06_R_a': {'accepted': True}}}
    atomic_json(output/'recommendations.json', report)
    final = {'generation': 'F06', 'evidence': evidence,
             'recommendations_sha256': sha256_file(output/'recommendations.json')}
    for flag in ('training_finalized', 'shap_finalized', 'reduction_finalized',
                 'lineages_compared', 'recommendations_finalized'):
        final[flag] = True
    atomic_json(output/'finalization.json', final)
    return folder, output, report, final


def test_acceptance_is_bound_to_unchanged_parent_artifacts(tmp_path):
    folder, _, _, _ = finalized_parent(tmp_path)
    assert verified_parent(tmp_path, 'F06_R_a')['acceptance']['accepted'] is True
    (folder/'values.parquet').write_text('changed after finalization')
    with pytest.raises(ValueError, match='evidence changed'):
        verified_parent(tmp_path, 'F06_R_a')


def test_finalization_flags_do_not_replace_measured_acceptance(tmp_path):
    _, output, report, final = finalized_parent(tmp_path)
    report['reduction_acceptance']['F06_R_a']['accepted'] = False
    atomic_json(output/'recommendations.json', report)
    final['recommendations_sha256'] = sha256_file(output/'recommendations.json')
    atomic_json(output/'finalization.json', final)
    with pytest.raises(ValueError, match='measured acceptance'):
        verified_parent(tmp_path, 'F06_R_a')


def test_acceptance_for_different_parent_and_partial_finalization_rejected(tmp_path):
    _, output, _, final = finalized_parent(tmp_path)
    with pytest.raises(ValueError, match='measured acceptance'):
        verified_parent(tmp_path, 'F06_R_b')
    final['shap_finalized'] = False
    atomic_json(output/'finalization.json', final)
    with pytest.raises(ValueError, match='not finalized'):
        verified_parent(tmp_path, 'F06_R_a')


def test_evidence_must_explicitly_bind_parent_inputs(tmp_path):
    _, output, report, final = finalized_parent(tmp_path)
    report['evidence'] = report['evidence'][:1]
    final['evidence'] = report['evidence']
    atomic_json(output/'recommendations.json', report)
    final['recommendations_sha256'] = sha256_file(output/'recommendations.json')
    atomic_json(output/'finalization.json', final)
    with pytest.raises(ValueError, match='bind parent fingerprint'):
        verified_parent(tmp_path, 'F06_R_a')


@pytest.mark.parametrize('new_count', [10, 12])
def test_progressive_trial_persists_only_pool_survivors_and_requires_screening(tmp_path, monkeypatch, new_count):
    import gridiron_ml.experiments.nextgen_progressive_artifacts as module
    from gridiron_ml.experiments.nextgen_designs import Formula, feature_record

    parent, output, report, final = finalized_parent(tmp_path)
    names = [f'old{i}' for i in range(62)] + [f'new{i}' for i in range(new_count)]
    records = [feature_record(Formula(n, (n,), 'identity', 'Fixture', 'fraction'),
                              'F06' if n.startswith('old') else 'F09', 'a', n,
                              endpoints=['fixture']) for n in names]
    frame = pd.DataFrame({'target_game_id': [1, 1], 'team': ['A', 'B'],
                          'season': [2020, 2020], **{n: [1., 2.] for n in names}})
    full = tmp_path/'fingerprints/F09_F_a'
    full.mkdir(parents=True)
    for folder, source, manifest in ((full, frame, records),
                                    (parent, frame[['target_game_id', 'team', 'season', *names[:60]]], records[:60])):
        source.to_parquet(folder/'values.parquet', index=False)
        atomic_json(folder/'feature_manifest.json', manifest)
        atomic_json(folder/'provenance.json', {
            'data_sha256': sha256_file(folder/'values.parquet'),
            'manifest_sha256': sha256_file(folder/'feature_manifest.json')})
    for item in report['evidence']:
        item['sha256'] = sha256_file(Path(item['path']))
    atomic_json(output/'recommendations.json', report)
    final['recommendations_sha256'] = sha256_file(output/'recommendations.json')
    atomic_json(output/'finalization.json', final)

    def checked(root, fingerprint):
        folder = root/'fingerprints'/fingerprint
        return (None, None, json.loads((folder/'feature_manifest.json').read_text()),
                json.loads((folder/'provenance.json').read_text()))

    monkeypatch.setattr(module, 'checked_fingerprint', checked)
    module.materialize_progressive_pool(tmp_path, 'F09_F_a')
    survivors = names[:60] + names[62:72]
    with pytest.raises(ValueError, match='Unknown surviving'):
        module.materialize_progressive_trial(tmp_path, 'F09_F_a', survivors + ['old60'], rationale='fixture')
    with pytest.raises(ValueError, match='not finished'):
        module.materialize_progressive_trial(tmp_path, 'F09_F_a', survivors, rationale='fixture')
    assert not (tmp_path/'fingerprints/F09_PR_a').exists()
    monkeypatch.setattr(module, 'verified_full_screening',
                        lambda *args: (pd.DataFrame(), pd.DataFrame(), []))
    result = module.materialize_progressive_trial(tmp_path, 'F09_F_a', survivors, rationale='fixture')
    assert result == {'fingerprint': 'F09_PR_a', 'features': 70, 'rows': 2, 'accepted': False}
    child = tmp_path/'fingerprints/F09_PR_a'
    actual = pd.read_parquet(child/'values.parquet')
    pd.testing.assert_frame_equal(actual, frame[['target_game_id', 'team', 'season', *survivors]])
    provenance = json.loads((child/'provenance.json').read_text())
    assert provenance['ancestry'] == ['F06_R_a']
    assert provenance['reduction_accepted'] is False
    with pytest.raises(ValueError, match='already exists'):
        module.materialize_progressive_trial(tmp_path, 'F09_F_a', survivors, rationale='fixture')
