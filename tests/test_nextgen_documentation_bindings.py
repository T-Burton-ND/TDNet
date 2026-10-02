import importlib.util
import json
from pathlib import Path

import pytest

from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file


spec = importlib.util.spec_from_file_location(
    'nextgen_consensus_diagnostics',
    Path(__file__).resolve().parents[1]/'scripts/nextgen_consensus_diagnostics.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_documentation_rejects_changed_observations_and_duplicate_definitions(tmp_path):
    folder = tmp_path/'docs/nextgen_fingerprints'
    folder.mkdir(parents=True)
    code = tmp_path/'source.py'
    code.write_text('fixture source')
    evidence = tmp_path/'observations.json'
    evidence.write_text('{"matches": 5}')
    kinds = ('graph', 'temporal', 'opponent', 'boxscore', 'efficiency', 'prior')
    for kind in kinds:
        data = {'common': {'implementation': 'source.py',
                           'implementation_sha256': sha256_file(code),
                           'supporting_evidence': [{'path': 'observations.json',
                                                   'sha256': sha256_file(evidence)}]},
                'features': {kind: {'units': 'fixture units'}}}
        (folder/f'inherited_{kind}_documentation.json').write_text(json.dumps(data))
    assert set(module.load_source_supplements(tmp_path)) == set(kinds)
    evidence.write_text('{"matches": 0}')
    with pytest.raises(ValueError, match='observations changed'):
        module.load_source_supplements(tmp_path)
    evidence.write_text('{"matches": 5}')
    code.write_text('changed computation')
    with pytest.raises(ValueError, match='implementation change'):
        module.load_source_supplements(tmp_path)
    code.write_text('fixture source')
    data['features'] = {'graph': {'units': 'conflicting units'}}
    (folder/'inherited_prior_documentation.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Ambiguous'):
        module.load_source_supplements(tmp_path)
