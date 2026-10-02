"""Accept a measured floor-sized reduced parent independently of other designs."""
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from .nextgen_contract import parse_fingerprint_id, parent_of
from .nextgen_reduction import check_survivors, measured_acceptance, source_importance_consensus
from .nextgen_results import collect_results
from .nextgen_screening import ROOT, checked_fingerprint
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file


def assert_floor_reached(full_records, candidate_records, shortfall_exceptions=None):
    """This fast path cannot finalize an arbitrary first passing trial."""
    counts = check_survivors([r['name'] for r in candidate_records], full_records,
                            shortfall_exceptions=shortfall_exceptions)
    available = Counter(r['generation'] for r in full_records)
    targets = {g: min(60 if g == 'F06' else 10, n) for g, n in available.items()}
    if counts != targets:
        raise ValueError('Independent acceptance requires the recorded generation floors')
    return targets


def accept_floor_reduction(root, fingerprint):
    """Write immutable acceptance evidence; never alter trained input provenance.

    Non-floor trials must continue the reduction search or use complete
    generation finalization. This optimization is for an already minimal
    representation under the recorded concrete-feature floor policy.
    """
    from .nextgen_finalize import verify_reduced_projection
    from .nextgen_progressive import progressive_pool
    from .nextgen_progressive_artifacts import verified_parent

    root = Path(root)
    generation, lineage, design = parse_fingerprint_id(fingerprint)
    if lineage not in ('R', 'PR'):
        raise ValueError('Only R/PR fingerprints can become progressive parents')
    full_id = f'{generation}_F_{design}'
    table = collect_results(root)
    runs = table.loc[table.row_type.eq('run')]
    points = json.loads((ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    data, records, provenance, evidence = {}, {}, {}, {}

    def bind(path, expected=None):
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError('Acceptance evidence changed')
        evidence[str(path)] = digest

    for name in (full_id, fingerprint):
        _, _, records[name], provenance[name] = checked_fingerprint(root, name)
        folder = root/'fingerprints'/name
        data[name] = pd.read_parquet(folder/'values.parquet')
        for filename in ('values.parquet', 'feature_manifest.json', 'provenance.json'):
            bind(folder/filename)
        shap = {}
        for row in runs.loc[runs.fingerprint_id.eq(name)].itertuples(index=False):
            directory = root/'experiments'/generation/row.run_id
            bind(directory/'result.json')
            if row.status == 'success':
                saved = json.loads((directory/'result.json').read_text())
                for path, digest in saved['execution_binding']['files'].items():
                    bind(Path(path), digest)
                if saved['execution_binding'].get('market_sha256') is not None:
                    bind(root/'canonical/evaluation_market_sidecar.parquet', saved['execution_binding']['market_sha256'])
                for filename in ('source_shap.parquet', 'predictions.parquet'):
                    bind(directory/filename, saved['output_sha256'][filename])
                shap[(row.model, row.hyperparameter_setpoint)] = pd.read_parquet(directory/'source_shap.parquet')
        source_importance_consensus(shap, points, records[name])
    acceptance = measured_acceptance(runs.loc[runs.fingerprint_id.eq(full_id)],
                                     runs.loc[runs.fingerprint_id.eq(fingerprint)], points, design=design)
    if not acceptance['accepted']:
        raise ValueError('Floor reduction fails measured acceptance')
    folder = root/'fingerprints'/fingerprint
    proposal_path = folder/'reduction_proposal.json'
    bind(proposal_path, provenance[fingerprint].get('reduction_proposal_sha256'))
    if not provenance[fingerprint].get('reduction_proposal_sha256'):
        raise ValueError('Reduction lacks proposal binding')
    proposal = json.loads(proposal_path.read_text())
    if proposal.get('full_reference') != full_id:
        raise ValueError('Reduction full reference changed')
    expected_full = {str(root/'experiments'/generation/r.run_id/'result.json')
                     for r in runs.loc[runs.fingerprint_id.eq(full_id)].itertuples()}
    if {e['path'] for e in proposal.get('full_screening_evidence', [])} != expected_full:
        raise ValueError('Proposal lacks complete reference evidence')
    for item in proposal['full_screening_evidence']:
        bind(Path(item['path']), item['sha256'])
    allowed = records[full_id]
    expected_parent = full_id
    if lineage == 'PR':
        expected_parent = parent_of(fingerprint)
        authorization = verified_parent(root, expected_parent)
        if (proposal.get('parent_authorization') != authorization
                or proposal.get('progressive_parent') != expected_parent):
            raise ValueError('Progressive parent authorization changed')
        pool_path = root/'progressive_pools'/fingerprint/'provenance.json'
        if not proposal.get('pool_provenance_sha256'):
            raise ValueError('Progressive proposal lacks pool binding')
        bind(pool_path, proposal['pool_provenance_sha256'])
        _, _, parent_records, _ = checked_fingerprint(root, expected_parent)
        _, allowed, pool = progressive_pool(
            full_id, expected_parent, data[full_id], records[full_id],
            pd.read_parquet(root/'fingerprints'/expected_parent/'values.parquet'), parent_records)
        if pool['excluded_full_game_ids']:
            raise ValueError('Progressive reference cohort differs')
        parent_receipt = Path(authorization.get('acceptance_path', authorization.get('finalization_path')))
        bind(parent_receipt)
        if 'finalization_path' in authorization:
            bind(parent_receipt.parent/'recommendations.json')
        for item in json.loads(parent_receipt.read_text())['evidence']:
            bind(Path(item['path']), item['sha256'])
    if provenance[fingerprint].get('ancestry') != [expected_parent]:
        raise ValueError('Reduction ancestry changed')
    exceptions = proposal.get('shortfall_exceptions', {})
    verify_reduced_projection(data[full_id], records[full_id], data[fingerprint], records[fingerprint],
                              allowed, shortfall_exceptions=exceptions)
    floors = assert_floor_reached(records[full_id], records[fingerprint], exceptions)
    if any(sha256_file(Path(path)) != digest for path, digest in evidence.items()):
        raise ValueError('Acceptance evidence changed during validation')
    receipt = {'fingerprint_id': fingerprint, 'full_reference': full_id, 'accepted': True,
               'minimum_floor_reached': True, 'floor_counts': floors, 'acceptance': acceptance,
               'shortfall_exceptions': exceptions,
               'selection_basis': 'All concrete generation floors reached; no smaller candidate under the recorded floor policy.',
               'evidence': [{'path': p, 'sha256': h} for p, h in sorted(evidence.items())]}
    path = root/'results'/generation/fingerprint/'acceptance.json'
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError('Existing acceptance differs; preserve and version its evidence')
    atomic_json(path, receipt)
    return receipt
