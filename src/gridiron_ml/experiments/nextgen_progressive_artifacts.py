"""Persist progressive pools only from verified, accepted parent evidence.

Pools are preparation artifacts, not screened or accepted fingerprints.
"""
import json
from pathlib import Path

import pandas as pd

from .nextgen_contract import parent_of, parse_fingerprint_id, validate_feature_manifest
from .nextgen_progressive import progressive_pool
from .nextgen_reduction import check_survivors
from .nextgen_screening import checked_fingerprint
from .nextgen_reduced_artifacts import ROOT, select_trial_frame, verified_full_screening
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file


def verified_parent(root, parent_id):
    """Bind acceptance to unchanged finalization, recommendations, and inputs."""
    root = Path(root)
    generation, lineage, design = parse_fingerprint_id(parent_id)
    if lineage != ('R' if generation == 'F06' else 'PR'):
        raise ValueError('Progressive parent must be the previous reduced lineage')
    folder = root/'results'/generation
    independent = folder/parent_id/'acceptance.json'
    if independent.exists():
        from .nextgen_acceptance import assert_floor_reached
        from .nextgen_reduction import measured_acceptance

        receipt = json.loads(independent.read_text())
        full_id = f'{generation}_F_{design}'
        if (receipt.get('fingerprint_id') != parent_id or receipt.get('full_reference') != full_id
                or receipt.get('accepted') is not True or receipt.get('minimum_floor_reached') is not True):
            raise ValueError('Independent parent lacks finalized floor acceptance')
        bound = {}
        for item in receipt.get('evidence', []):
            path = Path(item['path'])
            if str(path) in bound or sha256_file(path) != item['sha256']:
                raise ValueError('Independent parent evidence changed or is duplicated')
            bound[str(path)] = item['sha256']
        manifests = {}
        matrices = {}
        points = json.loads((ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
        for fingerprint in (full_id, parent_id):
            source = root/'fingerprints'/fingerprint
            for name in ('values.parquet', 'feature_manifest.json', 'provenance.json'):
                if str(source/name) not in bound:
                    raise ValueError('Independent acceptance lacks input evidence')
            manifests[fingerprint] = json.loads((source/'feature_manifest.json').read_text())
            rows = []
            for model in ('M2', 'M4'):
                for point in points[model]:
                    path = root/'experiments'/generation/f"{fingerprint}__{model}__{point['id']}"/'result.json'
                    if str(path) not in bound:
                        raise ValueError('Independent acceptance lacks terminal matrix evidence')
                    row = json.loads(path.read_text())
                    if (row.get('fingerprint_id'), row.get('model'), row.get('hyperparameter_setpoint')) != (fingerprint, model, point['id']):
                        raise ValueError('Independent parent result identity changed')
                    if row.get('status') == 'success':
                        for name in ('source_shap.parquet', 'predictions.parquet'):
                            digest = row.get('output_sha256', {}).get(name)
                            if not digest or bound.get(str(path.parent/name)) != digest:
                                raise ValueError('Independent acceptance lacks successful output evidence')
                        for source, digest in row['execution_binding']['files'].items():
                            if bound.get(source) != digest:
                                raise ValueError('Independent acceptance lacks execution evidence')
                        market = row['execution_binding'].get('market_sha256')
                        if market is not None and bound.get(str(root/'canonical/evaluation_market_sidecar.parquet')) != market:
                            raise ValueError('Independent acceptance lacks market evidence')
                    rows.append(row)
            matrices[fingerprint] = pd.DataFrame(rows)
        floors = assert_floor_reached(manifests[full_id], manifests[parent_id], receipt.get('shortfall_exceptions'))
        acceptance = measured_acceptance(matrices[full_id], matrices[parent_id], points, design=design)
        if floors != receipt.get('floor_counts') or not acceptance['accepted'] or acceptance != receipt.get('acceptance'):
            raise ValueError('Independent parent measured acceptance changed')
        return {'parent_id': parent_id, 'authorization_kind': 'floor_acceptance',
                'acceptance_path': str(independent), 'acceptance_sha256': sha256_file(independent),
                'acceptance': acceptance}
    final_path = folder/'finalization.json'
    final = json.loads(final_path.read_text())
    flags = ('training_finalized', 'shap_finalized', 'reduction_finalized',
             'lineages_compared', 'recommendations_finalized')
    if final.get('generation') != generation or not all(final.get(k) is True for k in flags):
        raise ValueError('Parent generation is not finalized')
    recommendation_path = folder/'recommendations.json'
    if sha256_file(recommendation_path) != final.get('recommendations_sha256'):
        raise ValueError('Parent recommendation binding changed')
    report = json.loads(recommendation_path.read_text())
    acceptance = report.get('reduction_acceptance', {}).get(parent_id, {})
    if report.get('generation') != generation or acceptance.get('accepted') is not True:
        raise ValueError('Progressive parent lacks measured acceptance')
    evidence = report.get('evidence')
    if not evidence or final.get('evidence') != evidence:
        raise ValueError('Parent finalization evidence is missing or inconsistent')
    bound = {}
    for item in evidence:
        path = Path(item['path'])
        if str(path) in bound or sha256_file(path) != item['sha256']:
            raise ValueError('Parent evidence changed or is duplicated')
        bound[str(path)] = item['sha256']
    for filename in ('values.parquet', 'feature_manifest.json', 'provenance.json'):
        path = root/'fingerprints'/parent_id/filename
        if str(path) not in bound:
            raise ValueError('Finalization does not bind parent fingerprint inputs')
    return {'parent_id': parent_id, 'finalization_path': str(final_path),
            'finalization_sha256': sha256_file(final_path),
            'recommendations_sha256': sha256_file(recommendation_path),
            'acceptance': acceptance}


def materialize_progressive_pool(root, full_id, *, shortfall_exceptions=None):
    """Preserve the entire new family and accepted prior survivors, unpruned.

Later screening/pruning must create an immutable trial under fingerprints;
this function deliberately writes only under progressive_pools.
"""
    root = Path(root)
    generation, lineage, design = parse_fingerprint_id(full_id)
    if generation == 'F06' or lineage != 'F':
        raise ValueError('A later-generation full fingerprint is required')
    child_id = f'{generation}_PR_{design}'
    parent_id = parent_of(child_id)
    dest = root/'progressive_pools'/child_id
    if dest.exists():
        raise ValueError('Progressive pool already exists; preserve immutable evidence')
    authorization = verified_parent(root, parent_id)
    _, _, full_records, full_prov = checked_fingerprint(root, full_id)
    _, _, parent_records, parent_prov = checked_fingerprint(root, parent_id)
    full_path = root/'fingerprints'/full_id/'values.parquet'
    parent_path = root/'fingerprints'/parent_id/'values.parquet'
    frame, records, audit = progressive_pool(
        full_id, parent_id, pd.read_parquet(full_path), full_records,
        pd.read_parquet(parent_path), parent_records)
    counts = check_survivors([r['name'] for r in records], records,
                            shortfall_exceptions=shortfall_exceptions)
    if (sha256_file(full_path) != full_prov['data_sha256']
            or sha256_file(parent_path) != parent_prov['data_sha256']
            or verified_parent(root, parent_id) != authorization):
        raise ValueError('Progressive sources changed during preparation')
    dest.mkdir(parents=True)
    atomic_json(dest/'feature_manifest.json', records)
    frame.to_parquet(dest/'values.parquet', index=False, compression='zstd')
    audit.update(status='unpruned_pool_not_screened', parent_authorization=authorization,
                 parent_data_sha256=parent_prov['data_sha256'],
                 full_data_sha256=full_prov['data_sha256'],
                 full_manifest_sha256=full_prov['manifest_sha256'],
                 data_sha256=sha256_file(dest/'values.parquet'),
                 manifest_sha256=sha256_file(dest/'feature_manifest.json'),
                 feature_counts_by_generation=counts, rows=len(frame),
                 shortfall_exceptions=shortfall_exceptions or {})
    atomic_json(dest/'provenance.json', audit)
    return audit


def materialize_progressive_trial(root, full_id, survivors, *, rationale,
                                  shortfall_exceptions=None):
    """Prune an authorized pool without resurrecting any previously removed feature.

    Full-reference SHAP informs a proposed selection; its relevance to this
    changed representation is provisional until the candidate's own screening.
    """
    root = Path(root)
    generation, lineage, design = parse_fingerprint_id(full_id)
    if (generation == 'F06' or lineage != 'F' or not isinstance(rationale, str)
            or not rationale.strip()):
        raise ValueError('A later full reference and scientific rationale are required')
    child_id = f'{generation}_PR_{design}'
    parent_id = parent_of(child_id)
    dest = root/'fingerprints'/child_id
    if dest.exists():
        raise ValueError('Progressive trial already exists; preserve immutable evidence')
    pool = root/'progressive_pools'/child_id
    audit_path = pool/'provenance.json'
    audit = json.loads(audit_path.read_text())
    authorization = verified_parent(root, parent_id)
    if (audit.get('fingerprint_id') != child_id or audit.get('full_reference') != full_id
            or audit.get('ancestry') != [parent_id]
            or audit.get('parent_authorization') != authorization):
        raise ValueError('Progressive pool ancestry or authorization changed')
    for name, key in [('values.parquet', 'data_sha256'),
                      ('feature_manifest.json', 'manifest_sha256')]:
        if sha256_file(pool/name) != audit.get(key):
            raise ValueError('Progressive pool content changed')
    _, _, full_records, full_prov = checked_fingerprint(root, full_id)
    _, _, parent_records, parent_prov = checked_fingerprint(root, parent_id)
    if (audit.get('full_data_sha256') != full_prov['data_sha256']
            or audit.get('full_manifest_sha256') != full_prov['manifest_sha256']
            or audit.get('parent_data_sha256') != parent_prov['data_sha256']):
        raise ValueError('Progressive source binding changed')
    expected, expected_records, recomputed = progressive_pool(
        full_id, parent_id,
        pd.read_parquet(root/'fingerprints'/full_id/'values.parquet'), full_records,
        pd.read_parquet(root/'fingerprints'/parent_id/'values.parquet'), parent_records)
    if recomputed['excluded_full_game_ids']:
        raise ValueError('Progressive full-reference comparison requires the same target cohort')
    records = json.loads((pool/'feature_manifest.json').read_text())
    frame = pd.read_parquet(pool/'values.parquet')
    if records != expected_records:
        raise ValueError('Progressive pool feature definitions changed')
    pd.testing.assert_frame_equal(frame, expected)
    reduced, retained, counts = select_trial_frame(
        frame, records, survivors, shortfall_exceptions=shortfall_exceptions,
        full_reference_names=[r['name'] for r in full_records])
    schema = json.loads((ROOT/'configs/experiments/nextgen_feature_manifest_schema_v1.json').read_text())
    validate_feature_manifest(retained, schema)
    medians, consensus, evidence = verified_full_screening(root, full_id, full_records, full_prov)
    if verified_parent(root, parent_id) != authorization:
        raise ValueError('Parent evidence changed during progressive preparation')
    for name, key in [('values.parquet', 'data_sha256'),
                      ('feature_manifest.json', 'manifest_sha256')]:
        if sha256_file(root/'fingerprints'/full_id/name) != full_prov[key]:
            raise ValueError('Full reference changed during progressive preparation')
    dest.mkdir(parents=True)
    atomic_json(dest/'feature_manifest.json', retained)
    reduced.to_parquet(dest/'values.parquet', index=False, compression='zstd')
    proposal = {'full_reference': full_id, 'progressive_parent': parent_id,
                'parent_authorization': authorization,
                'pool_provenance_sha256': sha256_file(audit_path),
                'rationale': rationale, 'survivor_counts': counts,
                'removed_pool_features': [r['name'] for r in records if r not in retained],
                'shortfall_exceptions': shortfall_exceptions or {},
                'full_screening_evidence': evidence, 'full_medians': medians.to_dict(orient='index'),
                'full_reference_source_importance': consensus.to_dict(orient='records'),
                'accepted': False, 'status': 'awaiting_candidate_screening_and_measured_acceptance'}
    atomic_json(dest/'reduction_proposal.json', proposal)
    provenance = dict(full_prov)
    provenance.update(fingerprint_id=child_id, ancestry=[parent_id],
                      parent_data_sha256=parent_prov['data_sha256'], full_reference=full_id,
                      data_sha256=sha256_file(dest/'values.parquet'),
                      manifest_sha256=sha256_file(dest/'feature_manifest.json'),
                      feature_count=len(retained), rows=len(reduced), reduction_accepted=False,
                      reduction_proposal_sha256=sha256_file(dest/'reduction_proposal.json'))
    atomic_json(dest/'provenance.json', provenance)
    return {'fingerprint': child_id, 'features': len(retained),
            'rows': len(reduced), 'accepted': False}
