"""Construct immutable late-reduction trials from measured full screening.

A materialized trial is not an accepted reduction. It must subsequently pass
M2/M4 screening and measured_acceptance before recommendation or inheritance.
"""
import json
from pathlib import Path

import pandas as pd

from .nextgen_contract import parse_fingerprint_id, validate_feature_manifest
from .nextgen_reduction import check_survivors, screening_medians, source_importance_consensus
from .nextgen_results import verify_successful_run
from .nextgen_screening import checked_fingerprint
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]


def select_trial_frame(frame, records, survivors, *, shortfall_exceptions=None,
                       full_reference_names=None):
    """Preserve row identity and metadata, removing every deselected column."""
    survivors = list(survivors)
    if not survivors or len(survivors) != len(set(survivors)):
        raise ValueError('Unique nonempty survivor list required')
    counts = check_survivors(survivors, records, shortfall_exceptions=shortfall_exceptions)
    all_names = [r['name'] for r in records]
    if not set(all_names) <= set(frame):
        raise ValueError('Source lacks declared full features')
    retained = [r for r in records if r['name'] in set(survivors)]
    names = [r['name'] for r in retained]
    # A progressive pool can already be reduced by its accepted ancestry.
    # In particular, a pool at every feature floor must not lose another column.
    inherited_reduction = (full_reference_names is not None
                           and set(all_names) < set(full_reference_names))
    if len(names) == len(all_names) and not inherited_reduction:
        raise ValueError('Reduction trial must remove at least one feature')
    metadata = [c for c in frame if c not in set(all_names)]
    return frame[metadata + names].copy(), retained, counts


def verified_full_screening(root, full_id, records, provenance):
    """Read the terminal full matrix and verify all successful source SHAP."""
    generation, lineage, _ = parse_fingerprint_id(full_id)
    if lineage != 'F':
        raise ValueError('Full screening reference required')
    points = json.loads((ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    runs, tables, evidence = [], {}, []
    for model in ('M2', 'M4'):
        for point in points[model]:
            run_id = f"{full_id}__{model}__{point['id']}"
            directory = root/'experiments'/generation/run_id
            path = directory/'result.json'
            if not path.exists():
                raise ValueError('Full screening has not finished every frozen configuration')
            row = json.loads(path.read_text())
            if (row.get('fingerprint_id'), row.get('model'), row.get('hyperparameter_setpoint')) != (full_id, model, point['id']):
                raise ValueError('Full screening result identity mismatch')
            runs.append(row)
            evidence.append({'path': str(path), 'sha256': sha256_file(path)})
            if row.get('status') == 'success':
                verify_successful_run(root, directory, row, provenance, points)
                tables[(model, point['id'])] = pd.read_parquet(directory/'source_shap.parquet')
    medians = screening_medians(pd.DataFrame(runs), points)
    consensus = source_importance_consensus(tables, points, records)
    return medians, consensus, evidence


def materialize_late_trial(root, full_id, survivors, *, rationale, shortfall_exceptions=None):
    """Require terminal frozen screening and verified SHAP for the full source.

    Only F06 R and later LR are handled here. Progressive ancestry needs its
    own expansion from a previously accepted reduced fingerprint.
    """
    root = Path(root)
    generation, lineage, design = parse_fingerprint_id(full_id)
    if lineage != 'F' or not isinstance(rationale, str) or not rationale.strip():
        raise ValueError('A full source and scientific reduction rationale are required')
    trial_id = f"{generation}_{'R' if generation == 'F06' else 'LR'}_{design}"
    dest = root/'fingerprints'/trial_id
    if dest.exists():
        raise ValueError('Trial already exists; archive its artifacts and evidence before revising')
    _, _, records, provenance = checked_fingerprint(root, full_id)
    medians, consensus, evidence = verified_full_screening(root, full_id, records, provenance)
    source = root/'fingerprints'/full_id/'values.parquet'
    frame, retained, counts = select_trial_frame(pd.read_parquet(source), records, survivors,
                                                shortfall_exceptions=shortfall_exceptions)
    schema = json.loads((ROOT/'configs/experiments/nextgen_feature_manifest_schema_v1.json').read_text())
    validate_feature_manifest(retained, schema)
    # Recheck after evidence reads, before creating the candidate directory.
    if sha256_file(source) != provenance['data_sha256']:
        raise ValueError('Full source changed during reduction preparation')
    dest.mkdir(parents=True)
    manifest = dest/'feature_manifest.json'
    atomic_json(manifest, retained)
    data = dest/'values.parquet'
    frame.to_parquet(data, index=False, compression='zstd')
    proposal = {'full_reference': full_id, 'rationale': rationale,
                'removed_features': [r['name'] for r in records if r not in retained],
                'survivor_counts': counts, 'shortfall_exceptions': shortfall_exceptions or {},
                'full_screening_evidence': evidence, 'full_medians': medians.to_dict(orient='index'),
                'source_importance': consensus.to_dict(orient='records'),
                'accepted': False, 'status': 'awaiting_candidate_screening_and_measured_acceptance'}
    atomic_json(dest/'reduction_proposal.json', proposal)
    child = dict(provenance)
    child.update(fingerprint_id=trial_id, ancestry=[full_id], parent_data_sha256=provenance['data_sha256'],
                 data_sha256=sha256_file(data), manifest_sha256=sha256_file(manifest),
                 feature_count=len(retained), rows=len(frame),
                 reduction_proposal_sha256=sha256_file(dest/'reduction_proposal.json'),
                 reduction_accepted=False)
    atomic_json(dest/'provenance.json', child)
    return {'fingerprint': trial_id, 'features': len(retained), 'rows': len(frame), 'accepted': False}
