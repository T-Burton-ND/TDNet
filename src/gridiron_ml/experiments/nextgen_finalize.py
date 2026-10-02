"""Finalize later generations from measured full, late, and progressive evidence."""
import json
from pathlib import Path

import pandas as pd

from .nextgen_contract import parent_of
from .nextgen_progressive import progressive_pool
from .nextgen_progressive_artifacts import verified_parent
from .nextgen_recommendations import rank_lineages
from .nextgen_reduction import check_survivors, measured_acceptance, screening_medians, source_importance_consensus
from .nextgen_results import collect_results
from .nextgen_screening import ROOT, checked_fingerprint
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file


def verify_reduced_projection(full_frame, full_records, candidate_frame, candidate_records,
                              allowed_records, *, shortfall_exceptions=None):
    """Check pair/floor semantics, unchanged definitions and identical targets."""
    names = [r['name'] for r in candidate_records]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate reduced feature')
    allowed = {r['name']: r for r in allowed_records}
    if any(allowed.get(r['name']) != r for r in candidate_records):
        raise ValueError('Reduced feature is outside its allowed ancestry or changed')
    if len(names) >= len(allowed):
        raise ValueError('Reduction did not prune its candidate pool')
    check_survivors(names, full_records, shortfall_exceptions=shortfall_exceptions)
    full_names = {r['name'] for r in full_records}
    columns = [c for c in full_frame if c not in full_names or c in names]
    if set(candidate_frame) != set(columns):
        raise ValueError('Reduced frame has undeclared or missing columns')
    pd.testing.assert_frame_equal(full_frame[columns], candidate_frame[columns])


def finalize_generation(root, generation):
    """Require complete measured lineage comparisons before authorizing descendants."""
    if generation not in ('F09', 'F10', 'F11', 'F12'):
        raise ValueError('Use the separate F06 finalizer for the baseline')
    root = Path(root)
    table = collect_results(root)
    runs = table.loc[table.row_type.eq('run')]
    summaries = table.loc[table.row_type.eq('fingerprint_architecture_summary')]
    points = json.loads((ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    evidence, acceptances, recommendations = {}, {}, []

    def bind(path):
        evidence[str(path)] = sha256_file(path)

    for design in 'abc':
        names = [f'{generation}_{v}_{design}' for v in ('F', 'LR', 'PR')]
        full_id, late_id, progressive_id = names
        data, manifests, provenances = {}, {}, {}
        for fingerprint in names:
            _, _, records, provenance = checked_fingerprint(root, fingerprint)
            folder = root/'fingerprints'/fingerprint
            data[fingerprint] = pd.read_parquet(folder/'values.parquet')
            manifests[fingerprint], provenances[fingerprint] = records, provenance
            for name in ('values.parquet', 'feature_manifest.json', 'provenance.json'):
                bind(folder/name)
            cells = runs.loc[runs.fingerprint_id.eq(fingerprint)]
            screening_medians(cells, points)
            shap = {}
            for row in cells.itertuples(index=False):
                directory = root/'experiments'/generation/row.run_id
                bind(directory/'result.json')
                if row.status == 'success':
                    shap[(row.model, row.hyperparameter_setpoint)] = pd.read_parquet(directory/'source_shap.parquet')
                    bind(directory/'source_shap.parquet')
                    bind(directory/'predictions.parquet')
            source_importance_consensus(shap, points, records)

        parent_id = parent_of(progressive_id)
        authorization = verified_parent(root, parent_id)
        parent_folder = root/'fingerprints'/parent_id
        _, _, parent_records, _ = checked_fingerprint(root, parent_id)
        _, pool_records, pool_audit = progressive_pool(
            full_id, parent_id, data[full_id], manifests[full_id],
            pd.read_parquet(parent_folder/'values.parquet'), parent_records)
        if pool_audit['excluded_full_game_ids']:
            raise ValueError('Progressive comparison has a different target cohort')
        parent_final = Path(authorization.get('acceptance_path', authorization.get('finalization_path')))
        bind(parent_final)
        if 'finalization_path' in authorization:
            bind(parent_final.parent/'recommendations.json')
        for item in json.loads(parent_final.read_text())['evidence']:
            bind(Path(item['path']))

        for fingerprint, allowed in ((late_id, manifests[full_id]), (progressive_id, pool_records)):
            folder = root/'fingerprints'/fingerprint
            proposal_path = folder/'reduction_proposal.json'
            if sha256_file(proposal_path) != provenances[fingerprint].get('reduction_proposal_sha256'):
                raise ValueError('Reduced proposal binding changed')
            proposal = json.loads(proposal_path.read_text())
            expected_parent = parent_id if fingerprint == progressive_id else full_id
            if (proposal.get('full_reference') != full_id
                    or provenances[fingerprint].get('ancestry') != [expected_parent]):
                raise ValueError('Reduced lineage ancestry changed')
            if fingerprint == progressive_id:
                if (proposal.get('progressive_parent') != parent_id
                        or proposal.get('parent_authorization') != authorization):
                    raise ValueError('Progressive parent authorization changed')
                pool_path = root/'progressive_pools'/progressive_id/'provenance.json'
                if sha256_file(pool_path) != proposal.get('pool_provenance_sha256'):
                    raise ValueError('Progressive proposal pool changed')
                bind(pool_path)
            proposal_evidence = proposal.get('full_screening_evidence', [])
            expected_results = {str(root/'experiments'/generation/r.run_id/'result.json')
                                for r in runs.loc[runs.fingerprint_id.eq(full_id)].itertuples()}
            if {item['path'] for item in proposal_evidence} != expected_results:
                raise ValueError('Reduction proposal does not bind the full screening matrix')
            for item in proposal_evidence:
                if sha256_file(Path(item['path'])) != item['sha256']:
                    raise ValueError('Full reference changed since reduction proposal')
            verify_reduced_projection(data[full_id], manifests[full_id], data[fingerprint],
                                      manifests[fingerprint], allowed,
                                      shortfall_exceptions=proposal.get('shortfall_exceptions'))
            acceptance = measured_acceptance(
                runs.loc[runs.fingerprint_id.eq(full_id)],
                runs.loc[runs.fingerprint_id.eq(fingerprint)], points, design=design)
            if not acceptance['accepted']:
                raise ValueError(f'{fingerprint} fails measured reduction acceptance')
            acceptances[fingerprint] = acceptance
            bind(proposal_path)
        ranked = rank_lineages(summaries, generation=generation, design=design,
                               eligible_fingerprints=set(names))
        ranked['status'] = 'success'
        ranked['max_design_year_used'] = 2025
        ranked['prospective_boundary_year'] = 2026
        ranked['run_id'] = ranked.fingerprint_id+'__recommendation'
        recommendations.extend(json.loads(ranked.to_json(orient='records')))

    bind(root/'results/source_semantics_audit.json')
    if any(sha256_file(Path(path)) != digest for path, digest in evidence.items()):
        raise ValueError('Finalization evidence changed during validation')
    output = root/'results'/generation
    output.mkdir(parents=True, exist_ok=True)
    report = {'generation': generation, 'rows': recommendations, 'reduction_acceptance': acceptances,
              'evidence': [{'path': p, 'sha256': h} for p, h in sorted(evidence.items())]}
    atomic_json(output/'recommendations.json', report)
    collect_results(root)
    final = {'generation': generation, 'training_finalized': True, 'shap_finalized': True,
             'reduction_finalized': True, 'lineages_compared': True, 'recommendations_finalized': True,
             'recommendations_sha256': sha256_file(output/'recommendations.json'),
             'incomplete_coverage': any(r['incomplete_coverage'] for r in recommendations),
             'evidence': report['evidence']}
    atomic_json(output/'finalization.json', final)
    return final
