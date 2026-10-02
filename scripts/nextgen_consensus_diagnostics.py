#!/usr/bin/env python3
"""Render the final global diagnostic set from all verified screening lineages."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import pandas as pd

from gridiron_ml.experiments.nextgen_contract import all_fingerprint_ids, parse_fingerprint_id
from gridiron_ml.experiments.nextgen_diagnostics import feature_identity, rank_diagnostic_features, render_diagnostic
from gridiron_ml.experiments.nextgen_reduction import screening_medians, source_importance_consensus
from gridiron_ml.experiments.nextgen_results import collect_results
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json, sha256_file

REQUIRED = ('offense_rush_ypa_q4_minus_q1', 'defense_rush_ypa_q4_minus_q1')


def load_source_supplements(repository=ROOT):
    """Verify source implementation and observational bindings before display."""
    repository = Path(repository)
    source_supplements = {}
    for filename in ('inherited_graph_documentation.json', 'inherited_temporal_documentation.json',
                     'inherited_opponent_documentation.json', 'inherited_boxscore_documentation.json',
                     'inherited_efficiency_documentation.json', 'inherited_prior_documentation.json'):
        supplement_path = repository/'docs/nextgen_fingerprints'/filename
        supplement = json.loads(supplement_path.read_text())
        common = supplement['common']
        for prefix in ('', 'additional_'):
            if prefix+'implementation' in common:
                if sha256_file(repository/common[prefix+'implementation']) != common[prefix+'implementation_sha256']:
                    raise ValueError('Source documentation needs review after implementation change')
        for evidence in common.get('supporting_implementations', []):
            if sha256_file(repository/evidence['path']) != evidence['sha256']:
                raise ValueError('Supporting source documentation needs review after implementation change')
        for evidence in common.get('supporting_evidence', []):
            if sha256_file(repository/evidence['path']) != evidence['sha256']:
                raise ValueError('Supporting documentation observations changed')
        for name, details in supplement['features'].items():
            if name in source_supplements:
                raise ValueError('Ambiguous supplemental source documentation')
            source_supplements[name] = {**common, **details,
                                       'supplement_sha256': sha256_file(supplement_path)}
    return source_supplements


def generate(root, limit=1000):
    root = Path(root)
    # This verifies run computation and recommendation evidence before plotting.
    table = collect_results(root)
    runs = table.loc[table.row_type.eq('run')]
    for generation in ('F06', 'F09', 'F10', 'F11', 'F12'):
        folder = root/'results'/generation
        final = json.loads((folder/'finalization.json').read_text())
        flags = ('training_finalized', 'shap_finalized', 'reduction_finalized',
                 'lineages_compared', 'recommendations_finalized')
        if final.get('generation') != generation or not all(final.get(k) is True for k in flags):
            raise ValueError('Global diagnostics require every generation finalized')
        if sha256_file(folder/'recommendations.json') != final.get('recommendations_sha256'):
            raise ValueError('Final recommendation binding changed')
        report = json.loads((folder/'recommendations.json').read_text())
        if not final.get('evidence') or final['evidence'] != report.get('evidence'):
            raise ValueError('Finalization evidence differs from verified recommendations')
    points = json.loads((ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    catalog, contexts, identities = {}, [], {}
    for fingerprint in all_fingerprint_ids():
        generation, _, _ = parse_fingerprint_id(fingerprint)
        folder = root/'fingerprints'/fingerprint
        provenance = json.loads((folder/'provenance.json').read_text())
        for filename, key in [('values.parquet', 'data_sha256'), ('feature_manifest.json', 'manifest_sha256')]:
            if sha256_file(folder/filename) != provenance[key]:
                raise ValueError('Diagnostic input binding changed')
        records = json.loads((folder/'feature_manifest.json').read_text())
        cells = runs.loc[runs.fingerprint_id.eq(fingerprint)]
        screening_medians(cells, points)
        shap = {(r.model, r.hyperparameter_setpoint): pd.read_parquet(
            root/'experiments'/generation/r.run_id/'source_shap.parquet')
            for r in cells.loc[cells.status.eq('success')].itertuples(index=False)}
        consensus = source_importance_consensus(shap, points, records)
        contexts.append({'fingerprint_id': fingerprint, 'records': records, 'consensus': consensus})
        catalog[fingerprint] = provenance
        identities[fingerprint] = {feature_identity(r) for r in records}
    ranked = rank_diagnostic_features(contexts, limit=limit, required_names=REQUIRED)
    source_supplements = load_source_supplements()
    destination = root/'feature_diagnostics'
    index_path = destination/'index.json'
    prior = json.loads(index_path.read_text()) if index_path.exists() else []
    legacy = {}
    for entry in prior:
        if entry.get('selection_basis') == 'explicit_required_rushing_diagnostic':
            legacy.setdefault(entry['feature'], []).append({k: entry[k] for k in ('png', 'markdown')})
        for earlier in entry.get('earlier_required_artifacts', []):
            if earlier not in legacy.setdefault(entry['feature'], []):
                legacy[entry['feature']].append(earlier)
    entries = []
    cached_fingerprint, cached_frame = None, None
    for item in ranked:
        candidates = [f for f in item['screening_contexts'] if parse_fingerprint_id(f)[1] == 'F']
        if not candidates:
            raise ValueError('Advanced feature has no checked full diagnostic source')
        source = min(candidates, key=lambda f: (-catalog[f]['rows'], f))
        if source != cached_fingerprint:
            cached_frame = pd.read_parquet(root/'fingerprints'/source/'values.parquet')
            cached_fingerprint = source
        absent = []
        for full in candidates:
            generation, _, design = parse_fingerprint_id(full)
            for variant in (('R',) if generation == 'F06' else ('LR', 'PR')):
                reduced = f'{generation}_{variant}_{design}'
                if item['identity'] not in identities[reduced]:
                    absent.append(reduced)
        survival = 'Present in '+', '.join(item['screening_contexts'])+'.'
        if absent:
            survival += ' Absent from corresponding reduced lineages (including inherited prior pruning): '+', '.join(sorted(absent))+'.'
        source_documentation = None
        if item['record']['generation'] == 'F06':
            source_documentation = source_supplements.get(item['record']['name'])
        entry = render_diagnostic(destination, item, cached_frame, fingerprint=source,
                                  data_sha256=catalog[source]['data_sha256'], survival_status=survival,
                                  source_documentation=source_documentation)
        entry.update(selection_basis='global_consensus_shap_with_required_rushing',
                     required_diagnostic=item['required_diagnostic'],
                     ranking_scope=item['ranking_scope'],
                     png_sha256=sha256_file(destination/entry['png']),
                     markdown_sha256=sha256_file(destination/entry['markdown']))
        if item['record']['name'] in legacy:
            entry['earlier_required_artifacts'] = legacy[item['record']['name']]
        entries.append(entry)
        if len(entries) % 25 == 0:
            print(f'Rendered {len(entries)}/{len(ranked)} verified feature diagnostics', flush=True)
    if len(entries) > 1000 or not set(REQUIRED) <= {e['feature'] for e in entries}:
        raise ValueError('Global diagnostic count or mandatory coverage mismatch')
    atomic_json(index_path, entries)
    return {'diagnostics': len(entries), 'index': str(index_path),
            'index_sha256': sha256_file(index_path), 'screening_fingerprints': len(contexts)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()
    root = Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    print(json.dumps(generate(root, args.limit), indent=2))
