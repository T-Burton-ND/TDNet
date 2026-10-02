"""Collect actual screening evidence into the declared ultra-wide result table."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_ml.pipeline.fetch.nextgen_acquisition import sha256_file

ROOT=Path(__file__).resolve().parents[3]
METRICS=('mae','rmse','brier','winner_accuracy','ats_accuracy','chalk_accuracy','upset_accuracy')


def verify_successful_run(root, directory, row, provenance, points):
    """Recheck successful evidence before including it in performance summaries."""
    from .nextgen_screening import execution_binding, verify_reusable_result
    if row.get('execution_binding', {}).get('scheduling_policy') == 'parallel_full_v1':
        from .nextgen_screening_parallel import execution_binding
    elif row.get('execution_binding', {}).get('scheduling_policy') == 'parallel_reduced_v1':
        from .nextgen_screening_reduced_parallel import execution_binding

    model=row['model']
    matches=[p for p in points.get(model, []) if p['id']==row['hyperparameter_setpoint']]
    if len(matches)!=1:
        raise ValueError('Run does not identify one frozen configuration')
    expected=f"{row['fingerprint_id']}__{model}__{row['hyperparameter_setpoint']}"
    if row['run_id']!=expected or directory.name!=expected:
        raise ValueError('Run identity differs from result directory')
    fingerprint=root/'fingerprints'/row['fingerprint_id']
    actual=sha256_file(fingerprint/'values.parquet')
    if actual!=provenance['data_sha256'] or actual!=row.get('data_sha256'):
        raise ValueError('Run input data changed since execution')
    binding=execution_binding(root,provenance,model,matches[0])
    verify_reusable_result(row,binding,directory)


def architecture_summaries(runs, points):
    summaries=[]
    if runs.empty:
        return summaries
    for (fingerprint,model),group in runs.groupby(['fingerprint_id','model']):
        expected={p['id'] for p in points[model]}
        if group.hyperparameter_setpoint.duplicated().any():
            raise ValueError('Duplicate run cell in results table')
        complete=(set(group.hyperparameter_setpoint)==expected and len(expected)==10 and group.status.eq('success').all())
        successful=group.loc[group.status.eq('success')]
        usable=(set(group.hyperparameter_setpoint)==expected and len(expected)==10
                and group.status.isin(['success','failed','skipped']).all() and len(successful)>=3)
        row={k:group.iloc[0].get(k) for k in ('fingerprint_id','generation','lineage','design','ancestry',
             'feature_count','feature_counts_by_generation_json','model','seed','feature_manifest_sha256',
             'prospective_boundary_year','max_design_year_used','acquisition_manifest_sha256','source_schema_version')}
        row.update(row_type='fingerprint_architecture_summary',run_id=f'{fingerprint}__{model}__summary',
                   status='success' if complete else 'incomplete',successful_run_count=int(group.status.eq('success').sum()),
                   observed_run_count=len(group),required_run_count=10,recommendation_eligible=False,
                   screening_usable=bool(usable),incomplete_coverage=not complete,
                   recommendation_selected=False,recommendation_basis='generation finalization required',
                   failure_reason=None if complete else 'Missing or unsuccessful frozen setpoint cells')
        for year in (2024,2025):
            for metric in METRICS:
                column=f'{metric}_{year}'
                values=pd.to_numeric(successful[column],errors='coerce') if column in successful else pd.Series(dtype=float)
                # The launch contract permits >=3 successes, explicitly flagged.
                row[column]=float(values.median()) if usable and len(values)>=3 and values.notna().all() else None
        summaries.append(row)
    return summaries


def verified_recommendations(root):
    """Keep finalized recommendations only while their recorded evidence matches."""
    rows=[]
    for path in sorted((root/'results').glob('F*/recommendations.json')):
        report=json.loads(path.read_text())
        if not report.get('evidence'):
            raise ValueError('Recommendation lacks measured evidence')
        for item in report['evidence']:
            source=Path(item['path'])
            if not source.exists() or sha256_file(source)!=item['sha256']:
                raise ValueError('Recommendation evidence changed; refinalization required')
        for row in report['rows']:
            if row.get('row_type')!='fingerprint_recommendation':
                raise ValueError('Invalid persisted recommendation row')
            rows.append(row)
    return rows


def collect_results(root: Path):
    schema=json.loads((ROOT/'configs/experiments/nextgen_result_schema_v1.json').read_text())
    points=json.loads((ROOT/'configs/experiments/nextgen_screening_setpoints_v1.json').read_text())
    rows=[]
    for directory in sorted((root/'experiments').glob('F*/*')):
        path=directory/'result.json'
        if not path.exists():
            path=directory/'progress.json'
        if not path.exists():
            continue
        row=json.loads(path.read_text())
        if row.get('row_type')!='run':
            continue
        fingerprint=root/'fingerprints'/row['fingerprint_id']
        manifest_path=fingerprint/'feature_manifest.json'
        if sha256_file(manifest_path)!=row['feature_manifest_sha256']:
            raise ValueError('Run manifest changed since execution')
        records=json.loads(manifest_path.read_text())
        counts=Counter(r['generation'] for r in records)
        provenance=json.loads((fingerprint/'provenance.json').read_text())
        if row.get('status')=='success':
            verify_successful_run(root,directory,row,provenance,points)
        row.update(ancestry=json.dumps(provenance.get('ancestry',sorted(counts))),
                   feature_counts_by_generation_json=json.dumps(dict(counts),sort_keys=True),
                   recommendation_eligible=False,recommendation_selected=False,
                   recommendation_tie_band_mae=.5,recommendation_basis='generation finalization required',
                   source_schema_version='nextgen_result_v1')
        # Preserve the run's acquisition binding; never substitute today's plan.
        row.setdefault('acquisition_manifest_sha256',None)
        row.setdefault('acquisition_summary_ref',None)
        for field in ('execution_binding','shap_report'):
            if isinstance(row.get(field),dict):
                row[field+'_json']=json.dumps(row.pop(field),sort_keys=True)
        for key,value in list(row.items()):
            if isinstance(value,(dict,list)):
                row[key]=json.dumps(value,sort_keys=True)
        rows.append(row)
    if not rows:
        raise ValueError('No actual screening runs exist; do not create fictional results')
    runs=pd.DataFrame(rows)
    if runs.run_id.duplicated().any():
        raise ValueError('Duplicate run identity')
    rows+=architecture_summaries(runs,points)
    rows+=verified_recommendations(root)
    table=pd.DataFrame(rows)
    for column in schema['required_columns']:
        if column not in table:
            table[column]=None
    if not table.status.isin(schema['status_values']).all():
        raise ValueError('Undeclared run status')
    if pd.to_numeric(table.max_design_year_used,errors='coerce').gt(2025).any():
        raise ValueError('Quarantined result evidence')
    dest=root/'results/nextgen_results.parquet'
    temporary=dest.with_suffix('.tmp.parquet')
    table.to_parquet(temporary,index=False)
    temporary.replace(dest)
    return table
