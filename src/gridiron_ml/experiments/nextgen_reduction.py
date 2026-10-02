"""Measured reduction acceptance and pair-closed ancestry constraints.

No proposal is accepted from SHAP alone: every candidate requires the frozen
M2/M4 attempt matrix and at least three successes per architecture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def screening_medians(runs: pd.DataFrame, frozen_points: dict):
    required={'model','hyperparameter_setpoint','status','mae_2024','mae_2025'}
    if not required <= set(runs):
        raise ValueError('Missing screening metrics')
    if set(runs.model)!={'M2','M4'} or not runs.status.isin(['success','failed','skipped']).all():
        raise ValueError('Terminal M2/M4 attempt matrix required')
    if runs.duplicated(['model','hyperparameter_setpoint']).any():
        raise ValueError('Duplicate screening cell')
    for model in ('M2','M4'):
        expected={p['id'] for p in frozen_points[model]}
        if len(expected)!=10 or set(runs.loc[runs.model.eq(model),'hyperparameter_setpoint'])!=expected:
            raise ValueError('Screening matrix does not match the ten frozen configurations')
        if len(runs.loc[runs.model.eq(model) & runs.status.eq('success')])<3:
            raise ValueError('At least three successes per architecture required')
    successful=runs.loc[runs.status.eq('success')]
    metrics=successful[['mae_2024','mae_2025']].to_numpy(float)
    if not np.isfinite(metrics).all() or (metrics<0).any():
        raise ValueError('Invalid or missing development MAE')
    return successful.groupby('model')[['mae_2024','mae_2025']].median()


def measured_acceptance(reference: pd.DataFrame, candidate: pd.DataFrame, frozen_points: dict,
                        *, design: str, tolerance=.25):
    if design not in {'a','b','c'} or tolerance<0:
        raise ValueError('Invalid reduction policy')
    ref=screening_medians(reference,frozen_points)
    cand=screening_medians(candidate,frozen_points)
    delta=cand-ref
    # A/B protect each architecture in each development year. C uses the
    # explicitly permitted mean across architecture medians and both years.
    accepted=bool((delta.to_numpy()<=tolerance+1e-12).all()) if design in {'a','b'} else bool(delta.to_numpy().mean()<=tolerance+1e-12)
    return {'accepted':accepted,'design':design,'tolerance_mae':tolerance,
            'reference_success_counts':reference.loc[reference.status.eq('success')].model.value_counts().to_dict(),
            'candidate_success_counts':candidate.loc[candidate.status.eq('success')].model.value_counts().to_dict(),
            'incomplete_coverage':not (reference.status.eq('success').all() and candidate.status.eq('success').all()),
            'delta_by_model_year':{m:{str(y):float(delta.loc[m,'mae_'+str(y)]) for y in (2024,2025)} for m in ('M2','M4')},
            'joint_mean_delta':float(delta.to_numpy().mean()),
            'reference_medians':ref.to_dict(orient='index'),'candidate_medians':cand.to_dict(orient='index')}


def pair_groups(records):
    by_name={r['name']:r for r in records}
    if len(by_name)!=len(records):
        raise ValueError('Duplicate feature name')
    groups=set()
    for name,r in by_name.items():
        counterpart=r['matchup_counterpart']
        if counterpart not in by_name or by_name[counterpart]['matchup_counterpart']!=name:
            raise ValueError('Non-reciprocal pruning pair')
        groups.add(tuple(sorted({name,counterpart})))
    return sorted(groups)


def check_survivors(names, records, *, shortfall_exceptions=None):
    """Validate concrete survivors; exceptions require documented scientific cause."""
    names=set(names)
    by_name={r['name']:r for r in records}
    if not names <= set(by_name):
        raise ValueError('Unknown surviving feature')
    for pair in pair_groups(records):
        if names.intersection(pair) and not set(pair)<=names:
            raise ValueError('Pruning split a reciprocal pair')
    exceptions=shortfall_exceptions or {}
    counts={generation:sum(by_name[n]['generation']==generation for n in names)
            for generation in {r['generation'] for r in records}}
    for generation,count in counts.items():
        floor=60 if generation=='F06' else 10
        if count<floor:
            exception=exceptions.get(generation,{})
            if (generation=='F06' or exception.get('legitimate_signal_count')!=count
                    or not isinstance(exception.get('evidence'),str) or not exception['evidence'].strip()):
                raise ValueError(f'Ancestry floor not met for {generation}: {count} < {floor}')
    return counts


def source_importance_consensus(tables: dict, frozen_points: dict, records):
    """Equal-weight configurations within each architecture, then architectures.

    Inputs are keyed by (model, setpoint). These ranks propose evaluations;
    they never authorize removal without measured_acceptance.
    """
    expected={(m,p['id']) for m in ('M2','M4') for p in frozen_points[m]}
    if not set(tables)<=expected or len(expected)!=20 or any(sum(k[0]==m for k in tables)<3 for m in ('M2','M4')):
        raise ValueError('At least three frozen source SHAP tables per architecture are required')
    names=[r['name'] for r in records]
    pair_groups(records)
    values={m:[] for m in ('M2','M4')}
    for (model,_),table in tables.items():
        if table.source_feature.duplicated().any() or set(table.source_feature)!=set(names):
            raise ValueError('Source SHAP feature schema mismatch')
        importance=table.set_index('source_feature').normalized_importance.reindex(names).astype(float)
        if not np.isfinite(importance).all() or importance.lt(0).any():
            raise ValueError('Invalid source importance')
        total=float(importance.sum())
        if not (np.isclose(total,1.,atol=1e-6) or total==0):
            raise ValueError('Source importance must be normalized within each run')
        values[model].append(importance)
    result=pd.DataFrame({'source_feature':names})
    for model in ('M2','M4'):
        result[model+'_importance']=pd.concat(values[model],axis=1).mean(axis=1).to_numpy()
    result['joint_importance']=result[['M2_importance','M4_importance']].mean(axis=1)
    result['conservative_importance']=result[['M2_importance','M4_importance']].max(axis=1)
    return result.sort_values(['conservative_importance','joint_importance','source_feature']).reset_index(drop=True)


def redundancy_report(frame: pd.DataFrame, records, *, minimum_joint_rows=100):
    """Report strong correlations without turning them into removal decisions."""
    if 'season' not in frame or frame.season.isna().any() or frame.season.gt(2025).any():
        raise ValueError('Redundancy analysis requires explicit pre-2026 seasons')
    names=[r['name'] for r in records]
    if not set(names)<=set(frame):
        raise ValueError('Missing redundancy feature columns')
    if minimum_joint_rows<2:
        raise ValueError('Correlation support must be at least two rows')
    data=frame[names].replace([np.inf,-np.inf],np.nan)
    missing=data.isna().mean()
    pearson=data.corr(method='pearson',min_periods=minimum_joint_rows)
    spearman=data.corr(method='spearman',min_periods=minimum_joint_rows)
    rows=[]
    for i,left in enumerate(names):
        for right in names[i+1:]:
            p,s=pearson.loc[left,right],spearman.loc[left,right]
            strength=max(abs(p) if np.isfinite(p) else 0,abs(s) if np.isfinite(s) else 0)
            if strength<.90:
                continue
            category='automatic_redundancy_candidate' if strength>=.995 else ('validate_representative' if strength>=.98 else 'report_only')
            rows.append({'left':left,'right':right,'pearson':p,'spearman':s,
                         'joint_rows':int(data[[left,right]].notna().all(axis=1).sum()),
                         'left_missing_fraction':float(missing[left]),'right_missing_fraction':float(missing[right]),
                         'category':category,'removal_authorized':False})
    return pd.DataFrame(rows,columns=['left','right','pearson','spearman','joint_rows',
                                     'left_missing_fraction','right_missing_fraction','category','removal_authorized'])
