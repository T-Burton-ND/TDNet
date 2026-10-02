"""Measured lineage recommendation ranking; never supplies missing run evidence."""
import numpy as np
import pandas as pd
from .nextgen_contract import parse_fingerprint_id
from .nextgen_results import METRICS


def rank_lineages(summaries, *, generation, design, eligible_fingerprints, tie_band=.5):
    """Rank one complete design comparison after reductions have been reviewed.

    Architecture medians receive equal weight; development years receive equal
    weight for the ranking while their metrics remain separate in output. A
    secondary criterion is used only if every practical-tie candidate has it.
    Callers must derive eligibility from actual reduction/finalization evidence.
    """
    variants=['F','R'] if generation=='F06' else ['F','LR','PR']
    expected={f'{generation}_{v}_{design}' for v in variants}
    for name in expected:parse_fingerprint_id(name)
    if tie_band<0 or not np.isfinite(tie_band):raise ValueError('Invalid practical tie band')
    if not set(eligible_fingerprints)<=expected:raise ValueError('Eligibility outside compared design')
    if 'max_design_year_used' not in summaries or summaries.max_design_year_used.isna().any() or summaries.max_design_year_used.gt(2025).any():
        raise ValueError('Explicit pre-2026 recommendation evidence required')
    selected=summaries.loc[summaries.fingerprint_id.isin(expected)]
    if set(selected.fingerprint_id)!=expected:raise ValueError('All requested lineages must be compared')
    if selected.duplicated(['fingerprint_id','model']).any():raise ValueError('Duplicate architecture summary')
    rows=[]
    for fingerprint in sorted(expected):
        group=selected.loc[selected.fingerprint_id.eq(fingerprint)]
        if set(group.model)!={'M2','M4'} or not group.screening_usable.eq(True).all():
            raise ValueError('Usable measured M2/M4 summaries required for every lineage')
        if not group.feature_count.nunique()==1:raise ValueError('Inconsistent feature count')
        row={'row_type':'fingerprint_recommendation','fingerprint_id':fingerprint,
             'generation':generation,'design':design,'lineage':parse_fingerprint_id(fingerprint)[1],
             'feature_count':int(group.feature_count.iloc[0]),
             'recommendation_eligible':fingerprint in eligible_fingerprints,
             'recommendation_selected':False,'recommendation_rank':None,
             'incomplete_coverage':bool(group.incomplete_coverage.any())}
        for metric in METRICS:
            for year in (2024,2025):
                key=f'{metric}_{year}'
                values=pd.to_numeric(group[key],errors='coerce') if key in group else pd.Series(dtype=float)
                row[key]=float(values.mean()) if len(values)==2 and np.isfinite(values).all() else None
            pair=[row[f'{metric}_{y}'] for y in (2024,2025)]
            row['ranking_'+metric]=float(np.mean(pair)) if all(v is not None for v in pair) else None
        if row['ranking_mae'] is None or row['ranking_mae']<0:raise ValueError('Measured MAE required')
        rows.append(row)
    table=pd.DataFrame(rows)
    eligible=table.loc[table.recommendation_eligible]
    if eligible.empty:raise ValueError('No scientifically eligible lineage')
    best=float(eligible.ranking_mae.min())
    tied=eligible.loc[eligible.ranking_mae.le(best+tie_band+1e-12)]
    criteria=[];ascending=[];skipped=[]
    for metric,lower in [('brier',True),('ats_accuracy',False),('upset_accuracy',False),('chalk_accuracy',False)]:
        key='ranking_'+metric
        if tied[key].notna().all():criteria.append(key);ascending.append(lower)
        else:skipped.append(metric)
    order=tied.sort_values(criteria+['feature_count','fingerprint_id'],ascending=ascending+[True,True]).index.tolist()
    order+=eligible.loc[~eligible.index.isin(order)].sort_values(['ranking_mae','feature_count','fingerprint_id']).index.tolist()
    table.loc[order,'recommendation_rank']=range(1,len(order)+1)
    table.loc[order[0],'recommendation_selected']=True
    table['recommendation_tie_band_mae']=tie_band
    table['within_practical_tie']=table.recommendation_eligible & table.ranking_mae.le(best+tie_band+1e-12)
    table['recommendation_basis']='Equal architecture/year mean MAE; practical tie then Brier, ATS, upset, chalk, feature count'
    table['unavailable_tiebreakers']=','.join(skipped)
    return table
