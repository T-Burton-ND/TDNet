import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_recommendations import rank_lineages


def fixture():
    return pd.DataFrame([dict(fingerprint_id=f'F09_{v}_a',model=m,screening_usable=True,
        incomplete_coverage=False,max_design_year_used=2025,feature_count=n,mae_2024=mae,mae_2025=mae,
        brier_2024=brier,brier_2025=brier,ats_accuracy_2024=.5,ats_accuracy_2025=.5,
        upset_accuracy_2024=.4,upset_accuracy_2025=.4,chalk_accuracy_2024=.8,chalk_accuracy_2025=.8)
        for v,n,mae,brier in [('F',100,10.,.2),('LR',80,10.4,.18),('PR',70,10.6,.1)] for m in ['M2','M4']])


def test_practical_tie_uses_brier_but_does_not_admit_outside_band():
    s=fixture();r=rank_lineages(s,generation='F09',design='a',eligible_fingerprints=set(s.fingerprint_id))
    assert r.loc[r.recommendation_selected,'fingerprint_id'].tolist()==['F09_LR_a']
    assert r.loc[r.fingerprint_id.eq('F09_PR_a'),'recommendation_rank'].iloc[0]==3
    assert 'mae_2024' in r and 'mae_2025' in r
    with pytest.raises(ValueError,match='All requested'):
        rank_lineages(s.iloc[2:],generation='F09',design='a',eligible_fingerprints=set(s.fingerprint_id))


def test_failed_reduction_is_ineligible_and_missing_metric_is_not_imputed():
    s=fixture();s.loc[s.fingerprint_id.eq('F09_LR_a'),'brier_2024']=None
    r=rank_lineages(s,generation='F09',design='a',eligible_fingerprints={'F09_F_a','F09_LR_a'})
    assert r.loc[r.recommendation_selected,'fingerprint_id'].tolist()==['F09_LR_a']
    assert set(r.unavailable_tiebreakers)=={'brier'}
    assert not r.loc[r.fingerprint_id.eq('F09_PR_a'),'recommendation_eligible'].iloc[0]
