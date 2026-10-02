import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_reduction import measured_acceptance, check_survivors


def matrix():
    points={m:[{'id':str(i)} for i in range(10)] for m in ('M2','M4')}
    runs=pd.DataFrame([dict(model=m,hyperparameter_setpoint=str(i),status='success',mae_2024=10.,mae_2025=11.) for m in points for i in range(10)])
    return points,runs


def test_ab_c_tradeoffs_are_distinct_and_require_full_matrix():
    points,ref=matrix(); cand=ref.copy()
    cand.loc[cand.model.eq('M2'),'mae_2024']+=.4
    cand.loc[cand.model.eq('M4'),'mae_2024']-=.4
    assert not measured_acceptance(ref,cand,points,design='a')['accepted']
    assert measured_acceptance(ref,cand,points,design='c')['accepted']
    with pytest.raises(ValueError,match='ten frozen'):
        measured_acceptance(ref,cand.iloc[:-1],points,design='b')
    cand.loc[0,'mae_2025']=float('nan')
    with pytest.raises(ValueError,match='missing'):
        measured_acceptance(ref,cand,points,design='c')


def test_pair_closure_and_floor_exceptions_are_explicit():
    records=[dict(name=f'f{i}',matchup_counterpart=f'f{i}',generation='F06') for i in range(60)]
    records += [dict(name='o',matchup_counterpart='d',generation='F11'),dict(name='d',matchup_counterpart='o',generation='F11')]
    base={r['name'] for r in records[:60]}
    with pytest.raises(ValueError,match='split'):
        check_survivors(base|{'o'},records)
    with pytest.raises(ValueError,match='floor'):
        check_survivors(base|{'o','d'},records)
    counts=check_survivors(base|{'o','d'},records,shortfall_exceptions={'F11':{'legitimate_signal_count':2,'evidence':'fixture independence audit'}})
    assert counts=={'F06':60,'F11':2}


def test_consensus_protects_features_important_to_either_architecture():
    from gridiron_ml.experiments.nextgen_reduction import source_importance_consensus
    points,_=matrix()
    records=[dict(name=n,matchup_counterpart=n,generation='F06') for n in ['x','y','z']]
    tables={(m,str(i)):pd.DataFrame({'source_feature':['x','y','z'],'normalized_importance':[.9,.09,.01] if m=='M2' else [.09,.9,.01]})
            for m in points for i in range(10)}
    out=source_importance_consensus(tables,points,records)
    assert out.iloc[0].source_feature=='z'
    assert out.iloc[-1].conservative_importance==pytest.approx(.9)
    for i in range(2,10):
        tables.pop(('M4',str(i)))
    with pytest.raises(ValueError,match='three'):
        source_importance_consensus(tables,points,records)


def test_redundancy_is_only_a_candidate_and_quarantines_2026():
    from gridiron_ml.experiments.nextgen_reduction import redundancy_report
    frame=pd.DataFrame({'season':[2024]*120,'x':range(120),'y':[2*i for i in range(120)]})
    records=[{'name':'x'},{'name':'y'}]
    report=redundancy_report(frame,records)
    assert report.iloc[0].category=='automatic_redundancy_candidate'
    assert not report.iloc[0].removal_authorized
    frame.loc[0,'season']=2026
    with pytest.raises(ValueError,match='pre-2026'):
        redundancy_report(frame,records)


def test_three_successes_proceed_with_explicit_incomplete_coverage():
    points,ref=matrix(); candidate=ref.copy()
    candidate.loc[candidate.hyperparameter_setpoint.astype(int)>=3,'status']='failed'
    report=measured_acceptance(ref,candidate,points,design='a')
    assert report['accepted'] and report['incomplete_coverage']
    assert report['candidate_success_counts']=={'M2':3,'M4':3}
    candidate.loc[candidate.hyperparameter_setpoint.eq('2'),'status']='failed'
    with pytest.raises(ValueError,match='three'):
        measured_acceptance(ref,candidate,points,design='a')
