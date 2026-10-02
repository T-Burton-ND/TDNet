import pandas as pd
import pytest

from gridiron_ml.experiments.nextgen_trial_selection import joint_floor_proposal


def fixture():
    names = [f'h{i}' for i in range(8)] + ['s1', 's2', 'p1', 'p2']
    records = [{'name': n, 'generation': 'F09', 'designs': ['c'],
                'matchup_counterpart': {'p1': 'p2', 'p2': 'p1'}.get(n, n)} for n in names]
    weights = [100.]*8 + [5., 5., 4.5, 4.5]
    values = [v/sum(weights) for v in weights]
    table = pd.DataFrame({'source_feature': names, 'M2_importance': values, 'M4_importance': values})
    return records, table


def test_exact_group_selection_beats_greedy_group_order_and_is_deterministic():
    records, table = fixture()
    result = joint_floor_proposal(records, table)
    # After the eight high-value singles, two singles beat the larger pair's score.
    assert result['survivors'] == [f'h{i}' for i in range(8)] + ['s1', 's2']
    assert result['counts_by_generation'] == {'F09': 10}
    assert result['accepted'] is False
    assert result == joint_floor_proposal(records[::-1], table.iloc[::-1])


def test_shortfall_requires_explicit_evidence_and_keeps_all_available_signals():
    records, table = fixture()
    records = records[:8]
    for r in records:
        r['generation'] = 'F12'
    with pytest.raises(ValueError, match='floor'):
        joint_floor_proposal(records, table)
    result = joint_floor_proposal(records, table, shortfall_exceptions={
        'F12': {'legitimate_signal_count': 8, 'evidence': 'Fixture source has eight signals.'}})
    assert result['counts_by_generation'] == {'F12': 8}
    assert len(result['survivors']) == 8


def test_rejects_missing_scores_wrong_design_and_cross_generation_pairs():
    records, table = fixture()
    with pytest.raises(ValueError, match='Missing'):
        joint_floor_proposal(records, table.iloc[1:])
    records[0]['designs'] = ['a']
    with pytest.raises(ValueError, match='design C'):
        joint_floor_proposal(records, table)
    records[0]['designs'] = ['c']
    records[-1]['generation'] = 'F12'
    with pytest.raises(ValueError, match='Cross-generation'):
        joint_floor_proposal(records, table, shortfall_exceptions={
            'F12': {'legitimate_signal_count': 1, 'evidence': 'Fixture'}})
