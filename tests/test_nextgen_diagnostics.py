import pandas as pd
import pytest

from gridiron_ml.experiments.nextgen_designs import Formula, feature_record
from gridiron_ml.experiments.nextgen_diagnostics import feature_identity, rank_diagnostic_features, render_diagnostic


def record(name, generation='F09'):
    return feature_record(Formula(name, (name,), 'identity', 'Fixture interpretation', 'fraction'),
                          generation, 'a', name, endpoints=['fixture'])


def context(fingerprint, x):
    return {'fingerprint_id': fingerprint, 'records': [record('x'), record('y')],
            'consensus': pd.DataFrame({'source_feature': ['x', 'y'],
                                       'M2_importance': [x, 1-x], 'M4_importance': [x, 1-x]})}


def test_global_ranking_balances_lineages_designs_and_generations():
    contexts = [context('F09_F_a', .8), context('F09_LR_a', .2),
                context('F09_F_b', .8), context('F10_F_a', .2)]
    ranked = rank_diagnostic_features(contexts)
    assert [r['record']['name'] for r in ranked] == ['y', 'x']
    assert ranked[1]['joint_importance'] == pytest.approx(.425)
    assert len(ranked[0]['screening_contexts']) == 4
    assert len(rank_diagnostic_features(contexts, limit=1)) == 1


def test_identity_merges_design_labels_but_not_equations():
    first = record('x')
    second = {**first, 'designs': ['b'], 'interpretation': 'Other prose'}
    assert feature_identity(first) == feature_identity(second)
    second['equation_excel'] = '=[x]*2'
    assert feature_identity(first) != feature_identity(second)


def test_required_diagnostic_shares_global_cap_and_keeps_its_true_rank():
    contexts = [context('F09_F_a', .8)]
    selected = rank_diagnostic_features(contexts, limit=1, required_names=['y'])
    assert len(selected) == 1 and selected[0]['record']['name'] == 'y'
    assert selected[0]['rank'] == 2 and selected[0]['required_diagnostic'] is True
    with pytest.raises(ValueError, match='exceed'):
        rank_diagnostic_features(contexts, limit=1, required_names=['x', 'y'])
    with pytest.raises(ValueError, match='lacks measured'):
        rank_diagnostic_features(contexts, required_names=['absent'])


def test_invalid_or_duplicate_contexts_and_atomic_only_budget_rejected():
    c = context('F09_F_a', .5)
    with pytest.raises(ValueError, match='Duplicate'):
        rank_diagnostic_features([c, c])
    with pytest.raises(ValueError, match='1000'):
        rank_diagnostic_features([c], limit=1001)
    c['consensus']['M2_importance'] = 2.
    with pytest.raises(ValueError, match='normalized'):
        rank_diagnostic_features([c])
    c = context('F06_F_a', .5)
    c['records'] = [record('x', 'F06'), record('y', 'F06')]
    with pytest.raises(ValueError, match='No measured advanced'):
        rank_diagnostic_features([c])


def test_plot_handles_constant_feature_and_rejects_2026(tmp_path):
    item = rank_diagnostic_features([context('F09_F_a', .8)])[0]
    frame = pd.DataFrame({'season': [2023, 2024, 2025], 'x': [1., 1., 1.],
                          'next_game_margin': [-1., 0., 1.], 'next_game_win': [0, 0, 1]})
    entry = render_diagnostic(tmp_path, item, frame, fingerprint='F09_F_a',
                              data_sha256='fixture', survival_status='Fixture only',
                              source_documentation={'units': 'documented fixture units'})
    assert (tmp_path/entry['png']).read_bytes().startswith(b'\x89PNG')
    text = (tmp_path/entry['markdown']).read_text()
    assert 'Pearson None; Spearman None' in text and 'Fixture only' in text
    assert 'Source computation supplement' in text and 'documented fixture units' in text
    assert item['record']['units'] == 'fraction'
    frame.loc[0, 'season'] = 2026
    with pytest.raises(ValueError, match='pre-2026'):
        render_diagnostic(tmp_path, item, frame, fingerprint='F09_F_a',
                          data_sha256='fixture', survival_status='Fixture only')
