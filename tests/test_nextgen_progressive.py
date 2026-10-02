import pandas as pd
import pytest

from gridiron_ml.experiments.nextgen_progressive import progressive_pool


def fixture():
    records = [{'name': n, 'generation': g, 'matchup_counterpart': n}
               for n, g in [('kept', 'F06'), ('removed', 'F06'), ('new', 'F09')]]
    full = pd.DataFrame({'target_game_id': [1, 1, 2, 2], 'team': ['A', 'B', 'C', 'D'],
                         'season': [2020]*4, 'kept': [1., 2., 3., 4.],
                         'removed': [5.]*4, 'new': [6.]*4})
    return full, records, full.drop(columns=['removed', 'new']), records[:1]


def test_pool_never_resurrects_removed_features_and_adds_full_new_family():
    full, records, parent, prior = fixture()
    result, retained, audit = progressive_pool('F09_F_a', 'F06_R_a', full, records, parent, prior)
    pd.testing.assert_frame_equal(result, full.drop(columns='removed'))
    assert [r['name'] for r in retained] == ['kept', 'new']
    assert audit['accepted'] is False and audit['ancestry'] == ['F06_R_a']


def test_pool_rejects_cross_design_or_late_parent_and_changed_values():
    full, records, parent, prior = fixture()
    for parent_id in ['F06_R_b', 'F06_F_a', 'F09_LR_a']:
        with pytest.raises(ValueError, match='ancestry'):
            progressive_pool('F09_F_a', parent_id, full, records, parent, prior)
    parent.loc[0, 'kept'] = 100.
    with pytest.raises(ValueError, match='values'):
        progressive_pool('F09_F_a', 'F06_R_a', full, records, parent, prior)


def test_pool_reports_cohort_loss_without_partial_games():
    full, records, parent, prior = fixture()
    result, _, audit = progressive_pool('F09_F_a', 'F06_R_a', full, records, parent.iloc[:2], prior)
    assert result.target_game_id.tolist() == [1, 1]
    assert audit['excluded_full_game_ids'] == [2]
    with pytest.raises(ValueError, match='paired'):
        progressive_pool('F09_F_a', 'F06_R_a', full, records, parent.iloc[:3], prior)


def test_new_family_cannot_reintroduce_a_removed_counterpart():
    full, records, parent, prior = fixture()
    records[-1]['matchup_counterpart'] = 'removed'
    with pytest.raises(ValueError, match='Pair closure'):
        progressive_pool('F09_F_a', 'F06_R_a', full, records, parent, prior)
