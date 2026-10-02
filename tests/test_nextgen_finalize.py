import pandas as pd
import pytest

from gridiron_ml.experiments.nextgen_finalize import verify_reduced_projection


def fixture():
    records = [{'name': f'x{i}', 'generation': 'F06' if i < 64 else 'F09',
                'matchup_counterpart': f'x{i}'} for i in range(76)]
    frame = pd.DataFrame({'target_game_id': [1, 1, 2, 2], 'team': ['A', 'B', 'C', 'D'],
                          'season': [2020]*4, 'next_game_margin': [7., -7., 3., -3.],
                          **{r['name']: [1., 2., 3., float('nan')] for r in records}})
    retained = records[:60] + records[64:74]
    columns = ['target_game_id', 'team', 'season', 'next_game_margin'] + [r['name'] for r in retained]
    return frame, records, frame[columns].copy(), retained


def test_projection_requires_unchanged_features_targets_and_metadata():
    full, records, candidate, kept = fixture()
    verify_reduced_projection(full, records, candidate, kept, records)
    candidate.loc[0, 'next_game_margin'] = 100.
    with pytest.raises(AssertionError):
        verify_reduced_projection(full, records, candidate, kept, records)
    candidate = full[candidate.columns].iloc[:2]
    with pytest.raises(AssertionError):
        verify_reduced_projection(full, records, candidate, kept, records)


def test_progressive_projection_cannot_resurrect_features_or_keep_unpruned_pool():
    full, records, candidate, kept = fixture()
    with pytest.raises(ValueError, match='allowed ancestry'):
        verify_reduced_projection(full, records, candidate, kept, records[1:])
    with pytest.raises(ValueError, match='did not prune'):
        verify_reduced_projection(full, records, candidate, kept, kept)
    candidate['undeclared'] = 1.
    with pytest.raises(ValueError, match='undeclared'):
        verify_reduced_projection(full, records, candidate, kept, records)


def test_projection_requires_documented_thin_generation_and_preserves_pair_closure():
    full, records, candidate, kept = fixture()
    kept = kept[:-1]
    candidate = candidate.drop(columns='x73')
    with pytest.raises(ValueError, match='floor'):
        verify_reduced_projection(full, records, candidate, kept, records)
    verify_reduced_projection(full, records, candidate, kept, records,
                              shortfall_exceptions={'F09': {'legitimate_signal_count': 9,
                                                           'evidence': 'Fixture thin-signal audit'}})
    records[0]['matchup_counterpart'] = 'x63'
    records[63]['matchup_counterpart'] = 'x0'
    with pytest.raises(ValueError, match='split'):
        verify_reduced_projection(full, records, candidate, kept, records,
                                  shortfall_exceptions={'F09': {'legitimate_signal_count': 9,
                                                               'evidence': 'Fixture audit'}})
