import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    'archive_audit', Path(__file__).resolve().parents[1]/'scripts/nextgen_archive_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_missing_events_and_disagreeing_yardage_are_not_repaired():
    plays = pd.DataFrame(dict(id=['1','2'],offense=['A','A'],defense=['B','B'],
                              play_type=['Rush','Rush'],yards_gained=[5,-2],period=[1,1]))
    stats = pd.DataFrame(dict(play_id=['1','999'],athlete_id=['x','y'],
                              stat_type=['Rush','Rush'],stat=[8,2],team=['A','A'],period=[1,1]))
    before = stats.copy(deep=True)
    result = audit.compare_game(stats,plays)
    assert result['expected_rush_plays']==2
    assert result['matched_rush_plays']==1
    assert result['missing_rush_plays']==1
    assert result['unmatched_attributed_rows']==1
    assert result['rush_yardage_mismatches']==1
    pd.testing.assert_frame_equal(before,stats)


def test_duplicate_play_keys_cannot_multiply_attribution_rows():
    plays = pd.DataFrame(dict(id=['1','1']))
    stats = pd.DataFrame(dict(play_id=['1']))
    with pytest.raises(ValueError,match='Duplicate canonical'):
        audit.compare_game(stats,plays)
