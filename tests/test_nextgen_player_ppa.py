import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_player_ppa import bind_player_ppa


def test_unique_game_binding_rejects_ambiguous_doubleheader_and_quarantine():
    schedule=pd.DataFrame([dict(id=i,season=2020,week=w,season_type='regular',completed=True,
        home_team='A',away_team='B',start_date=f'2020-09-{day:02d}T00:00:00Z')
        for i,w,day in [(1,1,1),(2,2,8),(3,2,10)]])
    records=pd.DataFrame([dict(season=2020,week=w,season_type='regular',id='p',team='A',opponent='B',position='QB',
        **{'average_p_p_a.all':.3,'average_p_p_a.pass':.4,'average_p_p_a.rush':None}) for w in [1,2]])
    joined,audit=bind_player_ppa(records,schedule)
    assert joined.game_id.tolist()==[1]
    assert joined.unit.tolist()==['qb']
    assert audit['unmatched_or_ambiguous_rows']==1
    assert joined.available.iloc[0]==pd.Timestamp('2020-09-03T00:00:00Z')
    duplicate,audit=bind_player_ppa(pd.concat([records.iloc[[0]],records.iloc[[0]]]),schedule)
    assert len(duplicate)==1 and audit['exact_duplicate_rows_removed']==1
    conflicting=records.iloc[[0]].copy()
    conflicting['average_p_p_a.pass']=99.
    clean,audit=bind_player_ppa(pd.concat([records.iloc[[0]],conflicting]),schedule)
    assert clean.empty and audit['conflicting_player_game_rows_excluded']==2
    records.loc[0,'season']=2026
    with pytest.raises(ValueError,match='pre-2026'):
        bind_player_ppa(records,schedule)
