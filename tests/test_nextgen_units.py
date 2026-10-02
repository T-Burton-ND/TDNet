import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_units import resolve_unit_production


def fixture():
    history=pd.DataFrame([dict(game_id=1,season=2020,team='A',athlete_id=p,metric=m,value=v)
        for p,m,v in [('1','rush_yards',20),('2','rush_yards',30),('3','tackles',4),('2','kick_return_yards',40),('unknown','tackles',2)]])
    roster=pd.DataFrame([dict(season=2020,team='A',athlete_id=p,position=pos,available_utc='2020-08-01T00:00:00Z',availability_evidence='fixture archive')
        for p,pos in [('1','QB'),('2','RB'),('3','LB')]])
    return history,roster,{(2020,'A'):'2020-09-01T00:00:00Z'}


def test_unit_allocation_preserves_qb_rushing_and_special_teams_roles():
    h,r,c=fixture(); totals,audit=resolve_unit_production(h,r,c)
    assert totals.set_index(['unit','metric']).value.to_dict()=={
        ('qb','rush_yards'):20,('rb','rush_yards'):30,('lb','tackles'):4,('special_teams','kick_return_yards'):40}
    assert audit['unassigned_rows']==1
    assert audit['assigned_fraction']==.8


def test_late_or_ambiguous_roster_is_rejected():
    h,r,c=fixture()
    with pytest.raises(ValueError,match='Ambiguous'):
        resolve_unit_production(h,pd.concat([r,r.iloc[[0]]]),c)
    r.loc[0,'available_utc']='2020-10-01T00:00:00Z'
    with pytest.raises(ValueError,match='Week 0'):
        resolve_unit_production(h,r,c)
