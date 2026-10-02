import json
from pathlib import Path
import pandas as pd
from gridiron_ml.experiments.nextgen_f11 import prior_staff_state,PriorStaffBuilder
from gridiron_ml.experiments.nextgen_contract import validate_feature_manifest


def test_prior_staff_freezes_history_without_current_coach_assignment(tmp_path):
    history=pd.DataFrame([dict(coach_id=i,team=t,season=y,games=10,wins=6,ties=0,points_for=200 if y==2019 else 9999,
        points_against=150,close_games=2,close_wins=1,available=pd.Timestamp(f'{y}-12-01T00:00:00Z'))
        for i,t,y in [(1,'A',2019),(2,'B',2019),(9,'A',2020)]])
    schedule=pd.DataFrame([dict(id=i,season=2020,start_date=date,home_team='A',away_team='B',home_classification='fbs',away_classification='fbs')
        for i,date in [(1,'2020-09-01T00:00:00Z'),(2,'2020-10-01T00:00:00Z')]])
    schedule=pd.concat([schedule,pd.DataFrame([dict(id=3,season=2020,start_date='2020-10-02T00:00:00Z',home_team='A',away_team='C',home_classification='fbs',away_classification=None)])],ignore_index=True)
    state,excluded=prior_staff_state(history,schedule)
    assert not excluded and len(state)==4
    assert state.prior_staff_regular_points_for_per_game.eq(20).all()
    assert state.feature_available_utc.eq(pd.Timestamp('2019-12-01T00:00:00Z')).all()
    schema=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/nextgen_feature_manifest_schema_v1.json').read_text())
    for design,count in [('a',7),('b',8),('c',5)]:
        b=PriorStaffBuilder(tmp_path,state,design)
        assert len(b.feature_columns)==count
        validate_feature_manifest(json.loads(b.manifest_path.read_text()),schema)
        assert 'coach_id' not in b.build_frame()
