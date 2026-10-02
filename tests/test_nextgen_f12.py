import json
from pathlib import Path
import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_f12 import special_team_games,special_team_state,SpecialTeamsBuilder,RATIOS
from gridiron_ml.experiments.nextgen_contract import validate_feature_manifest


def test_special_team_rates_use_matched_observations_and_only_prior_games(tmp_path):
    observations=pd.DataFrame([dict(game_id=g,team=t,metric=m,value=v) for g in [1,2] for t in ['A','B']
        for m,v in [('field_goals',8 if g==1 else 100),('field_goal_attempts',10 if g==1 else 100)]])
    games=special_team_games(observations)
    schedule=pd.DataFrame([dict(id=g,season=2020,season_type='regular',completed=True,
        start_date=d,home_team='A',away_team='B',home_classification='fbs',away_classification='fbs')
        for g,d in [(1,'2020-09-01T00:00:00Z'),(2,'2020-09-08T00:00:00Z')]])
    state,excluded=special_team_state(games,schedule)
    assert excluded==[1]
    assert state.st_field_goal_rate.tolist()==[.8,.8]
    assert state.st_punt_yards_per_punt.isna().all()
    schema=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/nextgen_feature_manifest_schema_v1.json').read_text())
    for d,count in [('a',7),('b',8),('c',6)]:
        b=SpecialTeamsBuilder(tmp_path,state,d)
        assert len(b.feature_columns)==count
        validate_feature_manifest(json.loads(b.manifest_path.read_text()),schema)
        assert len(b.build_frame())==2
