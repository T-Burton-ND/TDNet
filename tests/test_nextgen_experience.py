import numpy as np
import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_experience import appearance_index,usage_weighted_experience


def test_observed_experience_counts_unique_prior_games_including_prior_team():
    history=pd.DataFrame([dict(athlete_id=p,game_id=g,season=2020,available=pd.Timestamp(date),value=v,metric='rush_attempts')
        for p,g,date,v in [('a',1,'2020-08-01T00:00:00Z',2),('a',2,'2020-08-08T00:00:00Z',3),
                            ('b',2,'2020-08-08T00:00:00Z',1),('a',3,'2020-09-01T00:00:00Z',100)]])
    index=appearance_index(pd.concat([history,history.iloc[[0]]]))
    prior=history.loc[history.game_id.eq(2)]
    result,latest=usage_weighted_experience(prior,index,cutoff='2020-09-01T00:00:00Z')
    assert result['rush_attempts_weighted_observed_games']==1.75
    assert np.isnan(result['pass_attempts_weighted_observed_games'])
    assert latest==pd.Timestamp('2020-08-08T00:00:00Z')
    with pytest.raises(ValueError,match='unavailable'):
        usage_weighted_experience(history,index,cutoff='2020-09-01T00:00:00Z')


def test_missing_identity_and_missing_history_do_not_become_zero_experience():
    prior=pd.DataFrame({'athlete_id':[None],'metric':['rush_attempts'],'value':[3.],
                        'available':[pd.Timestamp('2020-08-08T00:00:00Z')]})
    result,_=usage_weighted_experience(prior,{},cutoff='2020-09-01T00:00:00Z')
    assert np.isnan(result['rush_attempts_weighted_observed_games'])
    prior['athlete_id']='unseen'
    result,_=usage_weighted_experience(prior,{},cutoff='2020-09-01T00:00:00Z')
    assert np.isnan(result['rush_attempts_weighted_observed_games'])


def test_state_is_paired_and_freezes_experience_before_target():
    from gridiron_ml.experiments.nextgen_experience import observed_experience_state
    schedule = pd.DataFrame([dict(id=i, season=2020, season_type='regular', completed=True,
        home_team='A', away_team='B', home_classification='fbs', away_classification='fbs',
        start_date=f'2020-09-{day:02d}T00:00:00Z') for i, day in [(1,1),(2,8),(3,15)]])
    observations = pd.DataFrame([dict(game_id=i, team=t, athlete_id=t, metric='rush_attempts', value=10.)
        for i in [1,2,3] for t in ['A','B']])
    state, audit = observed_experience_state(observations, schedule)
    assert audit['excluded_target_games'] == [1]
    assert state.target_game_id.tolist() == [2,2,3,3]
    assert state.rush_attempts_weighted_observed_games.tolist() == [1.,1.,2.,2.]
    assert (state.feature_available_utc < state.target_start_utc).all()


def test_experience_manifests_are_valid(tmp_path):
    import json
    from pathlib import Path
    from gridiron_ml.experiments.nextgen_f10_experience import ObservedExperienceBuilder
    from gridiron_ml.experiments.nextgen_contract import validate_feature_manifest
    schema = json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/nextgen_feature_manifest_schema_v1.json').read_text())
    for d, count in [('a',8),('b',9),('c',3)]:
        b = ObservedExperienceBuilder(tmp_path, pd.DataFrame(), d)
        assert len(b.feature_columns) == count
        validate_feature_manifest(json.loads(b.manifest_path.read_text()), schema)
