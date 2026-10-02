import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_coaching import regular_coach_seasons, coach_history_features


def test_coach_history_rebuilds_regular_scores_ignoring_season_aggregates():
    coaches = pd.DataFrame([{'year': y, 'team.school': 'A', 'coach.id': 1, 'attribution_complete': True,
                             'wins': 999, 'sp_offense': 999} for y in (2019,2020)])
    games = pd.DataFrame([dict(id=i, season=y, season_type=kind, completed=True,
        start_date=f'{y}-10-01T00:00:00Z', home_team='A', away_team='B', home_points=points, away_points=10)
        for i,y,kind,points in [(1,2019,'regular',20),(2,2019,'postseason',1000),(3,2020,'regular',500)]])
    history, _ = regular_coach_seasons(coaches,games)
    features=coach_history_features(history,1,target_season=2020,cutoff='2020-09-01T00:00:00Z')
    assert features['regular_points_for_per_game']==20
    assert features['observed_regular_games']==1
    assert features['regular_win_fraction']==1
    assert 'coach_id' not in features
    assert 'sp_offense' not in features


def test_split_coach_season_is_not_assigned_to_both_coaches():
    coaches = pd.DataFrame([{'year':2019,'team.school':'A','coach.id':i,'attribution_complete':True} for i in (1,2)])
    games=pd.DataFrame([dict(id=1,season=2019,season_type='regular',completed=True,start_date='2019-10-01T00:00:00Z',home_team='A',away_team='B',home_points=20,away_points=10)])
    history,audit=regular_coach_seasons(coaches,games)
    assert history.empty
    assert len(audit['excluded_team_seasons'])==1
