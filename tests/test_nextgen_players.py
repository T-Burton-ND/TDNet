import numpy as np
import pandas as pd
import pytest
from gridiron_ml.experiments.nextgen_players import flatten_player_boxes, match_players


def test_structured_counts_preserve_missing_and_fractional_stats():
    types = [{'name': 'C/ATT', 'athletes': [{'id': '4', 'name': 'QB', 'stat': '8/13'}, {'id': '5', 'name': 'QB2', 'stat': '--'}]}]
    frame = pd.DataFrame([{'id': 1, 'teams': [{'team': 'A', 'categories': [
        {'name': 'passing', 'types': types},
        {'name': 'defensive', 'types': [{'name': 'SACKS', 'athletes': [{'id': '8', 'name': 'DL', 'stat': '0.5'}]}]}
    ]}]}])
    values, audit = flatten_player_boxes(frame)
    assert values.value.tolist()[:2] == [8, 13]
    assert values.loc[values.athlete_id.eq('5'), 'value'].isna().all()
    assert values.iloc[-1].value == .5
    assert audit['invalid_values'] == 1
    with pytest.raises(ValueError, match='Duplicate'):
        flatten_player_boxes(pd.concat([frame, frame]))


def test_matching_does_not_override_conflicting_ids_or_ambiguous_names():
    def row(pid, name, team='A'):
        return dict(athlete_id=pid, name=name, team=team, season=2020, position='QB')
    roster = pd.DataFrame([row('1', 'John Smith'), row('2', 'Pat Doe'), row('3', 'Pat Doe')])
    players = pd.DataFrame([row('1', 'Different Name'), row(None, 'John Smith'), row('9', 'John Smith'), row(None, 'Pat Doe'), row('1', 'John Smith', 'B')])
    result = match_players(players, roster)
    assert result.match_method.tolist() == ['stable_id', 'exact_name', 'unmatched', 'unmatched', 'unmatched']


def test_prior_usage_excludes_target_and_recent_unavailable_games():
    from gridiron_ml.experiments.nextgen_players import player_usage_state
    schedule = pd.DataFrame([dict(id=i, season=2020, season_type='regular', completed=True,
        start_date=date, home_team='A', away_team='B', home_classification='fbs', away_classification='fbs')
        for i, date in [(1, '2020-09-01T12:00:00Z'), (2, '2020-09-07T12:00:00Z'), (3, '2020-09-08T12:00:00Z')]])
    observations = pd.DataFrame([dict(game_id=i, team=t, athlete_id=p, athlete_name=p,
        metric='rush_attempts', value=v) for i in [1, 2, 3] for t in ['A', 'B']
        for p,v in [('x', 3 if i==1 else 100), ('y', 1)]])
    state, _ = player_usage_state(observations, schedule)
    target = state.loc[state.target_game_id.eq(3)]
    assert target.rush_attempts_top_share.tolist() == [.75, .75]
    assert target.rush_attempts_hhi.tolist() == [.625, .625]
    assert target.latest_source_game_id.tolist() == [1, 1]
    assert target.tackles_hhi.isna().all()
    bad = schedule.copy(); bad.loc[0, 'season_type']='postseason'
    with pytest.raises(ValueError, match='regular-season'):
        player_usage_state(observations, bad)


def test_missing_identity_is_not_one_fictional_player():
    from gridiron_ml.experiments.nextgen_players import usage_concentration
    prior = pd.DataFrame({'athlete_id': ['1', None], 'metric': ['receptions']*2, 'value': [3., 1.]})
    assert np.isnan(usage_concentration(prior)['receptions_hhi'])


def test_returning_production_uses_frozen_membership_and_prior_season_only():
    from gridiron_ml.experiments.nextgen_players import returning_production
    history = pd.DataFrame([dict(athlete_id=p, team=t, season=y, available=pd.Timestamp(f'{y}-12-01T00:00:00Z'), metric='rush_attempts', value=v)
        for p,t,y,v in [('returner','A',2019,30), ('departed','A',2019,70),
                        ('transfer','B',2019,20), ('returner','A',2020,999)]])
    roster = pd.DataFrame([dict(athlete_id=p, team='A', season=2020, available_utc='2020-08-01T00:00:00Z', availability_evidence='fixture archived roster')
        for p in ['returner','transfer','freshman']])
    values, report = returning_production(history,roster,team='A',season=2020,week0_cutoff='2020-09-01T00:00:00Z')
    assert values['rush_attempts_returning_share']==.3
    assert values['rush_attempts_incoming_prior_total']==20
    assert np.isnan(values['pass_attempts_incoming_prior_total'])
    assert report['roster_with_observed_prior_production']==2
    roster.loc[0,'available_utc']='2020-09-02T00:00:00Z'
    with pytest.raises(ValueError, match='Week 0'):
        returning_production(history,roster,team='A',season=2020,week0_cutoff='2020-09-01T00:00:00Z')
