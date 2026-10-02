import numpy as np
import pandas as pd
from gridiron_ml.experiments.nextgen_offensive_units import offensive_unit_state


def test_rooms_use_only_prior_supported_game_records():
    schedule=pd.DataFrame([dict(id=i,season=2020,week=i,season_type='regular',completed=True,
        home_team='A',away_team='B',home_classification='fbs',away_classification='fbs',
        start_date=f'2020-09-{day:02d}T00:00:00Z') for i,day in [(1,1),(2,8),(3,15),(4,22)]])
    bound=pd.DataFrame([dict(game_id=g.id,season=2020,team=t,athlete_id=t,unit='qb',
        kickoff=pd.Timestamp(g.start_date),available=pd.Timestamp(g.start_date)+pd.Timedelta(hours=48),
        **{'average_p_p_a.pass':float(g.id) if g.id<4 else 999.,'average_p_p_a.rush':None})
        for g in schedule.itertuples(index=False) for t in ['A','B']])
    state,audit=offensive_unit_state(bound,schedule)
    assert audit['excluded_target_games']==[1]
    assert state.loc[state.target_game_id.eq(3),'qb_observed_player_ppa_mean'].isna().all()
    last=state.loc[state.target_game_id.eq(4)]
    assert last.qb_observed_player_ppa_mean.tolist()==[2.,2.]
    assert np.allclose(last.qb_observed_player_ppa_dispersion, np.std([1,2,3]))
    assert last.rb_observed_player_ppa_mean.isna().all()


def test_offensive_room_designs_have_valid_manifests(tmp_path):
    import json
    from pathlib import Path
    from gridiron_ml.experiments.nextgen_f12_offense import OffensiveRoomBuilder
    from gridiron_ml.experiments.nextgen_contract import validate_feature_manifest
    schema=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/nextgen_feature_manifest_schema_v1.json').read_text())
    state=pd.DataFrame({room+'_observed_player_ppa_'+stat:[.5] for room in ['qb','rb','wrte'] for stat in ['mean','dispersion']})
    for d,n in [('a',6),('b',8),('c',3)]:
        b=OffensiveRoomBuilder(tmp_path,state,d)
        assert len(b.feature_columns)==n
        validate_feature_manifest(json.loads(b.manifest_path.read_text()),schema)
        assert len(b.build_frame().columns)==n
