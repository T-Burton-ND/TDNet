import pandas as pd
from gridiron_ml.experiments.nextgen_recruiting import recruiting_cohort_state


def test_lagged_cohorts_never_use_current_class_or_claim_roster_membership():
    recruits=pd.DataFrame([dict(id=f'{t}{y}{i}',year=y,recruit_type='HighSchool',position='QB',
        committed_to=t,rating=.8 if y==2019 else 1.,stars=3 if y==2019 else 5)
        for t in ['A','B'] for y in [2019,2020] for i in range(3)])
    schedule=pd.DataFrame([dict(id=1,season=2020,home_team='A',away_team='B',home_classification='fbs',away_classification='fbs',start_date='2020-09-01T00:00:00Z')])
    state,audit=recruiting_cohort_state(recruits,schedule)
    assert state.recruit_history_qb_rating.round(5).tolist()==[.8,.8]
    assert state.recruit_history_qb_bluechip_share.tolist()==[0.,0.]
    assert state.recruit_history_ol_rating.isna().all()
    assert all(c['class_max']==2019 for c in audit['cohorts'])
    assert state.feature_available_utc.eq(pd.Timestamp('2020-01-01T00:00:00Z')).all()


def test_recruiting_design_manifests(tmp_path):
    import json
    from pathlib import Path
    from gridiron_ml.experiments.nextgen_recruiting import RecruitingHistoryBuilder
    from gridiron_ml.experiments.nextgen_contract import validate_feature_manifest
    units=['ol','qb','rb','wrte','front','lb','secondary','special_teams']
    state=pd.DataFrame({f'recruit_history_{u}_{m}':[.5] for u in units for m in ['rating','bluechip_share']})
    schema=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/nextgen_feature_manifest_schema_v1.json').read_text())
    for d,count in [('a',16),('b',17),('c',8)]:
        b=RecruitingHistoryBuilder(tmp_path,state,d)
        assert len(b.feature_columns)==count
        validate_feature_manifest(json.loads(b.manifest_path.read_text()),schema)
