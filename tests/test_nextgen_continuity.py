import numpy as np
import pandas as pd
from gridiron_ml.experiments.nextgen_continuity import observed_usage_continuity


def test_continuity_is_prior_target_dynamic_usage_not_future_roster():
    h=pd.DataFrame([dict(team='A',season=y,athlete_id=p,metric='pass_attempts',value=v,available=pd.Timestamp(t))
        for y,p,v,t in [(2020,'a',80,'2020-09-01T00:00:00Z'),(2020,'b',20,'2020-09-01T00:00:00Z'),
                         (2021,'a',10,'2021-09-01T00:00:00Z'),(2021,'c',30,'2021-09-01T00:00:00Z'),
                         (2021,'b',999,'2021-09-08T00:00:00Z')]])
    values,support=observed_usage_continuity(h,team='A',season=2021,cutoff='2021-09-08T00:00:00Z')
    assert values['pass_attempts_observed_returning_usage_share']==.25
    assert values['pass_attempts_prior_production_share_of_observed_returners']==.8
    assert support['pass_attempts']['overlapping_users']==1
    preseason,_=observed_usage_continuity(h,team='A',season=2021,cutoff='2021-08-01T00:00:00Z')
    assert all(np.isnan(v) for v in preseason.values())
    h.loc[2,'athlete_id']=None
    unknown,_=observed_usage_continuity(h,team='A',season=2021,cutoff='2021-09-08T00:00:00Z')
    assert np.isnan(unknown['pass_attempts_observed_returning_usage_share'])


def test_continuity_manifests(tmp_path):
    import json
    from pathlib import Path
    from gridiron_ml.experiments.nextgen_f10_continuity import ObservedContinuityBuilder
    from gridiron_ml.experiments.nextgen_contract import validate_feature_manifest
    schema=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/nextgen_feature_manifest_schema_v1.json').read_text())
    for d,n in [('a',16),('b',17),('c',8)]:
        b=ObservedContinuityBuilder(tmp_path,pd.DataFrame(),d)
        assert len(b.feature_columns)==n
        validate_feature_manifest(json.loads(b.manifest_path.read_text()),schema)
