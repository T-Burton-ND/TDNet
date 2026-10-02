#!/usr/bin/env python3
"""Measure historical roster identity/class fields without inventing availability."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from gridiron_ml.experiments.nextgen_f09 import verified_endpoint_records
from gridiron_ml.pipeline.fetch.nextgen_acquisition import atomic_json


def main():
    root=Path(json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())['artifact_root'])
    reports=[]
    for record in verified_endpoint_records(root,'/roster'):
        frame=pd.read_parquet(record['cache_path'])
        year=pd.to_numeric(frame.year,errors='coerce')
        plausible=year.between(1,6)
        reports.append({'season':record['year'],'rows':len(frame),'request_id':record['request_id'],
                        'source_sha256':record['sha256'],'year_values':sorted(year.dropna().unique().tolist()),
                        'class_like_rows':int(plausible.sum()),'request_season_in_year_rows':int(year.eq(record['year']).sum()),
                        'other_or_missing_year_rows':int((~plausible & ~year.eq(record['year'])).sum()),
                        'duplicate_team_ids':int(frame.duplicated(['team','id']).sum()),
                        'missing_id_rows':int(frame.id.isna().sum()),
                        'missing_position_rows':int(frame.position.isna().sum())})
    result={'partitions':reports,'week0_membership_verified':False,'class_year_usable_without_review':False,
            'finding':'Roster year mixes plausible class numbers with requested season values; do not treat the whole field as experience.',
            'availability_limit':'A season query and current acquisition timestamp do not establish historical Week-0 membership or position availability.',
            'allowed_alternative':'Derive observed prior regular-game experience from player-game IDs with strict pre-target availability; do not invent class years.'}
    atomic_json(root/'results/roster_semantics_audit.json',result)
    print(json.dumps({'partitions':len(reports),'rows':sum(r['rows'] for r in reports),
                      'request_season_in_year_rows':sum(r['request_season_in_year_rows'] for r in reports),
                      'week0_membership_verified':False}))


if __name__=='__main__':main()
