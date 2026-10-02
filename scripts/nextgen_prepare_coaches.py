#!/usr/bin/env python3
"""Prepare regular-only historical coach aggregates, not model-ready rows."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from gridiron_ml.experiments.nextgen_f09 import verified_endpoint_records
from gridiron_ml.experiments.nextgen_coaching import regular_coach_seasons
from gridiron_ml.pipeline.fetch.nextgen_acquisition import load_authoritative_schedule, sha256_file, atomic_json


def main():
    config=json.loads((ROOT/'configs/experiments/nextgen_fingerprints_v1.json').read_text())
    root=Path(config['artifact_root'])
    inventory=json.loads((ROOT/'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text())
    schedule,schedule_hash=load_authoritative_schedule(root,inventory)
    records=verified_endpoint_records(root,'/coaches/seasons')
    if {int(r['year']) for r in records} != set(range(2010,2026)):
        raise ValueError('Incomplete 2010–2025 coaching acquisition')
    frames=[]
    for r in records:
        frame=pd.read_parquet(r['cache_path'])
        if not frame.year.eq(r['year']).all():
            raise ValueError('Coach partition year mismatch')
        frames.append(frame)
    history,audit=regular_coach_seasons(pd.concat(frames,ignore_index=True),schedule)
    path=root/'canonical/coach_regular_history.parquet'
    tmp=path.with_suffix('.tmp.parquet')
    history.to_parquet(tmp,index=False,compression='zstd')
    tmp.replace(path)
    report={'model_ready':False,'rows':len(history),'coaches':int(history.coach_id.nunique()),
            'schedule_sha256':schedule_hash,'data_sha256':sha256_file(path),
            'code_sha256':sha256_file(ROOT/'src/gridiron_ml/experiments/nextgen_coaching.py'),
            'source_sha256':{r['request_id']:r['sha256'] for r in records}, **audit}
    atomic_json(root/'results/coach_regular_history_preparation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'source_sha256','excluded_team_seasons'}}))


if __name__=='__main__':main()
