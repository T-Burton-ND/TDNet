#!/usr/bin/env python3
"""Materialize resumable player observations; explicitly not model-ready features."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from gridiron_ml.experiments.nextgen_players import flatten_player_boxes, checked_player_history
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (
    COMPLETE, atomic_json, load_authoritative_schedule, sha256_file, verify_cache,
)


def prepare(root):
    parser_path = ROOT / 'src/gridiron_ml/experiments/nextgen_players.py'
    parser_hash = sha256_file(parser_path)
    preparation_hash = sha256_file(Path(__file__))
    inventory = json.loads((ROOT / 'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text())
    schedule, schedule_hash = load_authoritative_schedule(root, inventory)
    records = []
    for path in (root / 'request_ledger').glob('*/*.json'):
        record = json.loads(path.read_text())
        if record['endpoint'] == '/games/players' and record['status'] in COMPLETE and 2010 <= int(record['year']) <= 2025:
            records.append(record)
    dest = root / 'canonical/player_observations'
    dest.mkdir(parents=True, exist_ok=True)
    reports = []
    for record in sorted(records, key=lambda r: (r['year'], r.get('week') or 0)):
        path = dest / (record['request_id']+'.parquet')
        meta = path.with_suffix('.json')
        binding = {'raw_sha256': record['sha256'], 'schedule_sha256': schedule_hash,
                   'parser_sha256': parser_hash,
                   'preparation_sha256': preparation_hash}
        if path.exists() and meta.exists():
            report = json.loads(meta.read_text())
            if all(report.get(k)==v for k,v in binding.items()) and report.get('data_sha256')==sha256_file(path):
                reports.append(report)
                continue
        raw_path = Path(record['cache_path'])
        if verify_cache(raw_path, record) is None:
            raise ValueError('Player source cache failed verification')
        raw = pd.read_parquet(raw_path)
        expected = schedule.loc[schedule.season.eq(record['year']) & schedule.week.eq(record['week'])]
        raw = raw.loc[raw.id.isin(expected.id)]
        observations, audit = flatten_player_boxes(raw)
        checked = checked_player_history(observations, schedule)
        if sha256_file(parser_path) != parser_hash or sha256_file(Path(__file__)) != preparation_hash:
            raise RuntimeError('Source code changed during preparation; resume with a fresh process')
        temporary = path.with_suffix('.tmp.parquet')
        checked.to_parquet(temporary, index=False, compression='zstd')
        temporary.replace(path)
        report = {**binding, 'request_id': record['request_id'], 'year': record['year'], 'week': record['week'],
                  'rows': len(checked), 'games': int(raw.id.nunique()), 'audit': audit,
                  'missing_game_ids': sorted(set(expected.id)-set(raw.id)), 'data_sha256': sha256_file(path)}
        atomic_json(meta, report)
        reports.append(report)
    report = {'model_ready': False, 'scope': 'completed ledger partitions observed at process start',
              'partitions': reports, 'partition_count': len(reports),
              'observation_count': sum(r['rows'] for r in reports)}
    atomic_json(root / 'results/player_observation_preparation.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='partitions'}))


if __name__ == '__main__':
    config = json.loads((ROOT / 'configs/experiments/nextgen_fingerprints_v1.json').read_text())
    prepare(Path(config['artifact_root']))
