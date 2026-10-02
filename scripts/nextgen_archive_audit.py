#!/usr/bin/env python3
"""Audit the acquired archive against existing structured plays, without API calls."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (
    AcquisitionLedger, atomic_json, sha256_file, verify_cache,
)
from gridiron_ml.experiments.nextgen_microstructure import RUSH_TYPES


def compare_game(stats, plays):
    """Count source disagreements; preserve missingness and original values."""
    stats = stats.copy()
    plays = plays.copy()
    stats['play_key'] = stats.play_id.astype(str)
    plays['play_key'] = plays.id.astype(str)
    if plays.play_key.duplicated().any():
        raise ValueError('Duplicate canonical play identity')
    joined = stats.merge(plays[['play_key','offense','defense','play_type','yards_gained','period']],
                         on='play_key', how='left', validate='many_to_one',
                         suffixes=('_stat','_play'))
    rush = joined[joined.stat_type.eq('Rush')]
    expected = set(plays.loc[plays.play_type.isin(RUSH_TYPES),'play_key'])
    observed = set(rush.play_key)
    matched = rush[rush.play_type.notna()]
    return dict(rows=len(stats), canonical_plays=len(plays),
                unique_attributed_plays=int(stats.play_key.nunique()),
                unmatched_attributed_rows=int(joined.play_type.isna().sum()),
                missing_athlete_ids=int(stats.athlete_id.isna().sum()),
                expected_rush_plays=len(expected), matched_rush_plays=len(expected & observed),
                missing_rush_plays=len(expected-observed), extra_rush_plays=len(observed-expected),
                rush_yardage_mismatches=int(pd.to_numeric(matched.stat, errors='coerce').ne(
                    pd.to_numeric(matched.yards_gained, errors='coerce')).sum()),
                rush_team_mismatches=int(matched.team.ne(matched.offense).sum()),
                rush_period_mismatches=int(matched.period_stat.ne(matched.period_play).sum()),
                duplicate_actor_stat_rows=int(stats.duplicated(['play_key','athlete_id','stat_type']).sum()),
                stat_types_json=json.dumps(stats.stat_type.value_counts().to_dict(), sort_keys=True))


def main():
    root = Path('/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen')
    directory = root/'results/preflight/archive_completion_20260928'
    plan = json.loads((directory/'plan.json').read_text())
    ledger = AcquisitionLedger(root)
    # Original year/week play sources only; gap probes are separate evidence.
    manifest = [json.loads(line) for line in
                (root/'results/preflight/cfbd_request_manifest_v1.jsonl').read_text().splitlines()]
    play_sources = {(r['year'],r['week']): r for r in manifest if r['endpoint']=='/plays'}
    play_cache, reports, evidence = {}, [], []
    items = sorted([i for i in plan['items'] if i['endpoint']=='/plays/stats'],
                   key=lambda i: (i['year'],i['game_id']))
    for item in items:
        rec = ledger.read(item['request_id'])
        if rec is None or rec['status'] not in ('success_complete','skipped_existing_complete'):
            reports.append(dict(game_id=item['game_id'], season=item['year'],
                                request_id=item['request_id'], status=rec['status'] if rec else 'not_acquired'))
            continue
        path = Path(rec['cache_path'])
        if verify_cache(path, rec) is None:
            raise ValueError(f'Changed attribution cache {item["request_id"]}')
        stats = pd.read_parquet(path)
        if (not stats.game_id.eq(item['game_id']).all()
                or not stats.season.eq(item['year']).all() or stats.week.nunique()!=1):
            raise ValueError('Attribution response escapes planned game/year/week')
        week = int(stats.week.iloc[0])
        key = (item['year'],week)
        if key not in play_cache:
            play_cache.clear()
            source = play_sources[key]
            record = ledger.read(source['request_id'])
            if not record or verify_cache(Path(record['cache_path']),record) is None:
                raise ValueError('Changed canonical raw play source')
            play_cache[key] = pd.read_parquet(record['cache_path'])
        plays = play_cache[key]
        report = compare_game(stats, plays[plays.game_id.eq(item['game_id'])])
        report.update(game_id=item['game_id'],season=item['year'],week=week,
                      request_id=item['request_id'],status=rec['status'],
                      source_sha256=rec['sha256'])
        reports.append(report)
        evidence.append(dict(path=str(path),sha256=rec['sha256']))
    frame = pd.DataFrame(reports)
    dest = directory/'attribution_coverage.parquet'
    frame.to_parquet(dest,index=False)
    good = frame[frame.status.isin(['success_complete','skipped_existing_complete'])]
    totals = {c:int(good[c].sum()) for c in (
        'rows','canonical_plays','unique_attributed_plays','unmatched_attributed_rows','missing_athlete_ids',
        'expected_rush_plays','matched_rush_plays','missing_rush_plays','extra_rush_plays',
        'rush_yardage_mismatches','rush_team_mismatches','rush_period_mismatches','duplicate_actor_stat_rows')}
    summary = dict(at_utc=datetime.now(timezone.utc).isoformat(),
                   scope='All successfully cached planned 2012-2025 event-attribution responses; 2012 bulk excluded after samples.',
                   counts_by_year_status=[dict(season=int(y),status=s,games=int(n)) for (y,s),n in
                                          frame.groupby(['season','status']).size().items()],
                   totals=totals,
                   games_with_unmatched_rows=int(good.unmatched_attributed_rows.gt(0).sum()),
                   games_missing_canonical_play_coverage=int(good.canonical_plays.eq(0).sum()),
                   games_with_missing_rushes=int(good.missing_rush_plays.gt(0).sum()),
                   games_with_yardage_disagreements=int(good.rush_yardage_mismatches.gt(0).sum()),
                   per_game_report=dict(path=str(dest),sha256=sha256_file(dest)),
                   evidence=evidence,
                   semantic_limits=['Response completeness is not event completeness.',
                       'Missing event does not mean nonparticipation.',
                       'Retain source yardage disagreement; use original structured play outcome for future joins.',
                       'No data imputation, canonical mutation or model experiment performed.'])
    atomic_json(directory/'attribution_audit.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='evidence'},indent=2))


if __name__ == '__main__':
    main()
