#!/usr/bin/env python3
"""Acquire complete conference/week attribution batches and materialize game caches.

Only reconstruct games when both teams' complete conference filters are present.
Small groups and capped batches fall back to game requests. No model execution.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nextgen_archive_complete import PacedBudget
from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import CFBDClient, CallBudget, load_cfbd_key_file, write_parquet
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (
    AcquisitionLedger, atomic_json, execute_request, load_authoritative_schedule,
    make_request, quota_allows, sha256_file, verify_cache,
)


def build_batch_plan(root, directory):
    original = json.loads((directory/'plan.json').read_text())
    inventory = json.loads((ROOT/'configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json').read_text())
    endpoint = next(x for x in inventory['endpoints'] if x['endpoint']=='/plays/stats')
    schedule, digest = load_authoritative_schedule(root, inventory)
    schedule = schedule.set_index('id')
    manifest = [json.loads(line) for line in
                (root/'results/preflight/cfbd_request_manifest_v1.jsonl').read_text().splitlines()]
    names = {}
    for rec in manifest:
        if rec['endpoint']=='/conferences':
            frame = pd.read_parquet(rec['cache_path'])
            names[rec['year']] = dict(zip(frame.name, frame.abbreviation))
    ledger = AcquisitionLedger(root)
    games, candidates, direct, reused = {}, {}, set(), []
    for item in original['items']:
        if item['endpoint']!='/plays/stats' or item['year']<2013:
            continue
        gid = item['game_id']
        games[str(gid)] = item
        old = ledger.read(item['request_id'])
        if old and old['status'] in ('success_complete','skipped_existing_complete'):
            if not verify_cache(Path(old['cache_path']),old):
                raise ValueError('Changed successful game cache')
            reused.append(gid)
            continue
        g = schedule.loc[gid]
        confs = {g.home_conference,g.away_conference}
        if (g.home_classification!='fbs' or g.away_classification!='fbs'
                or any(not isinstance(names[g.season].get(c),str) for c in confs)):
            direct.add(gid)
        else:
            candidates[gid] = [(int(g.season),int(g.week),c,names[g.season][c]) for c in sorted(confs)]
    # Avoid a batch call that only replaces one or two game calls, particularly
    # when its counterpart would require another independent conference query.
    while candidates:
        degrees = Counter(k for keys in candidates.values() for k in keys)
        small = {g for g,keys in candidates.items() if any(degrees[k]<3 for k in keys)}
        if not small:
            break
        direct.update(small)
        for g in small:
            del candidates[g]
    groups = defaultdict(list)
    game_sources = defaultdict(list)
    for gid,keys in candidates.items():
        for key in keys:
            groups[key].append(gid)
    batches = []
    for (year,week,name,abbr),gids in sorted(groups.items()):
        params = dict(year=year,week=week,conference=abbr,seasonType='regular')
        item = make_request(endpoint,params,f'{year}-w{week:02d}-conference-{abbr}',root,year,week)
        item.update(conference_name=name,covered_game_ids=sorted(gids))
        batches.append(item)
        for gid in gids:
            game_sources[str(gid)].append(item['request_id'])
    result = dict(original_plan_sha256=sha256_file(directory/'plan.json'),schedule_sha256=digest,
                  games=games,batches=batches,game_sources=dict(game_sources),
                  direct_game_ids=sorted(direct),reused_game_ids=sorted(reused),
                  status='reviewed sample-equivalent batching; capped partitions require game fallback')
    atomic_json(directory/'batch_plan.json',result)
    return result


def assemble_game(item, sources, ledger):
    """Bind a cache to complete source partitions, with no fictional HTTP call."""
    old = ledger.read(item['request_id'])
    if old and old['status'] in ('success_complete','skipped_existing_complete'):
        if not verify_cache(Path(old['cache_path']),old):
            raise ValueError('Changed existing game cache')
        return old
    frames = []
    for rec in sources:
        if rec['status']!='success_complete' or not verify_cache(Path(rec['cache_path']),rec):
            raise ValueError('Cannot assemble from incomplete batch')
        frame = pd.read_parquet(rec['cache_path'])
        if (not frame.season.eq(item['year']).all()
                or not frame.week.eq(rec['week']).all()
                or not frame.conference.eq(rec['conference_name']).all()):
            raise ValueError('Batch response violates year/week/conference scope')
        frames.append(frame.loc[frame.game_id.eq(item['game_id'])])
    data = pd.concat(frames,ignore_index=True)
    # Distinct conference filters are disjoint in the observed API semantics;
    # do not drop source duplicates or average disputed event values.
    record = dict(item,status='needs_review',http_status=None,
                  attempt_count=old.get('attempt_count',0) if old else 0,
                  attempts_this_execution=0,origin='derived_complete_conference_partitions',
                  derived_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_partitions=[dict(request_id=r['request_id'],path=r['cache_path'],sha256=r['sha256'])
                                     for r in sources],
                  completeness_status='empty_partition_union' if data.empty else 'complete_query_scope_union',
                  error_summary='No attributed events in complete conference partitions' if data.empty else None)
    if old:
        record['prior_record'] = {k:old.get(k) for k in ('status','attempt_count','http_status','error_summary')}
    if not data.empty:
        path = Path(item['cache_path'])
        write_parquet(data,path,snake=True)
        meta = verify_cache(path)
        if meta is None:
            raise ValueError('Derived cache failed verification')
        record.update(meta,status='success_complete')
    ledger.write(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--workers',type=int,choices=(1,2,3,4),default=2)
    parser.add_argument('--interval',type=float,default=5.)
    args = parser.parse_args()
    if args.interval < 1.1:
        parser.error('Interval must be at least 1.1 seconds')
    root = Path('/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen')
    directory = root/'results/preflight/archive_completion_20260928'
    with (directory/'executor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = build_batch_plan(root,directory)
        print(json.dumps({k:len(plan[k]) for k in ('games','batches','direct_game_ids','reused_game_ids')}),flush=True)
        if not args.execute:
            return
        gate = json.loads((directory/'reviewed_gate.json').read_text())
        sample = json.loads((directory/'batch_validation.json').read_text())
        if (gate.get('archive_acquisition_authorized') is not True
                or gate.get('plan_sha256')!=plan['original_plan_sha256']
                or not sample['checks'] or not all(c['exact_row_match'] for c in sample['checks'])):
            raise ValueError('Archive/batch sample gate not satisfied')
        budget = CallBudget(root/'results/cfbd_api_call_budget.json',20000)
        start = budget.status()['reserved']
        os.environ['CFBD_API_KEY'] = load_cfbd_key_file(ROOT/'.env')
        quota = CFBDClient(call_budget=budget).get_json('/info',{},max_retries=0)
        quota = quota[0] if isinstance(quota,list) else quota
        if not quota_allows(quota['remainingCalls'],budget.status()['reserved'],reserve=10000,hard_limit=20000):
            raise RuntimeError('Provider reserve insufficient')
        if start + len(plan['batches']) + len(plan['direct_game_ids']) + 1 > 20000:
            raise RuntimeError('Minimum first attempts exceed local budget')
        initial_bytes = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if initial_bytes>=100*1024**3 or shutil.disk_usage(root).free<2*1024**3:
            raise RuntimeError('Storage guard')
        pace = PacedBudget(budget,20000,interval=args.interval)
        ledger = AcquisitionLedger(root)
        results, problems = {}, []
        bytes_added = 0
        started = datetime.now(timezone.utc).isoformat()

        def report(phase,completed=False):
            obj = dict(phase=phase,dispatch_finished=completed,started_utc=started,
                       updated_utc=datetime.now(timezone.utc).isoformat(),
                       budget=budget.status(),attempts_this_run=budget.status()['reserved']-start,
                       quota_at_start=quota,completed_requests=len(results),
                       statuses=dict(Counter(r['status'] for r in results.values())),
                       problems=problems,additional_cache_bytes=bytes_added,
                       batch_plan_sha256=sha256_file(directory/'batch_plan.json'))
            atomic_json(directory/'batch_status.json',obj)
            print(json.dumps(obj),flush=True)

        def acquire(item):
            old = ledger.read(item['request_id'])
            if old and old['status'] not in ('planned',):
                if old['status'] in ('success_complete','skipped_existing_complete'):
                    if not verify_cache(Path(old['cache_path']),old):
                        raise ValueError('Changed successful request cache')
                return {**old,'_reused_without_http':True}
            client = CFBDClient(call_budget=pace)
            client.s.hooks['response'] = [pace.observe_response]
            try:
                rec = execute_request(item,ledger,client)
                return rec
            finally:
                client.s.close()

        def fetch(items,phase):
            nonlocal bytes_added
            iterator = iter(items)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                active = {}
                for _ in range(args.workers):
                    item = next(iterator,None)
                    if item is not None:
                        active[pool.submit(acquire,item)] = item
                while active:
                    ready,_ = wait(active,timeout=30,return_when=FIRST_COMPLETED)
                    if not ready:
                        report(phase)
                        continue
                    for future in ready:
                        item = active.pop(future)
                        rec = future.result()
                        results[item['request_id']] = rec
                        bytes_added += rec.get('byte_size') or 0
                        if (not rec.get('_reused_without_http') and
                                (rec['status']=='failed_retryable' or rec.get('http_status') in (400,401,403,404,429))):
                            problems.append(dict(request_id=item['request_id'],status=rec['status'],error=rec.get('error_summary')))
                            pace.stop.set()
                        if initial_bytes+bytes_added>100*1024**3:
                            problems.append(dict(error='Storage limit'));pace.stop.set()
                        if len(results)%50==0:
                            if shutil.disk_usage(root).free<2*1024**3:
                                problems.append(dict(error='Free storage limit'));pace.stop.set()
                            report(phase)
                        if not pace.stop.is_set():
                            nxt = next(iterator,None)
                            if nxt is not None:
                                active[pool.submit(acquire,nxt)] = nxt

        report('conference_batches')
        fetch(plan['batches'],'conference_batches')
        if pace.stop.is_set():
            report('stopped',True)
            return
        direct = set(plan['direct_game_ids'])
        reconstructed = Counter()
        for gid,rids in plan['game_sources'].items():
            sources = [ledger.read(rid) for rid in rids]
            if all(r and r['status']=='success_complete' for r in sources):
                # Existing sample records may predate this plan's name annotation.
                byid = {r['request_id']:r for r in plan['batches']}
                sources = [{**r,'conference_name':byid[r['request_id']]['conference_name']} for r in sources]
                rec = assemble_game(plan['games'][gid],sources,ledger)
                reconstructed[rec['status']] += 1
            else:
                direct.add(int(gid))
        atomic_json(directory/'batch_reconstruction.json',dict(
            reconstructed=dict(reconstructed),fallback_and_direct_games=len(direct),
            source_batch_statuses=dict(Counter(r['status'] for r in results.values()))))
        # Small game queries use the previously selected conservative base pace;
        # provider throttling can raise it again independently in this phase.
        pace.interval = 1.1
        fetch([plan['games'][str(g)] for g in sorted(direct)],'game_fallbacks')
        report('complete' if not pace.stop.is_set() else 'stopped',True)


if __name__=='__main__':
    main()
