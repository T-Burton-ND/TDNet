#!/usr/bin/env python3
"""Resume the explicitly authorized historical raw archive; never train models.

Uses the existing immutable request identities, cache verification, and shared
20,000-attempt ledger. A separate reviewed archive gate supersedes the earlier
feature-adoption Stage E decision without asserting that attribution is complete.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import CFBDClient, CallBudget, load_cfbd_key_file
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (
    AcquisitionLedger, atomic_json, execute_request, quota_allows, sha256_file, verify_cache,
)


def pending_items(items, ledger):
    """Never repeat terminal requests, including reviewed empty/error responses."""
    pending, retained = [], []
    seen = set()
    queries = set()
    for item in items:
        rid = item['request_id']
        query = (item['endpoint'], item['parameters_json'])
        if rid in seen or query in queries:
            raise ValueError('Duplicate request identity or outbound query')
        seen.add(rid)
        queries.add(query)
        old = ledger.read(rid)
        if old and old['status'] != 'planned':
            if old['status'] in ('success_complete', 'skipped_existing_complete'):
                if not verify_cache(Path(old['cache_path']), old):
                    raise ValueError('Existing successful cache changed; review before refetch')
            retained.append(old)
        else:
            pending.append(item)
    return pending, retained


class PacedBudget:
    """Global pacing and provider-reserve ceiling apply to every HTTP retry too."""
    def __init__(self, budget, ceiling, interval=1.1):
        self.budget = budget
        self.ceiling = min(20000, ceiling)
        self.interval = interval
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.next_at = 0.

    def reserve(self, endpoint):
        with self.lock:
            if self.stop.is_set() or self.budget.status()['reserved'] >= self.ceiling:
                raise RuntimeError('Archive paused or provider/local reserve reached')
            time.sleep(max(0., self.next_at - time.monotonic()))
            if self.stop.is_set():
                raise RuntimeError('Archive paused')
            value = self.budget.reserve(endpoint)
            self.next_at = time.monotonic() + self.interval
            return value

    def observe_response(self, response, *args, **kwargs):
        if response.status_code == 429:
            try:
                delay = max(65., float(response.headers.get('Retry-After', 65)))
            except ValueError:
                delay = 65.
            with self.lock:
                self.interval = min(60.,max(5.,self.interval*2))
                self.next_at = max(self.next_at, time.monotonic() + delay)
            print(json.dumps({'rate_limited':True,'cooldown_seconds':delay,
                              'new_minimum_interval':self.interval,
                              'at_utc':datetime.now(timezone.utc).isoformat()}),flush=True)


def run(root, phase, workers, max_requests=None):
    directory = root / 'results/preflight/archive_completion_20260928'
    directory.mkdir(parents=True, exist_ok=True)
    # Lock covers plan read, quota check and the entire worker lifetime.
    with (directory/'executor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(root, directory, phase, workers, max_requests)


def run_locked(root, directory, phase, workers, max_requests):
    plan_path = directory/'plan.json'
    plan = json.loads(plan_path.read_text())
    gate = json.loads((directory/'reviewed_gate.json').read_text())
    if (gate.get('plan_sha256') != sha256_file(plan_path)
            or gate.get('archive_acquisition_authorized') is not True
            or gate.get('era_samples_reviewed') is not True
            or gate.get('no_training_or_canonical_mutation') is not True):
        raise ValueError('Missing reviewed archive authorization')
    items = [i for i in plan['items'] if (i['endpoint'] == '/plays/stats') == (phase == 'attribution')]
    excluded = set(gate.get('excluded_request_ids', []))
    items = [i for i in items if i['request_id'] not in excluded]
    ledger = AcquisitionLedger(root)
    pending, retained = pending_items(items, ledger)
    if max_requests is not None:
        pending = pending[:max_requests]
    budget = CallBudget(root/'results/cfbd_api_call_budget.json', 20000)
    start_reserved = budget.status()['reserved']
    if start_reserved + len(pending) + bool(pending) > 20000:
        raise RuntimeError('First attempts exceed remaining shared budget')
    os.environ['CFBD_API_KEY'] = load_cfbd_key_file(ROOT/'.env')
    quota = None
    if pending:
        quota = CFBDClient(call_budget=budget).get_json('/info', {}, max_retries=0)
        quota = quota[0] if isinstance(quota, list) else quota
        current = budget.status()['reserved']
        if not quota_allows(quota['remainingCalls'], current, reserve=10000, hard_limit=20000):
            raise RuntimeError('Provider reserve cannot cover remaining local budget')
        ceiling = min(20000, current + quota['remainingCalls'] - 10000)
    else:
        ceiling = 20000
    pace = PacedBudget(budget, ceiling)
    before_bytes = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if before_bytes >= 100 * 1024**3 or shutil.disk_usage(root).free < 2 * 1024**3:
        raise RuntimeError('Storage limit reached')
    counts = Counter(r['status'] for r in retained)
    done, added_bytes, stopping = 0, 0, None
    started = datetime.now(timezone.utc).isoformat()
    report_path = directory/f'{phase}_status.json'

    def report(finished=False):
        payload = dict(phase=phase, started_utc=started,
                       updated_utc=datetime.now(timezone.utc).isoformat(),
                       selected_requests=len(items), newly_selected=len(pending),
                       retained_without_http=len(retained), completed_this_run=done,
                       statuses=dict(counts), budget=budget.status(), quota_at_start=quota,
                       attempts_this_run=budget.status()['reserved']-start_reserved,
                       additional_cache_bytes=added_bytes,
                       excluded_requests=len(excluded), stop_reason=stopping,
                       dispatch_finished=finished,
                       all_selected_responses_complete=(finished and not stopping and
                           all(k in ('success_complete','skipped_existing_complete') for k in counts)),
                       plan_sha256=sha256_file(plan_path))
        atomic_json(report_path, payload)
        print(json.dumps(payload), flush=True)

    def acquire(item):
        client = CFBDClient(call_budget=pace)
        client.s.hooks['response'] = [pace.observe_response]
        try:
            return execute_request(item, ledger, client)
        finally:
            client.s.close()

    report()
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        for _ in range(workers):
            item = next(iterator, None)
            if item is not None:
                active[pool.submit(acquire, item)] = item
        while active:
            completed, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            if not completed:
                report()
                continue
            for future in completed:
                item = active.pop(future)
                rec = future.result()
                counts[rec['status']] += 1
                done += 1
                added_bytes += rec.get('byte_size') or 0
                code = rec.get('http_status')
                # Empty successful responses are retained as unavailable; continue
                # other games. Transport/auth/cap errors require adaptive review.
                if (rec['status'] in ('success_suspected_partial','failed_retryable')
                        or code in (400,401,403,404,429)):
                    stopping = f"Review {item['request_id']}: {rec['status']} HTTP {code}"
                    pace.stop.set()
                if before_bytes + added_bytes > 100 * 1024**3:
                    stopping = 'Storage soft limit reached'
                    pace.stop.set()
                if done % 100 == 0:
                    if shutil.disk_usage(root).free < 2 * 1024**3:
                        stopping = 'Insufficient free storage'
                        pace.stop.set()
                    report()
                if not pace.stop.is_set():
                    item = next(iterator, None)
                    if item is not None:
                        active[pool.submit(acquire, item)] = item
    report(finished=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('attribution','gaps'), required=True)
    parser.add_argument('--workers', type=int, choices=range(1,5), default=4)
    parser.add_argument('--max-requests', type=int)
    args = parser.parse_args()
    if args.max_requests is not None and args.max_requests < 1:
        parser.error('--max-requests must be positive')
    run(Path('/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen'),
        args.phase, args.workers, args.max_requests)
