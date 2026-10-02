#!/usr/bin/env python3
"""Reviewed recovery of empty conference unions and interrupted game queries.

Never repeats a successful query or a previously empty direct response.
Each reviewed query gets one outbound attempt; the original budget is retained.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import CFBDClient, CallBudget, load_cfbd_key_file
from gridiron_ml.pipeline.fetch.nextgen_acquisition import AcquisitionLedger, atomic_json, execute_request, quota_allows, sha256_file, verify_cache
from nextgen_archive_complete import PacedBudget


class SingleAttemptClient(CFBDClient):
    def get_json(self, endpoint, params, **kwargs):
        kwargs['max_retries'] = 0
        return super().get_json(endpoint, params, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path('/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen')
    directory = root / 'results/preflight/archive_completion_20260928'
    dest = directory / 'recovery_pass_1'
    with (directory / 'executor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((directory / 'batch_plan.json').read_text())
        receipt = json.loads((directory / 'completion_receipt.json').read_text())
        ledger = AcquisitionLedger(root)
        selected, retained = [], []
        for gap in receipt['unresolved_games']:
            item = plan['games'][str(gap['game_id'])]
            old = ledger.read(item['request_id'])
            if old['status'] in ('success_complete', 'skipped_existing_complete'):
                if not verify_cache(Path(old['cache_path']), old):
                    raise ValueError('Successful cache changed')
                retained.append(gap['game_id'])
            elif (old.get('origin') == 'derived_complete_conference_partitions'
                  and old.get('completeness_status') == 'empty_partition_union'
                  and old.get('attempt_count', 0) == 0):
                selected.append((item, old, 'First direct game query after empty conference union'))
            elif (old['status'] == 'failed_retryable' and
                  (old.get('http_status') == 429 or 'Archive paused' in (old.get('error_summary') or ''))):
                selected.append((item, old, 'One reviewed retry after cooldown under renewed user recovery authorization'))
            else:
                retained.append(gap['game_id'])
        print(json.dumps(dict(selected=len(selected), retained_without_http=len(retained))), flush=True)
        if not args.execute:
            return
        dest.mkdir(exist_ok=True)
        review_path = dest / 'review.json'
        if review_path.exists():
            raise RuntimeError('Recovery pass already started; inspect its records before another pass')
        review = dict(
            authorized_by='User: Alright, can you do what you can to complete the archive?',
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            receipt_sha256=sha256_file(directory / 'completion_receipt.json'),
            selected=[dict(item=i, prior_record=o, reason=why) for i, o, why in selected],
            retained_game_ids=retained,
            retry_policy='One manually reviewed attempt per selected query, including the exhausted throttle query; preserve cumulative history. No automatic retries.',
        )
        atomic_json(review_path, review)
        for name in ('attribution_audit.json', 'attribution_coverage.parquet', 'completion_receipt.json'):
            shutil.copy2(directory / name, dest / ('before_' + name))
        budget = CallBudget(root / 'results/cfbd_api_call_budget.json', 20000)
        start = budget.status()['reserved']
        os.environ['CFBD_API_KEY'] = load_cfbd_key_file(ROOT / '.env')
        client = SingleAttemptClient(call_budget=budget)
        quota = client.get_json('/info', {})
        quota = quota[0] if isinstance(quota, list) else quota
        if not quota_allows(quota['remainingCalls'], budget.status()['reserved'], reserve=10000, hard_limit=20000):
            raise RuntimeError('Insufficient provider reserve')
        if start + len(selected) + 2 > 20000:
            raise RuntimeError('Recovery would exceed shared budget')
        pace = PacedBudget(budget, min(20000, start + len(selected) + 2), interval=1.1)
        client.call_budget = pace
        # CFBDClient uses this public attribute to reserve each actual request.
        client.s.hooks['response'] = [pace.observe_response]
        results = []
        for item, old, reason in selected:
            # Bind this pass to the reviewed state before any mutation.
            if ledger.read(item['request_id']) != old:
                raise RuntimeError('Request state changed after review')
            rec = execute_request(item, ledger, client)
            results.append(dict(game_id=item['game_id'], request_id=item['request_id'],
                                status=rec['status'], rows=rec.get('row_count'),
                                attempts_this_execution=rec.get('attempts_this_execution'),
                                cumulative_attempts=rec['attempt_count'], reason=reason))
            atomic_json(dest / 'results.json', dict(results=results, budget=budget.status()))
            print(json.dumps(results[-1]), flush=True)
            if rec['status'] == 'failed_retryable' or rec.get('http_status') in (400, 401, 403, 404, 429):
                break
        client.s.close()
        atomic_json(dest / 'summary.json', dict(
            finished_at_utc=datetime.now(timezone.utc).isoformat(), selected=len(selected),
            completed=len(results), statuses=dict(Counter(r['status'] for r in results)),
            budget=budget.status(), quota_at_start=quota,
            attempts_this_pass=budget.status()['reserved']-start,
        ))


if __name__ == '__main__':
    main()
