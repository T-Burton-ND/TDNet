#!/usr/bin/env python3
"""Wait for the authorized archive acquisition, audit it, and save its receipt.

This does not run models, rebuild canonical data, or reset the API budget.
It makes only a counted final quota query; acquisition recovery needs review.
"""
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import CFBDClient, CallBudget, load_cfbd_key_file
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (
    AcquisitionLedger, atomic_json, sha256_file, verify_cache,
)
import nextgen_archive_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-start', required=True)
    args = parser.parse_args()
    root = Path('/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen')
    directory = root / 'results/preflight/archive_completion_20260928'
    receipt_path = directory / 'completion_receipt.json'
    with (directory / 'executor.lock').open('a') as lock:
        print('Waiting for the current acquisition executor to release its lock.', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        status = json.loads((directory / 'batch_status.json').read_text())
        if status['started_utc'] != args.expected_start:
            raise RuntimeError('Executor identity changed; review before finalizing')
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            if previous.get('executor_started_utc') == args.expected_start:
                print('This executor already has a receipt; no repeated quota call.', flush=True)
                return
        budget = CallBudget(root / 'results/cfbd_api_call_budget.json', 20000)
        receipt = dict(
            executor_started_utc=args.expected_start,
            executor_phase=status['phase'],
            dispatch_finished=status['dispatch_finished'],
            checked_at_utc=datetime.now(timezone.utc).isoformat(),
            status='verification_in_progress',
            budget_before_final_quota=budget.status(),
            no_model_experiments=True,
        )
        atomic_json(receipt_path, receipt)
        try:
            plan = json.loads((directory / 'batch_plan.json').read_text())
            if sha256_file(directory / 'batch_plan.json') != status['batch_plan_sha256']:
                raise ValueError('Batch plan changed since executor checkpoint')
            ledger = AcquisitionLedger(root)
            counts, unresolved = Counter(), []
            origins = Counter()
            checked_partitions = set()
            for game_id, item in plan['games'].items():
                record = ledger.read(item['request_id'])
                state = record['status'] if record else 'not_acquired'
                counts[state] += 1
                if state not in ('success_complete', 'skipped_existing_complete'):
                    unresolved.append(dict(
                        game_id=int(game_id), season=item['year'],
                        request_id=item['request_id'], status=state,
                        attempts=record.get('attempt_count', 0) if record else 0,
                        error=record.get('error_summary') if record else None,
                    ))
                    continue
                if not verify_cache(Path(record['cache_path']), record):
                    raise ValueError('Game cache failed final hash verification')
                origins[record.get('origin', 'direct_cached_response')] += 1
                for source in record.get('source_partitions', []):
                    identity = (source['path'], source['sha256'])
                    if identity not in checked_partitions:
                        if sha256_file(Path(source['path'])) != source['sha256']:
                            raise ValueError('Derived game source partition changed')
                        checked_partitions.add(identity)
            with (directory / 'completion_audit.log').open('w') as out:
                with redirect_stdout(out):
                    nextgen_archive_audit.main()
            audit = json.loads((directory / 'attribution_audit.json').read_text())
            receipt.update(
                approved_game_count=len(plan['games']),
                approved_game_statuses=dict(counts),
                successful_cache_origins=dict(origins),
                verified_source_partition_count=len(checked_partitions),
                unresolved_games=unresolved,
                audit_path=str(directory / 'attribution_audit.json'),
                audit_sha256=sha256_file(directory / 'attribution_audit.json'),
                audit_totals=audit['totals'],
                status=('all_approved_game_queries_cached' if not unresolved else
                        'audited_with_unresolved_games'),
                completeness_note='Query completeness is not complete event or participation coverage.',
            )
            if budget.status()['reserved'] < 20000:
                os.environ['CFBD_API_KEY'] = load_cfbd_key_file(ROOT / '.env')
                client = CFBDClient(call_budget=budget)
                try:
                    quota = client.get_json('/info', {}, max_retries=0)
                    receipt['quota_at_finish'] = quota[0] if isinstance(quota, list) else quota
                except Exception as exc:
                    receipt['quota_readback_error'] = type(exc).__name__
                finally:
                    client.s.close()
            else:
                receipt['quota_readback_skipped'] = 'Shared attempt cap reached'
        except Exception as exc:
            receipt.update(status='verification_failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            receipt['budget_at_finish'] = budget.status()
            receipt['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(receipt_path, receipt)
            print(json.dumps({k: v for k, v in receipt.items() if k != 'unresolved_games'}), flush=True)


if __name__ == '__main__':
    main()
