"""Avoid duplicate API spend and enforce the shared counter under concurrency."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gridiron_ml.pipeline.fetch.cfbd_fetch_v2 import CallBudget

spec = importlib.util.spec_from_file_location(
    'archive_executor', Path(__file__).resolve().parents[1]/'scripts/nextgen_archive_complete.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class Ledger:
    def __init__(self, rows):
        self.rows = rows

    def read(self, rid):
        return self.rows.get(rid)


def item(rid='a', params='{"gameId":1}'):
    return dict(request_id=rid, endpoint='/plays/stats', parameters_json=params)


def test_resume_does_not_repeat_empty_or_exhausted_requests():
    for status in ('needs_review', 'failed_final', 'failed_retryable', 'success_suspected_partial'):
        old = dict(status=status)
        pending, retained = archive.pending_items([item()], Ledger({'a': old}))
        assert pending == [] and retained == [old]


def test_success_is_verified_and_corruption_requires_review(monkeypatch):
    old = dict(status='success_complete', cache_path='/unused')
    monkeypatch.setattr(archive, 'verify_cache', lambda *args: {'valid': True})
    assert archive.pending_items([item()], Ledger({'a': old}))[0] == []
    monkeypatch.setattr(archive, 'verify_cache', lambda *args: None)
    with pytest.raises(ValueError, match='cache changed'):
        archive.pending_items([item()], Ledger({'a': old}))


def test_same_http_query_cannot_be_scheduled_under_two_ids():
    with pytest.raises(ValueError, match='Duplicate'):
        archive.pending_items([item(), item('b')], Ledger({}))


def test_concurrent_attempts_preserve_existing_shared_budget(tmp_path):
    shared = CallBudget(tmp_path/'budget.json', 20000)
    for _ in range(3):
        shared.reserve('/old')
    paced = archive.PacedBudget(shared, ceiling=8, interval=0)
    def attempt(_):
        try:
            paced.reserve('/new')
            return True
        except RuntimeError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(attempt, range(30))) == 5
    assert shared.status()['reserved'] == 8


def test_stop_blocks_further_spending(tmp_path):
    shared = CallBudget(tmp_path/'budget.json', 20000)
    paced = archive.PacedBudget(shared, 20000, interval=0)
    paced.stop.set()
    with pytest.raises(RuntimeError):
        paced.reserve('/new')
    assert shared.status()['reserved'] == 0


def test_rate_limit_cooldown_increases_pace_without_spending(tmp_path):
    from types import SimpleNamespace
    import time
    shared = CallBudget(tmp_path/'budget.json',20000)
    paced = archive.PacedBudget(shared,20000,interval=5)
    before = time.monotonic()
    paced.observe_response(SimpleNamespace(status_code=429,headers={'Retry-After':'90'}))
    assert paced.interval==10
    assert paced.next_at>=before+90
    assert shared.status()['reserved']==0
