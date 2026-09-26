"""The next-generation CFBD request counter must fail closed at its cap."""

import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cfbd_fetch_v2_budget_test", ROOT / "src/gridiron_ml/pipeline/fetch/cfbd_fetch_v2.py"
)
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


class FakeResponse:
    status_code = 200
    headers = {"Content-Type": "application/json"}

    def raise_for_status(self):
        return None

    def json(self):
        return []


class FakeSession:
    def __init__(self):
        self.calls = 0
        self.timeout = 60

    def get(self, *_args, **_kwargs):
        self.calls += 1
        return FakeResponse()


class StatusSession(FakeSession):
    def __init__(self, status):
        super().__init__()
        self.status = status

    def get(self, *_args, **_kwargs):
        self.calls += 1
        response = requests.Response()
        response.status_code = self.status
        response.url = "https://api.collegefootballdata.com/test"
        response.headers["Content-Type"] = "application/json"
        return response


class BudgetTests(unittest.TestCase):
    def test_current_nextgen_policy_has_no_higher_limit_path(self):
        sources = [
            ROOT / "src/gridiron_ml/pipeline/fetch/cfbd_fetch_v2.py",
            ROOT / "src/gridiron_ml/pipeline/fetch/nextgen_acquisition.py",
            ROOT / "src/gridiron_ml/experiments/nextgen_contract.py",
            ROOT / "scripts/nextgen_preflight.py",
            ROOT / "scripts/nextgen_cfbd_acquire.py",
            ROOT / "configs/experiments/nextgen_fingerprints_v1.json",
            ROOT / "configs/experiments/nextgen_acquisition_v1.json",
        ]
        for source in sources:
            self.assertNotIn("24000", source.read_text(), source.name)
            self.assertNotIn("24,000", source.read_text(), source.name)
            self.assertNotIn("migrate_20k_to_24k", source.read_text(), source.name)
        config = json.loads(sources[-2].read_text())["cfbd_api_call_budget"]
        self.assertEqual((config["hard_limit"], config["minimum_reserve"]), (20000, 10000))

    def test_parallel_reservations_stay_below_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = fetcher.CallBudget(Path(directory) / "budget.json", 5)

            def reserve(_):
                try:
                    budget.reserve("/plays")
                    return True
                except fetcher.CallBudgetExceeded:
                    return False

            with ThreadPoolExecutor(max_workers=12) as pool:
                results = list(pool.map(reserve, range(12)))
            self.assertEqual(sum(results), 5)
            self.assertEqual(budget.status()["reserved"], 5)

    def test_reservation_caps_actual_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "budget.json"
            budget = fetcher.CallBudget(ledger, 2)
            previous = os.environ.get("CFBD_API_KEY")
            os.environ["CFBD_API_KEY"] = "test-only-placeholder"
            try:
                client = fetcher.CFBDClient(call_budget=budget, max_retries=0)
            finally:
                if previous is None:
                    os.environ.pop("CFBD_API_KEY", None)
                else:
                    os.environ["CFBD_API_KEY"] = previous
            client.s = FakeSession()
            client.get_json("/teams/fbs", {"year": 2025})
            client.get_json("/teams/fbs", {"year": 2025})
            with self.assertRaises(fetcher.CallBudgetExceeded):
                client.get_json("/teams/fbs", {"year": 2025})
            self.assertEqual(client.s.calls, 2)
            self.assertEqual(budget.status()["reserved"], 2)
            self.assertEqual(fetcher.CallBudget(ledger, 2).status()["reserved"], 2)
            with self.assertRaises(ValueError):
                fetcher.CallBudget(ledger, 20001)
            ledger.unlink()
            with self.assertRaises(RuntimeError):
                budget.status()

    def test_twenty_thousand_is_hard_outbound_attempt_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "budget.json"
            budget = fetcher.CallBudget(ledger, 20000)
            ledger.write_text(json.dumps({"limit": 20000, "reserved": 19999,
                                          "experiment": "nextgen_fingerprints_v1"}))
            with patch.dict(os.environ, {"CFBD_API_KEY": "test-only-placeholder"}):
                client = fetcher.CFBDClient(call_budget=budget, max_retries=0)
            client.s = FakeSession()
            client.get_json("/info", {}, max_retries=0)
            with self.assertRaises(fetcher.CallBudgetExceeded):
                client.get_json("/plays", {"year": 2025, "week": 1}, max_retries=0)
            self.assertEqual(client.s.calls, 1)
            self.assertEqual(budget.status()["reserved"], 20000)

    def test_http_400_is_not_silent_and_retry_ceiling_is_three_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"CFBD_API_KEY": "test-only-placeholder"}):
                budget = fetcher.CallBudget(Path(directory) / "budget.json", 10)
                client = fetcher.CFBDClient(call_budget=budget)
                client.s = StatusSession(400)
                with self.assertRaises(requests.HTTPError):
                    client.get_json("/plays", {"year": 2025, "week": 1})
                self.assertEqual(client.s.calls, 1)
                client.s = StatusSession(503)
                with patch.object(fetcher.time, "sleep", return_value=None):
                    with self.assertRaises(requests.HTTPError):
                        client.get_json("/plays", {"year": 2025, "week": 1})
                self.assertEqual(client.s.calls, 3)
                self.assertEqual(budget.status()["reserved"], 4)


if __name__ == "__main__":
    unittest.main()
