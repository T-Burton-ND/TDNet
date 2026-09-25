"""The next-generation CFBD request counter must fail closed at its cap."""

import importlib.util
import os
import tempfile
import unittest
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


class BudgetTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
