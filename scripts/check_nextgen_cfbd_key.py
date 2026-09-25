#!/usr/bin/env python3
"""Check the repo-local CFBD key with exactly one budgeted API request."""

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/gridiron_ml/pipeline/fetch"))
from cfbd_fetch_v2 import CFBDClient, CallBudget, load_cfbd_key_file  # noqa: E402


def main() -> None:
    config = json.loads((ROOT / "configs/experiments/nextgen_fingerprints_v1.json").read_text())
    settings = config["cfbd_api_call_budget"]
    budget = CallBudget(Path(settings["ledger"]), settings["hard_limit"])
    os.environ["CFBD_API_KEY"] = load_cfbd_key_file(ROOT / ".env")
    client = CFBDClient(call_budget=budget, max_retries=0, strict_http_errors=True)
    rows = client.get_json("/teams/fbs", {"year": 2025}, max_retries=0)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("CFBD key check returned no FBS teams")
    state = budget.status()
    print(f"CFBD key accepted: {len(rows)} FBS teams returned; requests made: {client.api_calls}; "
          f"experiment budget: {state['reserved']}/{state['limit']} reserved")


if __name__ == "__main__":
    main()
