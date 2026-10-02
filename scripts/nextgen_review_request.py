#!/usr/bin/env python3
"""Record an evidence-backed disposition without claiming an anomaly is data.

This performs no network calls. A failed/empty/capped response is never promoted
to complete. The original observation is retained with the explicit review.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gridiron_ml.pipeline.fetch.nextgen_acquisition import AcquisitionLedger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--disposition", choices=["failed_final", "structurally_unavailable"], required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/experiments/nextgen_fingerprints_v1.json").read_text())
    ledger = AcquisitionLedger(Path(config["artifact_root"]))
    record = ledger.read(args.request_id)
    if record is None or record["status"] not in {"needs_review", "success_suspected_partial", "failed_retryable"}:
        raise ValueError("Only observed unresolved requests may receive a disposition")
    if not args.reason.strip() or not args.evidence.strip():
        raise ValueError("Both a substantive reason and evidence are required")
    record.setdefault("reviews", []).append({
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "prior_status": record["status"], "prior_error": record.get("error_summary"),
        "disposition": args.disposition, "reason": args.reason, "evidence": args.evidence,
    })
    record["status"] = args.disposition
    record["completeness_status"] = "reviewed_unavailable_not_complete"
    ledger.write(record)
    print(json.dumps({"request_id": args.request_id, "status": record["status"], "review": record["reviews"][-1]}, indent=2))


if __name__ == "__main__":
    main()
