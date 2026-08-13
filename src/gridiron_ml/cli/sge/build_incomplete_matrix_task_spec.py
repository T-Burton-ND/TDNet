#!/usr/bin/env python3
"""Print a compact SGE task spec for matrix cells that are not successful."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    args = parser.parse_args()
    incomplete = []
    for task_id in range(int(args.task_count)):
        status_path = args.run_root / f"task_{task_id:07d}" / "status.json"
        status = None
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text()).get("status")
            except (OSError, json.JSONDecodeError):
                status = None
        if status != "success":
            incomplete.append(task_id + 1)
    ranges = []
    if incomplete:
        start = previous = incomplete[0]
        for value in incomplete[1:]:
            if value == previous + 1:
                previous = value
                continue
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = value
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
    print(",".join(ranges))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
