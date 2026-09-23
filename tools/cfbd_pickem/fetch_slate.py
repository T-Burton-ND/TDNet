from __future__ import annotations

import argparse
from pathlib import Path

from tools.cfbd_pickem.common import (
    API_BASE_URL,
    ValidationError,
    api_request,
    load_api_key,
    utc_now,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the authenticated CFBD contest slate without changing picks."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    status, games = api_request("GET", "/api/picks", load_api_key())
    if status != 200 or not isinstance(games, list):
        raise ValidationError("GET /api/picks did not return the expected game list.")

    snapshot = {
        "fetched_at_utc": utc_now(),
        "endpoint": f"{API_BASE_URL}/api/picks",
        "http_status": status,
        "games": games,
    }
    write_json(args.output.resolve(), snapshot)
    print(
        f"Fetched {len(games)} contest games with GET /api/picks; "
        f"wrote {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()

