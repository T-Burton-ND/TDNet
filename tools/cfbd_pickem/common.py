from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


API_BASE_URL = "https://predictionsapi.collegefootballdata.com"
API_KEY_ENV = "CFBD_PICKEM_API_KEY"


class ValidationError(RuntimeError):
    """Raised when an export or submission safety condition is not satisfied."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_api_key() -> str:
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise ValidationError(
            f"{API_KEY_ENV} is unset. Load the key from a local ignored secret; "
            "do not pass it as a command-line argument."
        )
    return api_key


def api_request(
    method: str,
    path: str,
    api_key: str,
    *,
    body: bytes | None = None,
) -> tuple[int, Any]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "TDNet-CFBD-Pickem/1.0",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}{path}",
            data=body,
            headers=headers,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise ValidationError(f"CFBD {method} {path} failed: {exc}") from exc
    if not response.ok:
        raise ValidationError(
            f"CFBD {method} {path} failed with HTTP {response.status_code}: "
            f"{response.text}"
        )
    if not response.content:
        return response.status_code, None
    try:
        return response.status_code, response.json()
    except requests.JSONDecodeError as exc:
        raise ValidationError(
            f"CFBD {method} {path} returned invalid JSON."
        ) from exc
