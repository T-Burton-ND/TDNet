"""Repository path helpers shared by command-line entry points."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root without depending on a script's file depth."""

    configured = os.environ.get("TDNET_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate.resolve()
    raise RuntimeError("Could not locate the TDNet project root")
