"""Canonical in-season publication directory helpers."""

from __future__ import annotations

from pathlib import Path
import shutil


WEEK_DIRECTORIES = ("pre_game", "post_game", "analysis")


def ensure_week_directories(week_root: str | Path) -> dict[str, Path]:
    root = Path(week_root)
    root.mkdir(parents=True, exist_ok=True)
    output = {}
    for name in WEEK_DIRECTORIES:
        directory = root / name
        directory.mkdir(exist_ok=True)
        output[name] = directory
    return output


def require_week_directory(path: str | Path, expected: str) -> Path:
    target = Path(path)
    if target.name != expected:
        raise ValueError(
            f"Canonical weekly {expected} output must end in '/{expected}': {target}"
        )
    ensure_week_directories(target.parent)
    return target


def copy_top25_outputs(source: str | Path, package_root: str | Path) -> list[Path]:
    """Place Top-25 visuals/tables into the package's shared directories."""
    source_root = Path(source)
    package = Path(package_root)
    figures = package / "figures"
    tables = package / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source_root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".svg":
            continue
        destination_root = figures if path.suffix.lower() == ".png" else tables
        destination = destination_root / path.name
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied
