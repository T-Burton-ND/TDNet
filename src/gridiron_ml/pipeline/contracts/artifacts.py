"""Artifact naming and path conventions for TDNet data products."""

from __future__ import annotations

from pathlib import Path

CANONICAL_FINGERPRINT_FILENAME = "canonical_fingerprint.parquet"
LEGACY_FINGERPRINTS_FILENAME_TEMPLATE = "v{version}_gridiron_ml_fingerprints.parquet"
LEGACY_LABELS_FILENAME_TEMPLATE = "v{version}_gridiron_ml_labels.parquet"
TEAM_WEEK_FINGERPRINTS_FILENAME_TEMPLATE = "team_week_fingerprints_{season}.parquet"
TEAM_WEEK_LABELS_FILENAME_TEMPLATE = "team_week_labels_{season}.parquet"
METADATA_FILENAME = "metadata.json"
TEAM_GAME_TABLE_FILENAME_TEMPLATE = "team_game_table_{season}_{division}.{suffix}"
TEAM_GAME_TABLE_REGEX = r"team_game_table_(\d{4})_fbs.(csv|parquet)$"

FINGERPRINT_REBUILD_CLEANUP_PATTERNS = (
    CANONICAL_FINGERPRINT_FILENAME,
    LEGACY_FINGERPRINTS_FILENAME_TEMPLATE,
    LEGACY_LABELS_FILENAME_TEMPLATE,
    METADATA_FILENAME,
    "*/team_week_fingerprints_*.parquet",
    "*/team_week_labels_*.parquet",
    f"*/{METADATA_FILENAME}",
)


def fingerprint_version_dir(root: str | Path, version: int) -> Path:
    return Path(root) / "data" / "fingerprints" / f"v{int(version)}"


def canonical_fingerprint_path(fp_dir: str | Path) -> Path:
    return Path(fp_dir) / CANONICAL_FINGERPRINT_FILENAME


def legacy_fingerprints_filename(version: int) -> str:
    return LEGACY_FINGERPRINTS_FILENAME_TEMPLATE.format(version=int(version))


def legacy_labels_filename(version: int) -> str:
    return LEGACY_LABELS_FILENAME_TEMPLATE.format(version=int(version))


def legacy_fingerprints_path(fp_dir: str | Path, version: int) -> Path:
    return Path(fp_dir) / legacy_fingerprints_filename(version)


def legacy_labels_path(fp_dir: str | Path, version: int) -> Path:
    return Path(fp_dir) / legacy_labels_filename(version)


def season_fingerprint_dir(fp_dir: str | Path, season: int) -> Path:
    return Path(fp_dir) / str(int(season))


def team_week_fingerprints_filename(season: int) -> str:
    return TEAM_WEEK_FINGERPRINTS_FILENAME_TEMPLATE.format(season=int(season))


def team_week_labels_filename(season: int) -> str:
    return TEAM_WEEK_LABELS_FILENAME_TEMPLATE.format(season=int(season))


def team_week_fingerprints_path(fp_dir: str | Path, season: int) -> Path:
    return season_fingerprint_dir(fp_dir, season) / team_week_fingerprints_filename(season)


def team_week_labels_path(fp_dir: str | Path, season: int) -> Path:
    return season_fingerprint_dir(fp_dir, season) / team_week_labels_filename(season)


def metadata_path(directory: str | Path) -> Path:
    return Path(directory) / METADATA_FILENAME


def team_game_table_filename(season: int, *, division: str = "fbs", suffix: str = "parquet") -> str:
    return TEAM_GAME_TABLE_FILENAME_TEMPLATE.format(
        season=int(season),
        division=str(division),
        suffix=str(suffix).lstrip("."),
    )


def cleanup_fingerprint_artifacts(fp_dir: str | Path, version: int) -> list[Path]:
    """Delete known generated fingerprint files while leaving debug files intact."""
    directory = Path(fp_dir)
    if not directory.exists():
        return []

    removed: list[Path] = []
    for pattern in FINGERPRINT_REBUILD_CLEANUP_PATTERNS:
        resolved = pattern.format(version=int(version))
        for path in directory.glob(resolved):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed
