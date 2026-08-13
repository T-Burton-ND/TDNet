from pathlib import Path

import pandas as pd
import pytest
import yaml

from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.pipeline import build_full_pipeline as pipeline_mod


def write_config(tmp_path: Path, config: dict) -> Path:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_run_pipeline_resolves_root_and_default_paths_from_config_location(tmp_path: Path):
    config_path = write_config(
        tmp_path,
        {
            "root": ".",
            "years": {"values": [2024]},
            "raw_fetch": {"enabled": False},
            "team_game_tables": {"enabled": False},
            "fingerprints": {"enabled": False},
        },
    )

    summary = pipeline_mod.run_pipeline(config_path)

    assert summary["root"] == str(tmp_path.resolve())
    assert summary["raw_cache_dir"] == str((tmp_path / "data" / "raw" / "cfbd" / "v2").resolve())
    assert summary["team_game_tables_dir"] == str((tmp_path / "data" / "team_game_tables").resolve())
    assert summary["years"] == [2024]


def test_run_pipeline_rejects_week_specific_table_builds_when_fingerprints_enabled(tmp_path: Path):
    config_path = write_config(
        tmp_path,
        {
            "root": ".",
            "years": {"values": [2025]},
            "raw_fetch": {"enabled": False},
            "team_game_tables": {"enabled": True, "week": 8},
            "fingerprints": {
                "enabled": True,
                "versions": {"v0": {"enabled": True}},
            },
        },
    )

    with pytest.raises(ValueError, match="full-season team-game tables"):
        pipeline_mod.run_pipeline(config_path)


def test_run_fingerprint_stage_calls_enabled_versions_in_sorted_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[int, bool, bool, Path]] = []

    def make_builder(version: int):
        def _builder(*, root: Path, overwrite: bool, postseason: bool, team_game_tables_dir: Path) -> Path:
            calls.append((version, overwrite, postseason, team_game_tables_dir))
            return root / "data" / "fingerprints" / f"v{version}" / "canonical_fingerprint.parquet"

        return _builder

    monkeypatch.setattr(
        pipeline_mod,
        "FINGERPRINT_BUILDERS",
        {
            0: make_builder(0),
            1: make_builder(1),
            2: make_builder(2),
        },
    )

    summaries = pipeline_mod.run_fingerprint_stage(
        root=tmp_path,
        team_game_tables_dir=tmp_path / "tables",
        fingerprints_cfg={
            "enabled": True,
            "postseason": False,
            "overwrite": False,
            "versions": {
                "v2": {"enabled": True, "overwrite": True},
                "v0": {"enabled": True},
                "v1": {"enabled": False},
            },
        },
    )

    assert [call[0] for call in calls] == [0, 2]
    assert calls[0][1] is False
    assert calls[1][1] is True
    assert summaries[0]["version"] == 0
    assert summaries[1]["version"] == 2


def test_fingerprints_accepts_team_game_table_directory_override(tmp_path: Path):
    table_dir = tmp_path / "custom_tables"
    table_dir.mkdir(parents=True)

    pd.DataFrame({"keys_season": [2024], "keys_week": [1], "keys_team": ["Alpha"]}).to_parquet(
        table_dir / "team_game_table_2024_fbs.parquet",
        index=False,
    )

    fp = Fingerprints(version=0, root=tmp_path, team_game_tables_dir=table_dir)

    assert fp.team_game_tables_dir == table_dir
    assert fp._builder()._team_game_table_paths() == [table_dir / "team_game_table_2024_fbs.parquet"]
