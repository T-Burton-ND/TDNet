from pathlib import Path

import pandas as pd

from gridiron_ml.fingerprints.builders.v0 import V0FingerprintBuilder


def test_team_game_table_paths_prefer_parquet_and_ignore_weekly_files(tmp_path: Path):
    table_dir = tmp_path / "data" / "team_game_tables"
    table_dir.mkdir(parents=True)

    pd.DataFrame({"keys_season": [2024], "keys_week": [1], "keys_team": ["Alpha"]}).to_csv(
        table_dir / "team_game_table_2024_fbs.csv",
        index=False,
    )
    pd.DataFrame({"keys_season": [2024], "keys_week": [1], "keys_team": ["Alpha"]}).to_parquet(
        table_dir / "team_game_table_2024_fbs.parquet",
        index=False,
    )
    pd.DataFrame({"keys_season": [2024], "keys_week": [5], "keys_team": ["Alpha"]}).to_parquet(
        table_dir / "team_game_table_2024_week_05_fbs.parquet",
        index=False,
    )
    pd.DataFrame({"keys_season": [2025], "keys_week": [1], "keys_team": ["Beta"]}).to_csv(
        table_dir / "team_game_table_2025_fbs.csv",
        index=False,
    )

    builder = V0FingerprintBuilder(version=0, root=tmp_path)

    paths = builder._team_game_table_paths()

    assert [path.name for path in paths] == [
        "team_game_table_2024_fbs.parquet",
        "team_game_table_2025_fbs.csv",
    ]


def test_v0_builder_handles_parquet_style_categorical_team_keys(tmp_path: Path):
    builder = V0FingerprintBuilder(version=0, root=tmp_path)
    frame = pd.DataFrame(
        {
            "keys_season": [2025, 2025, 2025, 2025],
            "keys_week": [1, 2, 1, 2],
            "keys_team": pd.Categorical(["A", "A", "B", "B"]),
            "keys_game_id": [1, 2, 1, 2],
            "keys_opponent": ["B", "B", "A", "A"],
            "keys_season_type": ["regular", "regular", "regular", "regular"],
            "keys_game_date": [
                pd.Timestamp("2025-09-01"),
                "2025-09-08T00:00:00Z",
                pd.Timestamp("2025-09-01"),
                "2025-09-08",
            ],
            "target_points_for": [10, 20, 15, 25],
            "target_points_against": [7, 14, 17, 21],
            "target_team_margin": [3, 6, -2, 4],
        }
    )

    fingerprint_df, label_df, canonical_df = builder._build_season_from_team_game_table(
        frame,
        build_timestamp="debug",
    )
    normalized = builder._normalize_parquet_dtypes(canonical_df)

    assert fingerprint_df.shape[0] == 6
    assert label_df.shape[0] == 6
    assert canonical_df.shape[0] == 6
    assert str(normalized["keys_game_date"].dtype).startswith("datetime64")


def test_normalize_parquet_dtypes_accepts_mixed_timezone_dates(tmp_path: Path):
    builder = V0FingerprintBuilder(version=0, root=tmp_path)
    frame = pd.DataFrame(
        {
            "keys_game_date": [
                "2025-08-30",
                "2025-08-31T01:30:00Z",
                pd.Timestamp("2025-09-01"),
                pd.Timestamp("2025-09-02", tz="UTC"),
            ]
        }
    )

    normalized = builder._normalize_parquet_dtypes(frame)

    assert str(normalized["keys_game_date"].dtype).startswith("datetime64")
    assert normalized["keys_game_date"].notna().all()
