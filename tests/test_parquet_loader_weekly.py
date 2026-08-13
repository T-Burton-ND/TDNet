from pathlib import Path

import pandas as pd

from gridiron_ml.pipeline.pre_processing.parquet_loader import (
    load_and_flatten_parquet,
    load_game_team_stats_parquet,
)


def test_load_and_flatten_parquet_filters_to_requested_week(tmp_path: Path):
    parquet_path = tmp_path / "generic.parquet"
    pd.DataFrame(
        [
            {"season": 2025, "week": 1, "team": "Alpha", "meta": {"rank": 10}},
            {"season": 2025, "week": 2, "team": "Beta", "meta": {"rank": 20}},
        ]
    ).to_parquet(parquet_path, index=False)

    out = load_and_flatten_parquet(str(parquet_path), week=2)

    assert out["week"].tolist() == [2]
    assert out["team"].tolist() == ["Beta"]
    assert out["meta_rank"].tolist() == [20]


def test_load_game_team_stats_parquet_filters_to_requested_week(tmp_path: Path):
    parquet_path = tmp_path / "game_team_stats.parquet"
    pd.DataFrame(
        [
            {
                "id": 1001,
                "season": 2025,
                "week": 1,
                "teams": [
                    {"team": "Alpha", "teamId": 1, "conference": "X", "homeAway": "home", "points": 24, "stats": []},
                    {"team": "Beta", "teamId": 2, "conference": "X", "homeAway": "away", "points": 17, "stats": []},
                ],
            },
            {
                "id": 1002,
                "season": 2025,
                "week": 2,
                "teams": [
                    {"team": "Gamma", "teamId": 3, "conference": "Y", "homeAway": "home", "points": 31, "stats": []},
                    {"team": "Delta", "teamId": 4, "conference": "Y", "homeAway": "away", "points": 14, "stats": []},
                ],
            },
        ]
    ).to_parquet(parquet_path, index=False)

    out = load_game_team_stats_parquet(str(parquet_path), week=2)

    assert sorted(out["game_id"].unique().tolist()) == [1002]
    assert sorted(out["team"].tolist()) == ["Delta", "Gamma"]
    assert sorted(out["week"].unique().tolist()) == [2]
