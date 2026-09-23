import pandas as pd

from gridiron_ml.cli.publication.refresh_postgame_rank_inputs import _merge_week_cache
from gridiron_ml.pipeline.pre_processing.cleaners import main_clean
from gridiron_ml.pipeline.raw_weekly_builder import _build_team_week_from_games


def test_modern_games_points_become_team_centric_targets():
    games = pd.DataFrame(
        [
            {
                "id": 1,
                "season": 2026,
                "week": 1,
                "home_team": "Home",
                "away_team": "Away",
                "home_points": 31,
                "away_points": 17,
                "completed": True,
                "season_type": "regular",
            }
        ]
    )
    cleaned = main_clean(_build_team_week_from_games(games)).set_index("team")
    assert cleaned.at["Home", "points_for"] == 31
    assert cleaned.at["Home", "points_against"] == 17
    assert cleaned.at["Home", "team_margin"] == 14
    assert cleaned.at["Away", "team_margin"] == -14


def test_postgame_refresh_replaces_only_the_requested_week():
    existing = pd.DataFrame(
        {"week": [1, 2], "game_id": [10, 20], "value": ["old-1", "old-2"]}
    )
    current = pd.DataFrame(
        {"week": [2], "game_id": [20], "value": ["new-2"]}
    )
    merged = _merge_week_cache(existing, current, week=2).sort_values("week")
    assert merged["value"].tolist() == ["old-1", "new-2"]
