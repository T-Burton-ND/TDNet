import pandas as pd

from gridiron_ml.pipeline.pre_processing.cleaners import clean_coaches
from gridiron_ml.pipeline.pre_processing.parquet_loader import load_coaches_parquet


def _coach_row(year, wins, postseason_rank):
    return {
        "first_name": "Jane",
        "last_name": "Doe",
        "hire_date": "2020-01-01",
        "seasons": [
            {
                "year": year,
                "school": "State",
                "games": 12,
                "wins": wins,
                "losses": 12 - wins,
                "ties": 0,
                "spOffense": 20.0 + wins,
                "spDefense": 10.0,
                "spOverall": 15.0,
                "srs": 12.0,
                "preseasonRank": 10,
                "postseasonRank": postseason_rank,
            }
        ],
    }


def test_coach_loader_defaults_to_prior_season_only(tmp_path):
    coaches_dir = tmp_path / "coaches"
    coaches_dir.mkdir()
    pd.DataFrame([_coach_row(2023, wins=8, postseason_rank=12)]).to_parquet(
        coaches_dir / "2023.parquet",
        index=False,
    )
    pd.DataFrame([_coach_row(2024, wins=12, postseason_rank=1)]).to_parquet(
        coaches_dir / "2024.parquet",
        index=False,
    )

    out = load_coaches_parquet(str(coaches_dir / "2024.parquet"), target_season=2024)

    assert out.loc[0, "coach_career_total_wins"] == 8
    assert "coach_career_mean_postseason_rank" not in out.columns
    assert "coach_career_mean_postseason_rank_points" not in out.columns
    assert not any(col.startswith("coach_season_") for col in out.columns)


def test_clean_coaches_drops_leaky_rank_and_current_season_columns():
    cleaned = clean_coaches(
        pd.DataFrame(
            {
                "team": ["State"],
                "coach_season_games": [12],
                "coach_career_mean_postseason_rank_points": [25.0],
            }
        )
    )

    assert "team" in cleaned.columns
    assert "coach_season_games" not in cleaned.columns
    assert "coach_career_mean_postseason_rank_points" not in cleaned.columns
