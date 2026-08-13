import pandas as pd
import pytest

from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.fingerprints.builders.v0 import V0FingerprintBuilder
from gridiron_ml.fingerprints.features import DEFAULT_FEATURE_SPEC, FeatureSpec, split_frame


def _fake_team_game_table():
    rows = []

    def add_game(game_id, week, home, away, home_margin, home_ppa, away_ppa):
        rows.append(
            {
                "keys_season": 2024,
                "keys_week": week,
                "keys_game_id": game_id,
                "keys_team": home,
                "keys_opponent": away,
                "keys_conference": "Test",
                "game_is_home": True,
                "game_home_away": "home",
                "target_points_for": 28 + home_margin,
                "target_points_against": 28,
                "target_team_margin": home_margin,
                "offense_ppa": home_ppa,
            }
        )
        rows.append(
            {
                "keys_season": 2024,
                "keys_week": week,
                "keys_game_id": game_id,
                "keys_team": away,
                "keys_opponent": home,
                "keys_conference": "Test",
                "game_is_home": False,
                "game_home_away": "away",
                "target_points_for": 28,
                "target_points_against": 28 + home_margin,
                "target_team_margin": -home_margin,
                "offense_ppa": away_ppa,
            }
        )

    add_game(1, 1, "Alpha", "Beta", 7, 0.50, -0.50)
    add_game(2, 2, "Beta", "Alpha", 3, 0.20, -0.20)
    add_game(3, 3, "Alpha", "Beta", 10, 0.70, -0.70)
    add_game(4, 4, "Beta", "Alpha", -4, -0.10, 0.10)
    return pd.DataFrame(rows)


def _build_v0_frame(tmp_path):
    builder = V0FingerprintBuilder(version=0, root=tmp_path, team_game_tables_dir=tmp_path)
    _, _, canonical = builder._build_season_from_team_game_table(
        _fake_team_game_table(),
        build_timestamp="2026-06-09T00:00:00+00:00",
    )
    return canonical


def test_v0_rows_are_state_after_completed_week(tmp_path):
    frame = _build_v0_frame(tmp_path)
    alpha_week0 = frame.loc[(frame["keys_team"] == "Alpha") & (frame["keys_week"] == 0)].iloc[0]
    alpha_week1 = frame.loc[(frame["keys_team"] == "Alpha") & (frame["keys_week"] == 1)].iloc[0]

    assert alpha_week0["games_played"] == 0
    assert pd.isna(alpha_week0["offense_ppa"])
    assert pd.isna(alpha_week0["keys_game_id"])
    assert alpha_week0["y_next_margin"] == 7

    assert alpha_week1["games_played"] == 1
    assert alpha_week1["offense_ppa"] == pytest.approx(0.50)
    assert alpha_week1["y_margin_this_week"] == 7
    assert alpha_week1["y_next_margin"] == -3
    assert alpha_week1["next_game_id"] == 2


def test_default_target_is_next_margin_and_same_week_target_is_blocked(tmp_path):
    frame = _build_v0_frame(tmp_path)

    assert DEFAULT_FEATURE_SPEC.target_column == "y_next_margin"
    with pytest.raises(ValueError, match="same-row completed-game target"):
        split_frame(frame, FeatureSpec(target_column="y_margin_this_week"))


def test_fingerprints_training_block_rejects_same_week_target(tmp_path):
    root = tmp_path
    fp_dir = root / "data" / "fingerprints" / "v0"
    fp_dir.mkdir(parents=True)
    _build_v0_frame(tmp_path).to_parquet(fp_dir / "canonical_fingerprint.parquet", index=False)

    fp = Fingerprints(version=0, root=root)
    with pytest.raises(ValueError, match="y_next_margin"):
        fp.training_block([2024], feature_spec=FeatureSpec(target_column="y_margin_this_week"))
