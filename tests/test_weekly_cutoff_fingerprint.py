import pandas as pd

from scripts.publication.build_weekly_cutoff_fingerprint import build


def test_cutoff_state_uses_only_completed_prior_week(tmp_path):
    source = tmp_path / "source.parquet"
    games_path = tmp_path / "games.parquet"
    output = tmp_path / "cutoff.parquet"
    proof = tmp_path / "proof.json"
    pd.DataFrame({
        "keys_season": [2026, 2026, 2026, 2026],
        "keys_week": [7, 7, 8, 8],
        "keys_team": ["A", "B", "A", "B"],
        "keys_game_id": [1, 1, 2, 2],
        "games_played": [7, 7, 8, 8],
        "offense_ppa": [0.1, 0.2, 999.0, 999.0],
    }).to_parquet(source, index=False)
    pd.DataFrame({
        "id": [1, 2, 3], "season": [2026] * 3, "week": [7, 8, 8],
        "season_type": ["regular"] * 3,
        "home_classification": ["fbs"] * 3,
        "away_classification": ["fbs"] * 3,
        "start_date": ["2026-10-17T20:00:00Z", "2026-10-20T23:00:00Z", "2026-10-22T00:00:00Z"],
        "completed": [True, True, False],
        "home_points": [20, 24, None], "away_points": [10, 7, None],
    }).to_parquet(games_path, index=False)
    report = build(source, games_path, season=2026, week=8,
                   cutoff_utc="2026-10-21T20:00:00Z", output=output, proof=proof)
    frozen = pd.read_parquet(output)
    assert frozen.loc[frozen.keys_week.eq(8), "offense_ppa"].tolist() == [0.1, 0.2]
    assert report["future_target_games"] == 1
    assert report["already_started_target_games"] == 1
