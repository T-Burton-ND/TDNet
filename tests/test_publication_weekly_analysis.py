from pathlib import Path

import pandas as pd

from gridiron_ml.publication.output_layout import copy_top25_outputs, ensure_week_directories
from gridiron_ml.publication.weekly_analysis import build_matchup_signals, plot_matchup_signals


def test_week_layout_and_top25_outputs_are_split_by_type(tmp_path: Path):
    layout = ensure_week_directories(tmp_path / "week_01")
    assert set(layout) == {"pre_game", "post_game", "analysis"}
    source = tmp_path / "poll"
    source.mkdir()
    (source / "tdnet_top25.png").write_bytes(b"png")
    (source / "tdnet_top25.svg").write_text("svg")
    (source / "tdnet_top25.csv").write_text("rank,team\n1,A\n")
    copied = copy_top25_outputs(source, layout["pre_game"])
    assert layout["pre_game"] / "figures/tdnet_top25.png" in copied
    assert layout["pre_game"] / "tables/tdnet_top25.csv" in copied
    assert not (layout["pre_game"] / "figures/tdnet_top25.svg").exists()
    assert not (layout["pre_game"] / "top25").exists()


def test_matchup_analysis_uses_three_signals_and_png_only(tmp_path: Path):
    games = pd.DataFrame([
        {
            "game_id": 1, "away_team": "Away", "home_team": "Home",
            "pred_home_margin": 2.0, "predicted_margin": 2.0,
            "pred_home_win_probability": .56, "pred_winner": "Home",
            "tdnet_rank_home": 3, "tdnet_rank_away": pd.NA,
        },
        {
            "game_id": 2, "away_team": "Sick Away", "home_team": "Sick Home",
            "pred_home_margin": -.5, "predicted_margin": .5,
            "pred_home_win_probability": .49, "pred_winner": "Sick Away",
        },
    ])
    poll = pd.DataFrame({"rank": [3], "team": ["Home"]})
    features = pd.DataFrame([
        {"keys_team": team, "keys_season": 2026, "keys_week": 1,
         "roster_talent": talent, "offense_ppa": offense,
         "defense_ppa": defense, "statGen_turnovers": turnovers}
        for team, talent, offense, defense, turnovers in [
            ("Away", 1, .1, .3, 2), ("Home", 4, .4, .1, 0),
            ("Sick Away", 2, .2, .2, 1), ("Sick Home", 3, .3, .4, 3),
        ]
    ])
    signals = build_matchup_signals(games, poll, features, season=2026, week=0)
    assert signals.groupby("game_id").size().eq(3).all()
    output = plot_matchup_signals(signals, tmp_path / "matchups.png", season=2026, week=0)
    assert output.exists()
    assert not output.with_suffix(".svg").exists()
