import numpy as np
import pandas as pd
import pytest

from gridiron_ml.experiments.nextgen_f09 import trailing_game_state, f09_formulas


def fixtures():
    schedule = pd.DataFrame({
        "id": [1, 2, 3], "season": [2024, 2025, 2025], "week": [14, 1, 2],
        "start_date": ["2024-11-30T12:00:00Z", "2025-09-01T12:00:00Z", "2025-09-08T12:00:00Z"],
        "home_team": ["A"]*3, "away_team": ["B"]*3,
        "home_classification": ["fbs"]*3, "away_classification": ["fbs"]*3,
        "completed": [True]*3, "season_type": ["regular"]*3,
    })
    stats = pd.DataFrame({"game_id": [1, 1, 2, 2, 3, 3], "team": ["A", "B"]*3,
                          "offense_rush_ypa_q1__sum": [40, 40, 80, 80, 999, 999],
                          "offense_rush_ypa_q1__n": [10]*6})
    return stats, schedule


def test_prior_season_context_without_target_outcomes():
    stats, schedule = fixtures()
    frame, report = trailing_game_state(stats, schedule)
    first = frame.loc[frame.target_game_id.eq(2)]
    assert first.offense_rush_ypa_q1.tolist() == [4, 4]
    next_game = frame.loc[frame.target_game_id.eq(3)]
    assert next_game.offense_rush_ypa_q1.tolist() == [6, 6]
    assert next_game.latest_source_game_id.tolist() == [2, 2]
    assert report["excluded_no_prior_coverage_game_ids"] == [1]
    assert frame.feature_available_utc.lt(frame.target_start_utc).all()


def test_reporting_lag_rejects_unavailable_game_and_checks_all_sources():
    stats, schedule = fixtures()
    schedule.loc[schedule.id.eq(2), "start_date"] = "2025-09-07T12:00:00Z"
    frame, _ = trailing_game_state(stats, schedule)
    assert frame.loc[frame.target_game_id.eq(3), "offense_rush_ypa_q1"].tolist() == [4, 4]
    with pytest.raises(ValueError, match="scheduled participant"):
        trailing_game_state(stats.assign(team="C"), schedule)
    with pytest.raises(ValueError, match="regular-season"):
        trailing_game_state(stats, schedule.assign(season_type="postseason"))


def test_quarter_delta_and_slope_formulas_preserve_missingness():
    formulas = {f.name: f for f in f09_formulas("a")}
    x = pd.DataFrame({f"offense_rush_ypa_q{q}": [float(q), np.nan if q == 2 else float(q)] for q in range(1, 5)})
    delta = formulas["offense_rush_ypa_q4_minus_q1"].evaluate(x)
    slope = formulas["offense_rush_quarter_slope"].evaluate(x)
    assert delta.tolist() == [3, 3]
    assert slope.iloc[0] == pytest.approx(1) and np.isnan(slope.iloc[1])
