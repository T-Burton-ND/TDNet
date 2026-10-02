import pandas as pd
import pytest

from gridiron_ml.experiments.nextgen_baseline import align_baseline


def fixtures():
    schedule = pd.DataFrame([
        dict(id=1, season=2025, week=1, start_date="2025-09-01T12:00:00Z"),
        dict(id=2, season=2025, week=2, start_date="2025-09-08T12:00:00Z"),
    ]).assign(home_team="A", away_team="B", home_classification="fbs", away_classification="fbs",
              season_type="regular", completed=True, home_points=20, away_points=10)
    source = pd.DataFrame([
        dict(keys_team="A", offense_value=7.0), dict(keys_team="B", offense_value=3.0),
    ]).assign(keys_season=2025, keys_week=1, keys_game_id=1, next_game_id=2,
              y_next_margin=999, market_spread_close=999)
    return source, schedule


def test_baseline_preserves_features_and_uses_authoritative_next_outcomes():
    source, schedule = fixtures()
    result, report = align_baseline(source, schedule, ["offense_value"])
    assert result.offense_value.tolist() == [7, 3]
    assert result.next_game_margin.tolist() == [10, -10]
    assert "market_spread_close" not in result and "y_next_margin" not in result
    assert result.latest_source_game_id.tolist() == [1, 1]
    assert result.feature_available_utc.lt(result.target_start_utc).all()
    assert report["source_semantics_audit_required_before_training"]


def test_baseline_rejects_2026_and_source_target_overlap():
    source, schedule = fixtures()
    with pytest.raises(ValueError, match="quarantined"):
        align_baseline(source.assign(keys_season=2026), schedule, ["offense_value"])
    with pytest.raises(ValueError, match="No canonical"):
        align_baseline(source.assign(keys_game_id=2), schedule, ["offense_value"])


def test_weekly_graph_waits_for_other_teams_reporting_cutoff():
    source, schedule = fixtures()
    late = schedule.iloc[[0]].assign(id=3, home_team="C", away_team="D", start_date="2025-09-07T12:00:00Z")
    with pytest.raises(ValueError, match="No canonical"):
        align_baseline(source, pd.concat([schedule, late]), ["offense_value"])


def test_bye_snapshot_keeps_latest_state_and_requires_both_teams():
    source, schedule = fixtures()
    schedule.loc[schedule.id.eq(2), "week"] = 3
    bye = source.assign(keys_week=2, offense_value=[8, 4])
    result, _ = align_baseline(pd.concat([source, bye]), schedule, ["offense_value"])
    assert result.offense_value.tolist() == [8, 4]
    with pytest.raises(ValueError, match="No paired"):
        align_baseline(source.iloc[:1], schedule, ["offense_value"])
