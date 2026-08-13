import pandas as pd

from gridiron_ml.publication.manual_poll import (
    find_latest_model_poll,
    load_saved_ballot,
    save_manual_ballot,
    validate_ballot,
)


def test_manual_ballot_is_persistent_and_replaced_by_week(tmp_path):
    teams = [f"Team {i}" for i in range(1, 26)]
    path = save_manual_ballot(tmp_path, season=2026, week=1, teams=teams, ballot_name="me")
    assert path.exists()
    assert load_saved_ballot(tmp_path, season=2026, week=1, ballot_name="me") == teams
    assert load_saved_ballot(tmp_path, season=2026, week=2, ballot_name="me") == []


def test_latest_model_poll_prefers_latest_recorded_week(tmp_path):
    older = tmp_path / "data/weekly_reports/2026/week_01/tables"
    newer = tmp_path / "data/weekly_reports/2026/week_02/tables"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    rows = pd.DataFrame({"season": [2026, 2026], "week": [1, 1], "rank": [1, 2], "keys_team": ["A", "B"]})
    rows.to_csv(older / "weekly_poll_top25.csv", index=False)
    rows.assign(week=2).to_csv(newer / "weekly_poll_top25.csv", index=False)
    path, poll = find_latest_model_poll(tmp_path)
    assert path == newer / "weekly_poll_top25.csv"
    assert poll["keys_team"].tolist() == ["A", "B"]


def test_ballot_validation_rejects_duplicates():
    teams = [f"Team {i}" for i in range(1, 25)] + ["Team 24"]
    try:
        validate_ballot(teams)
    except ValueError as exc:
        assert "same team" in str(exc)
    else:
        raise AssertionError("duplicate ballot was accepted")
