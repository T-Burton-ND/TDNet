from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from gridiron_ml.f20.capture import PARSER_VERSION, capture, parse_tracker
from gridiron_ml.f20.model import (
    BASE_COLUMNS,
    FEATURE_COLUMNS,
    captured_features,
    feature_row,
)
from gridiron_ml.f20.variants import assemble_variant, select_compact_tdnet
from scripts.publication.issue_f20_forecasts import cutoff, issue


def page(*, duplicate: bool = False, bad_value: bool = False) -> bytes:
    expert = "oops" if bad_value else "6"
    row = f"<tr><td>Alabama</td><td>Texas A&M</td><td>4</td><td>5</td><td>6</td><td>7</td><td>{expert}</td></tr>"
    return ("<table><caption>Last updated: Thursday October 29, 5:00:00 PM CDT</caption>"
            "<tr><th>Home</th><th>Road</th><th>Line Open</th><th>Line</th><th>Midweek Line</th></tr>"
            "<tr><td>Alabama</td><td>Texas A&M</td><td>4</td><td>5</td><td>5</td></tr></table>"
            "<table><tr><th>home</th><th>road</th><th>Team Rankings</th><th>Massey Ratings</th>"
            "<th>ESPN FPI</th><th>FEI Projections</th><th>Sagarin Pred.</th></tr>"
            f"{row}{row if duplicate else ''}</table>").encode()


def schedule() -> pd.DataFrame:
    return pd.DataFrame([{"id": 1, "season": 2026, "week": 9,
                          "start_date": "2026-10-31T23:00:00Z", "home_team": "Alabama",
                          "away_team": "Texas A&M", "home_classification": "fbs",
                          "away_classification": "fbs", "season_type": "regular"}])


def models(path: Path, observation: dict) -> None:
    path.mkdir(parents=True)
    X = pd.concat([captured_features(observation)] * 2, ignore_index=True)
    hashes = {}
    for arm, cols in {"market_only": BASE_COLUMNS, "F20-A": FEATURE_COLUMNS}.items():
        model = DummyClassifier(strategy="prior").fit(X.loc[:, list(cols)], [0, 1])
        artifact = path / f"{arm}.joblib"
        joblib.dump(model, artifact)
        hashes[arm] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps({
        "training_seasons": list(range(2017, 2025)), "excluded_seasons": [2025, 2026],
        "historical_source_timing": "archived_pregame_unverified", "model_sha256": hashes}))


def test_capture_issues_one_official_and_keeps_revisions(tmp_path: Path) -> None:
    raw = page(duplicate=True)
    _, parsed = parse_tracker(raw)
    assert len(parsed) == 1  # Exact duplicate source rows do not duplicate a game.
    collected = datetime(2026, 10, 29, 22, 30, tzinfo=UTC)
    archive = tmp_path / "archive"
    receipt = capture(raw, schedule(), archive, collected_at=collected,
                      schedule_modified_at_utc="2026-10-29T20:00:00+00:00")
    assert receipt["pregame_games"] == 1
    assert receipt["site_updated_at_utc"] < receipt["collected_at_utc"]
    receipt_path = archive / "receipts" / f"{receipt['raw_sha256']}.{PARSER_VERSION}.json"
    assert capture(raw, schedule(), archive, collected_at=datetime(2026, 10, 30, tzinfo=UTC)) == receipt
    model_dir = tmp_path / "models"
    models(model_dir, receipt["observations"][0])
    first = issue(archive, receipt_path, model_dir, now=datetime(2026, 10, 30, 1, tzinfo=UTC))
    assert (first["issued"], first["official"]) == (1, 1)
    (archive / "forecasts/1/official.json").unlink()  # Simulate a crash before pointer creation.
    again = issue(archive, receipt_path, model_dir, now=datetime(2026, 10, 30, 2, tzinfo=UTC))
    assert (again["existing"], again["official"]) == (1, 1)
    second = issue(archive, receipt_path, model_dir, now=datetime(2026, 10, 30, 4, tzinfo=UTC), revision_id="r2")
    assert (second["issued"], second["official"], second["late_revision"]) == (1, 0, 1)
    pointer = json.loads((archive / "forecasts/1/official.json").read_text())
    record = json.loads(Path(pointer["record_file"]).read_text())
    assert record["issuance_class"] == "eligible_weekly"
    assert record["home_win_probability"]["F20-A"] == .5
    assert len(list((archive / "forecasts/1").glob("*.json"))) == 3


def test_cutoff_postgame_and_parse_rejections(tmp_path: Path) -> None:
    assert cutoff(datetime(2026, 10, 31, 23, tzinfo=UTC)).isoformat() == "2026-10-30T03:59:00+00:00"
    assert cutoff(datetime(2026, 11, 7, 23, tzinfo=UTC)).isoformat() == "2026-11-06T04:59:00+00:00"
    with pytest.raises(ValueError, match="Invalid tracker numeric"):
        capture(page(bad_value=True), schedule(), tmp_path, collected_at=datetime(2026, 10, 29, 22, 30, tzinfo=UTC))
    with pytest.raises(ValueError, match="after collection"):
        capture(page(), schedule(), tmp_path, collected_at=datetime(2026, 10, 29, 21, tzinfo=UTC))
    past = schedule()
    past.loc[0, "start_date"] = "2026-10-29T20:00:00Z"
    receipt = capture(page(), past, tmp_path, collected_at=datetime(2026, 10, 29, 22, 30, tzinfo=UTC))
    assert receipt["pregame_games"] == 0
    duplicate = pd.concat([schedule(), schedule()], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate schedule game ID"):
        capture(page(), duplicate, tmp_path / "duplicate", collected_at=datetime(2026, 10, 29, 22, 30, tzinfo=UTC))


def test_missing_expert_is_explicit(tmp_path: Path) -> None:
    raw = page().replace(b"<td>6</td></tr>", b"<td></td></tr>")
    receipt = capture(raw, schedule(), tmp_path, collected_at=datetime(2026, 10, 29, 22, 30, tzinfo=UTC))
    observation = receipt["observations"][0]
    assert observation["values"]["sagarin_pred"] == {
        "raw_value": "", "home_margin": None, "status": "missing"}
    assert captured_features(observation).iloc[0].expert_count == 4


def test_feature_whitelist_and_training_boundary() -> None:
    a = feature_row({"market_margin": 3, "espn_fpi": None, "score": 200})
    b = feature_row({"market_margin": 3, "espn_fpi": None, "score": -200})
    assert a == b
    assert a["missing_espn_fpi"] == 1
    from scripts.publication.fit_f20_baselines import fit_models
    with pytest.raises(ValueError, match="2017–2024"):
        fit_models(pd.DataFrame({"season": [2025], "home_win": [1]}))


def test_controlled_variants_and_selection_cutoff() -> None:
    frame = pd.DataFrame({"td_a": [0, 1, 0, 1], "td_b": [1, 2, 3, 4],
                          "target_leak": [0, 1, 0, 1]})
    manifest = [{"name": name, "temporal_cutoff": "before_target_game",
                 "prospective_allowed": True, "target_derived": name == "target_leak"}
                for name in frame]
    labels = pd.Series([0, 1, 0, 1])
    selected = select_compact_tdnet(frame, labels, pd.Series([2020, 2021, 2022, 2023]), manifest,
                                    max_features=2)
    assert set(selected) == {"td_a", "td_b"}
    with pytest.raises(ValueError, match="exclude 2025/2026"):
        select_compact_tdnet(frame, labels, pd.Series([2020, 2021, 2022, 2025]), manifest,
                             max_features=2)
    feature = feature_row({"market_margin": 2, "massey": 3})
    expert = pd.DataFrame([{**feature, "game_id": 1, "cutoff_utc": "2026-10-30T03:59:00Z",
                            "kickoff_utc": "2026-10-31T23:00:00Z"}])
    f19 = pd.DataFrame([{"game_id": 1, "feature_available_utc": "2026-10-30T02:00:00Z",
                         "td_a": 3.5, "f19_market": 4.0}])
    assert "market_margin" in assemble_variant("F20-A", expert)
    assert "td_a" in assemble_variant("F20-B", expert, f19=f19, selected_tdnet=("td_a",))
    full = assemble_variant("F20-C", expert, f19=f19)
    assert "f19_market" in full and "market_margin" not in full
    f19["actual_margin"] = 17
    with pytest.raises(ValueError, match="outcome"):
        assemble_variant("F20-C", expert, f19=f19)
