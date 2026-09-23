import csv
import json
from pathlib import Path

import pytest

from tools.cfbd_pickem.common import ValidationError
from tools.cfbd_pickem.export import export_submission


SOURCE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_start_time_utc",
    "home_team",
    "away_team",
    "neutral_site",
    "season_type",
    "pred_home_margin",
    "pred_winner",
    "predicted_margin",
    "model_count",
]


def _write_source(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": 123,
        "season": 2026,
        "week": 1,
        "game_start_time_utc": "2099-08-29T16:00:00Z",
        "home_team": "Home",
        "away_team": "Away",
        "neutral_site": "True",
        "season_type": "regular",
        "pred_home_margin": "3.1250000000000001",
        "pred_winner": "Home",
        "predicted_margin": "3.1250000000000001",
        "model_count": 36,
    }
    row.update(overrides)
    return row


def _write_slate(path: Path, **overrides: object) -> None:
    game: dict[str, object] = {
        "id": 123,
        "season": 2026,
        "seasonType": "regular",
        "week": 1,
        "homeTeam": "Home",
        "awayTeam": "Away",
        "homeId": 1,
        "awayId": 2,
        "pick": None,
    }
    game.update(overrides)
    path.write_text(
        json.dumps({"fetched_at_utc": "2026-08-28T20:00:00Z", "games": [game]}),
        encoding="utf-8",
    )


def _export(tmp_path: Path) -> dict[str, object]:
    return export_submission(
        source=tmp_path / "source.csv",
        cfbd_slate=tmp_path / "slate.json",
        output_dir=tmp_path / "out",
        season=2026,
        reader_week=0,
        provider_week=1,
    )


def test_export_preserves_precision_and_negates_home_margin(tmp_path: Path) -> None:
    _write_source(tmp_path / "source.csv", [_row()])
    _write_slate(tmp_path / "slate.json")

    audit = _export(tmp_path)

    payload_path = Path(audit["artifacts"]["api_payload"]["path"])
    assert '"pick": -3.1250000000000001' in payload_path.read_text(encoding="utf-8")
    csv_path = Path(audit["artifacts"]["csv"]["path"])
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "id,home,away,predicted",
        "123,Home,Away,-3.1250000000000001",
    ]
    assert audit["validation"]["passed"] is True
    assert audit["validation"]["cfbd_mapped_count"] == 1


def test_export_fails_when_cfbd_game_is_missing(tmp_path: Path) -> None:
    _write_source(tmp_path / "source.csv", [_row()])
    _write_slate(tmp_path / "slate.json", id=999)

    with pytest.raises(ValidationError, match="absent from the authenticated"):
        _export(tmp_path)


def test_export_fails_on_reversed_orientation(tmp_path: Path) -> None:
    _write_source(tmp_path / "source.csv", [_row()])
    _write_slate(tmp_path / "slate.json", homeTeam="Away", awayTeam="Home")

    with pytest.raises(ValidationError, match="does not exactly match"):
        _export(tmp_path)


def test_export_fails_on_duplicate_tdnet_game(tmp_path: Path) -> None:
    _write_source(tmp_path / "source.csv", [_row(), _row()])
    _write_slate(tmp_path / "slate.json")

    with pytest.raises(ValidationError, match="Duplicate TDNet game_id"):
        _export(tmp_path)


def test_full_slate_export_accepts_multiple_provider_weeks(tmp_path: Path) -> None:
    rows = [
        _row(),
        _row(
            game_id=456,
            week=2,
            game_start_time_utc="2099-09-12T16:00:00Z",
            home_team="Second Home",
            away_team="Second Away",
            neutral_site="False",
            pred_home_margin="-1.25",
            pred_winner="Second Away",
            predicted_margin="1.25",
        ),
    ]
    _write_source(tmp_path / "source.csv", rows)
    games = [
        {
            "id": 123,
            "season": 2026,
            "seasonType": "regular",
            "week": 1,
            "homeTeam": "Home",
            "awayTeam": "Away",
            "pick": None,
        },
        {
            "id": 456,
            "season": 2026,
            "seasonType": "regular",
            "week": 2,
            "homeTeam": "Second Home",
            "awayTeam": "Second Away",
            "pick": None,
        },
    ]
    (tmp_path / "slate.json").write_text(
        json.dumps({"games": games}), encoding="utf-8"
    )

    audit = export_submission(
        source=tmp_path / "source.csv",
        cfbd_slate=tmp_path / "slate.json",
        output_dir=tmp_path / "out",
        season=2026,
        provider_weeks=[1, 2],
        require_full_slate=True,
    )

    assert audit["validation"]["cfbd_mapped_count"] == 2
    assert audit["selection"]["cfbd_provider_weeks"] == [1, 2]


def test_full_slate_export_refuses_an_omitted_cfbd_game(tmp_path: Path) -> None:
    _write_source(tmp_path / "source.csv", [_row()])
    _write_slate(tmp_path / "slate.json")
    payload = json.loads((tmp_path / "slate.json").read_text(encoding="utf-8"))
    payload["games"].append(
        {
            "id": 999,
            "season": 2026,
            "seasonType": "regular",
            "week": 1,
            "homeTeam": "Other Home",
            "awayTeam": "Other Away",
            "pick": None,
        }
    )
    (tmp_path / "slate.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="would omit authenticated CFBD games"):
        export_submission(
            source=tmp_path / "source.csv",
            cfbd_slate=tmp_path / "slate.json",
            output_dir=tmp_path / "out",
            season=2026,
            provider_week=1,
            require_full_slate=True,
        )
