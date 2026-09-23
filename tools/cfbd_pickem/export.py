"""Validate and export exact TDNet predictions for CFBD Model Pick'em."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

from tools.cfbd_pickem.common import (
    API_BASE_URL,
    ValidationError,
    sha256_file,
    utc_now,
    write_json,
)


REQUIRED_SOURCE_COLUMNS = {
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
}
CSV_COLUMNS = ("id", "home", "away", "predicted")


@dataclass(frozen=True)
class Prediction:
    game_id: int
    season: int
    provider_week: int
    season_type: str
    kickoff_utc: str
    home_team: str
    away_team: str
    neutral_site: bool
    home_margin_text: str
    cfbd_margin_text: str
    pred_winner: str
    model_count: int


def _parse_bool(value: str, *, field: str, game_id: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValidationError(f"Game {game_id}: invalid {field} value {value!r}.")


def _decimal(value: str, *, field: str, game_id: int) -> Decimal:
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValidationError(
            f"Game {game_id}: {field} is not a decimal number: {value!r}."
        ) from exc
    if not parsed.is_finite():
        raise ValidationError(f"Game {game_id}: {field} must be finite.")
    return parsed


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value, "f")


def _provider_week_set(
    *, provider_week: int | None, provider_weeks: Sequence[int] | None
) -> set[int] | None:
    if provider_week is not None and provider_weeks is not None:
        raise ValidationError("Pass provider_week or provider_weeks, not both.")
    values = [provider_week] if provider_week is not None else provider_weeks
    if values is None:
        return None
    selected = {int(value) for value in values}
    if not selected:
        raise ValidationError("At least one provider week is required.")
    return selected


def load_predictions(
    source: Path,
    *,
    season: int,
    provider_week: int | None = None,
    provider_weeks: Sequence[int] | None = None,
) -> list[Prediction]:
    if not source.is_file():
        raise ValidationError(f"Prediction source artifact does not exist: {source}")
    selected_weeks = _provider_week_set(
        provider_week=provider_week, provider_weeks=provider_weeks
    )

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_SOURCE_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValidationError(
                "Prediction source artifact is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
        selected = []
        for row in reader:
            row_season = int(row["season"])
            row_week = int(row["week"])
            if row_season != season:
                continue
            if selected_weeks is not None and row_week not in selected_weeks:
                continue
            selected.append(row)

    if not selected:
        weeks = "all" if selected_weeks is None else sorted(selected_weeks)
        raise ValidationError(
            f"No predictions found for season={season}, provider_weeks={weeks}."
        )

    predictions: list[Prediction] = []
    seen_ids: set[int] = set()
    seen_matchups: set[tuple[int, int, frozenset[str]]] = set()
    for row in selected:
        game_id = int(row["game_id"])
        if game_id in seen_ids:
            raise ValidationError(f"Duplicate TDNet game_id: {game_id}")
        seen_ids.add(game_id)

        row_season = int(row["season"])
        row_week = int(row["week"])
        home_team = row["home_team"].strip()
        away_team = row["away_team"].strip()
        matchup = (row_season, row_week, frozenset((home_team, away_team)))
        if len(matchup[2]) != 2 or matchup in seen_matchups:
            raise ValidationError(
                f"Duplicate or invalid TDNet matchup: {away_team} at {home_team}"
            )
        seen_matchups.add(matchup)

        home_margin_raw = row["pred_home_margin"].strip()
        home_margin = _decimal(
            home_margin_raw, field="pred_home_margin", game_id=game_id
        )
        predicted_margin = _decimal(
            row["predicted_margin"], field="predicted_margin", game_id=game_id
        )
        winner = row["pred_winner"].strip()
        expected_winner = (
            home_team if home_margin > 0 else away_team if home_margin < 0 else ""
        )
        if not expected_winner:
            raise ValidationError(
                f"Game {game_id}: a zero projected margin has no unambiguous winner."
            )
        if winner != expected_winner:
            raise ValidationError(
                f"Game {game_id}: pred_winner={winner!r} conflicts with "
                f"pred_home_margin={home_margin_raw}."
            )
        if predicted_margin != abs(home_margin):
            raise ValidationError(
                f"Game {game_id}: predicted_margin does not equal "
                "abs(pred_home_margin)."
            )

        kickoff = row["game_start_time_utc"].strip()
        try:
            parsed_kickoff = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError(
                f"Game {game_id}: invalid game_start_time_utc {kickoff!r}."
            ) from exc
        if parsed_kickoff.tzinfo is None:
            raise ValidationError(f"Game {game_id}: kickoff must include a timezone.")

        model_count = int(row["model_count"])
        if model_count < 1:
            raise ValidationError(f"Game {game_id}: model_count must be positive.")

        predictions.append(
            Prediction(
                game_id=game_id,
                season=row_season,
                provider_week=row_week,
                season_type=row["season_type"].strip(),
                kickoff_utc=kickoff,
                home_team=home_team,
                away_team=away_team,
                neutral_site=_parse_bool(
                    row["neutral_site"], field="neutral_site", game_id=game_id
                ),
                home_margin_text=_decimal_text(home_margin),
                cfbd_margin_text=_decimal_text(-home_margin),
                pred_winner=winner,
                model_count=model_count,
            )
        )

    model_counts = {prediction.model_count for prediction in predictions}
    if len(model_counts) != 1:
        raise ValidationError(f"Inconsistent TDNet model_count values: {model_counts}")
    return predictions


def load_slate(snapshot_path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read CFBD slate snapshot: {snapshot_path}") from exc

    if isinstance(payload, list):
        return None, payload
    if not isinstance(payload, dict) or not isinstance(payload.get("games"), list):
        raise ValidationError("CFBD slate snapshot must be a game list or a snapshot object.")
    fetched_at = payload.get("fetched_at_utc")
    return str(fetched_at) if fetched_at else None, payload["games"]


def validate_mapping(
    predictions: list[Prediction],
    cfbd_games: list[dict[str, Any]],
    *,
    require_full_slate: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    games_by_id: dict[int, dict[str, Any]] = {}
    for game in cfbd_games:
        try:
            game_id = int(game["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("CFBD slate contains a game without a valid id.") from exc
        if game_id in games_by_id:
            raise ValidationError(f"Duplicate CFBD game id: {game_id}")
        games_by_id[game_id] = game

    matched: list[dict[str, Any]] = []
    missing: list[int] = []
    for prediction in predictions:
        game = games_by_id.get(prediction.game_id)
        if game is None:
            missing.append(prediction.game_id)
            continue
        identity = (
            int(game.get("season", -1)),
            int(game.get("week", -1)),
            str(game.get("seasonType", "")),
            str(game.get("awayTeam", "")),
            str(game.get("homeTeam", "")),
        )
        expected = (
            prediction.season,
            prediction.provider_week,
            prediction.season_type,
            prediction.away_team,
            prediction.home_team,
        )
        if identity != expected:
            raise ValidationError(
                f"Game {prediction.game_id}: CFBD identity/orientation {identity!r} "
                f"does not exactly match TDNet {expected!r}."
            )
        existing_pick = game.get("pick")
        if existing_pick is not None:
            try:
                if not math.isfinite(float(existing_pick)):
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Game {prediction.game_id}: CFBD returned an invalid existing pick."
                ) from exc
        matched.append(game)

    if missing:
        raise ValidationError(
            "TDNet games absent from the authenticated CFBD slate: "
            + ", ".join(str(game_id) for game_id in missing)
        )
    if len(matched) != len(predictions):
        raise ValidationError("Mapped game count does not equal TDNet game count.")

    source_ids = {prediction.game_id for prediction in predictions}
    selected_periods = {
        (prediction.season, prediction.provider_week, prediction.season_type)
        for prediction in predictions
    }
    in_selected_periods = [
        game
        for game in cfbd_games
        if (
            int(game.get("season", -1)),
            int(game.get("week", -1)),
            str(game.get("seasonType", "")),
        )
        in selected_periods
    ]
    unexpected = [
        game for game in in_selected_periods if int(game["id"]) not in source_ids
    ]
    outside_periods = [game for game in cfbd_games if int(game["id"]) not in source_ids]
    if require_full_slate and outside_periods:
        raise ValidationError(
            "Full-slate export would omit authenticated CFBD games: "
            + ", ".join(str(game["id"]) for game in outside_periods)
        )
    return matched, unexpected


def _validate_before_kickoff(predictions: list[Prediction]) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    kickoffs = [
        datetime.fromisoformat(prediction.kickoff_utc.replace("Z", "+00:00"))
        for prediction in predictions
    ]
    locked = [
        prediction
        for prediction, kickoff in zip(predictions, kickoffs)
        if kickoff <= now
    ]
    if locked:
        raise ValidationError(
            "Games are at or past kickoff: "
            + ", ".join(str(prediction.game_id) for prediction in locked)
        )
    return min(kickoffs).isoformat(), max(kickoffs).isoformat()


def _write_api_payload(path: Path, predictions: list[Prediction]) -> None:
    lines = ["{", '  "picks": [']
    for index, prediction in enumerate(predictions):
        comma = "," if index < len(predictions) - 1 else ""
        lines.append(
            "    {"
            f'"gameId": {prediction.game_id}, '
            f'"pick": {prediction.cfbd_margin_text}'
            f"}}{comma}"
        )
    lines.extend(["  ]", "}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not result:
        raise ValidationError("entry_name must contain a letter or number.")
    return result


def export_submission(
    *,
    source: Path,
    cfbd_slate: Path,
    output_dir: Path,
    season: int,
    reader_week: int | None = None,
    provider_week: int | None = None,
    provider_weeks: Sequence[int] | None = None,
    season_type: str = "regular",
    entry_name: str = "TDNet margin consensus",
    roster: str = "corrected-F6 wide-margin operational roster",
    rounding: str = "none",
    require_full_slate: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    cfbd_slate = cfbd_slate.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(
        source,
        season=season,
        provider_week=provider_week,
        provider_weeks=provider_weeks,
    )
    if {prediction.season_type for prediction in predictions} != {season_type}:
        raise ValidationError(
            f"Prediction season types do not exactly equal {season_type!r}."
        )
    earliest_kickoff, latest_kickoff = _validate_before_kickoff(predictions)
    fetched_at, cfbd_games = load_slate(cfbd_slate)
    matched, unexpected = validate_mapping(
        predictions, cfbd_games, require_full_slate=require_full_slate
    )

    slug = _slug(entry_name)
    stem = f"{season}_current_slate_{slug}_cfbd"
    csv_path = output_dir / f"{stem}_submission.csv"
    payload_path = output_dir / f"{stem}_submission.json"
    games_path = output_dir / f"{stem}_contest_games.json"
    audit_path = output_dir / f"{stem}_audit.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for prediction in predictions:
            writer.writerow(
                (
                    prediction.game_id,
                    prediction.home_team,
                    prediction.away_team,
                    prediction.cfbd_margin_text,
                )
            )
    _write_api_payload(payload_path, predictions)
    write_json(
        games_path,
        {
            "fetched_at_utc": fetched_at,
            "endpoint": f"{API_BASE_URL}/api/picks",
            "games": matched,
        },
    )

    existing_picks = {
        int(game["id"]): game.get("pick")
        for game in matched
        if game.get("pick") is not None
    }
    selected_weeks = sorted({prediction.provider_week for prediction in predictions})
    neutral_games = [prediction.game_id for prediction in predictions if prediction.neutral_site]
    audit: dict[str, Any] = {
        "schema_version": 2,
        "status": "prepared_not_submitted",
        "created_at_utc": utc_now(),
        "entry_name": entry_name,
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "authoritative_column": "pred_home_margin",
            "margin_convention": "home_score_minus_away_score",
            "roster": roster,
            "model_count": predictions[0].model_count,
        },
        "selection": {
            "season": season,
            "reader_week": reader_week,
            "cfbd_provider_weeks": selected_weeks,
            "season_type": season_type,
            "full_authenticated_slate_required": require_full_slate,
        },
        "cfbd_contract": {
            "api_base_url": API_BASE_URL,
            "read_endpoint": "GET /api/picks",
            "submit_endpoint": "POST /api/picks",
            "api_body_schema": {"picks": [{"gameId": "integer", "pick": "number"}]},
            "csv_columns": list(CSV_COLUMNS),
            "margin_convention": "away_score_minus_home_score",
            "authentication": "Authorization: Bearer <API key>",
            "neutral_site_required": False,
            "documented_deadline": "before kickoff",
            "cfbd_slate_fetched_at_utc": fetched_at,
        },
        "transformation": {
            "formula": "cfbd_pick = -pred_home_margin",
            "rounding": rounding,
            "decimal_text_preserved": True,
        },
        "validation": {
            "passed": True,
            "tdnet_game_count": len(predictions),
            "cfbd_mapped_count": len(matched),
            "duplicate_tdnet_games": 0,
            "duplicate_cfbd_games": 0,
            "missing_tdnet_games": [],
            "unexpected_cfbd_games": [int(game["id"]) for game in unexpected],
            "orientation_mismatches": [],
            "neutral_site_game_ids": neutral_games,
            "neutral_site_game_count": len(neutral_games),
            "existing_cfbd_picks_on_selected_games": existing_picks,
            "authenticated_cfbd_slate_game_count": len(cfbd_games),
            "earliest_kickoff_utc": earliest_kickoff,
            "latest_kickoff_utc": latest_kickoff,
            "all_games_before_kickoff_at_validation": True,
            "margin_sign_check": "passed",
            "precision_check": "passed",
        },
        "games": [
            {
                "game_id": prediction.game_id,
                "provider_week": prediction.provider_week,
                "away_team": prediction.away_team,
                "home_team": prediction.home_team,
                "kickoff_utc": prediction.kickoff_utc,
                "neutral_site": prediction.neutral_site,
                "tdnet_pred_home_margin": prediction.home_margin_text,
                "cfbd_pick_away_minus_home": prediction.cfbd_margin_text,
                "existing_cfbd_pick": next(
                    game.get("pick")
                    for game in matched
                    if int(game["id"]) == prediction.game_id
                ),
            }
            for prediction in predictions
        ],
        "artifacts": {
            "csv": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
            "api_payload": {
                "path": str(payload_path),
                "sha256": sha256_file(payload_path),
            },
            "cfbd_contest_games": {
                "path": str(games_path),
                "sha256": sha256_file(games_path),
            },
            "input_cfbd_slate": {
                "path": str(cfbd_slate),
                "sha256": sha256_file(cfbd_slate),
            },
        },
        "submission": None,
    }
    write_json(audit_path, audit)
    audit["audit_path"] = str(audit_path)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export validated TDNet predictions to CFBD without submitting them."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cfbd-slate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--reader-week", type=int)
    parser.add_argument("--provider-week", type=int, action="append")
    parser.add_argument("--season-type", default="regular")
    parser.add_argument("--entry-name", required=True)
    parser.add_argument("--roster", required=True)
    parser.add_argument("--rounding", default="none")
    parser.add_argument("--require-full-slate", action="store_true")
    args = parser.parse_args()

    audit = export_submission(
        source=args.source,
        cfbd_slate=args.cfbd_slate,
        output_dir=args.output_dir,
        season=args.season,
        reader_week=args.reader_week,
        provider_weeks=args.provider_week,
        season_type=args.season_type,
        entry_name=args.entry_name,
        roster=args.roster,
        rounding=args.rounding,
        require_full_slate=args.require_full_slate,
    )
    validation = audit["validation"]
    print("CFBD export prepared; no submission was attempted.")
    print(f"Entry: {audit['entry_name']}")
    print(f"TDNet games: {validation['tdnet_game_count']}")
    print(f"Mapped CFBD games: {validation['cfbd_mapped_count']}")
    print(f"Unexpected CFBD games: {validation['unexpected_cfbd_games']}")
    print(f"CSV: {audit['artifacts']['csv']['path']}")
    print(f"API payload: {audit['artifacts']['api_payload']['path']}")
    print(f"Audit: {audit['audit_path']}")


if __name__ == "__main__":
    main()
