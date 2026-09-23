"""Inference-only generation of CFBD contest picks from frozen TDNet rosters.

This module never trains, refits, calibrates, downloads, or submits. It uses
the same schedule-driven preseason fingerprint construction as the published
Week 0 package and requires exact reproduction of that eight-game freeze
before it writes an expanded contest-slate source artifact.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridiron_ml.publication.weekly import build_weekly_blog_package

from tools.cfbd_pickem.common import (
    ValidationError,
    sha256_file,
    utc_now,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEDULE = ROOT / "data/raw/cfbd/v2/games/2026.parquet"
DEFAULT_MARGIN_INVENTORY = (
    ROOT
    / "data/publication/2026/week_00/pre_game_inputs/"
    "margin_wide_f6_runtime_inventory.csv"
)
FROZEN_MARGIN_SOURCE = (
    ROOT / "publication/2026/week_00/pre_game/tables/all_games.csv"
)
PRESEASON_INPUTS = (
    ROOT / "data/raw/cfbd/v2/talent/2026.parquet",
    ROOT / "data/raw/cfbd/v2/returning/2026.parquet",
    ROOT / "data/raw/cfbd/v2/teams_fbs/2026.parquet",
    ROOT / "configs/features/feature_registry.yaml",
)
SOURCE_COLUMNS = (
    "entry_name",
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
    "raw_consensus_pred_home_margin",
    "raw_consensus_pred_home_win_probability",
    "submission_precision",
)


def _load_snapshot(path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read CFBD slate snapshot: {path}") from exc
    if isinstance(payload, list):
        return None, payload
    if not isinstance(payload, dict) or not isinstance(payload.get("games"), list):
        raise ValidationError("CFBD slate snapshot must contain a games list.")
    fetched_at = payload.get("fetched_at_utc")
    return str(fetched_at) if fetched_at else None, payload["games"]


def _contest_schedule(
    schedule_path: Path,
    contest_games: list[dict[str, Any]],
) -> pd.DataFrame:
    schedule = (
        pd.read_parquet(schedule_path)
        if schedule_path.suffix == ".parquet"
        else pd.read_csv(schedule_path)
    )
    id_column = "id" if "id" in schedule else "game_id"
    if id_column not in schedule:
        raise ValidationError("Schedule is missing id/game_id.")
    schedule = schedule.copy()
    schedule[id_column] = pd.to_numeric(schedule[id_column], errors="raise").astype(int)
    if schedule[id_column].duplicated().any():
        raise ValidationError("The local schedule contains duplicate game IDs.")

    contest_ids: list[int] = []
    seen: set[int] = set()
    for game in contest_games:
        game_id = int(game["id"])
        if game_id in seen:
            raise ValidationError(f"Duplicate CFBD contest game ID: {game_id}")
        seen.add(game_id)
        contest_ids.append(game_id)
    if not contest_ids:
        raise ValidationError("The CFBD contest slate is empty.")

    selected = schedule.loc[schedule[id_column].isin(contest_ids)].copy()
    missing = sorted(set(contest_ids) - set(selected[id_column]))
    if missing:
        raise ValidationError(
            "Contest games missing from the local schedule: "
            + ", ".join(map(str, missing))
        )
    selected = selected.set_index(id_column, drop=False).loc[contest_ids].reset_index(drop=True)

    orientation_reversed = []
    cfbd_home_teams = []
    cfbd_away_teams = []
    for row, game in zip(selected.to_dict("records"), contest_games):
        identity = (
            int(row["season"]),
            str(row["season_type"]),
            int(row["week"]),
            str(row["away_team"]),
            str(row["home_team"]),
        )
        expected = (
            int(game["season"]),
            str(game["seasonType"]),
            int(game["week"]),
            str(game["awayTeam"]),
            str(game["homeTeam"]),
        )
        reversed_expected = (
            expected[0],
            expected[1],
            expected[2],
            expected[4],
            expected[3],
        )
        reversed_orientation = identity == reversed_expected
        if identity != expected and not reversed_orientation:
            raise ValidationError(
                f"Game {game['id']}: local schedule identity/orientation {identity!r} "
                f"does not match CFBD {expected!r}."
            )
        if reversed_orientation and not bool(row.get("neutral_site", False)):
            raise ValidationError(
                f"Game {game['id']}: CFBD reverses a non-neutral local home/away "
                "orientation; refusing to infer a mapping."
            )
        orientation_reversed.append(reversed_orientation)
        cfbd_home_teams.append(str(game["homeTeam"]))
        cfbd_away_teams.append(str(game["awayTeam"]))
    selected["cfbd_home_team"] = cfbd_home_teams
    selected["cfbd_away_team"] = cfbd_away_teams
    selected["cfbd_orientation_reversed"] = orientation_reversed
    return selected


def _validate_eligibility(schedule: pd.DataFrame, *, now: datetime) -> None:
    kickoff_column = "start_date" if "start_date" in schedule else "game_start_time_utc"
    if kickoff_column not in schedule:
        raise ValidationError("Schedule is missing a kickoff timestamp.")
    kickoffs = pd.to_datetime(schedule[kickoff_column], utc=True, errors="coerce")
    if kickoffs.isna().any():
        ids = schedule.loc[kickoffs.isna(), "id"].astype(str).tolist()
        raise ValidationError("Invalid kickoff timestamps for games: " + ", ".join(ids))
    locked = schedule.loc[kickoffs.le(now), ["id", "away_team", "home_team"]]
    if not locked.empty:
        detail = ", ".join(
            f"{row.id} ({row.away_team} at {row.home_team})"
            for row in locked.itertuples()
        )
        raise ValidationError(f"Contest games are at or past kickoff: {detail}")


def _inventory_fingerprints(inventory: pd.DataFrame) -> list[dict[str, str]]:
    if "fingerprint_path" not in inventory:
        raise ValidationError("Frozen inventory is missing fingerprint_path.")
    records = []
    for raw in sorted(inventory["fingerprint_path"].dropna().astype(str).unique()):
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise ValidationError(f"Frozen fingerprint is missing: {path}")
        records.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    if not records:
        raise ValidationError("Frozen inventory resolved no fingerprint artifacts.")
    return records


def _verify_checkpoints(inventory: pd.DataFrame) -> list[str]:
    hashes: list[str] = []
    for row in inventory.to_dict("records"):
        enabled = str(row.get("use_in_weekly_consensus", True)).strip().casefold()
        if enabled not in {"true", "1", "yes", "y"}:
            continue
        path = Path(str(row["checkpoint_path"]))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise ValidationError(f"Frozen checkpoint is missing: {path}")
        actual = sha256_file(path)
        recorded = str(row.get("checkpoint_sha256", "")).strip().lower()
        if recorded and recorded != actual:
            raise ValidationError(
                f"Checkpoint hash mismatch for {path}: expected {recorded}, got {actual}."
            )
        hashes.append(actual)
    if len(hashes) != len(set(hashes)):
        raise ValidationError("The active inventory contains duplicate checkpoint hashes.")
    return sorted(hashes)


def _run_roster(
    *,
    inventory_path: Path,
    schedule: pd.DataFrame,
    staging: Path,
    expected_model_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    staging.mkdir(parents=True, exist_ok=True)
    schedule_path = staging / "contest_schedule.csv"
    schedule.to_csv(schedule_path, index=False)
    consensus_frames = []
    model_frames = []
    reports = []
    provider_weeks = sorted(pd.to_numeric(schedule["week"], errors="raise").astype(int).unique())
    for week in provider_weeks:
        report = build_weekly_blog_package(
            project_root=ROOT,
            season=2026,
            week=int(week),
            model_inventory_path=inventory_path,
            schedule_snapshot_path=schedule_path,
            # Every frozen roster used here is market-free.  An explicitly
            # absent path prevents later market snapshots from becoming input.
            market_lines_path=staging / "market_lines_intentionally_absent.parquet",
            output_root=staging / f"week_{week:02d}",
            # A two-game provider week can make a healthy model look constant.
            # Keep every frozen checkpoint here, then validate variation over
            # the complete 53-game contest slate below.
            include_collapsed_models=True,
            schedule_driven_matchups=True,
            render_social_assets=False,
        )
        manifest = report["manifest"]
        if int(manifest["model_count"]) != expected_model_count:
            raise ValidationError(
                f"Week {week}: expected {expected_model_count} models, got "
                f"{manifest['model_count']}."
            )
        if int(manifest["model_failure_count"]) != 0:
            raise ValidationError(
                f"Week {week}: model failures occurred; refusing to export."
            )
        consensus_frames.append(report["all_games"])
        model_frames.append(report["all_model_predictions"])
        reports.append(manifest)
    consensus = pd.concat(consensus_frames, ignore_index=True)
    model_predictions = pd.concat(model_frames, ignore_index=True)
    per_game_counts = model_predictions.groupby("game_id")["model_name"].nunique()
    bad_games = per_game_counts.loc[per_game_counts.ne(expected_model_count)]
    if not bad_games.empty:
        raise ValidationError(
            "Not every contest game has the complete frozen roster: "
            + repr(bad_games.to_dict())
        )
    per_model_variation = model_predictions.groupby("model_name")[
        "pred_home_margin"
    ].nunique()
    collapsed = per_model_variation.loc[per_model_variation.le(1)].index.astype(str).tolist()
    if collapsed:
        raise ValidationError(
            "Models are constant across the complete contest slate: " + repr(collapsed)
        )
    return consensus, model_predictions, reports


def _freeze_gate_margin(
    generated: pd.DataFrame,
    generated_model_predictions: pd.DataFrame,
) -> dict[str, Any]:
    frozen = pd.read_csv(FROZEN_MARGIN_SOURCE)
    columns = ["game_id", "away_team", "home_team", "pred_home_margin"]
    check = frozen[columns].merge(
        generated[columns],
        on=["game_id", "away_team", "home_team"],
        how="left",
        validate="one_to_one",
        suffixes=("_frozen", "_generated"),
    )
    if check["pred_home_margin_generated"].isna().any():
        raise ValidationError("Margin freeze gate has missing or mismatched games.")
    frozen_values = check["pred_home_margin_frozen"].to_numpy(dtype=float)
    generated_values = check["pred_home_margin_generated"].to_numpy(dtype=float)
    consensus_deltas = generated_values - frozen_values
    if np.max(np.abs(consensus_deltas)) > 1e-12:
        raise ValidationError(
            "Margin freeze reproduction failed; maximum absolute delta is "
            f"{np.max(np.abs(consensus_deltas))!r}."
        )

    frozen_long = pd.read_csv(
        FROZEN_MARGIN_SOURCE.with_name("all_game_model_predictions.csv")
    )
    frozen_ids = set(frozen["game_id"].astype(int))
    current_long = generated_model_predictions.loc[
        generated_model_predictions["game_id"].astype(int).isin(frozen_ids),
        ["game_id", "model_name", "pred_home_margin"],
    ]
    frozen_long = frozen_long[["game_id", "model_name", "pred_home_margin"]]
    model_check = frozen_long.merge(
        current_long,
        on=["game_id", "model_name"],
        how="outer",
        indicator=True,
        validate="one_to_one",
        suffixes=("_frozen", "_generated"),
    )
    if not model_check["_merge"].eq("both").all():
        raise ValidationError("Margin freeze model-level roster/game rows do not match.")
    model_delta = (
        pd.to_numeric(model_check["pred_home_margin_generated"], errors="raise")
        - pd.to_numeric(model_check["pred_home_margin_frozen"], errors="raise")
    )
    if model_delta.abs().max() > 1e-12:
        raise ValidationError(
            "Margin freeze model-level reproduction failed; maximum absolute delta is "
            f"{model_delta.abs().max()!r}."
        )
    return {
        "passed": True,
        "game_count": int(len(check)),
        "model_prediction_count": int(len(model_check)),
        "comparison": (
            "model-level and consensus pred_home_margin within 1e-12; final eight "
            "consensus values restored from the authoritative published source"
        ),
        "maximum_consensus_absolute_delta_before_restore": float(
            np.max(np.abs(consensus_deltas))
        ),
        "maximum_model_absolute_delta": float(model_delta.abs().max()),
        "source_path": str(FROZEN_MARGIN_SOURCE.resolve()),
        "source_sha256": sha256_file(FROZEN_MARGIN_SOURCE),
        "model_source_path": str(
            FROZEN_MARGIN_SOURCE.with_name("all_game_model_predictions.csv").resolve()
        ),
        "model_source_sha256": sha256_file(
            FROZEN_MARGIN_SOURCE.with_name("all_game_model_predictions.csv")
        ),
    }


def _restore_frozen_margin_consensus(generated: pd.DataFrame) -> pd.DataFrame:
    """Keep published Week 0 consensus values authoritative byte-for-byte."""
    frame = generated.copy()
    frozen = pd.read_csv(FROZEN_MARGIN_SOURCE).set_index("game_id")
    selected = frame["game_id"].astype(int).isin(frozen.index.astype(int))
    for index in frame.index[selected]:
        game_id = int(frame.at[index, "game_id"])
        value = float(frozen.at[game_id, "pred_home_margin"])
        frame.at[index, "pred_home_margin"] = value
        frame.at[index, "predicted_margin"] = abs(value)
        expected = frame.at[index, "home_team"] if value > 0 else frame.at[index, "away_team"]
        if frame.at[index, "pred_winner"] != expected:
            raise ValidationError(
                f"Game {game_id}: frozen margin sign conflicts with generated winner."
            )
    return frame


def _restore_frozen_margin_text(source: pd.DataFrame) -> pd.DataFrame:
    """Restore the exact decimal text published for the opening eight games."""
    frame = source.copy()
    frozen = pd.read_csv(
        FROZEN_MARGIN_SOURCE,
        dtype={"game_id": str, "pred_home_margin": str},
        keep_default_na=False,
    )
    exact = dict(zip(frozen["game_id"], frozen["pred_home_margin"]))
    for index in frame.index:
        game_id = str(int(frame.at[index, "game_id"]))
        if game_id not in exact:
            continue
        text = exact[game_id]
        frame.at[index, "pred_home_margin"] = text
        frame.at[index, "predicted_margin"] = str(abs(Decimal(text)))
    restored = frame.loc[
        frame["game_id"].astype(str).isin(exact), ["game_id", "pred_home_margin"]
    ]
    observed = {str(int(row.game_id)): str(row.pred_home_margin) for row in restored.itertuples()}
    if observed != exact:
        raise ValidationError(
            "The output source did not preserve the exact published Week 0 decimals."
        )
    return frame


def _submission_source(
    consensus: pd.DataFrame,
    contest_ids: list[int],
    *,
    entry_name: str,
    precision: str,
) -> pd.DataFrame:
    frame = consensus.copy()
    frame["game_id"] = pd.to_numeric(frame["game_id"], errors="raise").astype(int)
    if frame["game_id"].duplicated().any():
        raise ValidationError(f"{entry_name}: duplicate consensus game IDs.")
    missing = sorted(set(contest_ids) - set(frame["game_id"]))
    extra = sorted(set(frame["game_id"]) - set(contest_ids))
    if missing or extra:
        raise ValidationError(
            f"{entry_name}: slate mismatch; missing={missing}, unexpected={extra}."
        )
    frame = frame.set_index("game_id", drop=False).loc[contest_ids].reset_index(drop=True)
    raw_margin = pd.to_numeric(frame["pred_home_margin"], errors="coerce")
    if raw_margin.isna().any() or (~np.isfinite(raw_margin)).any():
        raise ValidationError(f"{entry_name}: non-finite home margin.")
    if raw_margin.eq(0).any():
        raise ValidationError(f"{entry_name}: exact zero margin is ambiguous.")
    sign_winner = frame["home_team"].where(raw_margin.gt(0), frame["away_team"])
    mismatched_winner = ~sign_winner.eq(frame["pred_winner"])
    if mismatched_winner.any():
        detail = frame.loc[
            mismatched_winner, ["game_id", "away_team", "home_team", "pred_winner"]
        ].to_dict("records")
        raise ValidationError(
            f"{entry_name}: probability winner conflicts with projected-margin sign: {detail}"
        )

    if precision != "full_float64":
        raise ValidationError(f"Unknown submission precision: {precision}")
    submission_margin = raw_margin.map(repr)

    output = pd.DataFrame(
        {
            "entry_name": entry_name,
            "game_id": frame["game_id"],
            "season": pd.to_numeric(frame["season"], errors="raise").astype(int),
            "week": pd.to_numeric(frame["week"], errors="raise").astype(int),
            "game_start_time_utc": frame["game_start_time_utc"],
            "home_team": frame["home_team"],
            "away_team": frame["away_team"],
            "neutral_site": frame["neutral_site"].astype(bool),
            "season_type": frame["season_type"],
            "pred_home_margin": submission_margin,
            "pred_winner": sign_winner,
            "predicted_margin": submission_margin.map(
                lambda value: str(abs(Decimal(value)))
            ),
            "model_count": pd.to_numeric(frame["model_count"], errors="raise").astype(int),
            "raw_consensus_pred_home_margin": raw_margin.map(repr),
            "raw_consensus_pred_home_win_probability": pd.to_numeric(
                frame["pred_home_win_probability"], errors="raise"
            ).map(repr),
            "submission_precision": precision,
        }
    )
    return output.loc[:, SOURCE_COLUMNS]


def _orient_consensus_to_cfbd(
    consensus: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Express local home-minus-away predictions in the contest orientation."""
    id_column = "id" if "id" in schedule else "game_id"
    mapping = schedule[
        [
            id_column,
            "cfbd_home_team",
            "cfbd_away_team",
            "cfbd_orientation_reversed",
        ]
    ].rename(columns={id_column: "game_id"})
    frame = consensus.merge(mapping, on="game_id", how="left", validate="one_to_one")
    if frame["cfbd_orientation_reversed"].isna().any():
        raise ValidationError("Consensus output is missing a CFBD orientation mapping.")
    reversed_rows = frame["cfbd_orientation_reversed"].astype(bool)
    local_home = frame["home_team"].copy()
    local_away = frame["away_team"].copy()
    frame.loc[reversed_rows, "pred_home_margin"] = -pd.to_numeric(
        frame.loc[reversed_rows, "pred_home_margin"], errors="raise"
    )
    frame.loc[reversed_rows, "pred_home_win_probability"] = 1.0 - pd.to_numeric(
        frame.loc[reversed_rows, "pred_home_win_probability"], errors="raise"
    )
    frame["home_team"] = frame["cfbd_home_team"]
    frame["away_team"] = frame["cfbd_away_team"]
    frame["local_schedule_home_team"] = local_home
    frame["local_schedule_away_team"] = local_away
    frame["predicted_margin"] = pd.to_numeric(
        frame["pred_home_margin"], errors="raise"
    ).abs()
    expected_winner = frame["home_team"].where(
        pd.to_numeric(frame["pred_home_margin"], errors="raise").gt(0),
        frame["away_team"],
    )
    if not expected_winner.eq(frame["pred_winner"]).all():
        bad = frame.loc[
            ~expected_winner.eq(frame["pred_winner"]),
            ["game_id", "away_team", "home_team", "pred_winner", "pred_home_margin"],
        ]
        raise ValidationError(
            "Winner changed under the CFBD orientation transform: "
            + repr(bad.to_dict("records"))
        )
    return frame


def generate(*, cfbd_slate: Path, schedule_path: Path, output_dir: Path) -> dict[str, Any]:
    cfbd_slate = cfbd_slate.resolve()
    schedule_path = schedule_path.resolve()
    output_dir = output_dir.resolve()
    fetched_at, contest_games = _load_snapshot(cfbd_slate)
    schedule = _contest_schedule(schedule_path, contest_games)
    generated_at = datetime.now(timezone.utc)
    _validate_eligibility(schedule, now=generated_at)
    contest_ids = [int(game["id"]) for game in contest_games]

    margin_inventory = pd.read_csv(DEFAULT_MARGIN_INVENTORY)
    margin_checkpoint_hashes = _verify_checkpoints(margin_inventory)
    margin_fingerprints = _inventory_fingerprints(margin_inventory)

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tdnet-cfbd-slate-") as temporary:
        staging = Path(temporary)
        margin_consensus, margin_long, margin_reports = _run_roster(
            inventory_path=DEFAULT_MARGIN_INVENTORY,
            schedule=schedule,
            staging=staging / "margin_consensus",
            expected_model_count=36,
        )
        margin_gate = _freeze_gate_margin(margin_consensus, margin_long)
        margin_consensus = _restore_frozen_margin_consensus(margin_consensus)
        margin_oriented = _orient_consensus_to_cfbd(margin_consensus, schedule)
        margin_source = _submission_source(
            margin_oriented,
            contest_ids,
            entry_name="TDNet corrected-F6 margin consensus",
            precision="full_float64",
        )
        margin_source = _restore_frozen_margin_text(margin_source)
        margin_path = output_dir / "2026_current_slate_margin_consensus_source.csv"
        margin_long_path = output_dir / "2026_current_slate_margin_model_predictions.csv.gz"
        schedule_output = output_dir / "2026_current_slate_schedule.csv"
        margin_source.to_csv(margin_path, index=False)
        margin_long.to_csv(margin_long_path, index=False, compression="gzip")
        schedule.to_csv(schedule_output, index=False)

    kickoff_column = "start_date" if "start_date" in schedule else "game_start_time_utc"
    kickoffs = pd.to_datetime(schedule[kickoff_column], utc=True, errors="raise")
    provider_counts = {
        str(int(week)): int(count)
        for week, count in schedule.groupby("week").size().items()
    }
    provenance_inputs = [
        schedule_path,
        cfbd_slate,
        DEFAULT_MARGIN_INVENTORY,
        *[path for path in PRESEASON_INPUTS if path.exists()],
    ]
    manifest: dict[str, Any] = {
        "schema": "tdnet-cfbd-margin-entry-generation-v1",
        "status": "generated_not_exported_not_submitted",
        "generated_at_utc": generated_at.isoformat(),
        "cfbd_slate_fetched_at_utc": fetched_at,
        "season": 2026,
        "provider_week_game_counts": provider_counts,
        "contest_game_count": len(contest_ids),
        "earliest_kickoff_utc": kickoffs.min().isoformat(),
        "latest_kickoff_utc": kickoffs.max().isoformat(),
        "all_games_before_kickoff_at_generation": True,
        "training_performed": False,
        "calibration_performed": False,
        "data_refresh_performed": False,
        "market_data_used": False,
        "cfbd_orientation_transformations": [
            {
                "game_id": int(row.get("id", row.get("game_id"))),
                "local_away_team": str(row["away_team"]),
                "local_home_team": str(row["home_team"]),
                "cfbd_away_team": str(row["cfbd_away_team"]),
                "cfbd_home_team": str(row["cfbd_home_team"]),
                "formula": "cfbd-oriented home margin = -local home margin",
            }
            for row in schedule.loc[
                schedule["cfbd_orientation_reversed"].astype(bool)
            ].to_dict("records")
        ],
        "fingerprint_policy": (
            "frozen canonical fingerprint plus schedule-driven 2026 preseason state; "
            "same-team carry-forward, conference-mean cold start, and existing local "
            "2026 talent/returning overlays"
        ),
        "freeze_gate": margin_gate,
        "entry": {
            "name": "TDNet corrected-F6 margin consensus",
            "roster": "corrected-F6 wide-margin operational roster",
            "model_count": 36,
            "submission_precision": "full float64 decimal representation; no rounding",
            "source_path": str(margin_path),
            "source_sha256": sha256_file(margin_path),
            "model_predictions_path": str(margin_long_path),
            "model_predictions_sha256": sha256_file(margin_long_path),
            "checkpoint_sha256s": margin_checkpoint_hashes,
            "fingerprints": margin_fingerprints,
            "weekly_reports": margin_reports,
        },
        "inputs": {
            str(path.resolve()): sha256_file(path)
            for path in provenance_inputs
        },
        "contest_schedule": {
            "path": str(schedule_output),
            "sha256": sha256_file(schedule_output),
        },
    }
    manifest_path = output_dir / "2026_current_slate_generation_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a complete CFBD source table from the frozen TDNet "
            "corrected-F6 margin roster, "
            "after exact Week 0 freeze reproduction. This never submits."
        )
    )
    parser.add_argument("--cfbd-slate", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(
        cfbd_slate=args.cfbd_slate,
        schedule_path=args.schedule,
        output_dir=args.output_dir,
    )
    print("Generated the margin-wide CFBD source table; no submission was attempted.")
    print(f"Contest games: {manifest['contest_game_count']}")
    print(f"Provider weeks: {manifest['provider_week_game_counts']}")
    print("Margin Week 0 freeze gate: PASS (published consensus restored exactly)")
    print(f"Manifest: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
