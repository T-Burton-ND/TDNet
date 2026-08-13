"""Immutable weekly prediction bundles, verification, and no-mutation scoring."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import gzip
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from gridiron_ml.pipeline.schemas import (
    validate_public_prediction_table,
    validate_scored_prediction_table,
)
from gridiron_ml.pipeline.schemas.validators import PUBLIC_PREDICTION_REQUIRED_COLUMNS


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def prepare_public_prediction_table(
    predictions: pd.DataFrame,
    *,
    prediction_deadline_utc: str,
    feature_manifest_sha256: str,
    data_snapshot_sha256: str,
    schedule_snapshot_sha256: str,
    git_commit: str,
    pipeline_version: str,
    environment_lock_sha256: str,
    kickoff_time_confirmed: bool,
) -> pd.DataFrame:
    """Add required provenance and stable prediction IDs to canonical long rows."""
    frame = predictions.copy()
    aliases = {
        "pred_margin": "pred_home_margin",
        "pred_proba_home_win": "pred_home_win_probability",
        "keys_game_id": "game_id",
        "next_game_id": "game_id",
        "keys_season": "season",
        "next_week": "week",
        "keys_team_home": "home_team",
        "keys_team_away": "away_team",
    }
    for source, target in aliases.items():
        if target not in frame and source in frame:
            frame[target] = frame[source]
    now = datetime.now(timezone.utc).isoformat()
    frame["created_at_utc"] = frame.get("created_at_utc", now)
    frame["prediction_deadline_utc"] = prediction_deadline_utc
    frame["feature_manifest_sha256"] = feature_manifest_sha256
    frame["data_snapshot_sha256"] = data_snapshot_sha256
    frame["schedule_snapshot_sha256"] = schedule_snapshot_sha256
    frame["git_commit"] = git_commit
    frame["pipeline_version"] = pipeline_version
    frame["environment_lock_sha256"] = environment_lock_sha256
    frame["kickoff_time_confirmed"] = bool(kickoff_time_confirmed)
    frame["pred_winner"] = frame.get("pred_winner", frame["home_team"].where(
        pd.to_numeric(frame["pred_home_win_probability"], errors="coerce") >= 0.5,
        frame["away_team"],
    ))
    frame["confidence"] = frame.get(
        "confidence",
        (pd.to_numeric(frame["pred_home_win_probability"], errors="coerce") - 0.5).abs() * 2,
    )
    optional_defaults = {
        "pred_total": np.nan,
        "model_rank_home": np.nan,
        "model_rank_away": np.nan,
        "top25_rank_home": np.nan,
        "top25_rank_away": np.nan,
        "vegas_spread_close_as_of_prediction": frame.get("market_spread_close", np.nan),
        "vegas_total_close_as_of_prediction": frame.get("market_over_under", np.nan),
        "vegas_home_win_probability_as_of_prediction": frame.get("market_win_probability", np.nan),
        "ap_rank_home": np.nan,
        "ap_rank_away": np.nan,
        "coaches_rank_home": np.nan,
        "coaches_rank_away": np.nan,
        "cfp_rank_home": np.nan,
        "cfp_rank_away": np.nan,
    }
    for column, default in optional_defaults.items():
        if column not in frame:
            frame[column] = default
    for column, default in [
        ("neutral_site", False),
        ("conference_game", False),
        ("season_type", "regular"),
        ("objective", "margin"),
    ]:
        if column not in frame:
            frame[column] = default
    if "checkpoint_sha256" not in frame:
        raise ValueError("Canonical predictions require checkpoint_sha256.")
    identifiers = []
    for _, row in frame.iterrows():
        payload = "|".join(
            map(
                str,
                [
                    row["season"],
                    row["week"],
                    row["game_id"],
                    row["model_name"],
                    row["objective"],
                    row["checkpoint_sha256"],
                    row["created_at_utc"],
                ],
            )
        )
        identifiers.append(sha256(payload.encode("utf-8")).hexdigest())
    frame["prediction_id"] = identifiers
    required_order = list(PUBLIC_PREDICTION_REQUIRED_COLUMNS) + ["kickoff_time_confirmed"]
    extra = [column for column in frame.columns if column not in required_order]
    return validate_public_prediction_table(frame.loc[:, required_order + extra])


def git_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_tree": run("rev-parse", "HEAD^{tree}"),
        "git_dirty": bool(run("status", "--porcelain")),
    }


def build_prediction_bundle(
    predictions: pd.DataFrame,
    *,
    output_root: str | Path,
    project_root: str | Path,
    supporting_files: dict[str, str | Path] | None = None,
    metadata: dict[str, Any] | None = None,
    allow_dirty_code: bool = False,
) -> dict[str, Any]:
    """Create a new content-hashed prediction bundle; never overwrite one."""
    frame = validate_public_prediction_table(predictions.copy())
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty prediction bundle: {root}")
    public = root / "public"
    private = root / "private"
    verification = root / "verification"
    for directory in [public, private, verification]:
        directory.mkdir(parents=True, exist_ok=True)
    state = git_state(project_root)
    if state["git_dirty"] and not allow_dirty_code:
        raise RuntimeError("Refusing to build a public bundle from a dirty worktree.")

    parquet_path = public / "predictions.parquet"
    csv_path = public / "predictions.csv.gz"
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False, compression="gzip")
    copied = {}
    for name, source in dict(supporting_files or {}).items():
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Supporting bundle file does not exist: {source_path}")
        destination = private / Path(name).name
        shutil.copy2(source_path, destination)
        copied[str(destination.relative_to(root))] = destination

    files = {
        str(path.relative_to(root)): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in [parquet_path, csv_path, *copied.values()]
    }
    manifest = {
        "bundle_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_rows": len(frame),
        "game_count": int(frame["game_id"].nunique()),
        "model_count": int(frame["model_name"].nunique()),
        **state,
        **dict(metadata or {}),
        "files": files,
    }
    manifest["manifest_sha256"] = sha256(canonical_json_bytes(manifest)).hexdigest()
    manifest_path = public / "prediction_manifest.json"
    _atomic_json(manifest_path, manifest)
    readme = public / "README.md"
    readme.write_text(
        "# TDNet Weekly Prediction Bundle\n\n"
        f"Rows: {len(frame)}  \n"
        f"Games: {manifest['game_count']}  \n"
        f"Models: {manifest['model_count']}  \n"
        f"Manifest SHA-256: `{manifest['manifest_sha256']}`\n",
        encoding="utf-8",
    )
    return {"root": root, "manifest": manifest, "manifest_path": manifest_path}


def verify_prediction_bundle(bundle_root: str | Path, *, require_clean_commit=False) -> dict[str, Any]:
    """Recompute hashes and prediction-time guardrails for a frozen bundle."""
    root = Path(bundle_root)
    manifest_path = root / "public" / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256")
    actual = sha256(canonical_json_bytes(manifest)).hexdigest()
    failures = []
    if claimed != actual:
        failures.append("manifest self-hash mismatch")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.exists():
            failures.append(f"missing file: {relative}")
            continue
        if sha256_file(path) != expected["sha256"]:
            failures.append(f"hash mismatch: {relative}")
        if path.stat().st_size != int(expected["size_bytes"]):
            failures.append(f"size mismatch: {relative}")
    predictions = pd.read_parquet(root / "public" / "predictions.parquet")
    try:
        validate_public_prediction_table(predictions)
    except Exception as exc:
        failures.append(f"prediction schema: {exc}")
    if require_clean_commit and manifest.get("git_dirty"):
        failures.append("bundle records a dirty source tree")
    return {
        "valid": not failures,
        "manifest_sha256": claimed,
        "prediction_rows": len(predictions),
        "failures": failures,
    }


def score_prediction_bundle(
    bundle_root: str | Path,
    results: pd.DataFrame,
    *,
    output_root: str | Path,
) -> dict[str, pd.DataFrame]:
    """Score frozen predictions into a separate directory without mutation."""
    bundle = Path(bundle_root)
    verification = verify_prediction_bundle(bundle)
    if not verification["valid"]:
        raise ValueError(f"Cannot score invalid bundle: {verification['failures']}")
    predictions_path = bundle / "public" / "predictions.parquet"
    before_hash = sha256_file(predictions_path)
    predictions = pd.read_parquet(predictions_path)
    required_results = {"game_id", "home_points", "away_points"}
    missing = required_results - set(results.columns)
    if missing:
        raise ValueError(f"Results missing columns: {sorted(missing)}")
    final = results.loc[:, sorted(required_results)].drop_duplicates("game_id")
    scored = predictions.merge(final, on="game_id", how="left", validate="many_to_one")
    scored["actual_home_margin"] = pd.to_numeric(scored["home_points"], errors="coerce") - pd.to_numeric(
        scored["away_points"], errors="coerce"
    )
    scored["actual_home_win"] = scored["actual_home_margin"] > 0
    scored["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
    scored["winner_correct"] = scored["pred_winner"].eq(
        scored["home_team"].where(scored["actual_home_win"], scored["away_team"])
    )
    scored["absolute_margin_error"] = (
        pd.to_numeric(scored["pred_home_margin"], errors="coerce") - scored["actual_home_margin"]
    ).abs()
    validate_scored_prediction_table(scored)
    metrics = (
        scored.groupby(["model_name", "objective"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            winner_accuracy=("winner_correct", "mean"),
            margin_mae=("absolute_margin_error", "mean"),
        )
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output / "scored_predictions.parquet", index=False)
    metrics.to_csv(output / "scorecard.csv", index=False)
    if sha256_file(predictions_path) != before_hash:
        raise RuntimeError("Frozen prediction file changed during scoring.")
    return {"scored_predictions": scored, "scorecard": metrics}


def _atomic_json(path: Path, payload: dict[str, Any]):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
