"""Audit the three named weekly rosters without promoting partial research."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from gridiron_ml.f20.scientific import MODEL_IDS as F20_MODEL_IDS

MODEL_IDS = frozenset({"M1", "M2", "M3", "M4", "M5", "M10"})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Frozen F20 component changed: {path}")
        return
    path.write_bytes(content)
    path.chmod(0o444)


def write_f20_prediction_component(*, output: Path, archive: Path, season: int,
                                   week: int, cutoff_utc: str) -> Path:
    """Bind the six issued F20 cells to the exact frozen weekly game slate."""
    if season != 2026:
        raise ValueError("F20 component is currently fitted for 2026 only")
    predictions = pd.read_parquet(output / "frozen_bundle/public/predictions.parquet",
                                  columns=["season", "week", "game_id"])
    if predictions.empty or not predictions.season.eq(season).all() or not predictions.week.eq(week).all():
        raise ValueError("Frozen F20 slate differs from publication bundle")
    game_ids = sorted(set(predictions.game_id.astype(int)))
    rows, consensus, sources = [], [], {}
    for game_id in game_ids:
        pointer_path = archive / "forecasts" / str(game_id) / "official.json"
        pointer = json.loads(pointer_path.read_text())
        record_path = Path(pointer["record_file"])
        record = json.loads(record_path.read_text())
        if (pointer.get("record_sha256") != _sha(record_path)
                or pointer.get("canonical_variant") != "F20"
                or pointer.get("weekly_cutoff_utc") != cutoff_utc
                or record.get("game_id") != game_id
                or record.get("season") != season or record.get("week") != week
                or record.get("issuance_class") != "eligible_weekly"
                or record.get("canonical_variant") != "F20"
                or record.get("weekly_cutoff_utc") != cutoff_utc):
            raise ValueError(f"F20 official record differs from frozen slate: {game_id}")
        probabilities = record["home_win_probability"]
        margins = record.get("pred_home_margin", {})
        if set(probabilities) != {*F20_MODEL_IDS, "F20"} or set(margins) != {*F20_MODEL_IDS, "F20"}:
            raise ValueError(f"F20 six-model panel is incomplete: {game_id}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               or not 0 <= value <= 1 for value in probabilities.values()):
            raise ValueError(f"F20 probability is invalid: {game_id}")
        if abs(probabilities["F20"] - sum(probabilities[model] for model in F20_MODEL_IDS) / 6) > 1e-12:
            raise ValueError(f"F20 consensus differs from its six model cells: {game_id}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in margins.values()):
            raise ValueError(f"F20 margin is invalid: {game_id}")
        if abs(margins["F20"] - sum(margins[model] for model in F20_MODEL_IDS) / 6) > 1e-10:
            raise ValueError(f"F20 margin consensus differs from its six model cells: {game_id}")
        for model_id in F20_MODEL_IDS:
            rows.append({"game_id": game_id, "fingerprint": "F20", "model_id": model_id,
                         "pred_home_win_probability": probabilities[model_id],
                         "pred_home_margin": margins[model_id]})
        consensus.append({"game_id": game_id, "fingerprint": "F20",
                          "pred_home_win_probability": probabilities["F20"],
                          "pred_home_margin": margins["F20"]})
        sources[str(game_id)] = {"official_pointer_sha256": _sha(pointer_path),
                                 "forecast_record_sha256": _sha(record_path)}
    component = output / "scientific/expanded_f0_f20/f20_component"
    cells_path = component / "f20_model_predictions.csv"
    consensus_path = component / "f20_consensus_predictions.csv"
    _write_immutable(cells_path, pd.DataFrame(rows).to_csv(index=False).encode())
    _write_immutable(consensus_path, pd.DataFrame(consensus).to_csv(index=False).encode())
    manifest = {"season": season, "week": week, "cutoff_utc": cutoff_utc,
                "games": len(game_ids), "model_cells": len(F20_MODEL_IDS),
                "models": list(F20_MODEL_IDS), "game_sources": sources,
                "model_predictions_sha256": _sha(cells_path),
                "consensus_predictions_sha256": _sha(consensus_path)}
    _write_immutable(component / "manifest.json",
                     (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return component / "manifest.json"


def _scientific_cells(path: Path, tiers: set[str]) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=["game_id", "fingerprint", "model_name"])
    games = set(frame["game_id"])
    if not games or set(frame["fingerprint"]) != tiers:
        raise ValueError(f"Scientific tier coverage changed: {path}")
    cells = frame.groupby(["fingerprint", "model_name"])["game_id"].agg(["nunique", "size"])
    if len(cells) != 6 * len(tiers) or not cells["nunique"].eq(len(games)).all() or not cells["size"].eq(len(games)).all():
        raise ValueError(f"Scientific model/game coverage changed: {path}")
    if any(not MODEL_IDS.issuperset({str(name).rsplit("_", 1)[-1]}) for _, name in cells.index):
        raise ValueError(f"Scientific architecture changed: {path}")
    return len(games), len(cells)


def write_weekly_roster_registry(
    *, output: Path, operations: Path, season: int, week: int,
    f20_research: dict | None = None,
) -> Path:
    """Record exactly three roster identities and actual research readiness."""
    scientific = output / "scientific"
    original = scientific / "scientific_all_game_predictions.csv"
    expanded_base = scientific / "expanded_f0_f20" / "f0_f8_component" / "scientific_all_game_predictions.csv"
    original_games, original_cells = _scientific_cells(original, {f"F{i}" for i in range(7)})
    expanded_games, expanded_cells = _scientific_cells(expanded_base, {f"F{i}" for i in range(9)})
    if original_games != expanded_games:
        raise ValueError("Frozen and expanded scientific components have different game coverage")
    frozen_bundle = output / "frozen_bundle" / "public" / "prediction_manifest.json"
    if not frozen_bundle.is_file():
        raise FileNotFoundError(frozen_bundle)
    f20_status = None if f20_research is None else f20_research.get("status")
    f20_result = f20_research.get("result", {}) if f20_research else {}
    canonical_issued = f20_status == "succeeded" and f20_result.get("status") == "captured_with_canonical_forecasts"
    f20_component = scientific / "expanded_f0_f20/f20_component/manifest.json"
    component_manifest = json.loads(f20_component.read_text()) if f20_component.is_file() else None
    if component_manifest and (component_manifest["season"] != season
                               or component_manifest["week"] != week
                               or component_manifest["games"] != original_games
                               or component_manifest["model_cells"] != len(F20_MODEL_IDS)
                               or _sha(f20_component.parent / "f20_model_predictions.csv") !=
                               component_manifest["model_predictions_sha256"]):
        raise ValueError("F20 component does not cover frozen scientific slate")
    registry = {
        "season": season,
        "week": week,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rosters": [
            {"id": "margin_wide_publication", "role": "canonical_publication",
             "status": "frozen", "bundle_manifest": str(frozen_bundle)},
            {"id": "original_frozen_scientific_f0_f6", "role": "scientific",
             "status": "complete", "games": original_games, "model_cells": original_cells,
             "predictions": str(original)},
            {"id": "expanded_scientific_f0_f20", "role": "scientific_research",
             "status": "incomplete_pending_f9_f19_prospective_path",
             "base_component": {"tiers": "F0-F8", "games": expanded_games,
                                "model_cells": expanded_cells, "predictions": str(expanded_base)},
             "missing_margin_tiers": [f"F{i}" for i in range(9, 20)],
             "f20": {"canonical_variant": "F20", "models": list(F20_MODEL_IDS),
                     "required_model_cells": len(F20_MODEL_IDS),
                     "completed_model_cells": len(F20_MODEL_IDS) if component_manifest else 0,
                     "component_manifest": str(f20_component) if component_manifest else None,
                     "run_status": f20_status,
                     "research_run_report": (
                         f20_result.get("run_report")
                         if f20_research else None
                     ),
                     "canonical_issued": canonical_issued,
                     "output_type": "home_win_probability_and_predicted_margin",
                     "included_in_margin_consensus": False},
             "complete_consensus_available": False},
        ],
    }
    operations.mkdir(parents=True, exist_ok=True)
    path = operations / "weekly_rosters.json"
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return path
