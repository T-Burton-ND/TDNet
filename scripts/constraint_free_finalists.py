#!/usr/bin/env python3
"""Select paired F18/F19 finalists using only completed historical receipts.

Every ranked candidate has the same 2022–2025 rolling-origin folds. The
one-year reducer screen is accounted for but cannot compete with confirmed
four-year candidates. This module never reads a 2026 feature or outcome.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from constraint_free_search import digest, write_json
from gridiron_ml.experiments.constraint_free import ROSTER

ROOT = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/broad_search")
TIERS = ("F18", "F19")
YEARS = (2022, 2023, 2024, 2025)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _fold_candidate(row: dict, *, track: str, path: Path, manifest: Path) -> dict:
    if row.get("status") != "success":
        raise ValueError(f"Candidate did not succeed: {path}")
    if tuple(f["year"] for f in row.get("folds", [])) != YEARS:
        raise ValueError(f"Candidate lacks common four-year folds: {path}")
    if row["tier"] not in TIERS or row["architecture"] not in ROSTER:
        raise ValueError(f"Candidate identity outside paired roster: {path}")
    if track in ("broad_stage1", "broad_stage2", "extra_stage2"):
        if row.get("manifest_sha256") != digest(manifest):
            raise ValueError(f"Receipt/manifest mismatch: {path}")
    return {
        "tier": row["tier"], "architecture": row["architecture"],
        "track": track, "receipt": str(path), "receipt_sha256": digest(path),
        "manifest": str(manifest), "manifest_sha256": digest(manifest),
        "representation": row.get("representation"),
        "genome": row.get("genome"), "genome_hash": row.get("genome_hash"),
        "mean_mae": float(row["mean_mae"]),
        "mean_brier": float(row["mean_brier"]),
        "worst_year_mae": float(row["worst_year_mae"]),
        "model_fits": int(row["model_fits"]),
    }


def _checked_tasks(manifest_path: Path, result_folder: Path) -> tuple[list[dict], dict]:
    manifest = _read(manifest_path)
    rows = []
    for task_id, task in enumerate(manifest["tasks"], start=1):
        path = result_folder / f"task_{task_id:03d}.json"
        if not path.exists():
            raise FileNotFoundError(f"Incomplete search: {path}")
        row = _read(path)
        if any(row.get(k) != value for k, value in task.items()):
            raise ValueError(f"Task identity mismatch: {path}")
        if row.get("manifest_sha256") != digest(manifest_path):
            raise ValueError(f"Task receipt/manifest mismatch: {path}")
        rows.append((path, row))
    return rows, manifest


def collect(root: Path = ROOT) -> tuple[list[dict], dict]:
    finalists_pool: list[dict] = []
    accounting: dict = {}
    for track, folder in (("broad_stage1", root / "retry1/stage1"),
                          ("broad_stage2", root / "retry1/stage2"),
                          ("extra_stage2", root / "extra_reducers/stage2")):
        manifest_path = folder / "manifest.json"
        rows, manifest = _checked_tasks(manifest_path, folder / "results")
        if track == "extra_stage2" and tuple(manifest["years_to_score"]) != YEARS:
            raise ValueError("Extra reducer confirmations use different folds")
        if track.startswith("broad") and tuple(manifest["development_years"]) != YEARS:
            raise ValueError("Broad candidates use different folds")
        accounting[track] = {}
        for tier in TIERS:
            cohort = [row for _, row in rows if row["tier"] == tier]
            accounting[track][tier] = {
                "candidates": len(cohort),
                "successful": sum(row["status"] == "success" for row in cohort),
                "failed": sum(row["status"] == "failed" for row in cohort),
                "model_fits": sum(int(row.get("model_fits", 0)) for row in cohort),
            }
        finalists_pool.extend(_fold_candidate(row, track=track, path=path,
                                              manifest=manifest_path)
                              for path, row in rows if row["status"] == "success")

    screen_folder = root / "extra_reducers/stage1"
    screen, screen_manifest = _checked_tasks(screen_folder / "manifest.json",
                                             screen_folder / "results")
    if tuple(screen_manifest["years_to_score"]) != (2025,):
        raise ValueError("Extra reducer screen year changed")
    accounting["extra_stage1"] = {}
    for tier in TIERS:
        cohort = [row for _, row in screen if row["tier"] == tier]
        accounting["extra_stage1"][tier] = {
            "candidates": len(cohort),
            "successful": sum(row["status"] == "success" for row in cohort),
            "failed": sum(row["status"] == "failed" for row in cohort),
            "model_fits": sum(int(row.get("model_fits", 0)) for row in cohort),
        }

    for track in ("ga", "bo"):
        folder = root / "advanced" / track
        manifest_path = folder / "manifest.json"
        manifest = _read(manifest_path)
        if tuple(manifest["development_years"]) != YEARS or manifest["track"] != track:
            raise ValueError(f"Advanced search manifest changed: {manifest_path}")
        accounting[track] = {}
        for tier in TIERS:
            tier_rows = []
            for architecture in ROSTER:
                task = {"tier": tier, "architecture": architecture}
                if task not in manifest["tasks"]:
                    raise ValueError(f"Missing advanced task {track} {task}")
                task_folder = folder / f"{tier}_{architecture}"
                summary = _read(task_folder / "summary.json")
                if summary.get("manifest_sha256") != digest(manifest_path):
                    raise ValueError(f"Advanced summary/manifest mismatch: {task_folder}")
                paths = sorted((task_folder / "candidates").glob("*.json"))
                if len(paths) != manifest["candidate_budget_per_task"]:
                    raise ValueError(f"Incomplete advanced candidates: {task_folder}")
                rows = [(path, _read(path)) for path in paths]
                for path, row in rows:
                    if row["tier"] != tier or row["architecture"] != architecture:
                        raise ValueError(f"Advanced candidate identity mismatch: {path}")
                    if row.get("genome_hash") != path.stem:
                        raise ValueError(f"Advanced candidate genome/path mismatch: {path}")
                if sorted(row["trial"] for _, row in rows) != list(
                        range(1, manifest["candidate_budget_per_task"] + 1)):
                    raise ValueError(f"Advanced candidate trial sequence is incomplete: {task_folder}")
                if summary["candidates"] != len(rows) or summary["successful_candidates"] != sum(
                        row["status"] == "success" for _, row in rows):
                    raise ValueError(f"Advanced summary count mismatch: {task_folder}")
                successful = [row for _, row in rows if row["status"] == "success"]
                if not successful:
                    raise ValueError(f"No successful advanced candidate: {task_folder}")
                best = min(successful, key=lambda row: (row["mean_mae"], row["mean_brier"],
                                                       row["worst_year_mae"]))
                declared = next((row for row in successful
                                 if row["genome_hash"] == summary["best_genome_hash"]), None)
                if declared is None or (declared["mean_mae"], declared["mean_brier"],
                                        declared["worst_year_mae"]) != (
                                            best["mean_mae"], best["mean_brier"],
                                            best["worst_year_mae"]):
                    raise ValueError(f"Advanced summary winner differs from receipts: {task_folder}")
                tier_rows.extend(rows)
                finalists_pool.extend(_fold_candidate(row, track=track, path=path,
                                                      manifest=manifest_path)
                                      for path, row in rows if row["status"] == "success")
            accounting[track][tier] = {
                "candidates": len(tier_rows),
                "successful": sum(row["status"] == "success" for _, row in tier_rows),
                "failed": sum(row["status"] == "failed" for _, row in tier_rows),
                "model_fits": sum(int(row.get("model_fits", 0)) for _, row in tier_rows),
            }
    return finalists_pool, accounting


def select(root: Path = ROOT) -> dict:
    candidates, accounting = collect(root)
    winners = []
    residual_comparison = []
    for tier in TIERS:
        for architecture in ROSTER:
            cell = [row for row in candidates if row["tier"] == tier
                    and row["architecture"] == architecture]
            if not cell:
                raise ValueError(f"No valid candidates for {tier} {architecture}")
            ranked = sorted(cell, key=lambda row: (row["mean_mae"], row["mean_brier"],
                                                    row["worst_year_mae"], row["track"],
                                                    row["receipt_sha256"]))
            winners.append({**ranked[0], "candidate_count": len(cell)})
            if tier == "F19":
                for mode in ("direct", "residual"):
                    subset = [row for row in ranked if row["genome"] is not None
                              and row["genome"]["target_mode"] == mode]
                    if not subset:
                        raise ValueError(f"F19 lacks {mode} candidate: {architecture}")
                    residual_comparison.append({"architecture": architecture,
                                                "mode": mode, **subset[0]})
    for track, tiers in accounting.items():
        if tiers["F18"]["candidates"] != tiers["F19"]["candidates"]:
            raise ValueError(f"Asymmetric {track} candidate budget")
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "minimum equal-year mean MAE; then Brier; then worst-year MAE; then stable identity",
        "years": list(YEARS), "universe": "F16_A_plus_distinct_F12_BC",
        "winners": winners, "f19_direct_residual_development": residual_comparison,
        "accounting": accounting,
        "development_warning": "2022–2025 scores were used for selection and are not unbiased estimates",
        "historical_f19_market_warning": "historical quote-level timestamps unverified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "finalists.json"
    if output.exists():
        raise FileExistsError(f"Selection receipt is immutable: {output}")
    result = select(args.root)
    write_json(output, result)
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "finalists": len(result["winners"]),
                      "accounting": result["accounting"]}, indent=2))


if __name__ == "__main__":
    main()
