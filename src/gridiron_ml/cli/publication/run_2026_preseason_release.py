#!/usr/bin/env python3
"""Publish the 2026 preseason package once the official AP Top 25 exists."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import pandas as pd

from gridiron_ml.cli._paths import project_root
from gridiron_ml.cli.publication.run_2026_preseason_rehearsal import materialize_runtime_inventory
from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.publication.polls import load_ap_top25
from gridiron_ml.publication.roster_poll import build_frozen_roster_poll
from gridiron_ml.publication.weekly import build_weekly_blog_package


ROOT = project_root()
MARKET_BEARING_TIERS = frozenset({"F7", "F8"})


def _load_ignored_cfbd_key() -> None:
    """Load only CFBD_API_KEY from the ignored repo-local .env when needed."""
    if os.environ.get("CFBD_API_KEY"):
        return
    path = ROOT / ".env"
    if not path.exists():
        raise EnvironmentError("CFBD_API_KEY is unset and the ignored repository .env is missing.")
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "CFBD_API_KEY":
            os.environ["CFBD_API_KEY"] = value.strip().strip("\"'")
            break
    if not os.environ.get("CFBD_API_KEY"):
        raise EnvironmentError("CFBD_API_KEY is not defined in the ignored repository .env.")


def _default_scientific_inventory() -> Path:
    local = ROOT / "models/season_2026_scientific_f0_f8_bundle/final_model_inventory.csv"
    if local.exists():
        return local
    configured = os.environ.get("TDNET_SCIENTIFIC_ROSTER_INVENTORY")
    if configured:
        return Path(configured)
    return Path(
        "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
        "scientific_roster_refits/f0_f8_margin_through_2025_v1/final_model_inventory.csv"
    )


def _scientific_prediction_inventory(source: Path, output: Path) -> Path:
    frame = pd.read_csv(source)
    expected = {
        (tier, level)
        for tier in [f"F{i}" for i in range(9)]
        for level in ("M1", "M2", "M3", "M4", "M5", "M10")
    }
    observed = set(zip(frame["feature_config"].astype(str), frame["model_level"].astype(str)))
    if len(frame) != 54 or observed != expected:
        raise ValueError("Scientific source must contain exactly 54 F0-F8 x M-level cells.")
    market = frame["market_bearing"].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    selected = frame.loc[~market & ~frame["feature_config"].astype(str).isin(MARKET_BEARING_TIERS)].copy()
    if len(selected) != 42:
        raise ValueError("Preseason scientific publication requires exactly 42 market-free F0-F6 cells.")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    return output


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "publication/2026/preseason")
    parser.add_argument("--scientific-inventory", type=Path, default=_default_scientific_inventory())
    parser.add_argument(
        "--wide-inventory", type=Path,
        default=Path(
            "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
            "corrected_f6_wide_margin_roster/through_2025_v1/final_model_inventory.csv"
        ),
    )
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.skip_refresh:
        _load_ignored_cfbd_key()
        subprocess.run(
            [
                "python", "-m", "gridiron_ml.pipeline.fetch.cfbd_fetch_v2",
                "--config", "configs/fetch/preseason_2026_watch.yaml", "--refresh",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            check=True,
        )

    rankings_path = ROOT / "data/raw/cfbd/v2/rankings/2026.parquet"
    schedule_path = ROOT / "data/raw/cfbd/v2/games/2026.parquet"
    ap = load_ap_top25(rankings_path, season=2026, week=1)
    if len(ap) != 25:
        available = []
        if rankings_path.exists():
            rankings = pd.read_parquet(rankings_path)
            for polls in rankings.get("polls", []):
                values = polls.tolist() if hasattr(polls, "tolist") else polls
                available.extend(str(item.get("poll")) for item in (values or []))
        status = {
            "status": "waiting_for_official_ap_top25",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "available_polls": sorted(set(available)),
        }
        print(json.dumps(status, indent=2, sort_keys=True))
        return 2

    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Preseason release output already exists: {output}; pass --overwrite explicitly.")
    output.mkdir(parents=True, exist_ok=True)
    ap_path = output / "inputs/ap_top25.csv"
    ap_path.parent.mkdir(parents=True, exist_ok=True)
    ap.to_csv(ap_path, index=False)

    scientific_source = args.scientific_inventory.resolve()
    scientific_market_free = _scientific_prediction_inventory(
        scientific_source, output / "inputs/scientific_market_free_inventory.csv"
    )
    roster_sources = {
        "scientific_market_free": scientific_market_free,
        "margin_wide_f6": args.wide_inventory.resolve(),
    }
    reports = {}
    for name, source in roster_sources.items():
        roster_root = output / name
        runtime = materialize_runtime_inventory(source, output / f"inputs/{name}_runtime_inventory.csv")
        poll_result = build_frozen_roster_poll(
            runtime,
            season=2026,
            week=0,
            output_dir=roster_root / "tdnet_preseason_top25",
            project_root=ROOT,
            logo_dir=ROOT / "data/meta/logos/by_team",
            objective="margin",
            reference_poll=ap,
            reference_label="AP",
        )
        weekly = build_weekly_blog_package(
            project_root=ROOT,
            season=2026,
            week=1,
            model_inventory_path=runtime,
            schedule_snapshot_path=schedule_path,
            output_root=roster_root / "week_01_predictions",
            top25_path=ap_path,
            top25_label="AP Top 25",
            ap_top25_path=ap_path,
            tdnet_top25_path=roster_root / "tdnet_preseason_top25/tdnet_top25.csv",
            logo_dir=ROOT / "data/meta/logos/by_team",
            include_collapsed_models=False,
            schedule_driven_matchups=True,
        )
        reports[name] = {
            "inventory": str(source),
            "runtime_inventory_sha256": sha256_file(runtime),
            "poll_model_count": int(poll_result["ballots"]["ballot_model"].nunique()),
            "poll_failures": int(len(poll_result["failures"])),
            "prediction_model_count": int(weekly["all_model_predictions"]["model_name"].nunique()),
            "predicted_game_count": int(len(weekly["all_games"])),
            "prediction_failures": int(weekly["manifest"]["model_failure_count"]),
        }

    manifest = {
        "status": "ready_for_owner_review_and_publication",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": 2026,
        "ap_poll_rows": 25,
        "ap_poll_sha256": sha256_file(ap_path),
        "schedule_sha256": sha256_file(schedule_path),
        "scientific_roster": "54 research cells; 42 market-free F0-F6 cells used for predictions/polls",
        "market_bearing_exclusions": ["F7", "F8"],
        "wide_margin_fingerprint": "F6",
        "preseason_data_policy": {
            "talent": "2025 carry-forward when 2026 CFBD talent is unavailable",
            "returning_production": "use live 2026 CFBD values when available",
            "raw_cfbd_cache_modified_by_fallback": False,
        },
        "reports": reports,
    }
    (output / "PRESEASON_RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "PRESEASON_BLOG_DRAFT.md").write_text(
        "# TDNet 2026 preseason poll and Week 1 outlook\n\n"
        "TDNet's preseason rankings are produced from models trained only through 2025. "
        "The official AP Top 25 is used as an external comparison, never as a TDNet model input.\n\n"
        "The scientific panel contains every prespecified M architecture at every fingerprint. "
        "Official predictions and polls use only the 42 market-free F0-F6 cells; F7 and F8 are "
        "withheld because they contain market information. The operational wide-margin roster "
        "uses corrected F6, the strongest market-free fingerprint under the prespecified margin-MAE criterion.\n\n"
        "The generated Top 25 table, consensus-spread graphic, descriptive AP-peer fingerprint graphic, "
        "and full ballot matrix appear in that order beside each roster poll. The fingerprint graphic is "
        "a descriptive comparison aid, not a causal explanation of AP voters or unencoded coaching changes.\n\n"
        "The generated comparison tables, ballot figures, Week 1 matchups, and model-failure "
        "reports live beside this draft and should be reviewed before public posting.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
