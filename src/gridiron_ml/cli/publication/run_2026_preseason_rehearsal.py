#!/usr/bin/env python3
"""Render marked 2026 preseason previews for the scientific and wide rosters."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from gridiron_ml.cli._paths import project_root
from gridiron_ml.publication.bundles import sha256_file
from gridiron_ml.publication.preseason_states import build_preseason_state_frame
from gridiron_ml.publication.roster_poll import build_frozen_roster_poll
from gridiron_ml.publication.weekly import build_weekly_blog_package


ROOT = project_root()
SURROGATE_LABEL = "Coaches Poll (AP surrogate; rehearsal only)"
SCIENTIFIC_INVENTORY = Path(
    "/groups/bsavoie2/tburton2/TDNet/publication_artifacts/"
    "scientific_roster_refits/f0_f8_margin_through_2025_v1/final_model_inventory.csv"
)
ROSTERS = {
    "scientific": SCIENTIFIC_INVENTORY,
    "margin_wide": ROOT / "models/season_2026_wide_margin_frozen_bundle/final_model_inventory.csv",
}


def _records(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else [value]


def extract_coaches_poll(rankings_path: Path, output: Path) -> pd.DataFrame:
    rankings = pd.read_parquet(rankings_path)
    rankings = rankings.loc[pd.to_numeric(rankings.get("season"), errors="coerce").eq(2026)]
    if rankings.empty:
        raise ValueError("No 2026 CFBD ranking snapshots are available.")
    rankings = rankings.sort_values("week")
    rows = []
    for record in rankings.iloc[::-1].to_dict("records"):
        for poll in _records(record.get("polls")):
            if str(poll.get("poll", "")).strip().casefold() != "coaches poll":
                continue
            for rank in _records(poll.get("ranks")):
                rows.append(
                    {
                        "rank": rank.get("rank"),
                        "team": rank.get("school"),
                        "points": rank.get("points"),
                        "first_place_votes": rank.get("firstPlaceVotes"),
                        "conference": rank.get("conference"),
                        "team_id": rank.get("teamId"),
                        "season": record.get("season"),
                        "week": record.get("week"),
                    }
                )
            if rows:
                poll_frame = pd.DataFrame(rows).sort_values("rank").head(25)
                output.parent.mkdir(parents=True, exist_ok=True)
                poll_frame.to_csv(output, index=False)
                return poll_frame
    raise ValueError("The latest 2026 rankings payload does not contain a Coaches Poll.")


def _endpoint_rows(endpoint: str, season: int = 2026) -> int:
    path = ROOT / f"data/raw/cfbd/v2/{endpoint}/{season}.parquet"
    return int(len(pd.read_parquet(path))) if path.exists() else 0


def materialize_runtime_inventory(source: Path, output: Path) -> Path:
    """Resolve portable bundle checkpoint paths without changing the freeze."""
    inventory = pd.read_csv(source)
    resolved = []
    for _, row in inventory.iterrows():
        primary = Path(str(row.get("checkpoint_path", "")))
        if not primary.is_absolute():
            primary = ROOT / primary
        portable_value = row.get("bundle_checkpoint_path")
        portable = None
        if portable_value is not None and not pd.isna(portable_value):
            portable = source.parent / str(portable_value)
        if not primary.exists() and portable is not None and portable.exists():
            primary = portable
        resolved.append(str(primary.resolve()))
    inventory["checkpoint_path"] = resolved
    family = inventory.get(
        "model_family", inventory.get("family", pd.Series("", index=inventory.index))
    ).astype(str).str.casefold()
    explicit_baseline = family.eq("naive")
    if explicit_baseline.any():
        inventory.loc[explicit_baseline, "use_in_weekly_consensus"] = False
        inventory.loc[explicit_baseline, "use_in_tdnet_poll"] = False
    feature_config = inventory.get(
        "feature_config", inventory.get("fingerprint", pd.Series("", index=inventory.index))
    ).astype(str)
    market = inventory.get("market_bearing", pd.Series(False, index=inventory.index))
    market = market.astype(str).str.casefold().isin({"1", "true", "yes", "y"})
    inventory = inventory.loc[~market & ~feature_config.isin({"F7", "F8"})].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output, index=False)
    return output


def _mark_csv(path: Path, roster: str) -> None:
    if not path.exists():
        return
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return
    frame.insert(0, "rehearsal_only", True)
    frame.insert(1, "roster", roster)
    frame.insert(2, "ranking_reference", SURROGATE_LABEL)
    frame.insert(3, "talent_source", "2025 carry-forward; 2026 CFBD unavailable")
    frame.insert(4, "returning_source", "live 2026 CFBD returning production")
    frame.to_csv(path, index=False)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "data/publication/2026/preseason_rehearsal_coaches_surrogate",
    )
    parser.add_argument("--week", type=int, default=1)
    args = parser.parse_args()

    schedule = ROOT / "data/raw/cfbd/v2/games/2026.parquet"
    rankings = ROOT / "data/raw/cfbd/v2/rankings/2026.parquet"
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    coaches_poll = extract_coaches_poll(rankings, output / "inputs/coaches_poll_surrogate.csv")

    reports = {}
    for roster, source_inventory in ROSTERS.items():
        roster_root = output / roster
        inventory = materialize_runtime_inventory(
            source_inventory, output / f"inputs/{roster}_runtime_inventory.csv"
        )
        poll_result = build_frozen_roster_poll(
            inventory,
            season=2026,
            week=0,
            output_dir=roster_root / "week_00_top25",
            project_root=ROOT,
            logo_dir=ROOT / "data/meta/logos/by_team",
            objective="margin",
            reference_poll=coaches_poll,
            reference_label="Coaches (rehearsal)",
        )
        weekly = build_weekly_blog_package(
            project_root=ROOT,
            season=2026,
            week=args.week,
            model_inventory_path=inventory,
            schedule_snapshot_path=schedule,
            output_root=roster_root / f"week_{args.week:02d}_predictions",
            top25_path=output / "inputs/coaches_poll_surrogate.csv",
            top25_label=SURROGATE_LABEL,
            tdnet_top25_path=roster_root / "week_00_top25/tdnet_top25.csv",
            logo_dir=ROOT / "data/meta/logos/by_team",
            include_collapsed_models=False,
            schedule_driven_matchups=True,
        )
        for csv_path in roster_root.rglob("*.csv"):
            _mark_csv(csv_path, roster)
        reports[roster] = {
            "inventory": str(inventory),
            "inventory_sha256": sha256_file(inventory),
            "source_inventory": str(source_inventory),
            "source_inventory_sha256": sha256_file(source_inventory),
            "top25_ballot_models": int(poll_result["ballots"]["ballot_model"].nunique()),
            "top25_model_failures": int(len(poll_result["failures"])),
            "prediction_models": int(weekly["all_model_predictions"]["model_name"].nunique()),
            "predicted_games": int(len(weekly["all_games"])),
            "model_failures": int(weekly["manifest"]["model_failure_count"]),
            "output_root": str(roster_root),
        }

    coverage_source = pd.read_parquet(
        ROOT / "data/experiments/opponent_adjusted_fingerprints/fingerprints/v1_7/canonical_fingerprint.parquet"
    )
    coverage_state = build_preseason_state_frame(coverage_source, season=2026, project_root=ROOT)
    coverage = {
        "teams": int(coverage_state["keys_team"].nunique()),
        "roster_talent_non_null_teams": int(
            coverage_state.loc[coverage_state["roster_talent"].notna(), "keys_team"].nunique()
        ),
        "returning_production_non_null_teams": int(
            coverage_state.loc[
                coverage_state["roster_return_percent_p_p_a"].notna(), "keys_team"
            ].nunique()
        ),
        "coach_career_non_null_teams": int(
            coverage_state.loc[coverage_state["coach_career_seasons"].notna(), "keys_team"].nunique()
        ),
        "missing_talent_teams": sorted(
            coverage_state.loc[coverage_state["roster_talent"].isna(), "keys_team"].astype(str)
        ),
        "missing_returning_teams": sorted(
            coverage_state.loc[
                coverage_state["roster_return_percent_p_p_a"].isna(), "keys_team"
            ].astype(str)
        ),
    }
    metadata = {
        "status": "rehearsal_only_not_for_release",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": 2026,
        "prediction_week": int(args.week),
        "schedule_pairing_policy": "all games with both teams in Week-0 FBS state",
        "schedule_pairing_travel_policy": "game-specific travel set missing; frozen fitted preprocessing applied",
        "ranking_reference": SURROGATE_LABEL,
        "coaches_poll_rows": int(len(coaches_poll)),
        "schedule_path": str(schedule),
        "schedule_sha256": sha256_file(schedule),
        "live_2026_endpoint_rows": {
            endpoint: _endpoint_rows(endpoint)
            for endpoint in ["games", "lines", "pregame_wp", "coaches", "rankings", "talent", "returning"]
        },
        "fallbacks": {
            "talent": "latest 2025 team value carried into derived Week-0 state",
            "returning_production": "live 2026 CFBD returning-production values",
            "raw_2026_cfbd_rows_modified_by_fallback": False,
        },
        "derived_week0_coverage": coverage,
        "next_api_call_config": "configs/fetch/preseason_2026_watch.yaml",
        "reports": reports,
    }
    (output / "REHEARSAL_MANIFEST.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# 2026 preseason rehearsal — not for release\n\n"
        f"Rank reference: **{SURROGATE_LABEL}**.\n\n"
        "CFBD has not published 2026 talent, so the rendered Week-0 states carry forward "
        "each team's latest 2025 talent value. Returning production uses live 2026 rows. "
        "These assumptions are derived data and were not written into the raw CFBD cache.\n\n"
        "Talent and returning-production coverage are reported in the rehearsal manifest. "
        "Missing values remain missing for each fitted model's stored preprocessing to handle.\n\n"
        "The margin-wide preview excludes explicit naive baselines and any learned checkpoint "
        "that collapses to a constant preseason score; exclusions are recorded in its failure tables.\n\n"
        "The scientific and margin-wide outputs are exploratory previews, not frozen or "
        "prospective publication artifacts.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
