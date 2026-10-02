"""Schedule-bound F09 play microstructure with trailing regular-game state."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .nextgen_artifacts import NextgenFeatureBuilder
from .nextgen_designs import Formula, feature_record, reciprocal_counterparts
from .nextgen_microstructure import game_sufficient_statistics, drive_sufficient_statistics
from gridiron_ml.pipeline.fetch.nextgen_acquisition import (
    COMPLETE, atomic_json, load_authoritative_schedule, sha256_file, verify_cache,
)

ROOT = Path(__file__).resolve().parents[3]
GLOSSARY = "https://www.footballstudyhall.com/2018/2/2/16963820/college-football-advanced-stats-glossary"


def verified_endpoint_records(root: Path, endpoint: str):
    records = []
    for path in (root / "request_ledger").glob("*/*.json"):
        record = json.loads(path.read_text())
        if record["endpoint"] != endpoint or record["status"] not in COMPLETE:
            continue
        if record.get("year") is not None and not 2010 <= int(record["year"]) <= 2025:
            continue
        if verify_cache(Path(record["cache_path"]), record) is None:
            raise ValueError(f"Source cache failed verification: {record['request_id']}")
        records.append(record)
    return sorted(records, key=lambda r: (r.get("year") or 0, r.get("week") or 0, r["request_id"]))


def prepare_game_statistics(root: Path):
    inventory = json.loads((ROOT / "configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json").read_text())
    schedule, schedule_hash = load_authoritative_schedule(root, inventory)
    allowed = set(schedule.id.astype(int))
    records = verified_endpoint_records(root, "/plays")
    if not records:
        raise ValueError("No verified plays partitions")
    from gridiron_ml.pipeline.fetch.nextgen_acquisition import AcquisitionLedger
    ledger = AcquisitionLedger(root)
    plan = root / "results/preflight/cfbd_request_manifest_v1.jsonl"
    for item in map(json.loads, plan.read_text().splitlines()):
        if item["endpoint"] not in {"/plays", "/drives"}:
            continue
        entry = ledger.read(item["request_id"]) or {}
        if entry.get("status") not in COMPLETE and not (
            entry.get("status") in {"failed_final", "structurally_unavailable"} and entry.get("reviews")
        ):
            raise ValueError("F09 materialization waits for every play partition or an explicit reviewed gap")
    drive_records = {int(r["year"]): r for r in verified_endpoint_records(root, "/drives")}
    drive_frames = {}
    # Every raw request gets one resumable compressed sufficient-statistic cache.
    out = root / "canonical/microstructure_game_statistics"
    out.mkdir(parents=True, exist_ok=True)
    parts, coverage = [], []
    for record in records:
        path = out / (record["request_id"]+".parquet")
        meta_path = path.with_suffix(".json")
        binding = {"raw_sha256": record["sha256"], "schedule_sha256": schedule_hash,
                   "code_sha256": sha256_file(ROOT / "src/gridiron_ml/experiments/nextgen_microstructure.py"),
                   "drives_sha256": drive_records.get(int(record["year"]), {}).get("sha256")}
        if path.exists() and meta_path.exists():
            old = json.loads(meta_path.read_text())
            if all(old.get(k) == v for k, v in binding.items()) and old.get("data_sha256") == sha256_file(path):
                parts.append(pd.read_parquet(path))
                coverage.append(old)
                continue
        raw = pd.read_parquet(record["cache_path"])
        raw = raw.loc[pd.to_numeric(raw.game_id).isin(allowed)].copy()
        # Partition scope and source IDs remain explicit even if provider sent extras.
        expected = schedule.loc[schedule.season.eq(record["year"]) & schedule.week.eq(record["week"])]
        raw = raw.loc[raw.game_id.isin(expected.id)]
        stats = game_sufficient_statistics(raw)
        year = int(record["year"])
        if year in drive_records:
            if year not in drive_frames:
                drive_frames.clear()  # retain only one year's source in memory
                drive_frames[year] = pd.read_parquet(drive_records[year]["cache_path"])
            drives = drive_frames[year]
            drives = drives.loc[drives.game_id.isin(raw.game_id)]
            drive_stats = drive_sufficient_statistics(drives, raw)
            stats = stats.merge(drive_stats, on=["game_id", "team"], how="left", validate="one_to_one")
        temp = path.with_suffix(".tmp.parquet")
        stats.to_parquet(temp, index=False, compression="zstd")
        temp.replace(path)
        report = {**binding, "request_id": record["request_id"], "season": record["year"],
                  "week": record["week"], "raw_rows": len(raw), "observed_games": int(raw.game_id.nunique()),
                  "expected_games": len(expected), "missing_game_ids": sorted(set(expected.id)-set(raw.game_id)),
                  "data_sha256": sha256_file(path)}
        atomic_json(meta_path, report)
        coverage.append(report)
        parts.append(stats)
    combined = pd.concat(parts, ignore_index=True)
    if combined.duplicated(["game_id", "team"]).any():
        raise ValueError("A source game/team appears in multiple play partitions")
    atomic_json(root / "results/f09_play_coverage.json", {"partitions": coverage, "total_source_team_games": len(combined)})
    return combined, schedule, schedule_hash


def trailing_game_state(stats: pd.DataFrame, schedule: pd.DataFrame, *, window=12, lag_hours=48):
    """Aggregate only games whose reconstructed reporting cutoff precedes target.

    A trailing 12-game state crosses seasons and supplies Week-0 context from
    the prior season. Games, not calendar weeks, are the window unit. Missing
    coverage is not filled with fictional production or zero denominators.
    """
    schedule = schedule.copy()
    if schedule.season.max() > 2025:
        raise ValueError("Quarantined season in F09 schedule")
    if not schedule.season_type.eq("regular").all() or not schedule.completed.eq(True).all():
        raise ValueError("F09 history must be completed regular-season games")
    schedule["kickoff"] = pd.to_datetime(schedule.start_date, utc=True)
    by_id = schedule.set_index("id")
    valid = stats.game_id.isin(by_id.index)
    if not valid.all():
        raise ValueError("Source statistic lacks regular-season authority")
    values = [c for c in stats if c.endswith(("__sum", "__n"))]
    source = stats.merge(schedule[["id", "kickoff", "home_team", "away_team"]], left_on="game_id", right_on="id", validate="many_to_one")
    if not (source.team.eq(source.home_team) | source.team.eq(source.away_team)).all():
        raise ValueError("A contributing team-game is not a scheduled participant")
    source["available"] = source.kickoff + pd.Timedelta(hours=lag_hours)
    history = {str(team): g.sort_values(["available", "game_id"]) for team, g in source.groupby("team")}
    targets = schedule.loc[schedule.home_classification.str.lower().eq("fbs") & schedule.away_classification.str.lower().eq("fbs")]
    rows, missing = [], []
    for game in targets.itertuples(index=False):
        pair = []
        for team in (game.home_team, game.away_team):
            h = history.get(team)
            if h is None:
                break
            prior = h.loc[h.available.lt(game.kickoff)].tail(window)
            if prior.empty:
                break
            latest = prior.iloc[-1]
            sums = prior[values].sum(min_count=1)
            rates = {}
            for name in [c.removesuffix("__sum") for c in values if c.endswith("__sum")]:
                count, total = sums[name+"__n"], sums[name+"__sum"]
                # Eight attempts supports quarter rates; ten for other sparse states.
                minimum = 8 if "rush_ypa_q" in name else 10
                rates[name] = float(total/count) if pd.notna(count) and count >= minimum else np.nan
            rates.update(season=int(game.season), season_type="regular", team=team,
                         target_game_id=int(game.id), target_start_utc=game.kickoff,
                         feature_kind="dynamic", latest_source_game_id=int(latest.game_id),
                         latest_source_game_utc=latest.kickoff, latest_source_season_type="regular",
                         feature_available_utc=latest.available, static_availability_documentation=None,
                         source_game_count=len(prior))
            pair.append(rates)
        if len(pair) == 2:
            rows.extend(pair)
        else:
            missing.append(int(game.id))
    if not rows:
        raise ValueError("No paired prior-game states")
    return pd.DataFrame(rows), {"window_games": window, "reporting_lag_hours": lag_hours,
                                 "excluded_no_prior_coverage_game_ids": missing}


def f09_formulas(design: str):
    formulas = []
    for side in ("offense", "defense"):
        def metric(name):
            return side+"_"+name
        formulas.extend([
            Formula(metric("rush_ypa_q4_minus_q1"), (metric("rush_ypa_q4"), metric("rush_ypa_q1")), "difference",
                    "Fourth-quarter rushing yards per attempt minus first-quarter rate across prior qualifying games; measures whether rushing production improves late.", "yards_per_rush"),
            Formula(metric("rush_quarter_slope"), tuple(metric(f"rush_ypa_q{q}") for q in range(1, 5)), "quarter_slope",
                    "Ordinary least-squares slope of rushing yards per attempt against quarters one through four; uses all four quarter rates to describe strengthening or weakening.", "yards_per_rush_per_quarter"),
            Formula(metric("second_minus_first_half_success"), (metric("second_half_success_rate"), metric("first_half_success_rate")), "difference",
                    "Second-half minus first-half efficiency, excluding garbage time; a compact adaptation/fatigue profile.", "fraction"),
        ])
        for name, interpretation, units in [
            ("middle_eight_success_rate", "Efficiency in the final four minutes of Q2 and first four of Q3.", "fraction"),
            ("two_minute_success_rate", "Efficiency during the last two minutes of each half in competitive play.", "fraction"),
            ("one_score_success_rate", "Efficiency while the pre-play score is within eight points.", "fraction"),
            ("backed_up_success_rate", "Efficiency when starting a play inside the offense's own ten-yard line.", "fraction"),
            ("red_zone_success_rate", "Efficiency inside the opponent twenty; location state retains garbage-time plays.", "fraction"),
            ("goal_to_go_success_rate", "Efficiency within ten yards with distance to gain at least distance to goal.", "fraction"),
            ("rush_line_yards", "Piecewise rushing yardage credited to line-created movement, with losses penalized and long gains capped.", "yards_per_rush"),
            ("rush_second_level_yards", "Rushing yardage between yards four and ten per attempt.", "yards_per_rush"),
            ("rush_open_field_yards", "Rushing yardage beyond ten yards per attempt.", "yards_per_rush"),
            ("rush_opportunity_rate", "Fraction of rushes gaining at least four yards.", "fraction"),
            ("drive_start_field_position", "Mean competitive-drive starting field position measured from own goal.", "yards_from_own_goal"),
            ("net_points_per_drive", "Own score gain minus opposing score gain per competitive possession.", "points_per_drive"),
            ("points_per_opportunity", "Own points per competitive drive reaching the opposing forty.", "points_per_opportunity"),
            ("scoring_opportunity_rate", "Share of competitive drives with a scrimmage snap at or inside the opposing forty.", "fraction"),
            ("quality_drive_rate", "Share of competitive drives reaching the opposing forty or gaining forty yards.", "fraction"),
            ("drive_seconds_per_play", "Competitive-drive elapsed seconds per qualifying scrimmage play; a pace proxy.", "seconds_per_play"),
        ]:
            n = metric(name)
            formulas.append(Formula(n, (n,), "identity", interpretation, units))
        if design == "b":
            for name in ("early_down_success_rate", "third_down_success_rate", "fourth_down_success_rate", "sack_rate"):
                n = metric(name)
                formulas.append(Formula(n, (n,), "identity", "Explicit competitive-play situational rate; richer B context.", "fraction"))
    return formulas


class MicrostructureBuilder(NextgenFeatureBuilder):
    generation = "F09"

    def __init__(self, root: Path, state: pd.DataFrame, design: str):
        if design not in {"a", "b", "c"}:
            raise ValueError("Unknown design")
        self.family = "play_microstructure_"+design
        self.state = state
        self.formulas = f09_formulas(design)
        # C retains the mandatory rushing difference; collapses compatible
        # pressure and field-position rates into three-input concepts.
        if design == "c":
            remove = set()
            extras = []
            for side in ("offense", "defense"):
                for concept, suffixes, meaning in (
                    ("pressure_success", ("middle_eight_success_rate", "two_minute_success_rate", "one_score_success_rate"),
                     "Equal-weight efficiency across clock-transition, late-half and close-score contexts; execution under game pressure."),
                    ("field_stress_success", ("backed_up_success_rate", "red_zone_success_rate", "goal_to_go_success_rate"),
                     "Equal-weight efficiency in constrained field-position states: backed up or near the opposing goal."),
                ):
                    inputs = tuple(side+"_"+x for x in suffixes)
                    remove.update(inputs)
                    extras.append(Formula(side+"_"+concept, inputs, "mean", meaning, "fraction"))
            self.formulas = [f for f in self.formulas if f.name not in remove]+extras
        self.feature_columns = tuple(f.name for f in self.formulas)
        counterparts = reciprocal_counterparts(list(self.feature_columns))
        records = []
        for formula in self.formulas:
            record = feature_record(formula, "F09", design, counterparts[formula.name], endpoints=["/plays", "/drives", "/games"])
            record.update(
                availability_rule="Latest contributing regular-game kickoff plus 48 hours, strictly before target; conservative reconstructed reporting cutoff, not archived publication time",
                aggregation_window="latest 12 completed regular games with available structured plays, crossing seasons",
                minimum_sample_rule="at least 8 rushes in each required quarter; at least 10 qualifying plays for each other rate; league-season coverage reviewed separately",
                garbage_time_handling="Q1 >28/Q2 >24/Q3 >21 pre-play lead; whole Q4 excluded only if every qualifying Q4 play >16; red-zone/goal-to-go location exceptions",
                source_inspiration=GLOSSARY if "success" in formula.name or "line_yards" in formula.name else "TDNet-derived from structured CFBD plays",
                code_path="src/gridiron_ml/experiments/nextgen_f09.py",
                raw_columns=["period", "down", "distance", "yards_gained", "yards_to_goal", "play_type", "offense_score", "defense_score", "clock.minutes", "clock.seconds"],
                rate_definition="sum qualifying event value / count qualifying prior plays; success thresholds 50/70/100 percent of distance by down",
                input_equations={n: f"=IF(SUM(Last12[{n}__n])>={8 if 'rush_ypa_q' in n else 10},SUM(Last12[{n}__sum])/SUM(Last12[{n}__n]),NA())" for n in formula.inputs},
            )
            if formula.operation == "identity":
                n = formula.name
                record["equation_excel"] = record["input_equations"][n]
                record["source_inputs"] = [n+"__sum", n+"__n"]
            records.append(record)
        self.manifest_path = root / "feature_families/F09" / self.family / "feature_manifest.json"
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(records, indent=2)+"\n"
        if self.manifest_path.exists() and self.manifest_path.read_text() != content:
            raise ValueError("Existing F09 manifest differs; version the family before changing formulas")
        self.manifest_path.write_text(content)

    def build_frame(self):
        metadata = ["season", "season_type", "team", "target_game_id", "target_start_utc", "feature_kind",
                    "latest_source_game_id", "latest_source_game_utc", "latest_source_season_type",
                    "feature_available_utc", "static_availability_documentation"]
        derived = pd.DataFrame({f.name: f.evaluate(self.state) for f in self.formulas})
        return pd.concat([self.state[metadata].reset_index(drop=True), derived.reset_index(drop=True)], axis=1)


def materialize_f09(root: Path):
    stats, schedule, schedule_hash = prepare_game_statistics(root)
    state, coverage = trailing_game_state(stats, schedule)
    report = {"schedule_sha256": schedule_hash, "coverage": coverage, "designs": {}}
    for design in "abc":
        builder = MicrostructureBuilder(root, state, design)
        path = builder.materialize(root)
        report["designs"][design] = {"path": str(path), "manifest": str(builder.manifest_path),
                                     "feature_count": len(builder.feature_columns), "rows": len(state)}
    atomic_json(root / "results/f09_materialization.json", report)
    return report


if __name__ == "__main__":
    config = json.loads((ROOT / "configs/experiments/nextgen_fingerprints_v1.json").read_text())
    print(json.dumps(materialize_f09(Path(config["artifact_root"])), indent=2))
