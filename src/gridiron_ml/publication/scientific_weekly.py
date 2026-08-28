"""Paper-only weekly outputs from the frozen scientific model roster."""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd

from gridiron_ml.td_run.poll_viz import (
    draw_team_logo,
    plot_ballot_logo_grid,
    resolve_team_logo_path,
)

from .bundles import sha256_file
from .figure_theme import TDNET_COLORS, apply_tdnet_theme
from .roster_poll import build_frozen_roster_poll
from .weekly import build_weekly_blog_package, format_eastern_kickoffs

SCIENTIFIC_ROSTER_LABEL = "Scientific roster (market-free F0–F6)"


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"1", "true", "yes", "y"})


def market_free_scientific_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return the confirmatory, market-free F0–F6 scientific cells."""
    frame = inventory.copy()
    if "market_bearing" in frame:
        frame = frame.loc[~_as_bool(frame["market_bearing"])].copy()
    if "feature_config" in frame:
        frame = frame.loc[~frame["feature_config"].astype(str).isin({"F7", "F8"})].copy()
    if "objective" in frame:
        frame = frame.loc[frame["objective"].astype(str).str.casefold().eq("margin")].copy()
    if "use_in_weekly_consensus" in frame:
        frame = frame.loc[_as_bool(frame["use_in_weekly_consensus"])].copy()
    if frame.empty:
        raise ValueError("The scientific inventory has no eligible market-free F0–F6 margin models.")
    return frame.reset_index(drop=True)


def scientific_prediction_table(
    games: pd.DataFrame,
    *,
    reader_week: int,
    phase: str,
) -> pd.DataFrame:
    """Add explicit straight-up and against-the-spread picks to game rows."""
    frame = games.copy()
    spread = pd.to_numeric(
        frame.get("vegas_spread_as_of_publish", frame.get("market_spread_close")),
        errors="coerce",
    )
    home_margin = pd.to_numeric(frame["pred_home_margin"], errors="coerce")
    home_edge = home_margin + spread

    ats_team = pd.Series(pd.NA, index=frame.index, dtype="object")
    ats_line = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    valid = spread.notna() & home_margin.notna()
    home_cover = valid & home_edge.gt(0)
    away_cover = valid & home_edge.lt(0)
    push = valid & home_edge.abs().le(1e-9)
    ats_team.loc[home_cover] = frame.loc[home_cover, "home_team"].astype(str)
    ats_line.loc[home_cover] = spread.loc[home_cover]
    ats_team.loc[away_cover] = frame.loc[away_cover, "away_team"].astype(str)
    ats_line.loc[away_cover] = -spread.loc[away_cover]
    ats_team.loc[push] = "Projected push"
    ats_line.loc[push] = 0.0

    def ats_label(team: object, line: object) -> str:
        if pd.isna(team):
            return "No market line"
        if str(team) == "Projected push":
            return "Projected push"
        return f"{team} {float(line):+g}"

    output = pd.DataFrame(
        {
            "phase": phase,
            "roster": SCIENTIFIC_ROSTER_LABEL,
            "reader_week": int(reader_week),
            "provider_week": pd.to_numeric(frame.get("week"), errors="coerce").astype("Int64"),
            "game_id": frame["game_id"],
            "game_start_time_utc": frame["game_start_time_utc"],
            "kickoff_eastern": format_eastern_kickoffs(frame["game_start_time_utc"]),
            "away_team": frame["away_team"],
            "home_team": frame["home_team"],
            "straight_up_pick": frame["pred_winner"],
            "predicted_winner_margin": pd.to_numeric(frame["predicted_margin"], errors="coerce").round(2),
            "predicted_home_win_probability": pd.to_numeric(
                frame["pred_home_win_probability"], errors="coerce"
            ).round(4),
            "home_team_market_spread": spread.round(2),
            "against_spread_pick": [ats_label(team, line) for team, line in zip(ats_team, ats_line)],
            "against_spread_team": ats_team,
            "against_spread_line": ats_line,
            "model_edge_vs_spread_points": home_edge.abs().round(2),
            "model_agreement": pd.to_numeric(frame["model_agreement"], errors="coerce").round(4),
            "scientific_model_count": pd.to_numeric(frame["model_count"], errors="coerce").astype("Int64"),
        }
    )
    output["__kickoff_sort"] = pd.to_datetime(
        output["game_start_time_utc"], utc=True, errors="coerce"
    )
    return output.sort_values(["__kickoff_sort", "game_id"], kind="stable").drop(
        columns="__kickoff_sort"
    ).reset_index(drop=True)


def scientific_model_game_predictions(
    model_predictions: pd.DataFrame,
    consensus_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve every model score and attach the corresponding consensus picks."""
    frame = model_predictions.copy()
    consensus_columns = [
        "game_id",
        "phase",
        "roster",
        "reader_week",
        "provider_week",
        "kickoff_eastern",
        "straight_up_pick",
        "predicted_winner_margin",
        "predicted_home_win_probability",
        "home_team_market_spread",
        "against_spread_pick",
        "against_spread_team",
        "against_spread_line",
        "model_edge_vs_spread_points",
        "model_agreement",
        "scientific_model_count",
    ]
    consensus = consensus_predictions[consensus_columns].rename(
        columns={
            column: f"consensus_{column}"
            for column in consensus_columns
            if column != "game_id"
        }
    )
    frame = frame.merge(consensus, on="game_id", how="left", validate="many_to_one")
    spread = pd.to_numeric(frame["consensus_home_team_market_spread"], errors="coerce")
    margin = pd.to_numeric(frame["pred_home_margin"], errors="coerce")
    edge = margin + spread
    home_cover = spread.notna() & edge.gt(0)
    away_cover = spread.notna() & edge.lt(0)
    model_ats_team = pd.Series(pd.NA, index=frame.index, dtype="object")
    model_ats_line = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    model_ats_team.loc[home_cover] = frame.loc[home_cover, "home_team"].astype(str)
    model_ats_line.loc[home_cover] = spread.loc[home_cover]
    model_ats_team.loc[away_cover] = frame.loc[away_cover, "away_team"].astype(str)
    model_ats_line.loc[away_cover] = -spread.loc[away_cover]
    frame["model_predicted_winner_margin"] = margin.abs().round(3)
    frame["model_against_spread_team"] = model_ats_team
    frame["model_against_spread_line"] = model_ats_line
    frame["model_edge_vs_spread_points"] = edge.abs().round(3)
    frame["model_against_spread_pick"] = [
        "No market line" if pd.isna(team) else f"{team} {float(line):+g}"
        for team, line in zip(model_ats_team, model_ats_line)
    ]
    priority = [
        "consensus_phase",
        "consensus_roster",
        "consensus_reader_week",
        "consensus_provider_week",
        "game_id",
        "consensus_kickoff_eastern",
        "away_team",
        "home_team",
        "model_name",
        "model_family",
        "fingerprint",
        "pred_home_margin",
        "pred_home_win_probability",
        "pred_winner",
        "model_predicted_winner_margin",
        "model_against_spread_pick",
        "model_edge_vs_spread_points",
        "consensus_straight_up_pick",
        "consensus_predicted_winner_margin",
        "consensus_against_spread_pick",
        "consensus_model_edge_vs_spread_points",
    ]
    ordered = [column for column in priority if column in frame]
    remainder = [column for column in frame if column not in ordered]
    return frame[ordered + remainder].sort_values(
        ["game_id", "model_name"], kind="stable"
    ).reset_index(drop=True)


def plot_scientific_predictions(
    predictions: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    week: int,
    phase: str,
    dpi: int = 200,
) -> Path:
    """Render paper-first scientific predictions with explicit SU and ATS picks."""
    apply_tdnet_theme()
    table = pd.DataFrame(
        {
            "Kickoff (ET)": predictions["kickoff_eastern"],
            "Matchup": predictions["away_team"].astype(str) + " at " + predictions["home_team"].astype(str),
            "Straight up": predictions["straight_up_pick"].astype(str),
            "Margin": predictions["predicted_winner_margin"].map(lambda value: f"{float(value):.1f}"),
            "Vegas (home)": predictions["home_team_market_spread"].map(
                lambda value: "No line" if pd.isna(value) else f"{float(value):+g}"
            ),
            "Against spread": predictions["against_spread_pick"],
            "Model edge": predictions["model_edge_vs_spread_points"].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.1f} pts"
            ),
        }
    )
    height = max(5.0, 0.52 * len(table) + 2.4)
    fig, axis = plt.subplots(figsize=(18, height))
    fig.patch.set_facecolor("#F7F4ED")
    axis.axis("off")
    plotted = axis.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.11, 0.28, 0.13, 0.07, 0.12, 0.17, 0.10],
    )
    plotted.auto_set_font_size(False)
    plotted.set_fontsize(11.5)
    plotted.scale(1, 1.55)
    for (row, _), cell in plotted.get_celld().items():
        cell.set_edgecolor("#D4D7DB")
        if row == 0:
            cell.set_facecolor(TDNET_COLORS["midnight_gridiron"])
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#EEF2F5")
    phase_label = phase.replace("_", " ").title()
    axis.set_title(
        f"{season} Week {week} • SCIENTIFIC ROSTER PICKS • {phase_label}",
        fontsize=21,
        weight="bold",
        pad=26,
        color=TDNET_COLORS["midnight_gridiron"],
    )
    fig.text(
        0.5,
        0.025,
        "PAPER-ONLY F0–F6 MARKET-FREE ENSEMBLE  •  STRAIGHT-UP AND AGAINST-THE-SPREAD PICKS SHOWN SEPARATELY  •  NOT BETTING ADVICE",
        ha="center",
        fontsize=10.5,
        weight="bold",
        color=TDNET_COLORS["slate"],
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def scientific_consensus_power_rankings(ballots: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every scientific ballot into one all-team power-rating table."""
    required = {"keys_team", "ballot_model", "ballot_rank", "power_rating_vs_average"}
    if not required.issubset(ballots):
        raise ValueError(
            "Scientific ballots must include per-model power_rating_vs_average values."
        )
    frame = ballots.copy()
    frame["power_rating_vs_average"] = pd.to_numeric(
        frame["power_rating_vs_average"], errors="coerce"
    )
    frame["ballot_rank"] = pd.to_numeric(frame["ballot_rank"], errors="coerce")
    power = (
        frame.groupby("keys_team", as_index=False)
        .agg(
            predicted_margin_vs_average_team=("power_rating_vs_average", "mean"),
            median_margin_vs_average_team=("power_rating_vs_average", "median"),
            minimum_model_margin_vs_average=("power_rating_vs_average", "min"),
            maximum_model_margin_vs_average=("power_rating_vs_average", "max"),
            model_margin_standard_deviation=("power_rating_vs_average", "std"),
            average_ballot_rank=("ballot_rank", "mean"),
            best_ballot_rank=("ballot_rank", "min"),
            worst_ballot_rank=("ballot_rank", "max"),
            scientific_models=("ballot_model", "nunique"),
            top25_votes=("top25_vote", "sum"),
            poll_points=("poll_points", "sum"),
            first_place_votes=("first_place_vote", "sum"),
        )
        .sort_values(
            ["predicted_margin_vs_average_team", "average_ballot_rank", "keys_team"],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    power.insert(0, "consensus_power_rank", range(1, len(power) + 1))
    points_order = power.sort_values(
        ["poll_points", "average_ballot_rank", "best_ballot_rank", "keys_team"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    points_rank = pd.Series(
        range(1, len(points_order) + 1), index=points_order["keys_team"].astype(str)
    )
    power.insert(
        1, "poll_points_rank", power["keys_team"].astype(str).map(points_rank).astype(int)
    )
    power.insert(2, "consensus_power_top25", power["consensus_power_rank"].le(25))
    power.insert(3, "poll_points_top25", power["poll_points_rank"].le(25))
    numeric = [
        "predicted_margin_vs_average_team",
        "median_margin_vs_average_team",
        "minimum_model_margin_vs_average",
        "maximum_model_margin_vs_average",
        "model_margin_standard_deviation",
        "average_ballot_rank",
    ]
    power[numeric] = power[numeric].round(3)
    return power


def validate_scientific_ballots(ballots: pd.DataFrame) -> None:
    """Require one complete, independent all-team ballot per scientific model."""
    required = {"ballot_model", "keys_team", "ballot_rank", "power_rating_vs_average"}
    if not required.issubset(ballots):
        raise ValueError(f"Scientific ballots are missing {sorted(required - set(ballots))}.")
    if ballots.duplicated(["ballot_model", "keys_team"]).any():
        raise ValueError("A scientific model ballot contains duplicate team rows.")
    expected_teams = int(ballots["keys_team"].nunique())
    for model, frame in ballots.groupby("ballot_model", sort=False):
        if len(frame) != expected_teams:
            raise ValueError(
                f"Scientific model {model!r} has {len(frame)} teams; expected {expected_teams}."
            )
        ranks = sorted(pd.to_numeric(frame["ballot_rank"], errors="raise").astype(int))
        if ranks != list(range(1, expected_teams + 1)):
            raise ValueError(f"Scientific model {model!r} does not have consecutive 1..N ranks.")
        if pd.to_numeric(frame["power_rating_vs_average"], errors="coerce").isna().any():
            raise ValueError(f"Scientific model {model!r} has missing average-team scores.")


def plot_scientific_power_top25(
    power: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    week: int,
    logo_dir: str | Path | None = None,
    dpi: int = 200,
) -> Path:
    """Render the Top 25 directly from the all-team consensus power table."""
    apply_tdnet_theme()
    frame = power.sort_values("consensus_power_rank").head(25).copy()
    table = pd.DataFrame(
        {
            "Rank": frame["consensus_power_rank"].astype(int),
            "Ballot pts": frame["poll_points"].astype(int),
            "Team": [
                "" if resolve_team_logo_path(team, logo_dir) else str(team)
                for team in frame["keys_team"]
            ],
            "vs avg": frame["predicted_margin_vs_average_team"].map(
                lambda value: f"{float(value):+.1f}"
            ),
            "Median": frame["median_margin_vs_average_team"].map(
                lambda value: f"{float(value):+.1f}"
            ),
            "Avg ballot": frame["average_ballot_rank"].map(lambda value: f"{float(value):.1f}"),
            "Top-25 votes": frame["top25_votes"].astype(int).astype(str)
            + "/"
            + frame["scientific_models"].astype(int).astype(str),
            "Model range": frame["minimum_model_margin_vs_average"].map(
                lambda value: f"{float(value):+.1f}"
            )
            + " to "
            + frame["maximum_model_margin_vs_average"].map(
                lambda value: f"{float(value):+.1f}"
            ),
        }
    )
    fig, axis = plt.subplots(figsize=(14.5, 12.0))
    fig.patch.set_facecolor("#F7F4ED")
    axis.axis("off")
    plotted = axis.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.075, 1.0, 0.84],
        colWidths=[0.06, 0.08, 0.23, 0.10, 0.10, 0.12, 0.14, 0.17],
    )
    plotted.auto_set_font_size(False)
    plotted.set_fontsize(11.2)
    plotted.scale(1, 1.5)
    for (row, _), cell in plotted.get_celld().items():
        cell.set_edgecolor("#D4D7DB")
        if row == 0:
            cell.set_facecolor(TDNET_COLORS["midnight_gridiron"])
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#EEF2F5")
            if row <= 4:
                cell.set_text_props(weight="bold")
    fig.canvas.draw()
    team_column = table.columns.get_loc("Team")
    for row_number, team in enumerate(frame["keys_team"].astype(str), start=1):
        logo = resolve_team_logo_path(team, logo_dir)
        if logo is None:
            continue
        cell = plotted[(row_number, team_column)]
        draw_team_logo(
            axis,
            logo,
            cell.get_x() + cell.get_width() / 2,
            cell.get_y() + cell.get_height() / 2,
            target_px=24,
        )
    axis.set_title(
        f"{season} Week {week}: SCIENTIFIC ROSTER Consensus Power Top 25 • Paper Only",
        fontsize=21,
        weight="bold",
        pad=22,
        color=TDNET_COLORS["midnight_gridiron"],
    )
    fig.text(
        0.5,
        0.02,
        "Power rating = consensus predicted margin versus the constructed average team. Positive is stronger than average.",
        ha="center",
        fontsize=10.5,
        color=TDNET_COLORS["slate"],
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def write_scientific_weekly_outputs(
    *,
    games: pd.DataFrame,
    poll: pd.DataFrame,
    ballots: pd.DataFrame,
    model_predictions: pd.DataFrame | None = None,
    model_metadata: pd.DataFrame | None = None,
    output_root: str | Path,
    season: int,
    week: int,
    phase: str,
    logo_dir: str | Path | None = None,
    source_inventory: str | Path | None = None,
    input_provenance: dict[str, str | Path | None] | None = None,
) -> dict[str, Path]:
    """Write three scientific CSVs and their readable publication figures."""
    if phase not in {"pre_game", "post_game"}:
        raise ValueError("phase must be 'pre_game' or 'post_game'.")
    apply_tdnet_theme()
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output.iterdir() if path.name != "README.md"]
    if existing:
        raise FileExistsError(
            "Scientific weekly artifacts are immutable once generated; refusing to overwrite: "
            + ", ".join(sorted(path.name for path in existing))
        )
    predictions = scientific_prediction_table(games, reader_week=week, phase=phase)
    validate_scientific_ballots(ballots)
    prediction_export = (
        scientific_model_game_predictions(model_predictions, predictions)
        if model_predictions is not None and not model_predictions.empty
        else predictions
    )
    ballot_export = ballots.copy()
    if model_metadata is not None and not model_metadata.empty:
        metadata = model_metadata.copy()
        label_column = "final_model_name" if "final_model_name" in metadata else "model_id"
        keep = [
            label_column,
            "model_id",
            "model_level",
            "model_family",
            "objective",
            "feature_config",
            "fingerprint",
            "checkpoint_sha256",
            "training_end_season",
            "selected_feature_count",
            "calibration_status",
            "calibrator_sha256",
        ]
        keep = list(dict.fromkeys(column for column in keep if column in metadata))
        metadata = metadata[keep].drop_duplicates(label_column).rename(
            columns={label_column: "ballot_model"}
        )
        ballot_export = ballot_export.merge(
            metadata, on="ballot_model", how="left", validate="many_to_one"
        )
    top_ballots = ballots.loc[
        pd.to_numeric(ballots["ballot_rank"], errors="coerce").between(1, 25, inclusive="both")
    ].copy()
    power = scientific_consensus_power_rankings(ballots)

    paths = {
        "predictions_csv": output / "scientific_all_game_predictions.csv",
        "predictions_png": output / "scientific_all_game_predictions.png",
        "ballots_csv": output / "scientific_full_ballots.csv",
        "ballots_png": output / "scientific_top25_ballots.png",
        "power_rankings_csv": output / "scientific_consensus_power_rankings.csv",
        "top25_png": output / "scientific_top25.png",
    }
    prediction_export.to_csv(paths["predictions_csv"], index=False)
    ballot_export.to_csv(paths["ballots_csv"], index=False)
    power.to_csv(paths["power_rankings_csv"], index=False)
    plot_scientific_predictions(
        predictions, paths["predictions_png"], season=season, week=week, phase=phase
    )
    plot_ballot_logo_grid(
        top_ballots,
        paths["ballots_png"],
        top_n=25,
        logo_dir=logo_dir,
        title=f"{season} Week {week}: SCIENTIFIC ROSTER Top 25 Ballots • Paper Only",
    )
    plot_scientific_power_top25(
        power,
        paths["top25_png"],
        season=season,
        week=week,
        logo_dir=logo_dir,
    )
    manifest = {
        "created_at_eastern": datetime.now(UTC).astimezone(
            ZoneInfo("America/New_York")
        ).isoformat(),
        "season": int(season),
        "week": int(week),
        "phase": phase,
        "roster": SCIENTIFIC_ROSTER_LABEL,
        "market_bearing_tiers_excluded": ["F7", "F8"],
        "scientific_model_count": int(ballots["ballot_model"].nunique()),
        "teams_per_model_ballot": int(ballots["keys_team"].nunique()),
        "full_ballot_rows": len(ballots),
        "artifact_policy": "paper_only_not_social_media",
        "source_inventory": str(source_inventory) if source_inventory else None,
        "source_inventory_sha256": (
            sha256_file(source_inventory) if source_inventory and Path(source_inventory).exists() else None
        ),
        "files": {name: path.name for name, path in paths.items()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# TDNet scientific weekly package\n\n"
        "Paper-only outputs from the frozen, market-free F0–F6 scientific roster. "
        "These files are separate from TDNet's operational and social-media products.\n\n"
        "- `scientific_all_game_predictions.csv` is model × game long form and preserves "
        "every model score plus consensus straight-up and against-the-spread picks.\n"
        "- `scientific_full_ballots.csv` preserves every model × team predicted margin "
        "against the constructed average team and its resulting ballot rank.\n"
        "- `scientific_consensus_power_rankings.csv` aggregates those model scores for "
        "every team; `predicted_margin_vs_average_team` is the consensus power rating.\n"
        "- The three PNGs are readable paper figures: all-game picks, Top-25 ballot slots, "
        "and the consensus Top 25.\n",
        encoding="utf-8",
    )
    generated_utc = datetime.now(UTC)
    kickoff = pd.to_datetime(games["game_start_time_utc"], utc=True, errors="coerce")
    earliest_kickoff = kickoff.min()
    source_inputs = {}
    for name, value in (input_provenance or {}).items():
        if value is None:
            source_inputs[name] = {"path": None, "sha256": None}
            continue
        source_path = Path(value)
        source_inputs[name] = {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path) if source_path.exists() else None,
        }
    generator_sources = [
        Path(__file__),
        Path(__file__).with_name("weekly.py"),
        Path(__file__).resolve().parents[1] / "td_run/evaluator.py",
    ]
    repository_root = Path(__file__).resolve().parents[3]
    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
        ).strip()
        dirty_paths = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository_root, text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        code_commit = None
        dirty_paths = []
    training_cutoff = None
    if model_metadata is not None and "training_end_season" in model_metadata:
        values = pd.to_numeric(model_metadata["training_end_season"], errors="coerce").dropna()
        training_cutoff = int(values.max()) if not values.empty else None
    payload = {
        "schema": "tdnet-scientific-weekly-reproducibility-v1",
        "phase": phase,
        "season": int(season),
        "week": int(week),
        "immutable": True,
        "code_commit": code_commit,
        "code_worktree_dirty_at_generation": bool(dirty_paths),
        "dirty_paths_at_generation": dirty_paths,
        "model_release": (
            Path(source_inventory).resolve().parent.name if source_inventory else None
        ),
        "training_cutoff_season": training_cutoff,
        "scientific_roster_size": int(ballots["ballot_model"].nunique()),
        "teams_per_model_ballot": int(ballots["keys_team"].nunique()),
        "full_ballot_rows": len(ballots),
        "post_game_overwrite_prohibited": True,
        "post_game_destination": "sibling post_game/scientific directory only",
        "generated_at_utc": generated_utc.isoformat(),
        "generated_at_eastern": generated_utc.astimezone(
            ZoneInfo("America/New_York")
        ).isoformat(),
        "earliest_game_kickoff_utc": (
            earliest_kickoff.isoformat() if pd.notna(earliest_kickoff) else None
        ),
        "generated_before_earliest_kickoff": (
            bool(generated_utc < earliest_kickoff.to_pydatetime())
            if pd.notna(earliest_kickoff) else None
        ),
        "source_inputs": source_inputs,
        "generator_sources": {
            str(path.relative_to(repository_root)): sha256_file(path)
            for path in generator_sources
        },
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in [*paths.values(), output / "manifest.json", output / "README.md"]
        },
    }
    (output / "scientific_reproducibility_payload.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def build_scientific_weekly_outputs(
    *,
    project_root: str | Path,
    inventory_path: str | Path,
    schedule_snapshot_path: str | Path,
    output_root: str | Path,
    season: int,
    week: int,
    phase: str,
    poll_week: int | None = None,
    prediction_week: int | None = None,
    market_lines_path: str | Path | None = None,
    reference_poll_path: str | Path | None = None,
) -> dict[str, Path]:
    """Run the frozen scientific roster and emit the compact weekly package."""
    root = Path(project_root).resolve()
    source_inventory = Path(inventory_path).resolve()
    inventory = market_free_scientific_inventory(pd.read_csv(source_inventory))
    reference = pd.read_csv(reference_poll_path) if reference_poll_path else pd.DataFrame()
    with tempfile.TemporaryDirectory(prefix="tdnet-scientific-weekly-") as temporary:
        staging = Path(temporary)
        runtime_inventory = staging / "scientific_runtime_inventory.csv"
        inventory.to_csv(runtime_inventory, index=False)
        poll_result = build_frozen_roster_poll(
            runtime_inventory,
            season=season,
            week=week if poll_week is None else poll_week,
            output_dir=staging / "poll",
            project_root=root,
            logo_dir=root / "data/meta/logos/by_team",
            objective="margin",
            reference_poll=reference,
            reference_label="AP",
            render_figures=False,
        )
        report = build_weekly_blog_package(
            project_root=root,
            season=season,
            week=week if prediction_week is None else prediction_week,
            model_inventory_path=runtime_inventory,
            schedule_snapshot_path=schedule_snapshot_path,
            market_lines_path=market_lines_path,
            output_root=staging / "report",
            ap_top25_path=reference_poll_path,
            tdnet_top25_path=staging / "poll/tdnet_top25.csv",
            logo_dir=root / "data/meta/logos/by_team",
            schedule_driven_matchups=True,
            render_social_assets=False,
        )
        return write_scientific_weekly_outputs(
            games=report["all_games"],
            poll=poll_result["poll"],
            ballots=poll_result["ballots"],
            model_predictions=report["all_model_predictions"],
            model_metadata=inventory,
            output_root=output_root,
            season=season,
            week=week,
            phase=phase,
            logo_dir=root / "data/meta/logos/by_team",
            source_inventory=source_inventory,
            input_provenance={
                "scientific_inventory": source_inventory,
                "schedule_snapshot": schedule_snapshot_path,
                "market_lines_snapshot": market_lines_path,
                "reference_poll": reference_poll_path,
            },
        )
