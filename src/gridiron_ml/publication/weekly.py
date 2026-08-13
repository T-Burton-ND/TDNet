"""Canonical weekly predictions and blog-ready logo/margin rendering."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import subprocess

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd

from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.experiments.opponent_adjusted import StaticFrameFingerprints
from gridiron_ml.models import load_model_checkpoint, normalize_model_family
from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.td_run.poll_viz import resolve_team_logo_path, load_team_logo_image

from .bundles import sha256_file
from .preseason_states import build_preseason_state_frame
from .polls import load_ap_top25
from .poll_recaps import plot_tdnet_vs_ap_poll
from .preseason_rankings import load_preseason_performance_rankings
from .team_labels import format_team_with_ap_rank


def frozen_model_set_sha256(checkpoint_hashes) -> str | None:
    """Stable, visible identifier for the exact checkpoint set behind a figure."""
    hashes = sorted({str(value).strip().lower() for value in checkpoint_hashes if str(value).strip() and str(value).lower() != "nan"})
    return sha256("\n".join(hashes).encode("utf-8")).hexdigest() if hashes else None


def _model_hash_stamp(model_set_sha256: str | None, checkpoint_count: int | None = None, generated_at_utc: str | None = None) -> str | None:
    if not model_set_sha256:
        return None
    count = f" · {int(checkpoint_count)} checkpoints" if checkpoint_count else ""
    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
    return f"Frozen model-set SHA-256: {model_set_sha256}{count} · Generated UTC: {generated}"


def build_weekly_blog_package(
    *,
    project_root: str | Path,
    season: int,
    week: int,
    model_inventory_path: str | Path,
    schedule_snapshot_path: str | Path,
    output_root: str | Path,
    top25_path: str | Path | None = None,
    top25_label: str | None = None,
    ap_top25_path: str | Path | None = None,
    tdnet_top25_path: str | Path | None = None,
    fingerprint_version: int = 0,
    logo_dir: str | Path | None = None,
    matchup_config: dict | None = None,
    preseason_ranking_path: str | Path | None = None,
    include_collapsed_models: bool = False,
    schedule_driven_matchups: bool = False,
) -> dict[str, object]:
    """Predict every scheduled game and render the Top-25 matchup slate.

    The canonical long and consensus tables are written before figures; the
    notebook only calls this function and displays its returned artifacts.
    """
    root = Path(project_root).resolve()
    output = Path(output_root)
    tables = output / "tables"
    figures = output / "figures"
    blog = output / "blog"
    metadata_dir = output / "metadata"
    for directory in [tables, figures, blog, metadata_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    logo_dir = Path(logo_dir or root / "data/meta/logos/by_team")
    inventory = pd.read_csv(model_inventory_path)
    schedule = pd.read_parquet(schedule_snapshot_path) if str(schedule_snapshot_path).endswith(".parquet") else pd.read_csv(schedule_snapshot_path)
    schedule = _normalize_schedule(schedule, season=season, week=week)
    model_entries, load_failures = _load_inventory_models(
        inventory, root, preseason_ranking_path=preseason_ranking_path
    )
    if not model_entries:
        raise FileNotFoundError("No model inventory checkpoints could be loaded.")

    matchup_builder = MatchupBuilder(
        **dict(matchup_config or {"representation": "unit_matchup", "safe_math": True})
    )
    matchup_cache = {}

    prediction_frames = []
    prediction_failures = []
    created_at = datetime.now(timezone.utc).isoformat()
    for entry in model_entries:
        try:
            fingerprint_label = entry.get("fingerprint")
            cache_key = fingerprint_label or f"v{int(fingerprint_version)}"
            if cache_key not in matchup_cache:
                fingerprints = _weekly_fingerprints(
                    root,
                    fingerprint_label=fingerprint_label,
                    fingerprint_path=entry.get("fingerprint_path"),
                    fallback_version=fingerprint_version,
                )
                if schedule_driven_matchups:
                    matchup_X, context = _build_schedule_driven_matchups(
                        fingerprints, schedule, matchup_builder, season=int(season), week=int(week)
                    )
                else:
                    X_block, meta_df, market_df = fingerprints.prediction_block(
                        season=int(season), predict_week=int(week), scheduled_only=True
                    )
                    matchup_X, matchup_meta, matchup_market = matchup_builder.matchups(
                        X_block, meta_df, market_df=market_df
                    )
                    context = pd.concat(
                        [matchup_meta.reset_index(drop=True), matchup_market.reset_index(drop=True)],
                        axis=1,
                    )
                    context = context.loc[:, ~context.columns.duplicated()].copy()
                    context = _normalize_matchup_context(context)
                    context = context.merge(schedule, on="game_id", how="left", suffixes=("", "_schedule"))
                    for column in ["home_team", "away_team", "game_start_time_utc"]:
                        schedule_column = f"{column}_schedule"
                        if schedule_column in context:
                            if column in context:
                                context[column] = context[schedule_column].combine_first(context[column])
                            else:
                                context[column] = context[schedule_column]
                matchup_cache[cache_key] = (matchup_X, context)
            matchup_X, context = matchup_cache[cache_key]
            prediction = entry["model"].predict(matchup_X)
            frame = context.copy()
            frame["model_name"] = entry["model_name"]
            frame["model_family"] = entry["model_family"]
            frame["objective"] = entry["objective"]
            frame["checkpoint_path"] = entry["checkpoint_path"]
            frame["checkpoint_sha256"] = entry["checkpoint_sha256"]
            frame["fingerprint"] = cache_key
            frame["roster_rank"] = int(entry["roster_rank"])
            frame["preseason_performance_rank"] = int(entry["preseason_performance_rank"])
            frame["pred_home_margin"] = pd.to_numeric(
                prediction["pred_margin"], errors="coerce"
            )
            frame["pred_home_win_probability"] = pd.to_numeric(
                prediction["pred_proba_home_win"], errors="coerce"
            )
            frame["pred_winner"] = frame["home_team"].where(
                frame["pred_home_win_probability"] >= 0.5, frame["away_team"]
            )
            frame["confidence"] = (
                frame["pred_home_win_probability"] - 0.5
            ).abs() * 2.0
            frame["created_at_utc"] = created_at
            prediction_frames.append(frame)
        except Exception as exc:
            prediction_failures.append(
                {
                    "model_name": entry["model_name"],
                    "checkpoint_path": entry["checkpoint_path"],
                    "reason": str(exc),
                }
            )
    if not prediction_frames:
        raise RuntimeError("All inventory models failed weekly prediction.")
    long = pd.concat(prediction_frames, ignore_index=True)
    figure_checkpoint_hashes = sorted(long["checkpoint_sha256"].dropna().astype(str).unique().tolist())
    model_set_sha256 = frozen_model_set_sha256(figure_checkpoint_hashes)
    # A one-game slate cannot distinguish a genuinely constant model from a
    # normal model. Keep the row so postseason/small-slate examples still
    # produce a complete package.
    if long["game_id"].nunique() <= 1:
        collapsed_models = []
    else:
        collapsed_models = (
            long.groupby("model_name")["pred_home_margin"].nunique()
            .loc[lambda values: values.le(1)]
            .index.astype(str)
            .tolist()
        )
    collapsed_models_detected = list(collapsed_models)
    if collapsed_models and not include_collapsed_models:
        prediction_failures.extend(
            {"model_name": name, "reason": "collapsed weekly predictions; excluded from consensus"}
            for name in collapsed_models
        )
        long = long.loc[~long["model_name"].astype(str).isin(collapsed_models)].copy()
    if long.empty:
        raise RuntimeError("Every inventory model collapsed to a constant weekly prediction.")
    consensus = summarize_weekly_predictions(long)
    ranking_column = "preseason_performance_rank" if "preseason_performance_rank" in long else "roster_rank"
    ap_path = ap_top25_path or top25_path
    ap_top25 = load_ap_top25(ap_path, season=season, week=week) if ap_path else pd.DataFrame(columns=["rank", "team"])
    if ap_path and ap_top25.empty:
        raise ValueError(
            f"No AP Top 25 snapshot is available at or before {season} Week {week}; "
            "refusing to substitute TDNet or a future poll."
        )
    tdnet_top25 = load_top25(tdnet_top25_path, season=season, week=week) if tdnet_top25_path else pd.DataFrame(columns=["rank", "team"])
    top25_games = select_top25_games(consensus, ap_top25)
    consensus = _add_poll_ranks(consensus, ap_top25, prefix="ap")
    if not tdnet_top25.empty:
        consensus = _add_poll_ranks(consensus, tdnet_top25, prefix="tdnet")
    closest_games = select_closest_games(consensus, count=10)

    long.to_parquet(tables / "all_game_model_predictions.parquet", index=False)
    long.to_csv(tables / "all_game_model_predictions.csv", index=False)
    consensus.to_parquet(tables / "all_games.parquet", index=False)
    consensus.to_csv(tables / "all_games.csv", index=False)
    top25_games.to_csv(tables / "ap_top25_games.csv", index=False)
    closest_games.to_csv(tables / "closest_games.csv", index=False)
    tdnet_top25.to_csv(tables / "tdnet_top25_snapshot.csv", index=False)
    pd.DataFrame(load_failures + prediction_failures).to_csv(
        tables / "model_failures.csv", index=False
    )

    figure_paths = {
        "top25_matchups": plot_top25_matchups(
            top25_games,
            figures / "top25_week_matchups.png",
            logo_dir=logo_dir,
            title=(
                f"{season} Week {week}: Top 25 Matchups"
                + f"\nRank source: {top25_label or 'AP Top 25'}"
            ),
            model_set_sha256=model_set_sha256,
            checkpoint_count=len(figure_checkpoint_hashes),
            generated_at_utc=created_at,
        ),
        "all_games_table": plot_all_games_table(
            consensus,
            figures / "all_games_predictions.png",
            title=f"{season} Week {week}: All TDNet Predictions",
            model_set_sha256=model_set_sha256,
            checkpoint_count=len(figure_checkpoint_hashes),
            generated_at_utc=created_at,
        ),
        "closest_games": plot_top25_matchups(
            closest_games,
            figures / "closest_games.png",
            logo_dir=logo_dir,
            title=f"{season} Week {week}: 10 Closest TDNet Consensus Games",
            model_set_sha256=model_set_sha256,
            checkpoint_count=len(figure_checkpoint_hashes),
            generated_at_utc=created_at,
        ),
    }
    if not tdnet_top25.empty and not ap_top25.empty:
        comparison_label = top25_label or "AP Top 25"
        figure_paths["tdnet_vs_ap_top25"] = plot_tdnet_vs_ap_poll(
            tdnet_top25, ap_top25, figures / "tdnet_vs_ap_top25.png",
            title=f"{season} Week {week}: TDNet Top 25 vs {comparison_label}",
            model_set_sha256=model_set_sha256,
            checkpoint_count=len(figure_checkpoint_hashes),
            generated_at_utc=created_at,
            reference_label=top25_label or "AP",
        )
    summary_md = render_weekly_summary_markdown(
        season=season,
        week=week,
        consensus=consensus,
        top25_games=top25_games,
        closest_games=closest_games,
        model_count=long["model_name"].nunique(),
        top25_label=top25_label,
    )
    (blog / "summary.md").write_text(summary_md, encoding="utf-8")
    (blog / "figure_captions.md").write_text(
        "# Figure captions\n\n"
        f"1. **Top 25 matchups.** Consensus frozen-model predictions for {len(top25_games)} games involving a team in {top25_label or 'the supplied Top 25 snapshot'}. Margins are signed from the home team's perspective.\n\n"
        f"2. **All games.** Consensus predictions for {len(consensus)} scheduled games.\n\n"
        f"3. **Closest games.** The ten scheduled games with the smallest absolute consensus predicted margins.\n",
        encoding="utf-8",
    )
    (blog / "alt_text.md").write_text(
        "# Alt text\n\n"
        "Top 25 matchup cards show each away and home team logo, poll rank when available, and the TDNet consensus predicted winner and margin.\n\n"
        "The all-games table lists kickoff, matchup, predicted winner, signed home margin, home win probability, and model agreement.\n\n"
        "Closest-game cards show the ten lowest-margin projected games with team logos, predicted winner, margin, home win probability, and model agreement.\n",
        encoding="utf-8",
    )
    manifest = {
        "season": int(season),
        "week": int(week),
        "created_at_utc": created_at,
        "game_count": len(consensus),
        "top25_game_count": len(top25_games),
        "closest_game_count": len(closest_games),
        "model_count": int(long["model_name"].nunique()),
        "frozen_model_set_sha256": model_set_sha256,
        "checkpoint_sha256s": figure_checkpoint_hashes,
        "model_failure_count": len(load_failures) + len(prediction_failures),
        "collapsed_models_skipped": [] if include_collapsed_models else collapsed_models,
        "collapsed_models_detected": collapsed_models_detected,
        "fingerprint_variants": sorted(map(str, long["fingerprint"].dropna().unique())),
        "collapsed_model_count": 0 if include_collapsed_models else len(collapsed_models),
        "preseason_ranking_path": str(Path(preseason_ranking_path).resolve()) if preseason_ranking_path else None,
        "preseason_ranking_sha256": sha256_file(preseason_ranking_path) if preseason_ranking_path else None,
        "top_selection_policy": "preseason_frozen_historical_performance_rank",
        "preseason_prior_policy": "same_team_carry_forward_then_conference_mean",
        "schedule_driven_matchups": bool(schedule_driven_matchups),
        "model_inventory_path": str(Path(model_inventory_path).resolve()),
        "model_inventory_sha256": sha256_file(model_inventory_path),
        "schedule_snapshot_path": str(Path(schedule_snapshot_path).resolve()),
        "schedule_snapshot_sha256": sha256_file(schedule_snapshot_path),
        "ap_top25_path": str(Path(ap_path).resolve()) if ap_path else None,
        "ap_top25_label": top25_label or "AP Top 25",
        "ranking_source_is_ap": (top25_label or "AP Top 25").strip().casefold() == "ap top 25",
        "ap_top25_sha256": sha256_file(ap_path) if ap_path else None,
        "tdnet_top25_path": str(Path(tdnet_top25_path).resolve()) if tdnet_top25_path else None,
        "tdnet_top25_sha256": sha256_file(tdnet_top25_path) if tdnet_top25_path else None,
        "figures": {name: str(path) for name, path in figure_paths.items()},
    }
    (metadata_dir / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "output_root": output,
        "all_model_predictions": long,
        "all_games": consensus,
        "top25": ap_top25,
        "ap_top25": ap_top25,
        "tdnet_top25": tdnet_top25,
        "top25_games": top25_games,
        "closest_games": closest_games,
        "figures": figure_paths,
        "manifest": manifest,
    }


def summarize_weekly_predictions(long: pd.DataFrame) -> pd.DataFrame:
    """Create one consensus row per game from game-level model predictions."""
    keys = [
        "game_id",
        "season",
        "week",
        "game_start_time_utc",
        "home_team",
        "away_team",
        "neutral_site",
        "conference_game",
        "season_type",
    ]
    keys = [key for key in keys if key in long.columns]
    summary = (
        long.groupby(keys, dropna=False, as_index=False)
        .agg(
            pred_home_margin=("pred_home_margin", "mean"),
            pred_home_margin_median=("pred_home_margin", "median"),
            pred_home_win_probability=("pred_home_win_probability", "mean"),
            model_count=("model_name", "nunique"),
            model_agreement=("pred_winner", lambda values: values.value_counts(normalize=True).iloc[0]),
        )
    )
    summary["pred_winner"] = summary["home_team"].where(
        summary["pred_home_win_probability"] >= 0.5, summary["away_team"]
    )
    summary["predicted_margin"] = summary["pred_home_margin"].abs()
    return summary.sort_values(["game_start_time_utc", "game_id"], ignore_index=True)


def load_top25(path: str | Path, *, season: int | None = None, week: int | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["rank", "team"])
    # Poll files often contain a full season. Prefer the requested snapshot;
    # otherwise use the latest available snapshot, never a mixture of weeks.
    if "season" in frame.columns and season is not None:
        matching_season = frame.loc[pd.to_numeric(frame["season"], errors="coerce") == int(season)]
        if not matching_season.empty:
            frame = matching_season
    if "week" in frame.columns:
        numeric_week = pd.to_numeric(frame["week"], errors="coerce")
        if week is not None and (numeric_week == int(week)).any():
            frame = frame.loc[numeric_week == int(week)]
        elif numeric_week.notna().any():
            frame = frame.loc[numeric_week == numeric_week.max()]
    rank_column = next((c for c in ["rank", "poll_rank", "ap_rank"] if c in frame.columns), None)
    team_column = next((c for c in ["team", "keys_team", "school"] if c in frame.columns), None)
    if rank_column is None or team_column is None:
        raise ValueError("Top-25 snapshot needs rank and team/keys_team columns.")
    out = frame.loc[:, [rank_column, team_column]].rename(
        columns={rank_column: "rank", team_column: "team"}
    )
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    return out.dropna().sort_values("rank").head(25).reset_index(drop=True)


def select_top25_games(games: pd.DataFrame, top25: pd.DataFrame) -> pd.DataFrame:
    if top25.empty:
        return games.iloc[0:0].copy()
    ranks = dict(zip(top25["team"].astype(str), top25["rank"].astype(int)))
    selected = games.loc[
        games["home_team"].astype(str).isin(ranks)
        | games["away_team"].astype(str).isin(ranks)
    ].copy()
    selected["home_rank"] = selected["home_team"].astype(str).map(ranks)
    selected["away_rank"] = selected["away_team"].astype(str).map(ranks)
    return selected.sort_values(
        ["away_rank", "home_rank", "game_start_time_utc"], na_position="last"
    ).reset_index(drop=True)


def select_closest_games(games: pd.DataFrame, *, count: int = 10) -> pd.DataFrame:
    """Return the games with the smallest absolute predicted winner margin."""
    if games is None or games.empty:
        return pd.DataFrame()
    frame = games.copy()
    margin = pd.to_numeric(frame.get("predicted_margin"), errors="coerce")
    frame = frame.loc[margin.notna()].copy()
    if frame.empty:
        return frame
    frame["__abs_predicted_margin"] = margin.loc[frame.index].abs()
    frame["closest_rank"] = frame["__abs_predicted_margin"].rank(method="first", ascending=True).astype(int)
    return (
        frame.sort_values(["__abs_predicted_margin", "game_start_time_utc", "game_id"], na_position="last")
        .drop(columns=["__abs_predicted_margin"])
        .head(int(count))
        .reset_index(drop=True)
    )


def _add_poll_ranks(games: pd.DataFrame, poll: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    out = games.copy()
    ranks = dict(zip(poll["team"].astype(str), poll["rank"].astype(int)))
    out[f"{prefix}_rank_home"] = out["home_team"].astype(str).map(ranks)
    out[f"{prefix}_rank_away"] = out["away_team"].astype(str).map(ranks)
    return out


def plot_top25_matchups(games, path, *, logo_dir, title, dpi=200, model_set_sha256=None, checkpoint_count=None, generated_at_utc=None):
    """Render logo matchup cards with explicit winner and margin labels."""
    games = pd.DataFrame(games)
    rows = max(1, len(games))
    fig, axes = plt.subplots(rows, 1, figsize=(11, 1.65 * rows + 0.8), squeeze=False)
    fig.patch.set_facecolor("#F6F3EC")
    for axis in axes[:, 0]:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
    if games.empty:
        axes[0, 0].text(0.5, 0.5, "No Top 25 games in this schedule snapshot", ha="center", va="center")
    for axis, (_, game) in zip(axes[:, 0], games.iterrows()):
        axis.add_patch(plt.Rectangle((0.01, 0.05), 0.98, 0.9, color="white", ec="#D8D2C4", lw=1.0))
        _draw_logo(axis, game["away_team"], logo_dir, 0.10, 0.5)
        _draw_logo(axis, game["home_team"], logo_dir, 0.90, 0.5)
        axis.text(0.19, 0.57, format_team_with_ap_rank(game, "away"), ha="left", va="center", fontsize=12, weight="bold")
        axis.text(0.81, 0.57, format_team_with_ap_rank(game, "home"), ha="right", va="center", fontsize=12, weight="bold")
        axis.text(0.50, 0.64, "at" if not bool(game.get("neutral_site", False)) else "vs", ha="center", color="#666")
        winner = str(game["pred_winner"])
        margin = float(game["predicted_margin"])
        axis.text(0.50, 0.39, f"TDNet: {winner} by {margin:.1f}", ha="center", va="center", fontsize=13, color="#8A2D2D", weight="bold")
        axis.text(0.50, 0.20, f"Agreement {float(game['model_agreement']):.0%}  •  Home win {float(game['pred_home_win_probability']):.0%}", ha="center", fontsize=8.5, color="#555")
    fig.suptitle(title, fontsize=17, weight="bold", y=0.995)
    stamp = _model_hash_stamp(model_set_sha256, checkpoint_count, generated_at_utc)
    if stamp:
        fig.text(0.5, 0.014, stamp, ha="center", va="bottom", fontsize=9, weight="bold", color="#28323C", family="monospace", bbox={"facecolor": "#FFFFFF", "edgecolor": "#AAB5C1", "boxstyle": "round,pad=0.35"})
    fig.tight_layout(rect=[0, 0.052 if stamp else 0, 1, 0.98])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_all_games_table(games, path, *, title, dpi=180, model_set_sha256=None, checkpoint_count=None, generated_at_utc=None):
    table = games.copy()
    if table.empty:
        table = pd.DataFrame({"Matchup": ["No scheduled games"]})
    else:
        kickoff = pd.to_datetime(table["game_start_time_utc"], utc=True, errors="coerce")
        table = pd.DataFrame(
            {
                "Kickoff (UTC)": kickoff.dt.strftime("%a %H:%M"),
                "Matchup": [
                    f"{format_team_with_ap_rank(game, 'away')} at {format_team_with_ap_rank(game, 'home')}"
                    for _, game in table.iterrows()
                ],
                "TDNet pick": table["pred_winner"],
                "Margin": table["predicted_margin"].map(lambda x: f"{x:.1f}"),
                "Home win": table["pred_home_win_probability"].map(lambda x: f"{x:.0%}"),
                "Agreement": table["model_agreement"].map(lambda x: f"{x:.0%}"),
            }
        )
    fig_height = max(3.0, 0.34 * len(table) + 1.4)
    fig, axis = plt.subplots(figsize=(13, fig_height))
    axis.axis("off")
    plot_table = axis.table(
        cellText=table.values,
        colLabels=table.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.13, 0.32, 0.20, 0.10, 0.11, 0.11][: len(table.columns)],
    )
    plot_table.auto_set_font_size(False)
    plot_table.set_fontsize(8.5)
    plot_table.scale(1, 1.25)
    for (row, _), cell in plot_table.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if row == 0:
            cell.set_facecolor("#22324A")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F5F8")
    axis.set_title(title, fontsize=16, weight="bold", pad=18)
    stamp = _model_hash_stamp(model_set_sha256, checkpoint_count, generated_at_utc)
    if stamp:
        fig.text(0.5, 0.014, stamp, ha="center", va="bottom", fontsize=9, weight="bold", color="#28323C", family="monospace", bbox={"facecolor": "#FFFFFF", "edgecolor": "#AAB5C1", "boxstyle": "round,pad=0.35"})
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def render_weekly_summary_markdown(*, season, week, consensus, top25_games, closest_games=None, model_count, top25_label=None):
    lines = [
        f"# TDNet {season} Week {week} predictions",
        "",
        f"This report covers **{len(consensus)} games** using consensus predictions from **{model_count} models**.",
        "",
        "Margins are reported as the predicted winner's advantage. The canonical tables retain signed home margins.",
        "",
        "## Ranked-team games",
        "",
        f"Ranked-team designation uses **{top25_label or 'the supplied Top 25 snapshot'}**.",
        "",
    ]
    if top25_games.empty:
        lines.append("No ranked-team games were present in the supplied Top 25 and schedule snapshots.")
    else:
        for _, game in top25_games.iterrows():
            lines.append(
                f"- {game['away_team']} at {game['home_team']}: **{game['pred_winner']} by {game['predicted_margin']:.1f}** "
                f"({game['model_agreement']:.0%} model agreement)."
            )
    lines.extend(["", "## Closest projected games", ""])
    closest_games = pd.DataFrame() if closest_games is None else closest_games
    if closest_games.empty:
        lines.append("No closest-game table was generated for this schedule snapshot.")
    else:
        for _, game in closest_games.iterrows():
            lines.append(
                f"- {game['away_team']} at {game['home_team']}: **{game['pred_winner']} by {game['predicted_margin']:.1f}** "
                f"({game['model_agreement']:.0%} model agreement)."
            )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "These are model predictions, not betting advice. Any comparison with market lines must identify whether a model used market features and must use lines captured before the prediction deadline.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_inventory_models(inventory, project_root, *, preseason_ranking_path=None):
    inventory = inventory.copy()
    if preseason_ranking_path is not None:
        ranking = load_preseason_performance_rankings(preseason_ranking_path)
        # Some publication callers persist the ranking column in their
        # objective-specific inventory and also pass the canonical ranking
        # file.  Drop the embedded copy before merging so pandas does not
        # create ``*_x``/``*_y`` columns and hide the required field.
        inventory = inventory.drop(columns=["preseason_performance_rank"], errors="ignore")
        inventory = inventory.merge(
            ranking[["model_id", "preseason_performance_rank"]],
            on="model_id", how="left", validate="one_to_one",
        )
        inventory["preseason_performance_rank"] = pd.to_numeric(
            inventory["preseason_performance_rank"], errors="coerce"
        )
        missing = inventory["preseason_performance_rank"].isna()
        if missing.any():
            raise ValueError(
                "Preseason ranking is missing inventory models: "
                + ", ".join(inventory.loc[missing, "model_id"].astype(str).tolist())
            )
        inventory = inventory.sort_values(
            ["preseason_performance_rank", "model_id"], kind="mergesort"
        ).reset_index(drop=True)
        # Family toggles in the owner notebook define the active roster.  The
        # frozen global order is therefore compressed after filtering so the
        # lead/Top-3 views always mean the best one/three active models.
        inventory["preseason_performance_rank"] = np.arange(1, len(inventory) + 1)
    if "roster_rank" not in inventory:
        score_column = next(
            (column for column in ["selection_brier_score", "eval_brier_score", "brier_score"] if column in inventory),
            None,
        )
        if score_column:
            inventory["__score"] = pd.to_numeric(inventory[score_column], errors="coerce").fillna(np.inf)
            inventory = inventory.sort_values(["__score", "final_model_name" if "final_model_name" in inventory else inventory.columns[0]])
        inventory["roster_rank"] = np.arange(1, len(inventory) + 1)
    entries = []
    failures = []
    for _, row in inventory.iterrows():
        if "use_in_weekly_consensus" in inventory.columns:
            enabled = row.get("use_in_weekly_consensus")
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() in {"1", "true", "yes", "y"}
            if not bool(enabled):
                continue
        raw_path = row.get("checkpoint_path")
        if pd.isna(raw_path):
            failures.append({"model_name": row.get("model", "unknown"), "reason": "missing checkpoint_path"})
            continue
        checkpoint = Path(str(raw_path))
        if not checkpoint.is_absolute():
            checkpoint = project_root / checkpoint
        try:
            model = load_model_checkpoint(checkpoint)
            entries.append(
                {
                    "model": model,
                    "model_name": str(row.get("final_model_name", row.get("model", getattr(model, "model_name", checkpoint.stem)))),
                    "model_family": normalize_model_family(row.get("family", getattr(model, "model_family", "unknown"))),
                    "objective": str(row.get("objective", "margin")),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "fingerprint": None if pd.isna(row.get("fingerprint")) else str(row.get("fingerprint")),
                    "fingerprint_path": None if pd.isna(row.get("fingerprint_path")) else str(row.get("fingerprint_path")),
                    "roster_rank": int(row.get("roster_rank")),
                    "preseason_performance_rank": int(row.get("preseason_performance_rank", row.get("roster_rank"))),
                }
            )
        except Exception as exc:
            failures.append({"model_name": row.get("model", checkpoint.stem), "checkpoint_path": str(checkpoint), "reason": str(exc)})
    return entries, failures


def _weekly_fingerprints(project_root, *, fingerprint_label, fingerprint_path=None, fallback_version):
    """Load the exact experiment fingerprint variant recorded by a checkpoint."""
    if fingerprint_path:
        path = Path(str(fingerprint_path))
        if not path.is_absolute():
            path = Path(project_root) / path
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint requires fingerprint path '{fingerprint_path}', but {path} is missing."
            )
        frame = pd.read_parquet(path)
        frame = _apply_preseason_priors(frame, project_root=project_root)
        return StaticFrameFingerprints(frame, postseason=False)
    if fingerprint_label:
        safe = str(fingerprint_label).strip().lower().replace(".", "_").replace("-", "_")
        path = (
            Path(project_root)
            / "data/experiments/opponent_adjusted_fingerprints/fingerprints"
            / safe
            / "canonical_fingerprint.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint requires fingerprint '{fingerprint_label}', but {path} is missing."
            )
        frame = pd.read_parquet(path)
        frame = _apply_preseason_priors(frame, project_root=project_root)
        return StaticFrameFingerprints(frame, postseason=False)
    return Fingerprints(version=int(fallback_version), postseason=False, root=project_root)


def _apply_preseason_priors(frame: pd.DataFrame, *, project_root: str | Path | None = None) -> pd.DataFrame:
    """Fill zero Week-0 feature rows from each team's latest prior-season state.

    Only scientific feature columns are carried. Game identity, next opponent,
    schedule, labels, market context, and fingerprint provenance remain those
    of the frozen 2026 row.
    """
    try:
        state = build_preseason_state_frame(frame, season=2026, project_root=project_root)
    except ValueError:
        return frame
    target = ~(
        pd.to_numeric(frame["keys_season"], errors="coerce").eq(2026)
        & pd.to_numeric(frame["keys_week"], errors="coerce").eq(0)
    )
    shared = [column for column in frame.columns if column in state.columns]
    return pd.concat([frame.loc[target], state.loc[:, shared]], ignore_index=True, sort=False)


def _normalize_schedule(schedule, *, season, week):
    frame = schedule.copy()
    frame = frame.loc[
        (pd.to_numeric(frame["season"], errors="coerce") == int(season))
        & (pd.to_numeric(frame["week"], errors="coerce") == int(week))
    ].copy()
    rename = {
        "id": "game_id",
        "start_date": "game_start_time_utc",
    }
    frame = frame.rename(columns=rename)
    required = ["game_id", "season", "week", "game_start_time_utc", "home_team", "away_team"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Schedule snapshot missing columns: {missing}")
    for column, default in [
        ("neutral_site", False),
        ("conference_game", False),
        ("season_type", "regular"),
    ]:
        if column not in frame:
            frame[column] = default
    return frame.loc[:, required + ["neutral_site", "conference_game", "season_type"]].drop_duplicates("game_id")


def _build_schedule_driven_matchups(fingerprints, schedule, matchup_builder, *, season, week):
    """Pair every scheduled game from one pre-week team-state snapshot.

    This avoids dropping games when CFBD assigns two games for one team to the
    same week. Game-specific home indicators are reconstructed; travel fields
    are set missing so each frozen model's fitted preprocessing handles them
    instead of reusing a different game's travel context.
    """
    snapshot = fingerprints.season_snapshot(season, max(int(week) - 1, 0))
    if len(snapshot) == 4:
        features, _, meta, _ = snapshot
    else:
        features, meta, _ = snapshot
    if "keys_team" not in meta:
        raise ValueError("Season snapshot lacks keys_team for schedule-driven pairing.")
    team_rows = pd.Series(meta.index.to_numpy(), index=meta["keys_team"].astype(str)).to_dict()
    eligible = schedule.loc[
        schedule["home_team"].astype(str).isin(team_rows)
        & schedule["away_team"].astype(str).isin(team_rows)
    ].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No scheduled games have both teams in the pre-week fingerprint state.")
    home = features.iloc[[team_rows[str(team)] for team in eligible["home_team"]]].reset_index(drop=True)
    away = features.iloc[[team_rows[str(team)] for team in eligible["away_team"]]].reset_index(drop=True)
    for column in ["game_is_home", "next_game_is_home"]:
        if column in home:
            home[column] = 1.0
            away[column] = 0.0
    for column in ["travel_distance_diff", "travel_tz_diff"]:
        if column in home:
            home[column] = np.nan
            away[column] = np.nan
    matchup = matchup_builder.build_many(home, away)
    context = eligible.copy()
    context["keys_season"] = context["season"]
    context["next_week"] = context["week"]
    context["next_game_id"] = context["game_id"]
    return matchup.reset_index(drop=True), context.reset_index(drop=True)


def _normalize_matchup_context(context):
    aliases = {
        "next_game_id": "game_id",
        "keys_game_id": "game_id",
        "keys_season": "season",
        "next_week": "week",
        "keys_week": "week",
        "keys_team_home": "home_team",
        "keys_team_away": "away_team",
    }
    for source, target in aliases.items():
        if target not in context.columns and source in context.columns:
            context[target] = context[source]
    if "game_id" not in context:
        raise ValueError("Matchup metadata does not contain a game identifier.")
    return context


def _draw_logo(axis, team, logo_dir, x, y):
    path = resolve_team_logo_path(team, logo_dir)
    if path is None:
        axis.text(x, y, str(team)[:3].upper(), ha="center", va="center", weight="bold")
        return
    image = load_team_logo_image(path)
    # Normalize the cropped visible mark, not the source canvas.
    longest_side = max(image.shape[:2])
    zoom = 52.0 / max(1, longest_side)
    axis.add_artist(AnnotationBbox(OffsetImage(image, zoom=zoom), (x, y), frameon=False))
