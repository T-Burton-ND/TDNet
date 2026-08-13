"""src.gridiron_ml.td_run.weekly_report.

Usage:
    Build blog-ready weekly poll, matchup, and team-comparison artifacts from
    trained TDNet checkpoints.

Logic flow:
    1. Load latest model checkpoints.
    2. Rank teams from post-previous-week fingerprints and optional manual poll.
    3. Predict target-week scheduled games with every loaded model.
    4. Summarize model consensus, score estimates, and matchup confidence.
    5. Save tables and PNG figures for notebook/blog use.

Responsibility:
    Provide a reusable workflow behind the weekly blog-output notebook.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_ml.fingerprints import Fingerprints
from gridiron_ml.td_run.matchups import MatchupBuilder
from gridiron_ml.models import load_model_checkpoint, model_label, normalize_model_family

from .poll_viz import plot_ballot_logo_grid, plot_weekly_top25_table, resolve_team_logo_path, draw_team_logo
from .season_vs_vegas import source_style
from .evaluator import TDEval


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class WeeklyReportBuilder:
    """Build weekly poll, matchup, and comparison outputs for a target week."""

    def __init__(
        self,
        *,
        root=None,
        fingerprint_version=0,
        postseason=False,
        matchup_config=None,
        models_root=None,
        output_root=None,
        logo_dir=None,
    ):
        """Initialize the report builder with project paths and shared config."""
        self.root = Path(root) if root is not None else PROJECT_ROOT
        self.fingerprint_version = int(fingerprint_version)
        self.postseason = bool(postseason)
        self.matchup_config = dict(matchup_config or {"representation": "unit_matchup", "safe_math": True})
        self.models_root = Path(models_root) if models_root is not None else self.root / "models"
        self.output_root = Path(output_root) if output_root is not None else self.root / "data" / "weekly_reports"
        self.logo_dir = Path(logo_dir) if logo_dir is not None else self.root / "data" / "meta" / "logos" / "by_team"

    def run(
        self,
        *,
        season,
        target_week,
        include_models="all",
        exclude_models=None,
        top_n=25,
        average_scope="season",
        manual_ballots=None,
        team_a=None,
        team_b=None,
        scheduled_only=True,
        default_total=52.5,
        output_dir=None,
        poll_output_dir=None,
        rebuild_poll_history=False,
    ):
        """Run the complete weekly report workflow and return generated artifacts."""
        season = int(season)
        target_week = int(target_week)
        snapshot_week = max(target_week - 1, 0)
        output_dir = Path(output_dir) if output_dir is not None else self.output_root / str(season) / f"week_{target_week:02d}"
        poll_output_dir = Path(poll_output_dir) if poll_output_dir is not None else output_dir.parent / "polls"
        poll_weeks = range(0, snapshot_week + 1) if bool(rebuild_poll_history) else [snapshot_week]
        tables_dir = output_dir / "tables"
        plots_dir = output_dir / "plots"
        tables_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        model_entries, checkpoint_inventory = discover_latest_checkpoints(
            models_root=self.models_root,
            include_models=include_models,
            exclude_models=exclude_models,
        )
        if not model_entries:
            raise FileNotFoundError(f"No loadable model checkpoints found under {self.models_root}.")

        fingerprints = Fingerprints(
            version=self.fingerprint_version,
            postseason=self.postseason,
            root=self.root,
        )
        matchup_builder = MatchupBuilder(**self.matchup_config)
        evaluator = TDEval(
            config=self._eval_config(model_entries[0]),
            fingerprints=fingerprints,
            matchup_builder=matchup_builder,
            model=model_entries[0]["model"],
        )

        weekly_poll_tables = evaluator.build_weekly_poll_outputs(
            models=[entry["model"] for entry in model_entries],
            season=season,
            weeks=poll_weeks,
            top_n=top_n,
            average_scope=average_scope,
            output_dir=poll_output_dir,
            logo_dir=self.logo_dir,
            manual_ballots=manual_ballots,
            merge_existing=not bool(rebuild_poll_history),
        )
        poll_df = evaluator.poll(
            models=[entry["model"] for entry in model_entries],
            season=season,
            week=snapshot_week,
            average_scope=average_scope,
            top_n=top_n,
            manual_ballots=manual_ballots,
        ).head(top_n)
        ballots_df = evaluator.poll_ballots_.copy()

        matchup_long, matchup_summary, skipped_predictions = self.predict_week_matchups(
            fingerprints=fingerprints,
            matchup_builder=matchup_builder,
            model_entries=model_entries,
            season=season,
            target_week=target_week,
            scheduled_only=scheduled_only,
            default_total=default_total,
        )

        comparison = {}
        if team_a and team_b:
            comparison = self.compare_teams(
                fingerprints=fingerprints,
                matchup_builder=matchup_builder,
                model_entries=model_entries,
                season=season,
                snapshot_week=snapshot_week,
                team_a=team_a,
                team_b=team_b,
                default_total=default_total,
            )

        tables = {
            "checkpoint_inventory": checkpoint_inventory,
            "current_poll_top25": poll_df,
            "current_poll_ballots": ballots_df,
            "upcoming_matchup_model_predictions": matchup_long,
            "upcoming_matchup_summary": matchup_summary,
            "skipped_model_predictions": skipped_predictions,
            **{f"poll_{name}": table for name, table in weekly_poll_tables.items()},
        }
        tables["blog_top25"] = blog_top25_table(poll_df)
        tables["blog_picks"] = blog_picks_table(matchup_summary)
        tables["blog_closest_games"] = blog_closest_games_table(matchup_summary, count=10)
        tables["fun_data_workspace"] = pd.DataFrame(
            columns=["topic", "table_or_figure", "note"]
        )
        if comparison:
            tables["team_comparison_feature_edges"] = comparison["feature_edges"]
            tables["team_comparison_model_predictions"] = comparison["model_predictions"]
            tables["team_comparison_summary"] = comparison["summary"]

        for name, table in tables.items():
            if isinstance(table, pd.DataFrame):
                table.to_csv(tables_dir / f"{name}.csv", index=False)

        figure_paths = {
            "current_poll_bar": plot_poll_top25_bar(
                poll_df,
                plots_dir / f"poll_top{top_n}_week_{snapshot_week:02d}.png",
                top_n=top_n,
                logo_dir=self.logo_dir,
                title=f"{season} Post-Week {snapshot_week} TDNet Poll",
            ),
            "current_ballot_grid": plot_ballot_logo_grid(
                ballots_df,
                plots_dir / f"model_ballot_grid_week_{snapshot_week:02d}.png",
                top_n=top_n,
                logo_dir=self.logo_dir,
                title=f"{season} Week {snapshot_week} Model Ballots",
            ),
            "weekly_poll_table": plot_weekly_top25_table(
                weekly_poll_tables.get("weekly_poll_top25"),
                plots_dir / f"weekly_poll_top{top_n}_through_week_{snapshot_week:02d}.png",
                top_n=top_n,
                logo_dir=self.logo_dir,
            ),
            "upcoming_matchups": plot_upcoming_matchups_table(
                matchup_summary,
                plots_dir / f"week_{target_week:02d}_matchup_predictions.png",
                title=f"{season} Week {target_week} TDNet Matchup Predictions",
            ),
            "closest_matchups": plot_upcoming_matchups_table(
                tables["blog_closest_games"],
                plots_dir / f"week_{target_week:02d}_closest_matchups.png",
                title=f"{season} Week {target_week} TDNet 10 Closest Games",
                max_rows=10,
            ),
        }
        if comparison:
            figure_paths["team_comparison_edges"] = plot_team_comparison_edges(
                comparison["feature_edges"],
                plots_dir / f"{safe_name(team_a)}_vs_{safe_name(team_b)}_feature_edges.png",
                title=f"{team_a} vs {team_b}: Biggest Fingerprint Edges",
            )
            figure_paths["team_comparison_predictions"] = plot_team_matchup_predictions(
                comparison["model_predictions"],
                plots_dir / f"{safe_name(team_a)}_vs_{safe_name(team_b)}_model_predictions.png",
                title=f"{team_a} vs {team_b}: Model Margins",
            )

        manifest = pd.DataFrame(
            [
                {
                    "season": season,
                    "target_week": target_week,
                    "snapshot_week": snapshot_week,
                    "output_dir": str(output_dir),
                    "poll_output_dir": str(poll_output_dir),
                    "poll_weeks_computed": ",".join(str(int(week)) for week in poll_weeks),
                    "rebuild_poll_history": bool(rebuild_poll_history),
                    "models_loaded": len(model_entries),
                    "matchups_predicted": len(matchup_summary),
                    "poll_rows": len(poll_df),
                }
            ]
        )
        manifest.to_csv(tables_dir / "manifest.csv", index=False)
        return {
            "output_dir": output_dir,
            "tables": tables,
            "figures": {key: value for key, value in figure_paths.items() if value is not None},
            "manifest": manifest,
        }

    def predict_week_matchups(
        self,
        *,
        fingerprints,
        matchup_builder,
        model_entries,
        season,
        target_week,
        scheduled_only=True,
        default_total=52.5,
    ):
        """Predict target-week matchups with each model and summarize consensus."""
        X_block, meta_df, market_df = fingerprints.prediction_block(
            season=season,
            predict_week=target_week,
            scheduled_only=scheduled_only,
        )
        matchup_X, matchup_meta, matchup_market = matchup_builder.matchups(
            X_block,
            meta_df,
            market_df=market_df,
        )
        base_df = concat_without_duplicate_columns(matchup_meta, matchup_market)
        prediction_frames = []
        skipped = []
        for entry in model_entries:
            try:
                pred = entry["model"].predict(matchup_X)
            except Exception as exc:
                skipped.append({"model": entry["name"], "family": entry["family"], "reason": str(exc)})
                continue
            frame = base_df.copy()
            frame["model"] = entry["name"]
            frame["family"] = entry["family"]
            frame["pred_margin"] = pd.to_numeric(pred["pred_margin"], errors="coerce")
            if "pred_proba_home_win" in pred.columns:
                frame["pred_proba_home_win"] = pd.to_numeric(pred["pred_proba_home_win"], errors="coerce")
            else:
                frame["pred_proba_home_win"] = 1.0 / (1.0 + np.exp(-frame["pred_margin"] / 14.0))
            frame["pred_pick_home"] = frame["pred_margin"] > 0
            prediction_frames.append(frame)

        long_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        skipped_df = pd.DataFrame(skipped)
        summary_df = summarize_matchup_predictions(long_df, default_total=default_total)
        return long_df, summary_df, skipped_df

    def compare_teams(
        self,
        *,
        fingerprints,
        matchup_builder,
        model_entries,
        season,
        snapshot_week,
        team_a,
        team_b,
        default_total=52.5,
    ):
        """Compare two team fingerprints and model-predict Team A home vs Team B."""
        X_a, _, meta_a, market_a = fingerprints.team_fingerprint(team_a, season, snapshot_week)
        X_b, _, meta_b, market_b = fingerprints.team_fingerprint(team_b, season, snapshot_week)
        matchup_X = matchup_builder.build_many(X_a, X_b)
        meta = pd.DataFrame(
            {
                "keys_season": [int(season)],
                "keys_week": [int(snapshot_week) + 1],
                "keys_team_home": [str(meta_a.loc[0, "keys_team"]) if "keys_team" in meta_a.columns else str(team_a)],
                "keys_team_away": [str(meta_b.loc[0, "keys_team"]) if "keys_team" in meta_b.columns else str(team_b)],
            }
        )
        prediction_frames = []
        for entry in model_entries:
            try:
                pred = entry["model"].predict(matchup_X)
            except Exception as exc:
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "model": [entry["name"]],
                            "family": [entry["family"]],
                            "error": [str(exc)],
                        }
                    )
                )
                continue
            frame = meta.copy()
            frame["model"] = entry["name"]
            frame["family"] = entry["family"]
            frame["pred_margin"] = pd.to_numeric(pred["pred_margin"], errors="coerce")
            frame["pred_proba_home_win"] = pd.to_numeric(pred.get("pred_proba_home_win", np.nan), errors="coerce")
            prediction_frames.append(frame)

        model_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        summary = summarize_matchup_predictions(model_predictions, default_total=default_total)
        feature_edges = compare_feature_edges(X_a, X_b, team_a=team_a, team_b=team_b)
        return {
            "feature_edges": feature_edges,
            "model_predictions": model_predictions,
            "summary": summary,
        }

    def _eval_config(self, model_entry):
        """Return a minimal TDEval config for report workflows."""
        return {
            "fingerprints": {
                "version": self.fingerprint_version,
                "postseason": self.postseason,
                "root": str(self.root),
            },
            "matchup": dict(self.matchup_config),
            "model": {
                "family": model_entry["family"],
            },
        }


def discover_latest_checkpoints(models_root, include_models="all", exclude_models=None):
    """Load the newest checkpoint per model name under a models directory."""
    root = Path(models_root)
    include = normalize_model_filter(include_models)
    exclude = set(normalize_model_filter(exclude_models) or [])
    candidates = []
    for path in sorted(root.glob("**/*.pkl")):
        try:
            model = load_model_checkpoint(path)
            name = model_label(model)
            family = normalize_model_family(getattr(model, "model_family", infer_family_from_path(path)))
        except Exception as exc:
            candidates.append(
                {
                    "name": path.stem,
                    "family": infer_family_from_path(path),
                    "path": path,
                    "mtime": path.stat().st_mtime,
                    "loadable": False,
                    "reason": str(exc),
                    "model": None,
                }
            )
            continue
        candidates.append(
            {
                "name": name,
                "family": family,
                "path": path,
                "mtime": path.stat().st_mtime,
                "loadable": True,
                "reason": "",
                "model": model,
            }
        )

    frame = pd.DataFrame([{k: v for k, v in row.items() if k != "model"} for row in candidates])
    loadable = [row for row in candidates if row["loadable"]]
    if include is not None:
        loadable = [row for row in loadable if row["name"] in include]
    if exclude:
        loadable = [row for row in loadable if row["name"] not in exclude]

    latest = {}
    for row in loadable:
        key = row["name"]
        if key not in latest or row["mtime"] > latest[key]["mtime"]:
            latest[key] = row
    entries = [
        {"name": row["name"], "family": row["family"], "path": row["path"], "model": row["model"]}
        for row in sorted(latest.values(), key=lambda item: (item["family"], item["name"]))
    ]
    return entries, frame


def blog_top25_table(poll_df):
    """Return the compact top-25 table used by the blog notebook."""
    if poll_df is None or poll_df.empty:
        return pd.DataFrame()
    keep = [
        col
        for col in [
            "rank",
            "keys_team",
            "poll_points",
            "first_place_votes",
            "average_rank",
            "best_rank",
            "worst_rank",
        ]
        if col in poll_df.columns
    ]
    return poll_df.loc[:, keep].head(25).reset_index(drop=True)


def blog_picks_table(matchup_summary):
    """Return the compact picks table used by the blog notebook."""
    if matchup_summary is None or matchup_summary.empty:
        return pd.DataFrame()
    keep = [
        col
        for col in [
            "keys_team_away",
            "keys_team_home",
            "score_projection",
            "predicted_winner",
            "winner_confidence",
            "model_agreement",
            "avg_pred_margin",
            "median_pred_margin",
            "model_count",
        ]
        if col in matchup_summary.columns
    ]
    return matchup_summary.loc[:, keep].reset_index(drop=True)


def blog_closest_games_table(matchup_summary, count=10):
    """Return the games with the smallest absolute consensus margin."""
    if matchup_summary is None or matchup_summary.empty:
        return pd.DataFrame()
    frame = matchup_summary.copy()
    frame["__abs_margin"] = pd.to_numeric(frame.get("avg_pred_margin"), errors="coerce").abs()
    sort_cols = [
        col
        for col in ["__abs_margin", "next_week", "keys_week", "keys_game_id", "next_game_id"]
        if col in frame.columns
    ]
    frame = frame.loc[frame["__abs_margin"].notna()].sort_values(
        sort_cols,
        na_position="last",
    )
    return blog_picks_table(frame.head(int(count))).reset_index(drop=True)


def normalize_model_filter(value):
    """Normalize include/exclude model filters."""
    if value in (None, "all"):
        return None
    if isinstance(value, str):
        return [value.strip()] if value.strip() else None
    out = [str(item).strip() for item in value if str(item).strip()]
    return out or None


def summarize_matchup_predictions(long_df, default_total=52.5):
    """Aggregate per-model matchup predictions into one consensus table."""
    if long_df is None or long_df.empty:
        return pd.DataFrame()

    frame = long_df.copy()
    frame["pred_margin"] = pd.to_numeric(frame["pred_margin"], errors="coerce")
    frame["pred_proba_home_win"] = pd.to_numeric(frame["pred_proba_home_win"], errors="coerce")
    group_cols = [
        col
        for col in [
            "keys_season",
            "next_week",
            "keys_week",
            "next_game_id",
            "keys_game_id",
            "keys_team_home",
            "keys_team_away",
            "market_over_under",
            "market_spread_close",
        ]
        if col in frame.columns
    ]
    if "keys_team_home" not in group_cols or "keys_team_away" not in group_cols:
        return pd.DataFrame()

    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values))
        margins = pd.to_numeric(group["pred_margin"], errors="coerce").dropna()
        probs = pd.to_numeric(group["pred_proba_home_win"], errors="coerce").dropna()
        avg_margin = float(margins.mean()) if len(margins) else np.nan
        avg_home_prob = float(probs.mean()) if len(probs) else np.nan
        home_pick = bool(avg_margin >= 0) if np.isfinite(avg_margin) else bool(avg_home_prob >= 0.5)
        model_picks = pd.to_numeric(group["pred_margin"], errors="coerce") >= 0
        agreement = float((model_picks == home_pick).mean()) if len(model_picks) else np.nan
        total = first_finite(row.get("market_over_under"), default_total)
        home_score = (float(total) + avg_margin) / 2.0 if np.isfinite(avg_margin) else np.nan
        away_score = (float(total) - avg_margin) / 2.0 if np.isfinite(avg_margin) else np.nan
        row.update(
            {
                "model_count": int(group["model"].nunique()) if "model" in group.columns else int(len(group)),
                "avg_pred_margin": round(avg_margin, 3) if np.isfinite(avg_margin) else np.nan,
                "median_pred_margin": round(float(margins.median()), 3) if len(margins) else np.nan,
                "std_pred_margin": round(float(margins.std(ddof=0)), 3) if len(margins) else np.nan,
                "avg_home_win_probability": round(avg_home_prob, 4) if np.isfinite(avg_home_prob) else np.nan,
                "predicted_winner": row["keys_team_home"] if home_pick else row["keys_team_away"],
                "winner_confidence": round(avg_home_prob if home_pick else 1.0 - avg_home_prob, 4) if np.isfinite(avg_home_prob) else np.nan,
                "model_agreement": round(agreement, 4) if np.isfinite(agreement) else np.nan,
                "pred_home_score": round(home_score, 1) if np.isfinite(home_score) else np.nan,
                "pred_away_score": round(away_score, 1) if np.isfinite(away_score) else np.nan,
                "score_projection": f"{row['keys_team_home']} {home_score:.1f}, {row['keys_team_away']} {away_score:.1f}" if np.isfinite(home_score) else "",
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [col for col in ["next_week", "keys_week", "keys_game_id", "next_game_id"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out.reset_index(drop=True)


def compare_feature_edges(X_a, X_b, team_a, team_b, top_n=24):
    """Return the largest raw fingerprint differences between two teams."""
    a = X_a.reset_index(drop=True).iloc[0]
    b = X_b.reset_index(drop=True).iloc[0]
    rows = []
    for feature in X_a.columns.intersection(X_b.columns):
        a_val = pd.to_numeric(pd.Series([a[feature]]), errors="coerce").iloc[0]
        b_val = pd.to_numeric(pd.Series([b[feature]]), errors="coerce").iloc[0]
        if not np.isfinite(a_val) or not np.isfinite(b_val):
            continue
        delta = float(a_val - b_val)
        rows.append(
            {
                "feature": feature,
                "team_a": team_a,
                "team_b": team_b,
                "team_a_value": float(a_val),
                "team_b_value": float(b_val),
                "delta_team_a_minus_team_b": delta,
                "abs_delta": abs(delta),
                "edge_team": team_a if delta >= 0 else team_b,
                "feature_group": feature_group(feature),
                "thing_to_watch": f"{team_a if delta >= 0 else team_b} edge in {str(feature).replace('_', ' ')}",
            }
        )
    return pd.DataFrame(rows).sort_values("abs_delta", ascending=False).head(top_n).reset_index(drop=True)


def plot_poll_top25_bar(poll_df, path, top_n=25, logo_dir=None, title=None):
    """Plot current top-25 poll points as a horizontal blog-ready figure."""
    if poll_df is None or poll_df.empty:
        return None
    import matplotlib.pyplot as plt

    frame = poll_df.head(top_n).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(10, max(7, len(frame) * 0.34)))
    y = np.arange(len(frame))
    ax.barh(y, pd.to_numeric(frame["poll_points"], errors="coerce"), color="#1EA7FF", alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{int(rank)}. {team}" for rank, team in zip(frame["rank"], frame["keys_team"])], fontsize=8)
    ax.set_xlabel("Poll Points")
    ax.set_title(title or f"Top {top_n} TDNet Poll")
    ax.grid(axis="x", color="#DDE2E7", linewidth=0.8)
    for idx, (_, row) in enumerate(frame.iterrows()):
        logo_path = resolve_team_logo_path(row["keys_team"], logo_dir)
        if logo_path is not None:
            draw_team_logo(ax, logo_path, -max(frame["poll_points"]) * 0.025, idx, target_px=16)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_upcoming_matchups_table(summary_df, path, title=None, max_rows=80):
    """Render matchup prediction summary as a compact table PNG."""
    if summary_df is None or summary_df.empty:
        return None
    keep = [
        col
        for col in [
            "keys_team_away",
            "keys_team_home",
            "score_projection",
            "predicted_winner",
            "winner_confidence",
            "model_agreement",
            "avg_pred_margin",
        ]
        if col in summary_df.columns
    ]
    return plot_dataframe_table(summary_df.loc[:, keep].head(max_rows), path, title=title or "Upcoming Matchups")


def plot_team_comparison_edges(edges_df, path, title=None, top_n=14):
    """Plot largest team fingerprint differences."""
    if edges_df is None or edges_df.empty:
        return None
    import matplotlib.pyplot as plt

    frame = edges_df.head(top_n).iloc[::-1].copy()
    colors = np.where(frame["delta_team_a_minus_team_b"] >= 0, "#1EA7FF", "#FF5FA2")
    fig, ax = plt.subplots(figsize=(10, max(5, len(frame) * 0.38)))
    ax.barh(np.arange(len(frame)), frame["delta_team_a_minus_team_b"], color=colors, alpha=0.9)
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(frame["feature"].map(lambda value: str(value).replace("_", " ")), fontsize=7)
    ax.axvline(0, color="#3A4450", linewidth=1.0)
    ax.set_title(title or "Biggest Fingerprint Edges")
    ax.set_xlabel("Team A minus Team B")
    ax.grid(axis="x", color="#DDE2E7", linewidth=0.8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_team_matchup_predictions(predictions_df, path, title=None):
    """Plot model margins for a selected Team A vs Team B matchup."""
    if predictions_df is None or predictions_df.empty or "pred_margin" not in predictions_df.columns:
        return None
    import matplotlib.pyplot as plt

    frame = predictions_df.copy()
    frame["pred_margin"] = pd.to_numeric(frame["pred_margin"], errors="coerce")
    frame = frame.dropna(subset=["pred_margin"]).sort_values("pred_margin")
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(4, len(frame) * 0.35)))
    colors = [source_style(row["model"], {}, idx)["color"] for idx, row in frame.reset_index(drop=True).iterrows()]
    ax.barh(np.arange(len(frame)), frame["pred_margin"], color=colors, alpha=0.9)
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(frame["model"], fontsize=8)
    ax.axvline(0, color="#3A4450", linewidth=1.0)
    ax.set_xlabel("Predicted margin for Team A as home team")
    ax.set_title(title or "Selected Matchup Model Margins")
    ax.grid(axis="x", color="#DDE2E7", linewidth=0.8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dataframe_table(frame, path, title=None):
    """Save a pandas dataframe as a simple PNG table."""
    if frame is None or frame.empty:
        return None
    import matplotlib.pyplot as plt

    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.3g}")
    fig_height = max(3.5, len(display) * 0.32 + 1.2)
    fig_width = max(10, len(display.columns) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display.astype(str).values,
        colLabels=[str(col).replace("_", " ").title() for col in display.columns],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDE2E7")
        if row == 0:
            cell.set_facecolor("#11214F")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F5F8")
    if title:
        ax.set_title(title, pad=18, weight="bold")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def concat_without_duplicate_columns(left, right):
    """Concatenate two aligned frames while dropping duplicated columns from right."""
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True) if right is not None else pd.DataFrame(index=left.index)
    keep = [col for col in right.columns if col not in left.columns]
    return pd.concat([left, right.loc[:, keep]], axis=1)


def first_finite(value, fallback):
    """Return value if it is finite, otherwise fallback."""
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if np.isfinite(parsed) else float(fallback)


def infer_family_from_path(path):
    """Infer model family from checkpoint path parts."""
    parts = [str(part).lower() for part in Path(path).parts]
    for family in ["linear", "stat", "tree", "knn"]:
        if family in parts:
            return family
    return "model"


def feature_group(feature):
    """Map a feature name to a broad football feature group."""
    text = str(feature).lower()
    if "statoff" in text or "offense" in text:
        return "offense"
    if "statdef" in text or "defense" in text:
        return "defense"
    if "statspe" in text or "special" in text or "kick" in text or "punt" in text:
        return "special_teams"
    if "turnover" in text or "penalt" in text or "fumble" in text or "interception" in text:
        return "discipline"
    if "talent" in text or "recruit" in text or "roster" in text:
        return "talent"
    return "general"


def safe_name(value):
    """Return a filesystem-safe lowercase identifier."""
    text = str(value).strip().lower()
    out = []
    for char in text:
        out.append(char if char.isalnum() else "_")
    return "_".join("".join(out).split("_"))
