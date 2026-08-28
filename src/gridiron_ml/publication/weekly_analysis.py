"""Descriptive weekly analysis figures kept separate from release assets."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .bundles import sha256_file
from .figure_theme import TDNET_COLORS, apply_tdnet_theme
from .poll_explainability import build_ap_peer_signal_proxy, plot_top25_discrepancy_features
from .preseason_states import build_preseason_state_frame
from .social_predictions import select_featured_games


MATCHUP_FEATURES = (
    ("roster_talent", "Roster talent", 1.0, "roster"),
    ("roster_return_percent_p_p_a", "Returning production", 1.0, "roster"),
    ("offense_ppa", "Offensive efficiency", 1.0, "offense"),
    ("defense_ppa", "Defensive efficiency", -1.0, "defense"),
    ("offense_success_rate", "Offensive success rate", 1.0, "offense"),
    ("defense_success_rate", "Defensive success prevention", -1.0, "defense"),
    ("statOff_yards_per_pass", "Yards per pass", 1.0, "offense"),
    ("statOff_yards_per_rush_attempt", "Yards per rush", 1.0, "offense"),
    ("statGen_turnovers", "Ball security", -1.0, "ball_security"),
    ("coach_career_mean_sp_offense", "Coach offensive history", 1.0, "coaching"),
    ("coach_career_mean_sp_defense", "Coach defensive history", 1.0, "coaching"),
)


def select_week_state(
    frame: pd.DataFrame, *, season: int, week: int, project_root: str | Path | None = None
) -> pd.DataFrame:
    if int(week) == 0:
        try:
            return build_preseason_state_frame(frame, season=season, project_root=project_root)
        except ValueError:
            pass
    state = frame.copy()
    if "keys_season" in state:
        season_rows = state[pd.to_numeric(state["keys_season"], errors="coerce").eq(season)]
        if not season_rows.empty:
            state = season_rows
    if "keys_week" in state:
        weeks = pd.to_numeric(state["keys_week"], errors="coerce")
        eligible = state.loc[weeks.le(max(1, week))]
        if not eligible.empty:
            state = eligible
    sort_columns = [column for column in ("keys_season", "keys_week", "keys_game_date") if column in state]
    if sort_columns:
        state = state.sort_values(sort_columns, kind="stable")
    return state.drop_duplicates("keys_team", keep="last").reset_index(drop=True)


def build_matchup_signals(
    games: pd.DataFrame,
    tdnet_poll: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    season: int,
    week: int,
    signals_per_game: int = 3,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    state = select_week_state(
        feature_frame, season=season, week=week, project_root=project_root
    ).set_index("keys_team")
    available = [item for item in MATCHUP_FEATURES if item[0] in state]
    numeric = state[[feature for feature, _, _, _ in available]].apply(pd.to_numeric, errors="coerce")
    scale = numeric.std(ddof=0).replace(0, np.nan)
    standardized = (numeric - numeric.median()) / scale
    featured, sickos = select_featured_games(games, tdnet_poll)
    selected = [(f"Featured {index}", game) for index, game in enumerate(featured, start=1)]
    if sickos is not None:
        selected.append(("Sickos game", sickos))
    rows = []
    for role, game in selected:
        if game.home_team not in state.index or game.away_team not in state.index:
            continue
        candidates = []
        for feature, label, direction, category in available:
            home_raw = numeric.at[game.home_team, feature]
            away_raw = numeric.at[game.away_team, feature]
            home_z = standardized.at[game.home_team, feature]
            away_z = standardized.at[game.away_team, feature]
            if pd.isna(home_z) or pd.isna(away_z):
                continue
            edge = float((home_z - away_z) * direction)
            candidates.append((abs(edge), feature, label, category, edge, home_raw, away_raw))
        candidates.sort(key=lambda row: (-row[0], row[1]))
        chosen = []
        used_categories = set()
        for candidate in candidates:
            category = candidate[3]
            if category in used_categories:
                continue
            chosen.append(candidate)
            used_categories.add(category)
            if len(chosen) == signals_per_game:
                break
        for _, feature, label, category, edge, home_raw, away_raw in chosen:
            rows.append({
                "game_id": game.game_id,
                "role": role,
                "away_team": game.away_team,
                "home_team": game.home_team,
                "feature": feature,
                "signal": label,
                "signal_category": category,
                "home_edge_z": edge,
                "home_raw": float(home_raw),
                "away_raw": float(away_raw),
                "edge_favors": game.home_team if edge >= 0 else game.away_team,
                "away_display": _format_matchup_value(feature, away_raw),
                "home_display": _format_matchup_value(feature, home_raw),
            })
    return pd.DataFrame(rows)


def plot_matchup_signals(signals: pd.DataFrame, path: str | Path, *, season: int, week: int) -> Path:
    if signals.empty:
        raise ValueError("No matchup signals are available to plot.")
    apply_tdnet_theme()
    games = signals[["game_id", "role", "away_team", "home_team"]].drop_duplicates().to_dict("records")
    fig, axes = plt.subplots(len(games), 1, figsize=(15.2, max(9.0, 3.25 * len(games))), squeeze=False)
    for axis, game in zip(axes[:, 0], games):
        frame = signals[signals["game_id"].astype(str).eq(str(game["game_id"]))].sort_values("home_edge_z")
        colors = [TDNET_COLORS["ion_blue"] if value >= 0 else TDNET_COLORS["edge_pink"] for value in frame["home_edge_z"]]
        labels = [
            f"{row.signal}\n{game['away_team']}: {row.away_display}  |  {game['home_team']}: {row.home_display}"
            for row in frame.itertuples(index=False)
        ]
        axis.barh(labels, frame["home_edge_z"], color=colors, height=.58)
        axis.axvline(0, color=TDNET_COLORS["slate"], lw=.9)
        axis.set_title(
            f"{game['role']}  •  {game['away_team']} at {game['home_team']}",
            loc="left", fontsize=14.5, weight="bold",
        )
        axis.set_xlabel(f"← {game['away_team']} edge     standardized preseason signal     {game['home_team']} edge →")
        axis.grid(axis="x", alpha=.35)
        axis.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle(f"TDNet {season} Week {week}: Key Matchup Signals", fontsize=21, weight="bold", x=.07, ha="left")
    fig.text(.07, .012, "Descriptive frozen-F6 preseason signals, selected by largest standardized team difference. These are context, not causal explanations or betting advice.", fontsize=10.5, color=TDNET_COLORS["slate"])
    fig.tight_layout(rect=[0, .04, 1, .95])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return target


def _format_matchup_value(feature: str, value: float) -> str:
    if "percent" in feature or "success_rate" in feature:
        return f"{float(value):.1%}"
    if feature == "roster_talent":
        return f"{float(value):.0f}"
    if "ppa" in feature:
        return f"{float(value):.3f}"
    if "yards_per" in feature:
        return f"{float(value):.2f}"
    if feature == "statGen_turnovers":
        return f"{float(value):.2f}"
    return f"{float(value):.1f}"


def build_weekly_analysis(
    *,
    games_path: str | Path,
    tdnet_poll_path: str | Path,
    ap_poll_path: str | Path,
    fingerprint_path: str | Path,
    feature_metadata_path: str | Path,
    output_root: str | Path,
    season: int,
    week: int,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    output = Path(output_root)
    figures, tables, metadata = output / "figures", output / "tables", output / "metadata"
    for directory in (figures, tables, metadata):
        directory.mkdir(parents=True, exist_ok=True)
    games = pd.read_csv(games_path)
    tdnet = pd.read_csv(tdnet_poll_path)
    ap = pd.read_csv(ap_poll_path)
    features = pd.read_parquet(fingerprint_path)
    feature_metadata = pd.read_csv(feature_metadata_path)
    poll = tdnet.rename(columns={"team": "keys_team"}).copy()
    ap_ranks = ap.set_index("team")["rank"]
    poll["ap_rank"] = poll["keys_team"].map(ap_ranks)
    state = select_week_state(features, season=season, week=week, project_root=project_root)
    disparity = build_ap_peer_signal_proxy(poll, state, feature_metadata, reference_poll=ap)
    disparity.to_csv(tables / "tdnet_vs_ap_disparity_signals.csv", index=False)
    if not disparity.empty:
        plot_top25_discrepancy_features(
            disparity,
            figures / "tdnet_vs_ap_disparity_signals.png",
            title=f"TDNet {season} Week {week}: Disparity Signals vs AP Peers",
        )
    matchup = build_matchup_signals(
        games, tdnet, features, season=season, week=week, project_root=project_root
    )
    matchup.to_csv(tables / "games_of_the_week_key_stats.csv", index=False)
    plot_matchup_signals(matchup, figures / "games_of_the_week_key_stats.png", season=season, week=week)
    manifest = {
        "season": season,
        "week": week,
        "analysis_only": True,
        "games_sha256": sha256_file(games_path),
        "tdnet_poll_sha256": sha256_file(tdnet_poll_path),
        "ap_poll_sha256": sha256_file(ap_poll_path),
        "fingerprint_sha256": sha256_file(fingerprint_path),
        "feature_metadata_sha256": sha256_file(feature_metadata_path),
        "matchup_selection": "same three featured games plus Sickos game as prediction social card",
        "matchup_signal_policy": "three largest standardized differences from fixed interpretable F6 feature shortlist",
        "disparity_policy": "descriptive AP-rank-peer proxy; not SHAP or causal attribution",
    }
    (metadata / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"disparity": disparity, "matchup": matchup, "manifest": manifest}
