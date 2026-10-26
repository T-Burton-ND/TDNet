#!/usr/bin/env python3
"""Fill Week 8 postgame graphic-type gaps across the three publication rosters.

Inputs are saved scored predictions and saved postgame polls. This is a replay
of figure construction, not a new prediction run. Expanded F20 margins remain
retrospective applications and its matched Week 8 cohort has 49 games.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
POST = ROOT / "publication/2026/week_08/post_game"
OUT = POST / "three_roster_graphics"
PRIVATE = ROOT / "data/private/postgame_2026_week08"
AP = PRIVATE / "ap_top25_week09.csv"
MATCHED = POST / "three_roster_comparison/matched_49_game_roster_consensus_results.csv"
F20_PRIOR = ROOT / "data/private/f20/replay_2026_week01_07_v3_margin/f20_replay.parquet"
F0_F19_PRIOR = PRIVATE / "prior_f0_f19_model_games.parquet"
NAVY, BLUE, TEAL, ORANGE, GOLD, PAPER = "#20324b", "#2d7192", "#3c8d93", "#d47638", "#d4a928", "#f7f3eb"
ROSTERS = {
    "margin_wide": ("Margin-wide", "Margin-wide", 33, 53, 378),
    "scientific_f0_f6": ("Frozen scientific F0–F6", "Scientific F0–F6", 42, 53, 378),
    "scientific_f0_f20": ("Scientific F0–F20 research", "Scientific F0–F20", 126, 49, 353),
}


@dataclass
class Roster:
    key: str
    label: str
    betting_label: str
    models_week: pd.DataFrame
    models_cumulative: pd.DataFrame
    games_week: pd.DataFrame
    games_cumulative: pd.DataFrame
    power: pd.DataFrame
    ballots: pd.DataFrame


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_models(frame: pd.DataFrame, *, week: int, source: str) -> pd.DataFrame:
    required = {"game_id", "model_name", "pred_home_margin", "pred_home_win_probability", "actual_home_margin"}
    if missing := required - set(frame):
        raise ValueError(f"{source}: missing {sorted(missing)}")
    result = frame.copy()
    result["publication_week"] = int(week)
    result["model_family"] = result.get("model_family", pd.Series("model", index=result.index)).fillna("model")
    result["fingerprint"] = result.get("fingerprint", pd.Series("", index=result.index)).fillna("")
    for column in ("pred_home_margin", "pred_home_win_probability", "actual_home_margin"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    if "market_spread_close" not in result:
        result["market_spread_close"] = np.nan
    result["market_spread_close"] = pd.to_numeric(result.market_spread_close, errors="coerce")
    if result.duplicated(["game_id", "model_name"]).any():
        raise ValueError(f"{source}: duplicate model/game")
    return result


def _consensus(models: pd.DataFrame, *, expected: int) -> pd.DataFrame:
    if models.duplicated(["game_id", "model_name"]).any():
        raise ValueError("Duplicate model/game in consensus cohort")
    grouped = models.groupby("game_id", as_index=False).agg(
        pred_home_margin=("pred_home_margin", "mean"),
        pred_home_win_probability=("pred_home_win_probability", "mean"),
        actual_home_margin=("actual_home_margin", "first"),
        market_spread_close=("market_spread_close", "first"),
        model_count=("model_name", "nunique"),
        publication_week=("publication_week", "first"),
    )
    if not grouped.model_count.eq(expected).all():
        raise ValueError(f"Expected {expected} models for every consensus game")
    grouped["actual_home_win"] = grouped.actual_home_margin.gt(0)
    grouped["su_correct"] = grouped.pred_home_win_probability.ge(.5).eq(grouped.actual_home_win)
    grouped["brier"] = (grouped.pred_home_win_probability - grouped.actual_home_win.astype(float)) ** 2
    grouped["margin_absolute_error"] = (grouped.pred_home_margin - grouped.actual_home_margin).abs()
    return grouped


def _published_consensus(*, key: str, week: int, raw_models: pd.DataFrame,
                         source_paths: list[Path]) -> pd.DataFrame:
    base = ROOT / f"publication/2026/week_{week:02d}"
    if key == "margin_wide":
        path = base / "post_game/tables/margin_wide_prediction_vs_actual.csv"
        frame = pd.read_csv(path)
        source_paths.append(path)
        result = frame[["game_id", "pred_home_margin", "pred_home_win_probability",
                        "actual_home_margin", "market_spread_close", "su_correct"]].copy()
    else:
        if week == 6:
            path = base / "pre_game/scientific/scientific_all_game_predictions.csv"
            frame = pd.read_csv(path).drop_duplicates("game_id")
            actual = raw_models[["game_id", "actual_home_margin"]].drop_duplicates("game_id")
            frame = frame.merge(actual, on="game_id", validate="one_to_one")
            source_paths.append(path)
            result = frame[["game_id", "consensus_predicted_home_win_probability",
                            "consensus_predicted_winner_margin", "consensus_straight_up_pick",
                            "home_team", "actual_home_margin", "market_spread_close"]].copy()
            result["pred_home_margin"] = np.where(
                result.consensus_straight_up_pick.eq(result.home_team),
                result.consensus_predicted_winner_margin,
                -result.consensus_predicted_winner_margin)
            result["su_correct"] = result.consensus_straight_up_pick.eq(
                result.home_team.where(result.actual_home_margin.gt(0), frame.away_team))
        else:
            path = base / "post_game/scientific/scientific_consensus_game_results.csv"
            saved = pd.read_csv(path)
            source_paths.append(path)
            probability = raw_models[["game_id", "consensus_predicted_home_win_probability"]].drop_duplicates("game_id")
            result = saved.merge(probability, on="game_id", validate="one_to_one")
            result = result.rename(columns={"consensus_predicted_home_margin": "pred_home_margin",
                                            "consensus_winner_correct": "su_correct",
                                            "consensus_home_team_market_spread": "market_spread_close"})
            result = result[["game_id", "pred_home_margin", "consensus_predicted_home_win_probability",
                             "actual_home_margin", "market_spread_close", "su_correct"]].copy()
        result = result.rename(columns={"consensus_predicted_home_win_probability": "pred_home_win_probability"})
    result["publication_week"] = int(week)
    for column in ("pred_home_margin", "pred_home_win_probability", "actual_home_margin", "market_spread_close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if (result.game_id.duplicated().any() or result[["pred_home_margin", "pred_home_win_probability",
                                                "actual_home_margin"]].isna().any().any()):
        raise ValueError(f"{key}: Week {week} consensus is incomplete")
    result["su_correct"] = result.su_correct.astype(bool)
    result["actual_home_win"] = result.actual_home_margin.gt(0)
    result["brier"] = (result.pred_home_win_probability - result.actual_home_win.astype(float)) ** 2
    result["margin_absolute_error"] = (result.pred_home_margin - result.actual_home_margin).abs()
    return result


def load_rosters() -> tuple[dict[str, Roster], list[Path]]:
    source_paths: list[Path] = [AP, MATCHED]
    rosters = {}
    for key in ("margin_wide", "scientific_f0_f6"):
        label, betting_label, expected, week_games, cumulative_games = ROSTERS[key]
        relative = ("tables/margin_wide_model_game_results.csv" if key == "margin_wide"
                    else "scientific/scientific_model_game_results.csv")
        current_path = POST / relative
        current = pd.read_csv(current_path)
        current_names = set(current.model_name)
        if len(current_names) != expected:
            raise ValueError(f"{key}: incorrect current model roster")
        frames = []
        consensus_frames = []
        for week in (0, 1, 2, 3, 4, 5, 6, 8):
            path = ROOT / f"publication/2026/week_{week:02d}/post_game/{relative}"
            if key == "scientific_f0_f6" and week == 6 and not path.exists():
                # The 42 cells are present in the frozen F0–F8 Week 6 component.
                path = ROOT / "publication/2026/week_06/post_game/scientific/full_f0_f8/scientific_model_game_results.csv"
            source_paths.append(path)
            raw = pd.read_csv(path)
            if key == "scientific_f0_f6" and week == 6:
                checkpoint = current[["model_name", "checkpoint_sha256"]].drop_duplicates()
                check = raw.loc[raw.model_name.isin(current_names), ["model_name", "checkpoint_sha256"]].drop_duplicates()
                if (len(check) != expected or not check.set_index("model_name").checkpoint_sha256.sort_index().equals(
                        checkpoint.set_index("model_name").checkpoint_sha256.sort_index())):
                    raise ValueError("Week 6 F0–F6 component does not use the frozen 42 checkpoints")
            frame = _normalize_models(raw.loc[raw.model_name.isin(current_names)], week=week, source=str(path))
            if frame.model_name.nunique() != expected:
                raise ValueError(f"{key}: Week {week} lacks a current-roster model")
            frames.append(frame)
            consensus_frames.append(_published_consensus(key=key, week=week, raw_models=raw,
                                                          source_paths=source_paths))
        cumulative = pd.concat(frames, ignore_index=True)
        weekly = frames[-1]
        consensus_week = consensus_frames[-1]
        consensus_cumulative = pd.concat(consensus_frames, ignore_index=True)
        if (consensus_week.game_id.nunique() != week_games
                or consensus_cumulative.game_id.nunique() != cumulative_games):
            raise ValueError(f"{key}: Week 8 or cumulative game count changed")
        power_path = POST / ("tables/margin_wide_consensus_power_rankings.csv" if key == "margin_wide"
                             else "scientific/scientific_consensus_power_rankings.csv")
        ballot_path = POST / ("tables/tdnet_model_ballots.csv" if key == "margin_wide"
                              else "scientific/scientific_full_ballots.csv")
        source_paths.extend([power_path, ballot_path])
        rosters[key] = Roster(key, label, betting_label, weekly, cumulative, consensus_week,
                              consensus_cumulative, pd.read_csv(power_path), pd.read_csv(ballot_path))

    key = "scientific_f0_f20"
    label, betting_label, expected, week_games, cumulative_games = ROSTERS[key]
    week_path = POST / "scientific/expanded_f0_f20/week08_f0_f20_126_model_games_matched.csv"
    power_path = POST / "scientific/expanded_f0_f20/power_rankings_research/scientific_136_team_power_rankings.csv"
    ballot_path = POST / "scientific/expanded_f0_f20/power_rankings_research/scientific_126_model_ballots.csv"
    source_paths.extend([week_path, power_path, ballot_path, F0_F19_PRIOR, F20_PRIOR,
                         PRIVATE / "prior_market_games.csv"])
    weekly = _normalize_models(pd.read_csv(week_path), week=8, source=str(week_path))
    prior = pd.read_parquet(F0_F19_PRIOR)
    replay = pd.read_parquet(F20_PRIOR)
    market = pd.read_csv(PRIVATE / "prior_market_games.csv")
    old_ids = set(replay.game_id) & set(prior.game_id) & set(market.loc[market.market_home_probability.notna(), "game_id"])
    if len(old_ids) != 304:
        raise ValueError("F0–F20 cumulative shared cohort is not 304 prior games")
    earlier = prior.loc[prior.game_id.isin(old_ids)].copy()
    earlier = earlier.merge(market[["game_id", "market_home_spread"]].drop_duplicates("game_id"),
                            on="game_id", validate="many_to_one")
    earlier = earlier.rename(columns={"market_home_spread": "market_spread_close"})
    earlier["model_family"] = earlier.model_id
    f20_rows = []
    for model_id in ("M1", "M2", "M3", "M4", "M5", "M10"):
        part = replay.loc[replay.game_id.isin(old_ids),
                          ["game_id", "cfbd_week", "actual_home_margin", f"margin_{model_id}", f"probability_{model_id}"]].copy()
        part = part.rename(columns={"cfbd_week": "week", f"margin_{model_id}": "pred_home_margin",
                                    f"probability_{model_id}": "pred_home_win_probability"})
        part["fingerprint"] = "F20"
        part["model_id"] = model_id
        part["model_name"] = f"scientific_F20_{model_id}"
        part["model_family"] = model_id
        part = part.merge(market[["game_id", "market_home_spread"]].drop_duplicates("game_id"),
                          on="game_id", validate="one_to_one")
        part = part.rename(columns={"market_home_spread": "market_spread_close"})
        f20_rows.append(part)
    earlier = pd.concat([earlier, *f20_rows], ignore_index=True)
    earlier["publication_week"] = earlier["week"].astype(int)
    cumulative = pd.concat([earlier, weekly], ignore_index=True)
    if (weekly.game_id.nunique() != week_games or weekly.model_name.nunique() != expected
            or len(weekly) != week_games * expected or len(cumulative) != cumulative_games * expected
            or cumulative.duplicated(["game_id", "model_name"]).any()):
        raise ValueError("F0–F20 Week 8 / cumulative panel is incomplete")
    consensus_week = _consensus(weekly, expected=expected)
    consensus_cumulative = _consensus(cumulative, expected=expected)
    rosters[key] = Roster(key, label, betting_label, weekly, cumulative, consensus_week,
                          consensus_cumulative, pd.read_csv(power_path), pd.read_csv(ballot_path))
    return rosters, list(dict.fromkeys(source_paths))


def _axes(*, nrows: int = 1, ncols: int = 1, width: float = 14, height: float = 7):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), facecolor=PAPER, constrained_layout=True)
    for axis in np.atleast_1d(axes).ravel():
        axis.set_facecolor(PAPER)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.18)
        axis.set_axisbelow(True)
    return fig, axes


def _finish(fig, path: Path, *, note: str = "") -> None:
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(.015, .065, .985, .90))
    if note:
        fig.text(.5, .008, note, ha="center", va="bottom", fontsize=9, color="#526173")
    fig.savefig(path, dpi=185, facecolor=fig.get_facecolor())
    plt.close(fig)


def _scope_note(roster: Roster, cumulative: bool = False) -> str:
    if roster.key == "scientific_f0_f20":
        return ("353 shared games, Weeks 1–6 and 8; F20 Week 7 unavailable. F20 replay / postgame-applied margins."
                if cumulative else "49 matched Week 8 games; F20 margins applied after games to archived pregame features.")
    return ("Weeks 0–6 and 8; Week 7 is excluded. Frozen F0–F6 Week 6 cells are recovered from its F0–F8 component."
            if roster.key == "scientific_f0_f6" and cumulative else
            "Weeks 0–6 and 8; Week 7 retrospective replay is excluded." if cumulative else
            f"{len(roster.games_week)} frozen Week 8 games.")


def render_game_results(roster: Roster, path: Path) -> None:
    games = roster.games_week.sort_values("game_id").copy()
    names = roster.models_week.drop_duplicates("game_id").set_index("game_id")[["home_team", "away_team"]]
    games = games.join(names, on="game_id")
    cells = []
    for row in games.itertuples():
        away = str(row.away_team)[:20]
        home = str(row.home_team)[:20]
        cells.append([f"{away} @ {home}", f"{row.pred_home_margin:+.1f}",
                      f"{row.actual_home_margin:+.0f}",
                      "H" if row.pred_home_win_probability >= .5 else "A",
                      "✓" if row.su_correct else "×",
                      f"{row.margin_absolute_error:.1f}"])
    half = (len(cells) + 1) // 2
    headers = ["Matchup", "Pred", "Actual", "Pick", "SU", "Error"]
    left = cells[:half]
    right = cells[half:] + [[""] * 6] * (half - len(cells[half:]))
    fig, axes = _axes(ncols=2, width=19, height=15)
    fig.suptitle(f"{roster.label} · Week 8 game results", fontsize=22, weight="bold", color=NAVY)
    for ax, data in zip(axes, (left, right)):
        ax.axis("off")
        artist = ax.table(cellText=data, colLabels=headers, cellLoc="left", colLoc="left",
                          bbox=[0, .035, 1, .95], colWidths=[.50, .11, .11, .07, .06, .11])
        artist.auto_set_font_size(False)
        artist.set_fontsize(8.8)
        for (row_index, _), cell in artist.get_celld().items():
            cell.set_edgecolor("#d5dbe1")
            cell.set_facecolor(NAVY if row_index == 0 else "#fff" if row_index % 2 else "#eef2f4")
            if row_index == 0:
                cell.set_text_props(color="white", weight="bold")
    _finish(fig, path, note=_scope_note(roster) + " SU follows the probability pick; Pred is home margin.")


def _model_metrics(models: pd.DataFrame) -> pd.DataFrame:
    frame = models.copy()
    actual_home = frame.actual_home_margin.gt(0)
    # Published SU picks can differ from a thresholded probability when the
    # original scorer used its saved winner pick. Preserve that graded result.
    graded = pd.Series(np.nan, index=frame.index)
    for column in ("model_winner_correct", "su_correct", "winner_correct"):
        if column in frame:
            graded = graded.fillna(pd.to_numeric(frame[column], errors="coerce").astype(float))
    frame["su_correct"] = graded.fillna(
        frame.pred_home_win_probability.ge(.5).eq(actual_home).astype(float)).astype(bool)
    frame["brier"] = (frame.pred_home_win_probability - actual_home.astype(float)) ** 2
    frame["margin_error"] = (frame.pred_home_margin - frame.actual_home_margin).abs()
    cover = frame.pred_home_margin + frame.market_spread_close
    actual_cover = frame.actual_home_margin + frame.market_spread_close
    frame["ats_decision"] = frame.market_spread_close.notna() & cover.ne(0) & actual_cover.ne(0)
    frame["ats_win"] = frame.ats_decision & np.sign(cover).eq(np.sign(actual_cover))
    return frame.groupby("model_name", as_index=False).agg(
        fingerprint=("fingerprint", "first"), family=("model_family", "first"),
        games=("game_id", "nunique"), su=("su_correct", "mean"),
        ats_wins=("ats_win", "sum"), ats_decisions=("ats_decision", "sum"),
        brier=("brier", "mean"), margin_mae=("margin_error", "mean"),
    ).assign(ats=lambda data: data.ats_wins / data.ats_decisions.replace(0, np.nan))


def render_scientific_cumulative_scorecard(roster: Roster, path: Path) -> None:
    metrics = _model_metrics(roster.models_cumulative)
    metrics["cell"] = metrics.fingerprint + "/" + metrics.model_name.str.extract(r"_(M\d+)$", expand=False)
    fingerprints = ([*(f"F{i}" for i in range(17)), "F17-market", "F18", "F19", "F20"]
                    if roster.key == "scientific_f0_f20" else [f"F{i}" for i in range(7)])
    architectures = ["M1", "M2", "M3", "M4", "M5", "M10"]
    fig, axes = _axes(nrows=2, ncols=2, width=15, height=12)
    for ax, (column, title, cmap) in zip(axes.flat, [
        ("su", "Straight-up accuracy", "YlGnBu"), ("ats", "ATS accuracy", "YlGnBu"),
        ("brier", "Brier · lower is better", "YlOrBr_r"),
        ("margin_mae", "Margin MAE · lower is better", "YlOrBr_r")]):
        grid = metrics.assign(architecture=metrics.model_name.str.extract(r"_(M\d+)$", expand=False)).pivot(
            index="fingerprint", columns="architecture", values=column).reindex(index=fingerprints, columns=architectures)
        values = grid.to_numpy(dtype=float)
        image = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(6), architectures)
        ax.set_yticks(range(len(fingerprints)), fingerprints, fontsize=8)
        ax.set_title(title, fontsize=14)
        fig.colorbar(image, ax=ax, fraction=.045, pad=.025)
    fig.suptitle(f"{roster.label} · cumulative {len(metrics)}-model scorecard",
                 fontsize=20, weight="bold", color=NAVY)
    _finish(fig, path, note=_scope_note(roster, cumulative=True))


def render_cumulative_performance(roster: Roster, path: Path) -> None:
    rows = []
    for week in sorted(roster.games_cumulative.publication_week.unique()):
        through = roster.games_cumulative.loc[roster.games_cumulative.publication_week.le(week)]
        if len(through) < 8:
            continue
        rows.append({"week": week, "su": through.su_correct.mean(), "brier": through.brier.mean(),
                     "margin_mae": through.margin_absolute_error.mean(), "games": len(through)})
    frame = pd.DataFrame(rows)
    fig, axes = _axes(ncols=3, width=17, height=5.5)
    for ax, (column, title, unit) in zip(axes, [
        ("su", "Straight-up accuracy", "%"), ("brier", "Brier score", ""),
        ("margin_mae", "Margin MAE", " points")]):
        ax.plot(frame.week, frame[column], color=TEAL, lw=3, marker="o", markersize=8)
        for row in frame.itertuples():
            label = f"{getattr(row, column):.0%}" if unit == "%" else f"{getattr(row, column):.3f}" if column == "brier" else f"{getattr(row, column):.1f}"
            ax.annotate(label, (row.week, getattr(row, column)), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(frame.week)
        ax.set_xlabel("Completed week")
        ax.set_ylabel(unit)
    fig.suptitle(f"{roster.label} · cumulative performance through Week 8", fontsize=19, weight="bold", color=NAVY)
    _finish(fig, path, note=_scope_note(roster, cumulative=True))


def render_calibration(roster: Roster, path: Path, *, cumulative: bool) -> None:
    models = roster.models_cumulative if cumulative else roster.models_week
    groups = models.copy()
    groups["group"] = groups.fingerprint.astype(str)
    groups["actual_home_win"] = groups.actual_home_margin.gt(0).astype(float)
    bins = np.linspace(0, 1, 6)
    fig, ax = _axes(width=11, height=8)
    for index, (name, frame) in enumerate(groups.groupby("group", sort=True)):
        frame = frame.copy()
        frame["bin"] = pd.cut(frame.pred_home_win_probability, bins=bins, include_lowest=True)
        points = frame.groupby("bin", observed=True).agg(pred=("pred_home_win_probability", "mean"),
                                                         observed=("actual_home_win", "mean"))
        ax.plot(points.pred, points.observed, marker="o", markersize=4, lw=1.4,
                color=plt.cm.turbo(index / max(1, groups.group.nunique() - 1)),
                alpha=.7, label=name if len(name) < 16 else name.replace("F17-market", "F17 market"))
    ax.plot([0, 1], [0, 1], ls="--", color=NAVY, lw=1.5, label="Ideal")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted home-win probability",
           ylabel="Observed home-win rate")
    ax.set_title(f"{roster.label} · {'cumulative' if cumulative else 'Week 8'} calibration", fontsize=17, color=NAVY)
    ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="upper left")
    _finish(fig, path, note=_scope_note(roster, cumulative=cumulative))


def render_parity(roster: Roster, path: Path, *, cumulative: bool) -> None:
    games = roster.games_cumulative if cumulative else roster.games_week
    pick = games.pred_home_margin.gt(0)
    result = games.actual_home_margin.gt(0)
    matrix = np.array([[int((~pick & ~result).sum()), int((~pick & result).sum())],
                       [int((pick & ~result).sum()), int((pick & result).sum())]])
    fig, ax = _axes(width=9, height=7.2)
    ax.grid(False)
    ax.imshow(matrix, cmap="YlGnBu", vmin=0)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value} games", ha="center", va="center", fontsize=20,
                color="white" if value > matrix.max() * .58 else NAVY, weight="bold")
    ax.set_xticks([0, 1], ["Away won", "Home won"])
    ax.set_yticks([0, 1], ["Picked away", "Picked home"])
    ax.set_title(f"{roster.label} · {'cumulative' if cumulative else 'Week 8'} margin parity", fontsize=17, color=NAVY)
    _finish(fig, path, note=_scope_note(roster, cumulative=cumulative))


def render_poll_extra(roster: Roster, path: Path, *, family: str) -> None:
    power = roster.power.sort_values("poll_points_rank").copy()
    ballots = roster.ballots[["keys_team", "ballot_model", "ballot_rank"]].copy()
    ap = pd.read_csv(AP).set_index("team")["rank"]
    fig, ax = _axes(width=12, height=9)
    top = power.head(25)
    if family == "rank_vs_margin":
        ax.scatter(power.poll_points_rank, power.predicted_margin_vs_average_team,
                   color=BLUE, s=47, alpha=.7)
        for row in power.head(15).itertuples():
            ax.annotate(str(row.keys_team), (row.poll_points_rank, row.predicted_margin_vs_average_team),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.axhline(0, ls="--", color=GOLD)
        ax.set(xlabel="TDNet ballot rank", ylabel="Predicted margin vs average team · points")
        title = "Ballot rank versus power margin"
    elif family == "poll_disagreement":
        variation = ballots.loc[ballots.keys_team.isin(top.keys_team)].groupby("keys_team").ballot_rank.std().sort_values()
        ax.barh(variation.index, variation.values, color=TEAL)
        ax.invert_yaxis()
        ax.set(xlabel="SD of model ballot rank", ylabel="Team")
        title = "Model ballot disagreement · Top 25"
    elif family == "top25_spread":
        stats = ballots.loc[ballots.keys_team.isin(top.keys_team)].groupby("keys_team").ballot_rank.agg(["min", "median", "max"])
        stats = stats.reindex(top.keys_team)
        y = np.arange(len(stats))
        ax.hlines(y, stats["min"], stats["max"], color="#90aaba", lw=2)
        ax.scatter(stats["median"], y, color=ORANGE, s=38, zorder=3)
        ax.set_yticks(y, stats.index)
        ax.invert_yaxis()
        ax.set(xlabel="Model ballot rank · range and median", ylabel="Team")
        title = "Top 25 model ballot spread"
    elif family == "ap_comparison":
        joined = top.assign(ap_rank=top.keys_team.map(ap)).dropna(subset=["ap_rank"])
        y = np.arange(len(joined))
        ax.hlines(y, joined.poll_points_rank, joined.ap_rank, color="#90aaba", lw=2)
        ax.scatter(joined.poll_points_rank, y, color=BLUE, s=60, label="TDNet")
        ax.scatter(joined.ap_rank, y, color=ORANGE, s=60, label="Week 9 AP")
        ax.set_yticks(y, joined.keys_team)
        ax.invert_yaxis()
        ax.set(xlabel="Rank", ylabel="Team")
        ax.legend(frameon=False)
        title = "TDNet versus Week 9 AP Top 25"
    else:
        raise ValueError(family)
    ax.set_title(f"{roster.label} · {title}", fontsize=16, color=NAVY)
    note = ("F0–F20 ranking is a descriptive 353-game prediction-network fit, not a live Week 9 state poll."
            if roster.key == "scientific_f0_f20" else
            "Week 9 AP reference; TDNet ballot points and model power margins are distinct summaries.")
    _finish(fig, path, note=note)


def _betting_data(roster: Roster) -> pd.DataFrame:
    matched = pd.read_csv(MATCHED)
    frame = matched.loc[matched.roster.eq(roster.betting_label)].copy()
    if len(frame) != 49 or frame.game_id.duplicated().any():
        raise ValueError(f"{roster.key}: matched betting ledger is not 49 games")
    models = roster.models_week.loc[roster.models_week.game_id.isin(frame.game_id)].copy()
    models["model_cover"] = np.sign(models.pred_home_margin + models.market_spread_close)
    consensus_side = np.sign(frame.set_index("game_id").pred_home_margin +
                             frame.set_index("game_id").market_spread_close)
    models["consensus_side"] = models.game_id.map(consensus_side)
    models["supports"] = models.model_cover.eq(models.consensus_side)
    support = models.groupby("game_id").supports.mean()
    count = models.groupby("game_id").model_name.nunique()
    if not count.eq(ROSTERS[roster.key][2]).all():
        raise ValueError("Betting support lacks model votes")
    frame["ats_confidence"] = frame.game_id.map(support)
    frame["moneyline_confidence"] = frame.pred_home_win_probability.where(
        frame.pred_home_win_probability.ge(.5), 1 - frame.pred_home_win_probability)
    frame = frame.sort_values(["game_start_time_utc", "game_id"]).reset_index(drop=True)
    return frame


def render_betting(roster: Roster, path: Path, *, family: str) -> None:
    frame = _betting_data(roster)
    ats = family.startswith("ats") or family.startswith("scaled_ats")
    scaled = family.startswith("scaled")
    sweep = family.endswith("sweep")
    confidence = frame.ats_confidence if ats else frame.moneyline_confidence
    profit_10 = frame.ats_profit_10 if ats else frame.moneyline_profit_10
    eligible = frame.ats_result.isin(["Win", "Loss", "Push"]) if ats else frame.moneyline_quoted.astype(bool)
    name = "ATS · assumed -110" if ats else "Moneyline · saved pregame odds"
    fig, ax = _axes(width=12, height=6.8)
    if sweep:
        cutoffs = np.round(np.arange(.50, .951, .025), 3)
        profits, counts = [], []
        for cutoff in cutoffs:
            selected = eligible & confidence.ge(cutoff)
            count = int(selected.sum())
            counts.append(count)
            profits.append(float(profit_10.loc[selected].sum()) if count >= 10 else np.nan)
        ax.plot(cutoffs, profits, color=TEAL, marker="o", lw=2.5)
        ax.axhline(0, color=NAVY, lw=1)
        ax.set(xlabel="Minimum confidence", ylabel="Net profit · flat $10 per qualifying game")
        for x, y, n in zip(cutoffs[::3], profits[::3], counts[::3]):
            if np.isfinite(y):
                ax.annotate(f"n={n}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        title = f"{name} confidence threshold sweep"
    else:
        stake = (np.clip(25 * (confidence - .499) / (1 - .499), 0, 25)
                 if scaled else pd.Series(10.0, index=frame.index))
        stake = stake.where(eligible, 0)
        profit = profit_10 * stake / 10
        ax.plot(np.arange(1, 50), profit.cumsum(), color=TEAL, lw=3)
        ax.axhline(0, color=NAVY, lw=1)
        ax.set(xlabel="Completed game in kickoff order", ylabel="Cumulative net profit · $",
               xlim=(1, 49))
        title = f"{name} · {'confidence-scaled' if scaled else 'flat $10'} bankroll"
    ax.set_title(f"{roster.label} · {title}", fontsize=17, color=NAVY)
    note = "Same 49 Week 8 games for all three rosters. Descriptive in-sample results; F20 margins applied after games."
    _finish(fig, path, note=note)


def render_violins(roster: Roster, path: Path) -> None:
    metrics = _model_metrics(roster.models_week)
    group = "fingerprint"
    if roster.key == "margin_wide":
        family = metrics.family.astype(str)
        metrics["violin_group"] = np.select(
            [family.eq("linear"), family.isin(["tree", "boosted", "temporal"]),
             family.isin(["kernel", "knn"]), family.eq("ensemble")],
            ["Linear", "Tree / temporal", "Kernel / KNN", "Ensemble"], default="Other")
        group = "violin_group"
    names = sorted(metrics[group].unique())
    fig, axes = _axes(nrows=2, ncols=2, width=15, height=10)
    for ax, (column, title) in zip(axes.flat, [("su", "Straight-up accuracy"),
                                               ("ats", "ATS accuracy"),
                                               ("brier", "Brier score"),
                                               ("margin_mae", "Margin MAE · points")]):
        values = [metrics.loc[metrics[group].eq(name), column].dropna().to_numpy() for name in names]
        violin = ax.violinplot(values, showmedians=True, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(TEAL)
            body.set_alpha(.7)
        violin["cmedians"].set_color(NAVY)
        ax.set_xticks(range(1, len(names) + 1), names, rotation=35 if group == "family" else 0,
                      ha="right" if group == "family" else "center", fontsize=8)
        ax.set_title(title, fontsize=14)
    fig.suptitle(f"{roster.label} · Week 8 model distributions", fontsize=19, color=NAVY, weight="bold")
    _finish(fig, path, note=_scope_note(roster))


def render_sunday_performance(roster: Roster, path: Path) -> None:
    metrics = _model_metrics(roster.models_week).sort_values("margin_mae").head(12)
    labels = metrics.model_name.str.replace("scientific_", "", regex=False).str.replace("margin_", "", regex=False)
    fig, axes = _axes(ncols=2, width=16, height=7)
    y = np.arange(len(metrics))
    axes[0].barh(y, metrics.margin_mae, color=TEAL)
    axes[1].barh(y, metrics.su, color=BLUE)
    for ax in axes:
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
    axes[0].set(xlabel="Margin MAE · points", title="Best 12 by margin error")
    axes[1].set(xlabel="Straight-up accuracy", title="Same 12 models · SU")
    fig.suptitle(f"{roster.label} · Week 8 Sunday performance", fontsize=19, color=NAVY, weight="bold")
    _finish(fig, path, note=_scope_note(roster))


def render_rotation(roster: Roster, png: Path, gif: Path) -> None:
    metrics = _model_metrics(roster.models_cumulative)
    groups = metrics.family if roster.key == "margin_wide" else metrics.fingerprint
    names = sorted(groups.unique())
    cmap = plt.get_cmap("turbo")
    colors = {name: cmap(i / max(1, len(names) - 1)) for i, name in enumerate(names)}
    fig = plt.figure(figsize=(12.7, 9.2), facecolor=PAPER)
    ax = fig.add_axes([.06, .14, .83, .77], projection="3d", facecolor=PAPER)
    for row, group in zip(metrics.itertuples(), groups):
        ax.scatter(row.su * 100, row.ats * 100, row.brier, s=145, color=colors[group],
                   edgecolors=PAPER, linewidths=.8, depthshade=False)
    for values, setter, floor in [(metrics.su * 100, ax.set_xlim, 1.0),
                                  (metrics.ats * 100, ax.set_ylim, 1.0),
                                  (metrics.brier, ax.set_zlim, .003)]:
        low, high = values.min(), values.max()
        pad = max((high - low) * .11, floor)
        setter(low - pad, high + pad)
    ax.set(xlabel="SU accuracy · % ↑", ylabel="ATS accuracy · % ↑", zlabel="Brier ↓")
    ax.tick_params(labelsize=11)
    ax.set_box_aspect((1.15, 1.15, .85), zoom=1.12)
    ax.view_init(elev=23, azim=-55)
    fig.suptitle(f"{roster.label} · cumulative SU, ATS, Brier", fontsize=21, weight="bold", color=NAVY)
    fig.text(.5, .08, f"One marker per model · {len(metrics)} cells · {_scope_note(roster, cumulative=True)}",
             ha="center", fontsize=10, color="#526173")
    fig.savefig(png, dpi=170, facecolor=PAPER)
    animation = FuncAnimation(fig, lambda step: ax.view_init(elev=23, azim=-55 + step * 15),
                              frames=24, interval=125, blit=False)
    animation.save(gif, writer=PillowWriter(fps=8), dpi=75, savefig_kwargs={"facecolor": PAPER})
    plt.close(fig)


EXISTING = {
    "margin_wide": {
        "weekly_game_results": "figures/margin_wide_prediction_scorecard.png",
        "weekly_model_scorecard": "figures/margin_wide_model_scorecard.png",
        "cumulative_model_scorecard": "figures/cumulative_margin_wide_performance.png",
        "cumulative_performance": "figures/cumulative_margin_wide_performance.png",
        "weekly_calibration": [f"figures/calibration_by_model_group/{name}_calibration.png" for name in
                               ("linear", "neighbors_and_kernels", "trees_and_temporal", "ensembles_and_other")],
        "cumulative_calibration": "figures/margin_wide_cumulative_model_calibration.png",
        "weekly_parity": "figures/team_margin_parity.png",
        "cumulative_parity": "figures/cumulative_team_margin_parity.png",
        "all_team_power": "figures/margin_wide_all_team_power_rankings.png",
        "top25": "figures/tdnet_top25.png",
        "ballots": "figures/tdnet_model_ballots.png",
        "top10_4x5": "figures/tdnet_top10_social_4x5.png",
        "top10_16x9": "figures/tdnet_top10_social_16x9.png",
        "rank_vs_margin": "figures/margin_wide_rank_vs_projected_margin.png",
        "poll_disagreement": "figures/model_consensus_disagreement.png",
        "top25_spread": "figures/top25_consensus_spread.png",
        "ap_comparison": "figures/tdnet_vs_ap_top25.png",
        "sunday_performance": "figures/sunday_performance.png",
    },
    "scientific_f0_f6": {
        "weekly_game_results": "scientific/scientific_consensus_game_results.png",
        "weekly_model_scorecard": "scientific/scientific_model_scorecard.png",
        "weekly_calibration": [f"scientific/calibration_by_fingerprint/f{i}_calibration.png" for i in range(7)],
        "all_team_power": "scientific/scientific_all_team_power_rankings.png",
        "top25": "scientific/scientific_tdnet_top25.png",
        "ballots": "scientific/scientific_top25_ballots.png",
        "top10_4x5": "scientific/scientific_research_ballot_social_4x5.png",
        "top10_16x9": "scientific/scientific_research_ballot_social_16x9.png",
    },
    "scientific_f0_f20": {
        "weekly_model_scorecard": "scientific/expanded_f0_f20/week08_126_model_scorecard.png",
        "all_team_power": "scientific/expanded_f0_f20/power_rankings_research/scientific_all_team_power_rankings.png",
        "top25": "scientific/expanded_f0_f20/power_rankings_research/scientific_top25.png",
        "ballots": "scientific/expanded_f0_f20/power_rankings_research/scientific_top25_ballots.png",
        "top10_4x5": "scientific/expanded_f0_f20/power_rankings_research/scientific_top10_research_4x5.png",
        "top10_16x9": "scientific/expanded_f0_f20/power_rankings_research/scientific_top10_research_16x9.png",
        "violin": "scientific/expanded_f0_f20/week08_126_model_violins.png",
        "rotation_png": "scientific/expanded_f0_f20/cumulative/cumulative_su_ats_brier_3d_matched_353_games_week08.png",
        "rotation_gif": "scientific/expanded_f0_f20/cumulative/cumulative_su_ats_brier_3d_matched_353_games_week08.gif",
    },
}

FAMILIES = (
    "weekly_game_results", "weekly_model_scorecard", "cumulative_model_scorecard",
    "cumulative_performance", "weekly_calibration", "cumulative_calibration",
    "weekly_parity", "cumulative_parity", "all_team_power", "top25", "ballots",
    "top10_4x5", "top10_16x9", "rank_vs_margin", "poll_disagreement",
    "top25_spread", "ap_comparison", "ats_bankroll", "ml_bankroll",
    "scaled_ats_bankroll", "scaled_ml_bankroll", "ats_sweep", "ml_sweep",
    "violin", "rotation_png", "rotation_gif", "sunday_performance",
)


def _make_missing(roster: Roster, family: str, path: Path) -> None:
    if family == "weekly_game_results":
        render_game_results(roster, path)
    elif family == "cumulative_model_scorecard":
        render_scientific_cumulative_scorecard(roster, path)
    elif family == "cumulative_performance":
        render_cumulative_performance(roster, path)
    elif family in {"weekly_calibration", "cumulative_calibration"}:
        render_calibration(roster, path, cumulative=family.startswith("cumulative"))
    elif family in {"weekly_parity", "cumulative_parity"}:
        render_parity(roster, path, cumulative=family.startswith("cumulative"))
    elif family in {"rank_vs_margin", "poll_disagreement", "top25_spread", "ap_comparison"}:
        render_poll_extra(roster, path, family=family)
    elif family in {"ats_bankroll", "ml_bankroll", "scaled_ats_bankroll", "scaled_ml_bankroll",
                    "ats_sweep", "ml_sweep"}:
        render_betting(roster, path, family=family)
    elif family == "violin":
        render_violins(roster, path)
    elif family == "sunday_performance":
        render_sunday_performance(roster, path)
    else:
        raise ValueError(f"No renderer for missing {roster.key}/{family}")


def _validate_published_week08(rosters: dict[str, Roster], sources: list[Path]) -> None:
    """Require the model values used in new charts to match saved scorecards."""
    comparisons = {
        "margin_wide": (POST / "tables/margin_wide_model_scorecard.csv", "model_id",
                        {"su": "su_accuracy", "ats": "ats_accuracy_excluding_pushes",
                         "brier": "brier_score", "margin_mae": "margin_mae"}),
        "scientific_f0_f6": (POST / "scientific/scientific_model_scorecard.csv", "model_name",
                             {"su": "straight_up_accuracy", "margin_mae": "margin_mae",
                              "ats_wins": "ats_wins"}),
        "scientific_f0_f20": (
            POST / "scientific/expanded_f0_f20/full_roster_scorecards/week08_matched_49.csv",
            "label", {"su": "su_accuracy", "ats": "ats_accuracy", "brier": "brier",
                      "margin_mae": "margin_mae", "ats_wins": "ats_wins"}),
    }
    for key, (path, index, columns) in comparisons.items():
        observed = _model_metrics(rosters[key].models_week)
        saved = pd.read_csv(path)
        if key == "scientific_f0_f20":
            observed["label"] = observed.fingerprint + "/" + observed.model_name.str.extract(
                r"_(M\d+)$", expand=False)
            saved = saved.loc[saved.row_type.eq("model")]
        observed = observed.set_index(index if index == "label" else "model_name")
        saved = saved.set_index(index)
        if set(observed.index) != set(saved.index):
            raise ValueError(f"{key}: Week 8 published model roster differs")
        for calculated, published in columns.items():
            error = (observed[calculated] - saved[published]).abs().max()
            if not np.isfinite(error) or error > 1e-8:
                raise ValueError(f"{key}: Week 8 {calculated} differs from saved scorecard ({error})")
        sources.append(path)


def main() -> None:
    rosters, sources = load_rosters()
    _validate_published_week08(rosters, sources)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    generated: list[Path] = []
    for key, roster in rosters.items():
        folder = OUT / key
        folder.mkdir(parents=True, exist_ok=True)
        for family in FAMILIES:
            existing = EXISTING[key].get(family)
            if existing:
                paths = [POST / value for value in (existing if isinstance(existing, list) else [existing])]
                if not all(path.is_file() for path in paths):
                    raise FileNotFoundError(f"{key}/{family}: an existing Week 8 graphic is missing")
                status = "existing_week08"
            elif family == "rotation_png":
                png, gif = folder / "rotation_png.png", folder / "rotation_gif.gif"
                render_rotation(roster, png, gif)
                paths = [png]
                generated.extend([png, gif])
                status = "generated_from_saved_results"
            elif family == "rotation_gif":
                paths = [folder / "rotation_gif.gif"]
                if not paths[0].exists():
                    raise FileNotFoundError(paths[0])
                status = "generated_from_saved_results"
            else:
                path = folder / f"{family}.png"
                _make_missing(roster, family, path)
                paths = [path]
                generated.append(path)
                status = "generated_from_saved_results"
            records.append({"roster": key, "graphic_type": family, "status": status,
                            "week08_games": len(roster.games_week),
                            "cumulative_games": len(roster.games_cumulative),
                            "files": ";".join(str(path.relative_to(POST)) for path in paths)})
    matrix = pd.DataFrame(records)
    if len(matrix) != len(FAMILIES) * 3 or matrix.duplicated(["roster", "graphic_type"]).any():
        raise ValueError("Three-roster graphic matrix is incomplete")
    for row in matrix.itertuples():
        for name in row.files.split(";"):
            if not (POST / name).exists():
                raise FileNotFoundError(name)
    matrix_path = OUT / "coverage_matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    readme_path = OUT / "README.md"
    readme_path.write_text(
        "# Week 8 postgame · all three roster graphic types\n\n"
        "`coverage_matrix.csv` maps 27 graphic types to every one of the three roster views. "
        "Entries marked `existing_week08` point to verified graphics already in the Week 8 package; "
        "new files here fill the other cells. The rejected `lost_graphics/` review styles are not included. "
        "All new figures are built from saved scored forecasts and polls, without refitting models.\n\n"
        "- Margin-wide: 33 models, 53 frozen Week 8 games, 378 games from Weeks 0–6 and 8.\n"
        "- Original scientific F0–F6: 42 models, 53 frozen Week 8 games, 378 games from Weeks 0–6 and 8. "
        "The saved Week 6 F0–F8 component supplies its 42 F0–F6 cells; all checkpoint hashes match the frozen Week 8 roster. "
        "This fills a gap in the prior 320-game F0–F6 cumulative figure without changing that earlier artifact.\n"
        "- Expanded scientific F0–F20: 126 models on 49 Week 8 games; cumulative matched graphics use 353 shared "
        "games from Weeks 1–6 and 8. F20 has no Week 7 coverage. Its earlier expert-input replay is retrospective; "
        "Week 8 F20 margins were applied after games to archived pregame features. Its power rankings are a descriptive "
        "353-game prediction-network fit, not a live Week 9 state poll.\n\n"
        "The six betting graphic types use the same 49 Week 8 games for all three rosters. ATS assumes -110; "
        "moneyline uses saved pregame odds. Flat stakes risk $10 per eligible game. Scaled stakes use $0 at "
        "49.9% confidence and $25 at 100%; ATS confidence is the fraction of models backing the consensus spread side, "
        "while moneyline confidence is the consensus picked-team probability. Threshold sweep points require "
        "at least ten eligible games and are descriptive in-sample results. Vegas has no ATS record against its own line.\n\n"
        "`manifest.json` records source and generated-image SHA-256 hashes. The regular weekly renderer's files "
        "were left intact; this completeness pass adds the missing views for the latest postgame package.\n",
        encoding="utf-8",
    )
    manifest = {"schema": "tdnet-week08-three-roster-graphic-completeness-v1", "completed_week": 8,
                "graphic_types": list(FAMILIES), "rosters": {key: {"models": ROSTERS[key][2],
                    "week08_games": len(roster.games_week), "cumulative_games": len(roster.games_cumulative)}
                    for key, roster in rosters.items()},
                "sources": {str(path.relative_to(ROOT)): digest(path) for path in sources},
                "files": {str(path.relative_to(OUT)): digest(path) for path in [*generated, matrix_path, readme_path]}}
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"graphic_types": len(FAMILIES), "coverage_rows": len(matrix),
                      "generated_files": len(generated), "week08_games": {key: len(roster.games_week)
                      for key, roster in rosters.items()}}))


if __name__ == "__main__":
    main()
