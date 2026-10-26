#!/usr/bin/env python3
"""Rebuild retired margin-roster diagnostics from saved 2026 game forecasts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "publication/2026/week_08/post_game/lost_graphics/margin_roster"
WEEK7 = ROOT / "publication/2026/week_07/post_game/retrospective_replay"
WEEK7_MARKET = ROOT / "data/private/postgame_2026_week08/prior_market_games.csv"
POSTGAME_FIGURE = re.compile(r"^publication/2026/week_(\d{2})/post_game/figures/(.+\.(?:png|gif))$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_inventory() -> pd.DataFrame:
    history = subprocess.check_output(
        ["git", "log", "--all", "--format=", "--name-only", "--diff-filter=ACMR", "--", "publication/2026"],
        cwd=ROOT, text=True,
    ).splitlines()
    by_name: dict[str, set[int]] = {}
    for path in history:
        match = POSTGAME_FIGURE.match(path)
        if match:
            by_name.setdefault(match.group(2), set()).add(int(match.group(1)))
    current = set()
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "publication/2026/week_08/post_game/figures"],
        cwd=ROOT, text=True,
    ).splitlines()
    for path in paths:
        if "/post_game/figures/" in path and path.lower().endswith((".png", ".gif")):
            current.add(path.split("/post_game/figures/", 1)[1])
    if len(by_name) != 28 or len(current) != 26:
        raise ValueError(f"Margin history/current figure census changed: {len(by_name)} / {len(current)}")
    missing = set(by_name) - current
    if missing != {"exploratory_postgame_diagnostics.png", "tdnet_vs_ap_preseason_top25.png"}:
        raise ValueError(f"Unexpected retired margin figure names: {missing}")
    rows = []
    for name in sorted(by_name):
        status = "still_generated_week08" if name in current else (
            "retired_result_diagnostic" if name == "exploratory_postgame_diagnostics.png"
            else "preseason_AP_name_variant"
        )
        rows.append({"figure_relative_name": name,
                     "committed_postgame_weeks": ",".join(map(str, sorted(by_name[name]))),
                     "week08_figure_exists": name in current,
                     "classification": status})
    return pd.DataFrame(rows)


def load_game_records() -> tuple[pd.DataFrame, dict[str, Path]]:
    frames = []
    sources: dict[str, Path] = {}
    for week in (*range(7), 8):
        root = ROOT / f"publication/2026/week_{week:02d}/post_game/tables"
        consensus_path = root / "margin_wide_prediction_vs_actual.csv"
        model_path = root / "margin_wide_model_game_results.csv"
        consensus = pd.read_csv(consensus_path)
        models = pd.read_csv(model_path)
        sources[f"week{week:02d}_consensus"] = consensus_path
        sources[f"week{week:02d}_models"] = model_path
        frames.append(_combine_week(consensus, models, week=week, source_class="saved_weekly_postgame"))

    consensus_path = WEEK7 / "week07_replay_consensus_games.csv"
    model_path = WEEK7 / "week07_replay_model_games.parquet"
    consensus = pd.read_csv(consensus_path)
    models = pd.read_parquet(model_path)
    market = pd.read_csv(WEEK7_MARKET)
    market = market.loc[market.week.eq(7), ["game_id", "market_home_spread"]].rename(
        columns={"market_home_spread": "market_spread_close"})
    consensus = consensus.merge(market, on="game_id", validate="one_to_one")
    sources.update({"week07_replay_consensus": consensus_path,
                    "week07_replay_models": model_path,
                    "week07_retrospective_market": WEEK7_MARKET})
    frames.append(_combine_week(consensus, models, week=7,
                                source_class="retrospective_week07_replay"))
    games = pd.concat(frames, ignore_index=True).sort_values(["week", "game_id"]).reset_index(drop=True)
    if len(games) != 437 or games.game_id.nunique() != 437:
        raise ValueError("Expected 437 distinct scored margin-roster games through Week 8")
    expected = {0: 8, 1: 39, 2: 49, 3: 57, 4: 58, 5: 56, 6: 58, 7: 59, 8: 53}
    if games.groupby("week").size().to_dict() != expected:
        raise ValueError("Margin-roster weekly counts differ from saved postgame/replay records")
    if not games[["pred_home_margin", "pred_home_win_probability", "actual_home_margin", "model_margin_sd"]].notna().all().all():
        raise ValueError("Missing saved prediction, result, or model disagreement")
    published = pd.read_csv(ROOT / "publication/2026/week_08/post_game/tables/cumulative_margin_wide_performance.csv")
    row = published.loc[published.series_type.eq("consensus") & published.through_week.eq(8)].iloc[0]
    no_replay = games.loc[games.week.ne(7)]
    wins = int(no_replay.su_correct.sum())
    brier = float(((no_replay.pred_home_win_probability - no_replay.actual_home_margin.gt(0).astype(float)) ** 2).mean())
    if len(no_replay) != int(row.games) or wins != int(row.su_wins) or not np.isclose(brier, row.brier_score, atol=2e-6):
        raise ValueError("Week 8 cumulative diagnostic source does not reconcile to published consensus")
    return games, sources


def _combine_week(consensus: pd.DataFrame, models: pd.DataFrame, *, week: int,
                  source_class: str) -> pd.DataFrame:
    if consensus.game_id.duplicated().any() or models.duplicated(["game_id", "model_name"]).any():
        raise ValueError(f"Week {week} has duplicate game or model forecast")
    provider_week = 1 if week == 0 else week
    if not consensus.week.eq(provider_week).all() or not models.week.eq(provider_week).all():
        raise ValueError(f"Week {week} source has inconsistent provider week")
    group = models.groupby("game_id").agg(
        model_count_verified=("model_name", "nunique"),
        model_margin_sd=("pred_home_margin", lambda values: values.std(ddof=0)),
    ).reset_index()
    fields = ["game_id", "week", "pred_home_margin", "pred_home_win_probability",
              "model_count", "actual_home_margin", "market_spread_close"]
    out = consensus[fields].merge(group, on="game_id", validate="one_to_one")
    out["week"] = week  # Week 0 is the opening publication slate of CFBD Week 1.
    if not out.model_count.eq(out.model_count_verified).all():
        raise ValueError(f"Week {week} consensus model count differs from individual models")
    if not out.model_count.eq(36 if week == 0 else 33).all():
        raise ValueError(f"Week {week} published margin roster has unexpected size")
    probability = pd.to_numeric(out.pred_home_win_probability, errors="coerce")
    if not probability.between(0, 1).all():
        raise ValueError(f"Week {week} margin-roster probability outside [0,1]")
    picked_home = out.pred_home_margin.gt(0)
    out["su_correct"] = picked_home.eq(out.actual_home_margin.gt(0))
    out["confidence"] = np.where(picked_home, probability, 1 - probability)
    out["tdnet_margin_abs_error"] = (out.pred_home_margin - out.actual_home_margin).abs()
    out["vegas_margin_abs_error"] = (-out.market_spread_close - out.actual_home_margin).abs()
    out["source_class"] = source_class
    if "su_correct" in consensus and not out.su_correct.eq(consensus.set_index("game_id").loc[out.game_id, "su_correct"].astype(bool).to_numpy()).all():
        raise ValueError(f"Week {week} winner grades differ from saved consensus")
    return out


def summarize(games: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    confidence = pd.cut(games.confidence, bins=[0, .6, .75, 1.0], include_lowest=True,
                        labels=["≤60%", "60–75%", "75–100%"])
    for label in confidence.cat.categories:
        part = games.loc[confidence.eq(label)]
        rows.append({"scope": scope, "panel": "confidence_calibration", "bucket": label,
                     "games": len(part), "tdnet_value": part.confidence.mean(),
                     "reference_value": part.su_correct.mean(), "value_units": "fraction",
                     "reference_label": "Observed SU accuracy"})
    with_line = games.loc[games.market_spread_close.notna()].copy()
    spread_bin = pd.cut(with_line.market_spread_close.abs(), bins=[0, 3.5, 7, 14, np.inf],
                        include_lowest=True, labels=["≤3.5", "3.5–7", "7–14", ">14"])
    for label in spread_bin.cat.categories:
        part = with_line.loc[spread_bin.eq(label)]
        rows.append({"scope": scope, "panel": "margin_error_by_spread", "bucket": label,
                     "games": len(part), "tdnet_value": part.tdnet_margin_abs_error.mean(),
                     "reference_value": part.vegas_margin_abs_error.mean(), "value_units": "points",
                     "reference_label": "Vegas spread MAE"})
    tertile = pd.qcut(games.model_margin_sd.rank(method="first"), q=3,
                      labels=["Low", "Middle", "High"])
    for label in tertile.cat.categories:
        part = games.loc[tertile.eq(label)]
        rows.append({"scope": scope, "panel": "margin_error_by_model_disagreement", "bucket": label,
                     "games": len(part), "tdnet_value": part.tdnet_margin_abs_error.mean(),
                     "reference_value": np.nan, "value_units": "points",
                     "reference_label": ""})
    result = pd.DataFrame(rows)
    if len(result) != 10:
        raise ValueError(f"{scope}: expected three confidence, four spread, and three disagreement bins")
    for panel in result.panel.unique():
        expected = int(games.market_spread_close.notna().sum()) if panel == "margin_error_by_spread" else len(games)
        if result.loc[result.panel.eq(panel), "games"].sum() != expected:
            raise ValueError(f"{scope}: {panel} bucket counts do not cover intended games")
    return result


def render(summary: pd.DataFrame, *, scope: str, subtitle: str, note: str, path: Path) -> None:
    navy, blue, orange, brass, green, red = "#20324b", "#2d7192", "#d47638", "#d4a928", "#78a992", "#c34c4d"
    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.9), facecolor="#f7f3eb")
    fig.suptitle(f"TDNet margin roster · {subtitle}", color=navy, fontsize=22, fontweight="bold", y=.985)
    for ax in axes:
        ax.set_facecolor("#f7f3eb")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10)
    frame = summary.loc[summary.panel.eq("confidence_calibration")]
    x = np.arange(len(frame))
    axes[0].bar(x-.18, frame.tdnet_value, .35, color=blue, label="Mean picked-team confidence")
    axes[0].bar(x+.18, frame.reference_value, .35, color=orange, label="Observed SU accuracy")
    axes[0].set_xticks(x, [f"{row.bucket}\n(n={row.games})" for row in frame.itertuples()])
    axes[0].set_ylim(0, 1.04)
    axes[0].set_ylabel("Rate", fontsize=11)
    axes[0].set_title("Calibration by confidence", fontsize=15)
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False)
    frame = summary.loc[summary.panel.eq("margin_error_by_spread")]
    x = np.arange(len(frame))
    axes[1].bar(x-.18, frame.tdnet_value, .35, color=blue, label="TDNet consensus")
    axes[1].bar(x+.18, frame.reference_value, .35, color=brass, label="Vegas spread")
    axes[1].set_xticks(x, [f"{row.bucket}\n(n={row.games})" for row in frame.itertuples()])
    axes[1].set_ylabel("Margin MAE · points", fontsize=11)
    axes[1].set_title("Error by absolute market spread", fontsize=15)
    axes[1].legend(loc="upper left", fontsize=9, frameon=False)
    frame = summary.loc[summary.panel.eq("margin_error_by_model_disagreement")]
    x = np.arange(len(frame))
    axes[2].bar(x, frame.tdnet_value, .7, color=[green, brass, red])
    axes[2].set_xticks(x, [f"{row.bucket}\n(n={row.games})" for row in frame.itertuples()])
    axes[2].set_ylabel("Consensus margin MAE · points", fontsize=11)
    axes[2].set_title("Error by model disagreement", fontsize=15)
    fig.text(.5, .055, note, ha="center", va="center", fontsize=10.2, color="#4b5867")
    fig.tight_layout(rect=[.02, .105, .98, .91], w_pad=2.2)
    fig.savefig(path, dpi=205, facecolor=fig.get_facecolor())
    plt.close(fig)


def _scorecard_games(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Load only the 33 models retained in the current margin roster."""
    source_paths: dict[str, Path] = {}
    current_path = ROOT / "publication/2026/week_08/post_game/tables/margin_wide_model_game_results.csv"
    current_names = set(pd.read_csv(current_path, usecols=["model_name"]).model_name)
    if len(current_names) != 33:
        raise ValueError("Expected 33 current margin models")
    pieces = []
    for week in range(9):
        path = (WEEK7 / "week07_replay_model_games.parquet" if week == 7 else
                ROOT / f"publication/2026/week_{week:02d}/post_game/tables/margin_wide_model_game_results.csv")
        frame = pd.read_parquet(path) if week == 7 else pd.read_csv(path)
        source_paths[f"scorecard_week{week:02d}"] = path
        frame = frame.loc[frame.model_name.isin(current_names)].copy()
        expected_ids = set(games.loc[games.week.eq(week), "game_id"])
        if (set(frame.model_name) != current_names or set(frame.game_id) != expected_ids
                or frame.duplicated(["game_id", "model_name"]).any()
                or len(frame) != len(expected_ids) * 33):
            raise ValueError(f"Week {week} scorecard models do not cover the 33-model game cohort")
        frame["week"] = week
        pieces.append(frame[["week", "game_id", "model_name", "pred_home_margin",
                             "pred_home_win_probability", "actual_home_margin"]])
    models = pd.concat(pieces, ignore_index=True)
    models = models.merge(games[["week", "game_id", "market_spread_close"]],
                          on=["week", "game_id"], validate="many_to_one")
    return models, source_paths


def _ats_result(pred_margin: pd.Series, actual_margin: pd.Series, spread: pd.Series) -> pd.Series:
    edge = pred_margin + spread
    actual_cover = actual_margin + spread
    result = pd.Series("ungraded", index=pred_margin.index, dtype="string")
    graded = spread.notna() & edge.ne(0)
    result.loc[graded & actual_cover.eq(0)] = "push"
    result.loc[graded & actual_cover.ne(0) & np.sign(edge).eq(np.sign(actual_cover))] = "win"
    result.loc[graded & actual_cover.ne(0) & np.sign(edge).ne(np.sign(actual_cover))] = "loss"
    return result


def _score_row(frame: pd.DataFrame, *, scope: str, kind: str, name: str) -> dict[str, object]:
    actual_home = frame.actual_home_margin.gt(0)
    probability = frame.pred_home_win_probability
    spread = frame.market_spread_close
    ats = _ats_result(frame.pred_home_margin, frame.actual_home_margin, spread)
    return {"scope": scope, "series_type": kind, "model": name, "games": len(frame),
            "su_wins": int(frame.pred_home_margin.gt(0).eq(actual_home).sum()),
            "ats_wins": int(ats.eq("win").sum()), "ats_losses": int(ats.eq("loss").sum()),
            "ats_pushes": int(ats.eq("push").sum()), "ats_ungraded": int(ats.eq("ungraded").sum()),
            "brier_games": int(probability.notna().sum()),
            "brier_score": float(((probability - actual_home.astype(float)) ** 2).mean()),
            "margin_games": int(frame.pred_home_margin.notna().sum()),
            "margin_mae": float((frame.pred_home_margin - frame.actual_home_margin).abs().mean())}


def _scorecards(games: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    published = pd.read_csv(ROOT / "publication/2026/week_08/post_game/tables/cumulative_margin_wide_performance.csv")
    baseline = published.loc[published.series_type.eq("vegas") & published.through_week.eq(8)].iloc[0]
    market = pd.read_csv(WEEK7_MARKET).loc[lambda data: data.week.eq(7)]
    replay_brier = market.brier_error.dropna()
    replay_wins = int(market.winner_correct.sum())
    scopes = {"week08_current_53": games.loc[games.week.eq(8)],
              "cumulative_with_week07_replay_437": games,
              "cumulative_excluding_week07_replay_378": games.loc[games.week.ne(7)]}
    rows = []
    for scope, cohort in scopes.items():
        scoped_models = models.loc[models.week.isin(cohort.week.unique())]
        for name, frame in scoped_models.groupby("model_name", sort=True):
            rows.append(_score_row(frame, scope=scope, kind="model", name=name))
        consensus = cohort.copy()
        consensus["model"] = "TDNet margin-wide consensus"
        rows.append(_score_row(consensus, scope=scope, kind="consensus", name="TDNet margin-wide consensus"))
        with_spread = cohort.loc[cohort.market_spread_close.notna()].copy()
        if scope == "week08_current_53":
            saved_models = pd.read_csv(ROOT / "publication/2026/week_08/post_game/tables/margin_wide_model_game_results.csv")
            probability = saved_models.drop_duplicates("game_id").set_index("game_id").market_win_probability
            vegas_probability = with_spread.game_id.map(probability)
            vegas_brier_n = int(vegas_probability.notna().sum())
            vegas_brier = float(((vegas_probability - with_spread.actual_home_margin.gt(0).astype(float).to_numpy()) ** 2).mean())
            vegas_su = int((-with_spread.market_spread_close).gt(0).eq(with_spread.actual_home_margin.gt(0)).sum())
        else:
            vegas_brier_n = int(baseline.brier_games)
            vegas_brier = float(baseline.brier_score)
            vegas_su = int(baseline.su_wins)
            if scope == "cumulative_with_week07_replay_437":
                vegas_brier = (vegas_brier * vegas_brier_n + float(replay_brier.sum())) / (vegas_brier_n + len(replay_brier))
                vegas_brier_n += len(replay_brier)
                vegas_su += replay_wins
        rows.append({"scope": scope, "series_type": "vegas", "model": "Vegas closing-line reference",
                     "games": len(cohort), "su_wins": vegas_su,
                     "ats_wins": pd.NA, "ats_losses": pd.NA, "ats_pushes": pd.NA, "ats_ungraded": pd.NA,
                     "brier_games": vegas_brier_n, "brier_score": vegas_brier,
                     "margin_games": len(with_spread), "margin_mae": float(with_spread.vegas_margin_abs_error.mean())})
    result = pd.DataFrame(rows)
    if result.groupby("scope").size().to_dict() != {scope: 35 for scope in scopes}:
        raise ValueError("Each standalone scorecard must have 33 models, consensus, and Vegas")
    for scope, cohort in scopes.items():
        section = result.loc[result.scope.eq(scope)]
        if not section.loc[section.series_type.eq("model"), "games"].eq(len(cohort)).all():
            raise ValueError(f"{scope}: a model is missing scored games")
    return result


def render_scorecard(scorecard: pd.DataFrame, *, subtitle: str, note: str, path: Path) -> None:
    ranked = scorecard.loc[scorecard.series_type.eq("model")].sort_values(["margin_mae", "model"])
    ordered = pd.concat([scorecard.loc[scorecard.series_type.eq("consensus")],
                         scorecard.loc[scorecard.series_type.eq("vegas")], ranked], ignore_index=True)
    cells = []
    for rank, row in enumerate(ordered.itertuples(), start=1):
        ats = "—" if row.series_type == "vegas" else f"{row.ats_wins}–{row.ats_losses}–{row.ats_pushes}"
        cells.append([str(rank - 2) if row.series_type == "model" else "—",
                      row.model.removeprefix("margin_"), str(row.games),
                      f"{row.su_wins}/{row.games}", ats, str(row.brier_games),
                      f"{row.brier_score:.3f}", str(row.margin_games), f"{row.margin_mae:.2f}"])
    fig, ax = plt.subplots(figsize=(17.5, 16.6), facecolor="#f7f3eb")
    ax.axis("off")
    ax.set_position([.025, .045, .95, .87])
    table = ax.table(cellText=cells, colLabels=["Rank", "Model", "Games", "SU", "ATS W–L–P",
                                            "Brier n", "Brier", "Margin n", "Margin MAE"],
                     cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1],
                     colWidths=[.045, .34, .065, .095, .12, .075, .08, .08, .10])
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d2d6da")
        cell.set_facecolor("#23354f" if row_index == 0 else "#fbe0eb" if row_index == 1
                           else "#fff2d0" if row_index == 2 else "#fff" if row_index % 2 else "#edf1f4")
        if row_index == 0:
            cell.set_text_props(color="white", weight="bold")
    fig.suptitle(f"TDNet margin roster · {subtitle}", fontsize=22, weight="bold", color="#20324b", y=.98)
    fig.text(.5, .015, note, ha="center", fontsize=10, color="#4b5867")
    fig.savefig(path, dpi=175, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    inventory = history_inventory()
    games, sources = load_game_records()
    models, model_sources = _scorecard_games(games)
    sources.update(model_sources)
    sources["published_cumulative_baseline"] = ROOT / "publication/2026/week_08/post_game/tables/cumulative_margin_wide_performance.csv"
    cohorts = {
        "week08_current_53": games.loc[games.week.eq(8)],
        "cumulative_with_week07_replay_437": games,
        "cumulative_excluding_week07_replay_378": games.loc[games.week.ne(7)],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inventory_path = OUTPUT / "history_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    standalone_path = OUTPUT / "historical_standalone_formats.csv"
    historical_standalones = [
        "publication/2026/week_07/post_game/retrospective_replay/cumulative/margin_wide/cumulative_model_scorecard.png",
        "publication/2026/week_07/post_game/retrospective_replay/cumulative/margin_wide/cumulative_performance.png",
    ]
    committed = set(subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "publication/2026/week_07/post_game/retrospective_replay/cumulative/margin_wide"],
        cwd=ROOT, text=True).splitlines())
    if not set(historical_standalones).issubset(committed):
        raise ValueError("Historical standalone margin figures are not both committed")
    pd.DataFrame([
        {"historical_figure": "week_07/post_game/retrospective_replay/cumulative/margin_wide/cumulative_model_scorecard.png",
         "week08_equivalent": "embedded in figures/cumulative_margin_wide_performance.png",
         "review_action": "regenerated as standalone current and cumulative scorecards"},
        {"historical_figure": "week_07/post_game/retrospective_replay/cumulative/margin_wide/cumulative_performance.png",
         "week08_equivalent": "figures/cumulative_margin_wide_performance.png",
         "review_action": "already generated in Week 8 composite"},
    ]).to_csv(standalone_path, index=False)
    ledgers = []
    captions = {
        "week08_current_53": ("Week 8 · 53 frozen games", "Week 8 margin-wide consensus; 33 archived model margins per game. Buckets are descriptive."),
        "cumulative_with_week07_replay_437": ("2026 through Week 8 · 437 scored games", "Includes 59 Week 7 forecasts replayed after games; Week 7 Vegas quote times are unverified."),
        "cumulative_excluding_week07_replay_378": ("2026 through Week 8 · 378 games without Week 7 replay", "Weeks 0–6 and 8 only. Week 0 used 36 models; the later saved margin rosters used 33."),
    }
    figures = []
    for scope, cohort in cohorts.items():
        ledger = summarize(cohort, scope)
        ledgers.append(ledger)
        title, note = captions[scope]
        path = OUTPUT / f"{scope}.png"
        render(ledger, scope=scope, subtitle=title, note=note, path=path)
        figures.append(path)
    ledger_path = OUTPUT / "diagnostic_buckets.csv"
    pd.concat(ledgers, ignore_index=True).to_csv(ledger_path, index=False, float_format="%.9f")
    scorecard = _scorecards(games, models)
    scorecard_path = OUTPUT / "standalone_model_scorecards.csv"
    scorecard.to_csv(scorecard_path, index=False, float_format="%.9f")
    for scope, _cohort in cohorts.items():
        title, note = captions[scope]
        path = OUTPUT / f"{scope}_model_scorecard.png"
        render_scorecard(scorecard.loc[scorecard.scope.eq(scope)], subtitle=title,
                         note=note, path=path)
        figures.append(path)
    readme = """# Lost margin-roster postgame graphics · Week 8 review

The Git-history audit found 28 distinct committed names under `publication/2026/week_*/post_game/figures/`. The Week 8 margin postgame output has 26. The retired result graphic is Week 3's `exploratory_postgame_diagnostics.png` (first committed in `295896e`). The other absent name, `tdnet_vs_ap_preseason_top25.png`, was a Week 0 AP naming variant and is represented by the current `tdnet_vs_ap_top25.png`. The deleted Week 4 files used figure names that are again generated in Week 8. `history_inventory.csv` lists every historical name and its status. Scientific, pregame, social, and intro artwork are outside this margin-roster **postgame result** audit.

The audit also found two margin-roster Week 7 replay figures committed outside the regular `figures/` directory. Its cumulative performance plot is covered by Week 8's `cumulative_margin_wide_performance.png`, which includes Brier, SU, and ATS trajectories. Its **standalone cumulative model scorecard** survives only as a small panel inside that composite. `historical_standalone_formats.csv` records both. The standalone scorecard format has been freshly rebuilt below for Week 8 and both cumulative cohorts. It ranks the same 33 current margin models by margin MAE and includes the consensus and Vegas reference. The scorecards show SU, ATS, Brier, and margin MAE. Vegas ATS is undefined against its own spread.

These are **fresh calculations**, not copies of the Week 3 image. Each figure rebuilds its three historical panels: published picked-team confidence versus observed winner accuracy; TDNet and Vegas margin MAE by absolute spread; and consensus margin MAE by the standard deviation of that game's individual model margins. `diagnostic_buckets.csv` has every plotted value and game count. The published winner follows the consensus margin sign, while picked-team confidence follows its archived probability; a few near-50% forecasts have a probability just across the opposite side. The first confidence bin therefore includes values below 50% and is labeled ≤60%, followed by 60–75% and 75–100%. Spread bins are ≤3.5, 3.5–7, 7–14, and >14 points; disagreement uses within-cohort thirds of model-margin SD. A missing Vegas line is omitted only from the spread panel. The panels are descriptive post-outcome diagnostics, not a validated betting rule or evidence that any subgroup will repeat.

| Figure | Games | Source scope |
| --- | ---: | --- |
| `week08_current_53.png` | 53 | Week 8 frozen margin-wide predictions and certified final scores. |
| `cumulative_with_week07_replay_437.png` | 437 | Weeks 0–8; includes 59 Week 7 forecasts replayed after those games. |
| `cumulative_excluding_week07_replay_378.png` | 378 | Weeks 0–6 and 8; matches the existing Week 8 cumulative margin-wide consensus count. |

Each scope also has a `*_model_scorecard.png` in the same directory. `standalone_model_scorecards.csv` contains its underlying 35 rows per scope (33 models, consensus, Vegas). Model Brier and margin MAE are calculated from the saved per-game predictions; the no-replay Vegas Brier and SU reconcile to the published Week 8 closing-line baseline. The 437-game Vegas row extends that baseline with the 59 archived Week 7 market entries. Vegas may have fewer valid probability or margin observations than a model's game count; Brier n and margin n in the CSV show those denominators. ATS W–L–P omits games without a market spread or with no model side.

The 437-game cumulative figure uses the saved Week 7 retrospective market spread, whose original quote time was not verified. The Week 0 margin roster had 36 models; Weeks 1–8 used 33. The script verifies unique games, model counts, bucket totals, and that the 378-game cohort's SU and Brier match the published Week 8 cumulative consensus. This directory is for owner review; no graphic is added to the weekly renderer by this script.
"""
    readme_path = OUTPUT / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    outputs = [inventory_path, standalone_path, ledger_path, scorecard_path, readme_path, *figures]
    manifest = {"schema": "tdnet-lost-margin-postgame-diagnostics-v1",
                "history": {"committed_figure_names": len(inventory), "week08_current_names": int(inventory.week08_figure_exists.sum()),
                            "retired_result_name": "exploratory_postgame_diagnostics.png",
                            "historical_commit": "295896e", "extra_historical_standalone_formats": 2},
                "cohorts": {name: len(frame) for name, frame in cohorts.items()},
                "week07_status": "retrospective replay after games; historical market quote times unverified",
                "sources": {str(path.relative_to(ROOT)): sha256(path) for path in sources.values()},
                "files": {path.name: sha256(path) for path in outputs}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"figures": [p.name for p in figures], "cohorts": manifest["cohorts"],
                      "history": manifest["history"]}))


if __name__ == "__main__":
    main()
