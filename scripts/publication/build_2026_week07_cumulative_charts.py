"""Publish cumulative 2026 charts with retrospective replay boundaries.

The original rosters have frozen Weeks 0–6 and a Week 7 replay. The expanded
F0–F19 roster has separate retrospective Week 6–7 diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "publication/2026"
REPLAY = PUB / "week_07/post_game/retrospective_replay"
OUTPUT = REPLAY / "cumulative"
ROSTERS = {
    "margin_wide": ("Margin roster", 33),
    "scientific_f0_f6": ("Original scientific F0–F6", 42),
    "scientific_f0_f8": ("Full scientific F0–F8", 49),
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(
    frame: pd.DataFrame, *, week: int, name: str, is_consensus: bool
) -> pd.DataFrame:
    actual_margin = pd.to_numeric(frame["actual_home_margin"], errors="raise")
    probability = pd.to_numeric(frame["pred_home_win_probability"], errors="coerce")
    predicted_margin = pd.to_numeric(frame["pred_home_margin"], errors="coerce")
    if "winner_correct" in frame:
        winner_correct = frame["winner_correct"].astype("boolean")
    elif "su_correct" in frame:
        winner_correct = frame["su_correct"].astype("boolean")
    else:
        raise ValueError(f"Missing graded winner outcome in {name}, Week {week}")
    result = pd.DataFrame(
        {
            "week": week,
            "game_id": frame["game_id"].astype(str),
            "model_name": name if is_consensus else frame["model_name"].astype(str),
            "series_type": "consensus" if is_consensus else "model",
            "margin_error": (predicted_margin - actual_margin).abs(),
            "winner_correct": winner_correct,
            "brier_error": (probability - actual_margin.gt(0).astype(float)).pow(2),
            "phase": "retrospective_replay" if week == 7 else "frozen_pre_game",
        }
    )
    if result[["margin_error", "brier_error"]].isna().any().any():
        raise ValueError(f"Missing prediction in {name}, Week {week}")
    return result


def _load_margin() -> tuple[pd.DataFrame, list[Path]]:
    sources: list[Path] = []
    replay_models_path = REPLAY / "week07_replay_model_games.parquet"
    replay_consensus_path = REPLAY / "week07_replay_consensus_games.csv"
    replay_models = pd.read_parquet(replay_models_path)
    current_names = set(replay_models["model_name"].astype(str))
    if len(current_names) != 33:
        raise ValueError("Week 7 margin roster is not 33 models")
    pieces: list[pd.DataFrame] = []
    for week in range(8):
        if week == 7:
            model_path, consensus_path = replay_models_path, replay_consensus_path
            models = replay_models
            consensus = pd.read_csv(consensus_path)
        else:
            base = PUB / f"week_{week:02d}/post_game"
            model_path = base / "scoring/scored_predictions.parquet"
            consensus_path = base / "tables/margin_wide_prediction_vs_actual.csv"
            models = pd.read_parquet(model_path)
            consensus = pd.read_csv(consensus_path)
        sources.extend((model_path, consensus_path))
        models = models.loc[models["model_name"].astype(str).isin(current_names)]
        pieces.append(_normalize(models, week=week, name="", is_consensus=False))
        pieces.append(
            _normalize(
                consensus,
                week=week,
                name="TDNet margin-wide consensus",
                is_consensus=True,
            )
        )
    return pd.concat(pieces, ignore_index=True), sources


def _load_scientific(subdir: str | None) -> tuple[pd.DataFrame, list[Path]]:
    sources: list[Path] = []
    pieces: list[pd.DataFrame] = []
    for week in reversed(range(8)):
        if week == 7:
            base = REPLAY / "scientific"
        else:
            base = PUB / f"week_{week:02d}/post_game/scientific"
        if subdir:
            base /= subdir
        path = base / "scientific_model_game_results.csv"
        sources.append(path)
        data = pd.read_csv(path)
        models = data.copy()
        models["winner_correct"] = models["model_winner_correct"]
        pieces.append(_normalize(models, week=week, name="", is_consensus=False))
        consensus = data.drop_duplicates("game_id").copy()
        magnitude = pd.to_numeric(
            consensus["consensus_predicted_winner_margin"], errors="raise"
        )
        consensus["pred_home_margin"] = np.where(
            consensus["consensus_straight_up_pick"].eq(consensus["home_team"]),
            magnitude,
            -magnitude,
        )
        consensus["pred_home_win_probability"] = consensus[
            "consensus_predicted_home_win_probability"
        ]
        consensus["winner_correct"] = consensus["consensus_straight_up_pick"].eq(
            consensus["actual_winner"]
        )
        pieces.append(
            _normalize(
                consensus,
                week=week,
                name="Full F0–F8 scientific consensus" if subdir else "Scientific consensus",
                is_consensus=True,
            )
        )
    return pd.concat(pieces, ignore_index=True), sources


def _performance(games: pd.DataFrame, expected_models: int) -> pd.DataFrame:
    if games.loc[games.week.eq(7) & games.series_type.eq("model"), "model_name"].nunique() != expected_models:
        raise ValueError("Current roster size does not match expected model count")
    if games.duplicated(["week", "game_id", "model_name"]).any():
        raise ValueError("Duplicate game/model result")
    if games.loc[games.week.eq(7) & games.series_type.eq("consensus"), "game_id"].nunique() != 59:
        raise ValueError("Week 7 replay must contain 59 supported games")
    if games.week.nunique() != 8:
        raise ValueError("Expected results for Weeks 0–7")
    rows: list[dict[str, object]] = []
    for week in range(8):
        through = games.loc[games.week.le(week)]
        for (series_type, model_name), group in through.groupby(
            ["series_type", "model_name"], sort=True
        ):
            if week == 7 and not group.week.eq(7).any():
                continue
            rows.append(
                {
                    "through_week": week,
                    "series_type": series_type,
                    "model_name": model_name,
                    "games": len(group),
                    "brier_games": int(group.brier_error.notna().sum()),
                    "margin_mae": group.margin_error.mean(),
                    "su_wins": int(group.winner_correct.sum()),
                    "su_accuracy": group.winner_correct.mean(),
                    "brier_score": group.brier_error.mean(),
                    "contains_retrospective_replay": week == 7,
                    "week_07_provenance": "retrospective_replay_after_games" if week == 7 else "",
                }
            )
    return pd.DataFrame(rows)


def _vegas_history(roster: str) -> pd.DataFrame:
    """Load the published comparator through Week 6, never extend into replay."""
    base = PUB / "week_06/post_game/scientific"
    if roster == "scientific_f0_f8":
        base /= "full_f0_f8"
    source = base / "scientific_rolling_cumulative_performance.csv"
    data = pd.read_csv(source)
    vegas = data.loc[data.scope.eq("cumulative") & data.series_type.eq("vegas")].copy()
    if set(vegas.through_week) != set(range(7)):
        raise ValueError("Expected published Vegas comparator through Week 6")
    vegas["model_name"] = "Vegas baseline · through Week 6"
    vegas["contains_retrospective_replay"] = False
    vegas["week_07_provenance"] = "No historical market snapshot for Week 7 replay"
    return vegas[
        ["through_week", "series_type", "model_name", "games", "brier_games",
         "margin_mae", "su_wins", "su_accuracy", "brier_score",
         "contains_retrospective_replay", "week_07_provenance"]
    ]


def _plot_trajectory(data: pd.DataFrame, title: str, path: Path) -> None:
    metrics = (
        ("margin_mae", "Margin MAE", False),
        ("su_accuracy", "Straight-up accuracy", True),
        ("brier_score", "Brier score", False),
    )
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), facecolor="#F7F4ED")
    colors = LinearSegmentedColormap.from_list("fingerprint", ["#EA5A9B", "#8753AE", "#2272A7"])
    scientific = "scientific" in title.lower()
    max_stage = 8 if "F0–F8" in title else 6
    for axis, (metric, label, percent) in zip(axes, metrics):
        for name, group in data.loc[data.series_type.eq("model")].groupby("model_name"):
            stage = int(name.split("_F", 1)[1].split("_", 1)[0]) if scientific else 0
            color = colors(stage / max_stage) if scientific else "#647D98"
            frozen = group.loc[group.through_week.le(6)].sort_values("through_week")
            axis.plot(frozen.through_week, frozen[metric], color=color, alpha=0.35, lw=1)
            endpoint = group.loc[group.through_week.ge(6)].sort_values("through_week")
            axis.plot(endpoint.through_week, endpoint[metric], color=color, alpha=0.45, lw=1, ls=":")
        consensus = data.loc[data.series_type.eq("consensus")].sort_values("through_week")
        frozen = consensus.loc[consensus.through_week.le(6)]
        endpoint = consensus.loc[consensus.through_week.ge(6)]
        axis.plot(frozen.through_week, frozen[metric], color="#D62F76", lw=3, marker="o", label="Consensus · frozen")
        axis.plot(endpoint.through_week, endpoint[metric], color="#D62F76", lw=3, ls=":", marker="o", label="Week 7 replay endpoint")
        vegas = data.loc[data.series_type.eq("vegas")].sort_values("through_week")
        axis.plot(vegas.through_week, vegas[metric], color="#BD8B17", lw=2.5,
                  ls="--", marker="s", label="Vegas · available Weeks 0–6")
        axis.set_title(label + (" · higher is better" if percent else " · lower is better"),
                       loc="left", weight="bold", fontsize=14, color="#15294C")
        axis.set_xticks(range(8))
        axis.set_xlim(-0.2, 7.35)
        axis.set_xlabel("Completed week")
        if percent:
            axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.subplots_adjust(top=.84, bottom=.2 if scientific else .13, wspace=.22)
    fig.suptitle(f"TDNet 2026 {title} · cumulative performance", fontsize=21,
                 weight="bold", color="#15294C", y=.96)
    if scientific:
        bar_axis = fig.add_axes([.26, .115, .48, .022])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, max_stage), cmap=colors),
                           cax=bar_axis, orientation="horizontal")
        bar.set_ticks(range(max_stage + 1))
        bar.set_ticklabels([f"F{i}" for i in range(max_stage + 1)])
        bar.set_label("Fingerprint generation (pink → blue)")
    fig.text(0.5, 0.02,
             "Weeks 0–6: frozen forecasts and archived Vegas. Dotted Week 7 endpoint: retrospective replay after games. "
             "Vegas stops at Week 6; its Brier series has fewer eligible games. ATS is omitted.",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_scorecard(scorecard: pd.DataFrame, title: str, path: Path) -> None:
    shown = scorecard.reset_index(drop=True).copy()
    table = pd.DataFrame(
        {
            "Rank": ["—" if x != "model" else str(i - 1) for i, x in enumerate(shown.series_type, start=0)],
            "Model": shown.model_name.str.replace("scientific_", "", regex=False),
            "Games": shown.games.astype(str),
            "W7 MAE": shown.week07_margin_mae.map(lambda x: "—" if pd.isna(x) else f"{x:.2f}"),
            "W7 SU": shown.apply(lambda row: "—" if pd.isna(row.week07_su_wins)
                                 else f"{int(row.week07_su_wins)}/{int(row.week07_games)}", axis=1),
            "Brier n": shown.brier_games.astype(str),
            "Cumulative MAE": shown.margin_mae.map(lambda x: f"{x:.2f}"),
            "Cumulative SU": shown.su_wins.astype(str) + "/" + shown.games.astype(str),
            "SU %": shown.su_accuracy.map(lambda x: f"{x:.1%}"),
            "Brier": shown.brier_score.map(lambda x: f"{x:.3f}"),
        }
    )
    fig, ax = plt.subplots(figsize=(18, max(10, len(table) * 0.38 + 2.5)), facecolor="#F7F4ED")
    ax.axis("off")
    ax.set_position([.025, .028, .95, .91])
    artist = ax.table(cellText=table.values, colLabels=table.columns, cellLoc="left", colLoc="left",
                      bbox=[0, .01, 1, .98],
                      colWidths=[0.045, 0.30, 0.06, 0.07, 0.07, 0.075, 0.10, 0.10, 0.09, 0.09])
    artist.auto_set_font_size(False)
    artist.set_fontsize(9)
    for (row, _), cell in artist.get_celld().items():
        cell.set_edgecolor("#D4D7DB")
        cell.set_facecolor("#22324A" if row == 0 else "#FDE2F0" if row == 1 else "#FFFFFF" if row % 2 else "#EEF2F5")
        if row == 0:
            cell.set_text_props(color="white", weight="bold")
    ax.set_title(f"2026 {title} · cumulative through Week 7", fontsize=19, weight="bold", pad=16)
    fig.text(0.5, 0.008,
             "W7 columns cover 59 retrospective games; cumulative model rows include Week 7. "
             "Vegas stops at Week 6 (325 games). Brier n shows valid probabilities; ATS is omitted.",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_sources: dict[str, str] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for roster, (title, expected) in ROSTERS.items():
        games, sources = _load_margin() if roster == "margin_wide" else _load_scientific(None if roster == "scientific_f0_f6" else "full_f0_f8")
        performance = _performance(games, expected)
        vegas = _vegas_history(roster)
        performance = pd.concat([performance, vegas], ignore_index=True)
        scorecard = performance.loc[performance.through_week.eq(7)].copy()
        scorecard = pd.concat([scorecard, vegas.loc[vegas.through_week.eq(6)]], ignore_index=True)
        scorecard["sort_key"] = scorecard.series_type.map({"consensus": 0, "model": 1})
        scorecard.loc[scorecard.series_type.eq("vegas"), "sort_key"] = 0.5
        scorecard = scorecard.sort_values(["sort_key", "margin_mae", "model_name"]).drop(columns="sort_key")
        week7 = games.loc[games.week.eq(7)].groupby(["series_type", "model_name"]).agg(
            week07_games=("game_id", "nunique"),
            week07_margin_mae=("margin_error", "mean"),
            week07_su_wins=("winner_correct", "sum"),
        ).reset_index()
        scorecard = scorecard.merge(week7, on=["series_type", "model_name"],
                                    how="left", validate="one_to_one")
        if scorecard.loc[scorecard.series_type.ne("vegas"), "week07_games"].ne(59).any():
            raise ValueError("A roster scorecard row is missing Week 7 results")
        directory = OUTPUT / roster
        directory.mkdir(exist_ok=True)
        paths = {
            "cumulative_csv": directory / "cumulative_performance.csv",
            "scorecard_csv": directory / "cumulative_model_scorecard.csv",
            "performance_png": directory / "cumulative_performance.png",
            "scorecard_png": directory / "cumulative_model_scorecard.png",
        }
        performance.to_csv(paths["cumulative_csv"], index=False)
        scorecard.to_csv(paths["scorecard_csv"], index=False)
        _plot_trajectory(performance, title, paths["performance_png"])
        _plot_scorecard(scorecard, title, paths["scorecard_png"])
        artifacts[roster] = {key: str(path.relative_to(ROOT)) for key, path in paths.items()}
        all_sources.update({str(path.relative_to(ROOT)): _hash(path) for path in sources})
        comparison_source = PUB / "week_06/post_game/scientific"
        if roster == "scientific_f0_f8":
            comparison_source /= "full_f0_f8"
        comparison_source /= "scientific_rolling_cumulative_performance.csv"
        all_sources[str(comparison_source.relative_to(ROOT))] = _hash(comparison_source)
    expanded_output = OUTPUT / "scientific_f0_f19"
    expanded_manifest = REPLAY / "scientific/full_f0_f19/backfill_manifest.json"
    if not expanded_manifest.exists():
        raise FileNotFoundError("Build the Week 6–7 F0–F19 retrospective backfill first")
    all_sources[str(expanded_manifest.relative_to(ROOT))] = _hash(expanded_manifest)
    artifacts["scientific_f0_f19"] = {
        "coverage_figure_png": str((expanded_output / "cumulative_available_performance.png").relative_to(ROOT)),
        "performance_png": str((expanded_output / "cumulative_performance.png").relative_to(ROOT)),
        "scorecard_png": str((expanded_output / "cumulative_model_scorecard.png").relative_to(ROOT)),
        "scorecard_csv": str((expanded_output / "cumulative_model_scorecard.csv").relative_to(ROOT)),
        "latest_scored_week": "7",
        "status": "Weeks 6–7 are retrospective, not issued pregame F0–F19 forecasts; Week 7 F7/F17/F19 market inputs were retrieved after games.",
    }
    manifest = {
        "season": 2026,
        "latest_completed_week": 7,
        "frozen_forecast_weeks": list(range(7)),
        "retrospective_replay_week": 7,
        "expanded_retrospective_backfill_weeks": [6, 7],
        "replay_games": 59,
        "ats_through_week": 6,
        "artifacts": artifacts,
        "input_sha256": all_sources,
        "output_sha256": {
            path: _hash(ROOT / path)
            for item in artifacts.values()
            for path in item.values()
            if isinstance(path, str) and path.startswith("publication/")
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    readme = [
        "# 2026 cumulative performance through Week 7",
        "",
        "Margin (33), original scientific F0–F6 (42), and full scientific F0–F8 have current model and consensus CSVs and figures here. The full roster had 54 models through Week 6 and 49 replayable cells in Week 7; five F7 market-dependent models stop at Week 6. Weeks 0–6 are frozen pregame forecasts. Week 7 is a 59-game retrospective replay made after the games; the dotted endpoint is not a prospective result. Published Vegas MAE and Brier baselines appear through Week 6 only; their Week 7 point is missing because no historical Week 7 market snapshot was frozen. The Week 7 scorecards retain a separate 325-game Vegas row clearly labeled as stopping at Week 6. Vegas Brier uses fewer eligible games than the model rows. ATS is omitted from these new figures.",
        "",
        "The expanded F0–F19 research cohort (120 model cells) now has a common-cohort scorecard through Week 7 in `scientific_f0_f19/`. Weeks 6–7 were reconstructed after the games from score-free states and are marked as retrospective. Week 7 F7, F17-market, and F19 used market lines retrieved after the games; Week 7 F8 had missing market fields imputed. One Week 6 F17/F19 game needed an ex-post market fallback. The archived Vegas comparator stops at Week 6. This backfill is exploratory and cannot be counted as prospective forecast performance. A separate complete 53-game Week 8 pregame research forecast is at `publication/2026/week_08/pre_game/scientific/full_f0_f19/`; it has no outcome scores yet.",
        "",
        "The margin, F0–F6, and F0–F8 cumulative cards now include Week 7 MAE and straight-up results next to the full-season columns. The scientific trajectory colors identify fingerprint stages. The expanded F0–F19 card adds descriptive Week 7 ATS and upset-call precision from after-game lines, with that provenance printed on the figure.",
        "",
        "`manifest.json` lists every figure, table, and input hash.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(readme) + "\n")
    print(json.dumps({"output": str(OUTPUT), "rosters": list(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
