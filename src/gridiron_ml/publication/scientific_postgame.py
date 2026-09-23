"""Score frozen scientific pregame predictions without rerunning any model."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd

from .bundles import sha256_file
from .figure_theme import TDNET_COLORS, apply_tdnet_theme


def score_scientific_predictions(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return model-game rows, consensus-game rows, and a 42-model scorecard."""
    required_predictions = {
        "game_id",
        "home_team",
        "away_team",
        "model_name",
        "pred_home_margin",
        "pred_winner",
        "model_against_spread_team",
        "consensus_straight_up_pick",
        "consensus_predicted_winner_margin",
        "consensus_against_spread_team",
        "consensus_home_team_market_spread",
    }
    missing = required_predictions - set(predictions)
    if missing:
        raise ValueError(
            f"Scientific pregame predictions are missing {sorted(missing)}."
        )
    required_results = {"game_id", "home_points", "away_points"}
    missing_results = required_results - set(results)
    if missing_results:
        raise ValueError(f"Postgame results are missing {sorted(missing_results)}.")

    frame = predictions.copy()
    frame["__game_key"] = frame["game_id"].astype(str)
    final = results.copy()
    final["__game_key"] = final["game_id"].astype(str)
    if final["__game_key"].duplicated().any():
        raise ValueError("Postgame results contain duplicate game IDs.")
    final = final[["__game_key", "home_points", "away_points"]]
    frame = frame.merge(final, on="__game_key", how="left", validate="many_to_one")
    if frame[["home_points", "away_points"]].isna().any(axis=1).any():
        missing_ids = sorted(
            frame.loc[
                frame[["home_points", "away_points"]].isna().any(axis=1), "game_id"
            ]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Scientific results are incomplete for games {missing_ids}.")

    frame["actual_home_margin"] = pd.to_numeric(frame["home_points"]) - pd.to_numeric(
        frame["away_points"]
    )
    frame["actual_winner"] = frame["home_team"].where(
        frame["actual_home_margin"].gt(0), frame["away_team"]
    )
    frame["model_winner_correct"] = frame["pred_winner"].eq(frame["actual_winner"])
    frame["model_absolute_margin_error"] = (
        pd.to_numeric(frame["pred_home_margin"], errors="coerce")
        - frame["actual_home_margin"]
    ).abs()

    spread = pd.to_numeric(frame["consensus_home_team_market_spread"], errors="coerce")
    cover_margin = frame["actual_home_margin"] + spread
    actual_cover_team = pd.Series(pd.NA, index=frame.index, dtype="object")
    actual_cover_team.loc[cover_margin.gt(0)] = frame.loc[
        cover_margin.gt(0), "home_team"
    ]
    actual_cover_team.loc[cover_margin.lt(0)] = frame.loc[
        cover_margin.lt(0), "away_team"
    ]
    actual_cover_team.loc[cover_margin.eq(0)] = "Push"
    frame["actual_against_spread_winner"] = actual_cover_team

    def ats_result(pick: object, actual: object, line: object) -> str:
        if pd.isna(line) or pd.isna(pick):
            return "No line"
        if actual == "Push":
            return "Push"
        return "Win" if str(pick) == str(actual) else "Loss"

    frame["model_ats_result"] = [
        ats_result(pick, actual, line)
        for pick, actual, line in zip(
            frame["model_against_spread_team"], actual_cover_team, spread
        )
    ]
    frame["consensus_ats_result"] = [
        ats_result(pick, actual, line)
        for pick, actual, line in zip(
            frame["consensus_against_spread_team"], actual_cover_team, spread
        )
    ]

    if "game_start_time_utc" in frame:
        frame["__kickoff_sort"] = pd.to_datetime(
            frame["game_start_time_utc"], utc=True, errors="coerce"
        )
        frame = frame.sort_values(["__kickoff_sort", "game_id"], kind="stable")
    consensus = frame.drop_duplicates("__game_key").copy()
    consensus["consensus_predicted_home_margin"] = consensus[
        "consensus_predicted_winner_margin"
    ].where(
        consensus["consensus_straight_up_pick"].eq(consensus["home_team"]),
        -pd.to_numeric(consensus["consensus_predicted_winner_margin"], errors="coerce"),
    )
    consensus["consensus_winner_correct"] = consensus["consensus_straight_up_pick"].eq(
        consensus["actual_winner"]
    )
    consensus["consensus_absolute_margin_error"] = (
        pd.to_numeric(consensus["consensus_predicted_home_margin"], errors="coerce")
        - consensus["actual_home_margin"]
    ).abs()
    consensus_columns = [
        "game_id",
        "away_team",
        "home_team",
        "away_points",
        "home_points",
        "actual_winner",
        "consensus_straight_up_pick",
        "consensus_winner_correct",
        "consensus_predicted_home_margin",
        "actual_home_margin",
        "consensus_absolute_margin_error",
        "consensus_against_spread_team",
        "consensus_home_team_market_spread",
        "actual_against_spread_winner",
        "consensus_ats_result",
    ]
    consensus = consensus[consensus_columns].reset_index(drop=True)

    scorecard = (
        frame.groupby("model_name", as_index=False)
        .agg(
            games=("game_id", "nunique"),
            straight_up_accuracy=("model_winner_correct", "mean"),
            margin_mae=("model_absolute_margin_error", "mean"),
            ats_wins=("model_ats_result", lambda values: int((values == "Win").sum())),
            ats_losses=(
                "model_ats_result",
                lambda values: int((values == "Loss").sum()),
            ),
            ats_pushes=(
                "model_ats_result",
                lambda values: int((values == "Push").sum()),
            ),
        )
        .sort_values(
            ["margin_mae", "straight_up_accuracy", "model_name"],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    scorecard.insert(0, "scorecard_rank", range(1, len(scorecard) + 1))
    internal_columns = [
        column for column in ("__game_key", "__kickoff_sort") if column in frame
    ]
    return frame.drop(columns=internal_columns), consensus, scorecard


def _plot_consensus_results(
    consensus: pd.DataFrame,
    path: Path,
    *,
    season: int,
    week: int,
    model_label: str = "Scientific consensus",
) -> Path:
    """Render the scientific consensus in the wide Sunday-scorecard style."""
    apply_tdnet_theme()
    frame = consensus.copy().reset_index(drop=True)

    su_wins = int(frame["consensus_winner_correct"].fillna(False).astype(bool).sum())
    su_losses = int(len(frame) - su_wins)
    su_accuracy = su_wins / len(frame) if len(frame) else float("nan")
    ats = frame["consensus_ats_result"].fillna("").astype(str).str.casefold()
    ats_wins = int(ats.eq("win").sum())
    ats_losses = int(ats.eq("loss").sum())
    ats_pushes = int(ats.eq("push").sum())
    ats_decisions = ats_wins + ats_losses
    ats_accuracy = ats_wins / ats_decisions if ats_decisions else float("nan")
    margin_mae = pd.to_numeric(
        frame["consensus_absolute_margin_error"], errors="coerce"
    ).mean()

    def projected(row: pd.Series) -> str:
        margin = float(row["consensus_predicted_home_margin"])
        winner = str(row["home_team"] if margin >= 0 else row["away_team"])
        return f"{winner} by {abs(margin):.1f}"

    def closing_line(row: pd.Series) -> str:
        spread = pd.to_numeric(
            pd.Series([row["consensus_home_team_market_spread"]]), errors="coerce"
        ).iloc[0]
        if pd.isna(spread):
            return "—"
        if float(spread) == 0:
            return "Pick'em"
        return f"{row['home_team']} {float(spread):+g}"

    table = pd.DataFrame(
        {
            "Matchup": frame["away_team"].astype(str) + " at " + frame["home_team"].astype(str),
            "Scientific projected": frame.apply(projected, axis=1),
            "Final": frame["away_team"].astype(str)
            + " "
            + frame["away_points"].astype(int).astype(str)
            + " – "
            + frame["home_points"].astype(int).astype(str)
            + " "
            + frame["home_team"].astype(str),
            "SU": frame["consensus_winner_correct"].map({True: "W", False: "L"}),
            "Closing line": frame.apply(closing_line, axis=1),
            "Scientific ATS side": frame["consensus_against_spread_team"].fillna("—"),
            "ATS": ats.map({"win": "W", "loss": "L", "push": "P"}).fillna("—"),
        }
    )
    fig, axis = plt.subplots(figsize=(16, max(4.0, len(table) * 0.36 + 2.25)))
    fig.patch.set_facecolor("#F7F4ED")
    axis.axis("off")
    plotted = axis.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.045, 1.0, 0.86],
        colWidths=[0.22, 0.22, 0.22, 0.055, 0.12, 0.12, 0.055],
    )
    plotted.auto_set_font_size(False)
    plotted.set_fontsize(8.2)
    plotted.scale(1, 1.28)
    su_col = table.columns.get_loc("SU")
    ats_col = table.columns.get_loc("ATS")
    for (row, column), cell in plotted.get_celld().items():
        cell.set_edgecolor("#D4D7DB")
        if row == 0:
            cell.set_facecolor(TDNET_COLORS["midnight_gridiron"])
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#EEF2F5")
            if column in (su_col, ats_col):
                value = table.iloc[row - 1, column]
                cell.set_facecolor(
                    {"W": "#DCEFE1", "L": "#F6DDDA", "P": "#FFF0C7", "—": "#E8E8E8"}[value]
                )
                cell.set_text_props(
                    weight="bold",
                    ha="center",
                    color="#183321" if value == "W" else "#702820",
                )
    ats_summary = f"{ats_accuracy:.1%}, pushes excluded" if ats_decisions else "no decisions"
    axis.set_title(
        f"{season} Week {week}: TDNet Sunday Scorecard — {model_label}\n"
        f"Week: SU {su_wins}–{su_losses} ({su_accuracy:.1%})"
        f"     •     ATS {ats_wins}–{ats_losses}–{ats_pushes} ({ats_summary})"
        f"     •     Margin MAE {margin_mae:.1f} pts",
        fontsize=17,
        weight="bold",
        pad=22,
        color=TDNET_COLORS["midnight_gridiron"],
    )
    fig.text(
        0.5,
        0.015,
        "Projected margins and picks come from the frozen pregame consensus. ATS uses the captured closing home-team spread; pushes and missing lines are excluded from ATS accuracy.",
        ha="center",
        fontsize=8.5,
        color="#555B63",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_scientific_postgame_evaluation(
    *,
    pregame_package: str | Path,
    results_path: str | Path,
    output_root: str | Path,
    season: int,
    week: int,
) -> dict[str, Path]:
    """Write a postgame evaluation whose only predictions are frozen pregame rows."""
    pregame = Path(pregame_package)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output.iterdir() if path.name != "README.md"]
    if existing:
        raise FileExistsError(
            "Scientific postgame artifacts are immutable once generated; refusing to overwrite: "
            + ", ".join(sorted(path.name for path in existing))
        )
    prediction_path = pregame / "scientific_all_game_predictions.csv"
    results_source = Path(results_path)
    predictions = pd.read_csv(prediction_path)
    results = (
        pd.read_parquet(results_source)
        if results_source.suffix == ".parquet"
        else pd.read_csv(results_source)
    )
    model_games, consensus, scorecard = score_scientific_predictions(
        predictions, results
    )
    expected_models = int(predictions["model_name"].nunique())
    if len(scorecard) != expected_models:
        raise RuntimeError(
            f"Scientific postgame scorecard has {len(scorecard)} models; expected {expected_models}."
        )
    paths = {
        "model_game_results_csv": output / "scientific_model_game_results.csv",
        "consensus_results_csv": output / "scientific_consensus_game_results.csv",
        "model_scorecard_csv": output / "scientific_model_scorecard.csv",
        "consensus_results_png": output / "scientific_consensus_game_results.png",
    }
    model_games.to_csv(paths["model_game_results_csv"], index=False)
    consensus.to_csv(paths["consensus_results_csv"], index=False)
    scorecard.to_csv(paths["model_scorecard_csv"], index=False)
    _plot_consensus_results(
        consensus,
        paths["consensus_results_png"],
        season=season,
        week=week,
        model_label=(
            "Full F0–F8 scientific consensus"
            if output.name == "full_f0_f8"
            else "F0–F6 scientific consensus"
        ),
    )
    generated = datetime.now(UTC)
    manifest = {
        "schema": "tdnet-scientific-postgame-evaluation-v1",
        "season": int(season),
        "week": int(week),
        "generated_at_eastern": generated.astimezone(
            ZoneInfo("America/New_York")
        ).isoformat(),
        "predictions_recalculated": False,
        "frozen_pregame_predictions": str(prediction_path),
        "frozen_pregame_predictions_sha256": sha256_file(prediction_path),
        "results_source": str(results_source),
        "results_sha256": sha256_file(results_source),
        "scientific_model_count": len(scorecard),
        "game_count": int(consensus["game_id"].nunique()),
        "files": {
            name: {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    manifest_path = output / "scientific_postgame_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# TDNet scientific postgame evaluation\n\n"
        f"This directory scores the frozen pregame {expected_models}-model scientific predictions. "
        "No model was rerun and no pregame artifact was overwritten.\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths
