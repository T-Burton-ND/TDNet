from pathlib import Path

import pandas as pd

from gridiron_ml.publication.scientific_performance import (
    VEGAS_LABEL,
    build_scientific_rolling_performance,
    cumulative_scientific_scorecard,
    write_scientific_cumulative_artifacts,
)


def _write_week(
    root: Path,
    week: int,
    game_id: int,
    actual_margin: float,
    scientific_subdir: str | None = None,
    include_f7: bool = False,
) -> None:
    output = root / f"week_{week:02d}" / "post_game" / "scientific"
    if scientific_subdir:
        output = output / scientific_subdir
    output.mkdir(parents=True)
    home_won = actual_margin > 0
    rows = []
    models = [
        ("scientific_F0_M1", 4.0, 0.7),
        ("scientific_F1_M1", -2.0, 0.4),
    ]
    if include_f7:
        models.append(("scientific_F7_M1", 3.0, 0.65))
    for model, predicted_margin, probability in models:
        rows.append(
            {
                "game_id": game_id,
                "home_team": "Home",
                "away_team": "Away",
                "model_name": model,
                "model_family": "linear",
                "fingerprint": model.split("_")[1],
                "pred_home_win_probability": probability,
                "model_absolute_margin_error": abs(predicted_margin - actual_margin),
                "model_winner_correct": (predicted_margin > 0) == home_won,
                "model_ats_result": "Win" if model.endswith("F0_M1") else "Loss",
                "actual_home_margin": actual_margin,
                "actual_winner": "Home" if home_won else "Away",
                "consensus_straight_up_pick": "Home",
                "consensus_predicted_winner_margin": 1.0,
                "consensus_predicted_home_win_probability": 0.55,
                "consensus_ats_result": "Win",
                "consensus_home_team_market_spread": -3.0,
            }
        )
    pd.DataFrame(rows).to_csv(output / "scientific_model_game_results.csv", index=False)


def test_scientific_cumulative_scorecard_includes_vegas_and_accumulates(tmp_path: Path):
    root = tmp_path / "publication" / "2026"
    _write_week(root, 0, 10, 7.0)
    _write_week(root, 1, 11, -1.0)

    performance = build_scientific_rolling_performance(
        publication_root=root,
        season=2026,
        completed_week=1,
        lines_path=tmp_path / "missing-lines.parquet",
    )
    scorecard = cumulative_scientific_scorecard(performance, completed_week=1)

    assert set(scorecard["series_type"]) == {"model", "consensus", "vegas"}
    vegas = scorecard.set_index("model_name").loc[VEGAS_LABEL]
    assert vegas["games"] == 2
    assert vegas["margin_mae"] == 4.0
    assert vegas["su_accuracy"] == 0.5
    assert vegas["ats_accuracy"] == 0.5
    assert pd.isna(vegas["ats_wins"])
    model = scorecard.set_index("model_name").loc["scientific_F0_M1"]
    assert model["games"] == 2
    assert model["su_accuracy"] == 0.5


def test_scientific_cumulative_can_track_full_consensus_in_sibling_subdirectory(
    tmp_path: Path,
):
    root = tmp_path / "publication" / "2026"
    _write_week(root, 4, 20, 7.0, scientific_subdir="full_f0_f8")
    performance = build_scientific_rolling_performance(
        publication_root=root,
        season=2026,
        completed_week=4,
        lines_path=tmp_path / "missing-lines.parquet",
        scientific_subdir="full_f0_f8",
        consensus_label="Full F0–F8 scientific consensus",
        consensus_fingerprint="F0–F8",
    )
    consensus = performance.loc[performance["series_type"].eq("consensus")]
    assert set(consensus["model_name"]) == {"Full F0–F8 scientific consensus"}
    assert set(consensus["fingerprint"]) == {"F0–F8"}


def test_full_cohort_keeps_f7_audit_rows_but_omits_them_from_scorecard(tmp_path: Path):
    root = tmp_path / "publication" / "2026"
    _write_week(
        root,
        0,
        20,
        7.0,
        scientific_subdir="full_f0_f8",
        include_f7=True,
    )
    output = root / "week_00" / "post_game" / "scientific" / "full_f0_f8"
    write_scientific_cumulative_artifacts(
        publication_root=root,
        season=2026,
        completed_week=0,
        output_root=output,
        scientific_subdir="full_f0_f8",
        consensus_label="Full F0–F8 scientific consensus",
        consensus_fingerprint="F0–F8",
    )

    rolling = pd.read_csv(output / "scientific_rolling_cumulative_performance.csv")
    scorecard = pd.read_csv(output / "scientific_cumulative_model_scorecard.csv")
    assert "F7" in set(rolling["fingerprint"])
    assert "F7" not in set(scorecard["fingerprint"])
