from pathlib import Path

import pandas as pd
import pytest

from gridiron_ml.publication.ats_bankroll import (
    american_odds_win_profit,
    build_confidence_scaled_ats_ledger,
    build_confidence_scaled_moneyline_ledger,
    build_confidence_threshold_sweep,
    build_consensus_ats_bet_ledger,
    build_consensus_moneyline_bet_ledger,
    confidence_scaled_stake,
    summarize_consensus_ats_bankroll,
    summarize_consensus_bankroll,
    summarize_profitable_confidence_thresholds,
)


def test_confidence_stake_curve_has_requested_anchors():
    assert confidence_scaled_stake(0.499) == 0.0
    assert confidence_scaled_stake(0.6994) == pytest.approx(10.0)
    assert confidence_scaled_stake(1.0) == 25.0
    assert confidence_scaled_stake(0.2) == 0.0
    assert confidence_scaled_stake(1.2) == 25.0


def _write_sources(root: Path) -> None:
    postgame = root / "week_00" / "post_game"
    (postgame / "tables").mkdir(parents=True)
    (postgame / "scientific" / "full_f0_f8").mkdir(parents=True)
    base = pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "away_team": ["A1", "A2", "A3", "A4"],
            "home_team": ["H1", "H2", "H3", "H4"],
            "actual_winner": ["H1", "H2", "H3", "H4"],
            "ats_pick": ["H1", "A2", "H3", "H4"],
            "market_spread_close": [-3.0, 2.5, -7.0, -1.0],
            "ats_result": ["win", "loss", "push", "win"],
            "pred_winner": ["H1", "A2", "H3", "H4"],
        }
    )
    base.to_csv(postgame / "tables" / "margin_wide_prediction_vs_actual.csv", index=False)
    scientific = base.rename(
        columns={
            "ats_pick": "consensus_against_spread_team",
            "market_spread_close": "consensus_home_team_market_spread",
            "ats_result": "consensus_ats_result",
            "pred_winner": "consensus_straight_up_pick",
        }
    )
    scientific.to_csv(
        postgame / "scientific" / "scientific_consensus_game_results.csv", index=False
    )
    scientific.to_csv(
        postgame
        / "scientific"
        / "full_f0_f8"
        / "scientific_consensus_game_results.csv",
        index=False,
    )
    margin_models = []
    for model_name, first_pick in (("m1", "H1"), ("m2", "A1")):
        model = base.copy()
        model["model_name"] = model_name
        model.loc[model["game_id"].eq(1), "ats_pick"] = first_pick
        model["pred_home_win_probability"] = [0.7, 0.3, 0.8, 0.6]
        margin_models.append(model)
    pd.concat(margin_models, ignore_index=True).to_csv(
        postgame / "tables" / "margin_wide_model_game_results.csv", index=False
    )
    scientific_models = pd.concat(margin_models, ignore_index=True).rename(
        columns={"ats_pick": "model_against_spread_team"}
    )
    scientific_models["consensus_predicted_home_win_probability"] = scientific_models[
        "pred_home_win_probability"
    ]
    scientific_models.to_csv(
        postgame / "scientific" / "scientific_model_game_results.csv", index=False
    )
    scientific_models.to_csv(
        postgame
        / "scientific"
        / "full_f0_f8"
        / "scientific_model_game_results.csv",
        index=False,
    )


def _write_lines(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "lines": [
                [
                    {"provider": "A", "homeMoneyline": -150, "awayMoneyline": 130},
                    {"provider": "B", "homeMoneyline": -140, "awayMoneyline": 120},
                ],
                [{"provider": "A", "homeMoneyline": -130, "awayMoneyline": 120}],
                [{"provider": "A", "homeMoneyline": 110, "awayMoneyline": -120}],
                [{"provider": "A", "homeMoneyline": None, "awayMoneyline": None}],
            ],
        }
    ).to_parquet(path, index=False)


def test_flat_stake_ats_and_cfbd_moneyline_ledgers(tmp_path: Path):
    root = tmp_path / "publication" / "2026"
    _write_sources(root)
    lines = tmp_path / "lines.parquet"
    _write_lines(lines)

    ats = build_consensus_ats_bet_ledger(
        publication_root=root, season=2026, completed_week=0
    )
    ats_weekly = summarize_consensus_ats_bankroll(ats)
    assert len(ats) == 12
    assert set(ats["odds_source"]) == {
        "standard_-110_assumption_cfbd_has_no_spread_side_price"
    }
    margin_ats = ats_weekly.loc[
        ats_weekly["strategy"].eq("Margin-wide consensus")
    ].iloc[0]
    assert margin_ats["weekly_staked"] == 40.0
    assert margin_ats["weekly_net_profit"] == pytest.approx(
        2 * american_odds_win_profit(10, -110) - 10
    )

    moneyline = build_consensus_moneyline_bet_ledger(
        publication_root=root,
        season=2026,
        completed_week=0,
        lines_path=lines,
    )
    moneyline_weekly = summarize_consensus_bankroll(
        moneyline, result_column="moneyline_result"
    )
    assert len(moneyline) == 12
    assert moneyline["bet_placed"].sum() == 9
    best_home = moneyline.loc[
        moneyline["game_id"].eq("1")
        & moneyline["strategy"].eq("Margin-wide consensus")
    ].iloc[0]
    assert best_home["american_odds"] == -140
    assert best_home["odds_provider"] == "B"
    margin_ml = moneyline_weekly.loc[
        moneyline_weekly["strategy"].eq("Margin-wide consensus")
    ].iloc[0]
    assert margin_ml["weekly_staked"] == 30.0
    assert margin_ml["weekly_net_profit"] == pytest.approx(10 / 1.4 - 10 + 11)

    confidence_ats = build_confidence_scaled_ats_ledger(
        publication_root=root, season=2026, completed_week=0
    )
    first_ats = confidence_ats.loc[
        confidence_ats["game_id"].eq("1")
        & confidence_ats["strategy"].eq("Margin-wide consensus")
    ].iloc[0]
    assert first_ats["ats_confidence"] == 0.5
    assert first_ats["stake"] == pytest.approx(confidence_scaled_stake(0.5))

    confidence_moneyline = build_confidence_scaled_moneyline_ledger(
        publication_root=root,
        season=2026,
        completed_week=0,
        lines_path=lines,
    )
    first_moneyline = confidence_moneyline.loc[
        confidence_moneyline["game_id"].eq("1")
        & confidence_moneyline["strategy"].eq("Margin-wide consensus")
    ].iloc[0]
    assert first_moneyline["moneyline_confidence"] == 0.7
    assert first_moneyline["stake"] == pytest.approx(confidence_scaled_stake(0.7))

    sweep = build_confidence_threshold_sweep(
        confidence_moneyline,
        result_column="moneyline_result",
        thresholds=pd.Series([0.5, 0.75]).to_numpy(),
    )
    margin = sweep.loc[sweep["strategy"].eq("Margin-wide consensus")]
    assert margin.loc[margin["confidence_cutoff"].eq(0.5), "bets"].iloc[0] == 3
    assert margin.loc[margin["confidence_cutoff"].eq(0.75), "bets"].iloc[0] == 1
    summary = summarize_profitable_confidence_thresholds(sweep, min_bets=1)
    assert set(summary["strategy"]) == {
        "Margin-wide consensus",
        "F0–F6 scientific consensus",
        "Full F0–F8 scientific consensus",
    }
