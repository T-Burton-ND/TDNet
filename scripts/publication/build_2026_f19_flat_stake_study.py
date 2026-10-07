"""Audit a $10 flat-stake F19 betting replay through completed 2026 Week 5.

Run from the repository root with ``python3 scripts/publication/build_2026_f19_flat_stake_study.py``.
The forecast is a retrospective, frozen scientific research artifact; only the
Week 4 moneyline snapshot has documented pregame quote capture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "publication/2026/week_05/post_game/scientific/current_season_f0_f19"
OUTPUT = SOURCE / "f19_flat_stake_betting"
MODEL = SOURCE / "scientific_2026_model_game_results.parquet"
CONSENSUS = SOURCE / "scientific_2026_consensus_game_results.parquet"
MARKET = SOURCE / "provenance/scientific_2026_pregame_market_snapshots.csv"
RAW_ODDS = ROOT / "data/raw/cfbd/v2/lines/2026.parquet"
FROZEN_ODDS = ROOT / "publication/2026/week_04/pre_game/metadata/cfbd_moneyline_snapshot.csv"
SOURCE_MANIFEST = SOURCE / "manifest.json"
STAKE = 10.0
ATS_ODDS = -110.0  # Explicit hypothetical price: CFBD does not archive spread-side prices.
STRATEGIES = ["M1", "M2", "M3", "M4", "M5", "M10", "F19 equal consensus"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_forecasts() -> pd.DataFrame:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    for path, key in [(MODEL, "model_games_parquet"), (CONSENSUS, "consensus_games_parquet")]:
        expected = manifest["outputs"][key]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"Source changed since the full scientific export: {path}")
    models = pd.read_parquet(MODEL)
    models = models.loc[models.fingerprint.eq("F19") & models.model_id.isin(STRATEGIES[:-1])].copy()
    models["strategy"] = models.model_id
    consensus = pd.read_parquet(CONSENSUS)
    consensus = consensus.loc[consensus.series_id.eq("fingerprint_F19")].copy()
    consensus["strategy"] = STRATEGIES[-1]
    # Preserve the eight F19-unavailable games explicitly for every strategy.
    missing = models.loc[models.model_id.eq("M1") & models.forecast_status.ne("available")].copy()
    missing["strategy"] = STRATEGIES[-1]
    missing[["pred_margin", "home_win_probability", "market_home_spread"]] = np.nan
    frame = pd.concat([models, consensus, missing], ignore_index=True)
    if len(frame) != 7 * 271 or frame.duplicated(["strategy", "game_id"]).any():
        raise ValueError("Expected exactly one row per strategy and each of 271 games.")
    counts = frame.groupby("strategy").forecast_status.value_counts()
    for strategy in STRATEGIES:
        if counts.get((strategy, "available"), 0) != 263:
            raise ValueError(f"Expected 263 available forecasts for {strategy}.")
    market = pd.read_csv(MARKET, dtype={"target_game_id": str})
    frame["game_id"] = frame.game_id.astype(str)
    frame = frame.merge(
        market[["target_game_id", "target_start_utc", "market_home_spread"]].rename(
            columns={"target_game_id": "game_id", "market_home_spread": "archived_spread"}
        ),
        on="game_id", how="left", validate="many_to_one",
    )
    check = frame.loc[frame.forecast_status.eq("available")]
    if not np.allclose(check.market_home_spread, check.archived_spread):
        raise ValueError("Model source spread differs from archived pregame spread.")
    if check[["pred_margin", "home_win_probability", "actual_home_margin", "market_home_spread"]].isna().any().any():
        raise ValueError("Available forecast is missing a settlement input.")
    if not check.home_win_probability.between(0, 1).all():
        raise ValueError("Forecast probability outside [0, 1].")
    return frame.sort_values(["strategy", "target_start_utc", "game_id"], kind="stable")


def raw_quotes() -> dict[str, dict[str, tuple[float, str] | None]]:
    frame = pd.read_parquet(RAW_ODDS)
    output = {}
    for row in frame.itertuples(index=False):
        offers = row.lines.tolist() if hasattr(row.lines, "tolist") else row.lines
        game = {}
        for side, field in [("home", "homeMoneyline"), ("away", "awayMoneyline")]:
            options = []
            for offer in offers or []:
                value = offer.get(field)
                if value is not None and np.isfinite(float(value)) and float(value) != 0:
                    options.append((float(value), str(offer.get("provider") or "unknown")))
            game[side] = max(options, key=lambda quote: quote[0]) if options else None
        output[str(row.id)] = game
    return output


def frozen_quotes() -> dict[str, dict[str, tuple[float, str] | None]]:
    frame = pd.read_csv(FROZEN_ODDS, dtype={"game_id": str})
    if not frame.snapshot_timing.eq("pregame").all():
        raise ValueError("Frozen moneyline file has a non-pregame row.")
    output = {}
    for row in frame.itertuples(index=False):
        output[row.game_id] = {
            side: (float(value), str(provider)) if pd.notna(value) else None
            for side, value, provider in [
                ("home", row.home_moneyline, row.home_moneyline_provider),
                ("away", row.away_moneyline, row.away_moneyline_provider),
            ]
        }
    return output


def american_win_profit(stake: float, odds: float) -> float:
    if not np.isfinite(odds) or odds == 0:
        raise ValueError("Invalid American odds.")
    return stake * (100 / abs(odds) if odds < 0 else odds / 100)


def money(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def ledger(forecasts: pd.DataFrame, market: str, scenario: str,
           raw: dict, frozen: dict) -> pd.DataFrame:
    rows = []
    for row in forecasts.itertuples(index=False):
        base = {key: getattr(row, key) for key in [
            "strategy", "game_id", "week", "target_start_utc", "home_team", "away_team",
            "forecast_status", "pred_margin", "home_win_probability", "market_home_spread",
            "actual_home_margin",
        ]}
        base.update(market=market, scenario=scenario, stake_usd=0.0, net_profit_usd=0.0,
                    gross_return_usd=0.0, bet_side=None, outcome=None,
                    american_odds=np.nan, odds_provider=None, odds_source=None,
                    no_bet_reason=None)
        if row.forecast_status != "available":
            base["no_bet_reason"] = "no_F19_forecast_or_pregame_spread"
            rows.append(base)
            continue
        if market == "ATS":
            edge = float(row.pred_margin + row.market_home_spread)
            if np.isclose(edge, 0, atol=1e-10):
                base["no_bet_reason"] = "zero_predicted_spread_edge"
                rows.append(base)
                continue
            side = "home" if edge > 0 else "away"
            realized_edge = float(row.actual_home_margin + row.market_home_spread)
            outcome = "push" if np.isclose(realized_edge, 0, atol=1e-10) else (
                "win" if (realized_edge > 0) == (side == "home") else "loss"
            )
            odds, provider, source = ATS_ODDS, None, "assumed_standard_minus_110"
        else:
            probability = float(row.home_win_probability)
            if np.isclose(probability, 0.5, atol=1e-12):
                base["no_bet_reason"] = "exact_50_percent_win_probability"
                rows.append(base)
                continue
            side = "home" if probability > 0.5 else "away"
            if scenario == "frozen_pregame_subset" and int(row.week) != 4:
                base["no_bet_reason"] = "no_frozen_pregame_moneyline_snapshot"
                rows.append(base)
                continue
            quotes = frozen if int(row.week) == 4 else raw
            quote = quotes.get(str(row.game_id), {}).get(side)
            if quote is None:
                base["no_bet_reason"] = "selected_side_moneyline_missing"
                rows.append(base)
                continue
            odds, provider = quote
            source = ("frozen_pregame_week4" if int(row.week) == 4
                      else "retrospective_CFBD_quote_timing_unverified")
            outcome = "win" if (row.actual_home_margin > 0) == (side == "home") else "loss"
            if row.actual_home_margin == 0:
                raise ValueError("Tied game needs an explicit moneyline settlement policy.")
        profit = american_win_profit(STAKE, odds) if outcome == "win" else (
            -STAKE if outcome == "loss" else 0.0
        )
        base.update(bet_side=side, outcome=outcome, american_odds=odds,
                    odds_provider=provider, odds_source=source, stake_usd=STAKE,
                    net_profit_usd=profit, gross_return_usd=STAKE + profit)
        rows.append(base)
    output = pd.DataFrame(rows)
    output["bet_placed"] = output.stake_usd.gt(0)
    output["cumulative_staked_usd"] = output.groupby("strategy").stake_usd.cumsum()
    output["cumulative_net_profit_usd"] = output.groupby("strategy").net_profit_usd.cumsum()
    output["cumulative_roi"] = output.cumulative_net_profit_usd / output.cumulative_staked_usd.replace(0, np.nan)
    return output


def summarize(frame: pd.DataFrame, by_week: bool = False) -> pd.DataFrame:
    keys = ["market", "scenario", "strategy"] + (["week"] if by_week else [])
    group = frame.groupby(keys, sort=False, dropna=False)
    result = group.agg(target_games=("game_id", "size"), bets=("bet_placed", "sum"),
                       staked_usd=("stake_usd", "sum"), net_profit_usd=("net_profit_usd", "sum"),
                       gross_return_usd=("gross_return_usd", "sum")).reset_index()
    for label, plural in [("win", "wins"), ("loss", "losses"), ("push", "pushes")]:
        result[plural] = group.outcome.apply(lambda x, label=label: int(x.eq(label).sum())).to_numpy()
    result["no_bets"] = result.target_games - result.bets
    result["roi"] = result.net_profit_usd / result.staked_usd.replace(0, np.nan)
    result["strategy"] = pd.Categorical(result.strategy, categories=STRATEGIES, ordered=True)
    result = result.sort_values(["market", "scenario", "strategy"] + (["week"] if by_week else []))
    result["strategy"] = result.strategy.astype(str)
    if by_week:
        result["cumulative_net_profit_usd"] = result.groupby(
            ["market", "scenario", "strategy"], sort=False
        ).net_profit_usd.cumsum()
        result["ending_bankroll_from_1000_usd"] = 1000 + result.cumulative_net_profit_usd
    else:
        result["ending_bankroll_from_1000_usd"] = 1000 + result.net_profit_usd
    return result


def plot_bankroll(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.subplots_adjust(left=0.06, right=0.93, top=0.87, bottom=0.19, wspace=0.31)
    colors = plt.get_cmap("tab10").colors
    for axis, (market, scenario) in zip(axes, [("ATS", "all_eligible"), ("moneyline", "quoted_odds")]):
        subset = frame.loc[(frame.market == market) & (frame.scenario == scenario)]
        for index, strategy in enumerate(STRATEGIES):
            bets = subset.loc[subset.strategy.eq(strategy) & subset.bet_placed]
            axis.plot(np.arange(len(bets) + 1), np.r_[0, bets.cumulative_net_profit_usd],
                      label=strategy, color=colors[index], linewidth=2 if index == 6 else 1.4)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=f"{market.upper()}: $10 per settled pick", xlabel="Bets placed",
                 ylabel="Cumulative net profit ($)")
        axis.grid(alpha=0.2)
        # Every strategy stakes the same $10 on the same number of eligible games.
        # Use a separate scale so cumulative stakes do not compress profit curves.
        stake_path = subset.loc[subset.strategy.eq(STRATEGIES[0]) & subset.bet_placed]
        stake_axis = axis.twinx()
        stake_line, = stake_axis.plot(
            np.arange(len(stake_path) + 1), np.r_[0, stake_path.cumulative_staked_usd],
            color="grey", linestyle=":", linewidth=2.2, alpha=0.8,
            label="Total wagered",
        )
        stake_axis.set_ylim(0, float(stake_path.cumulative_staked_usd.iloc[-1]) * 1.08)
        stake_axis.set_ylabel("Cumulative wagered ($)", color="dimgray")
        stake_axis.tick_params(axis="y", colors="dimgray")
        if axis is axes[0]:
            handles, labels = axis.get_legend_handles_labels()
            axis.legend(handles + [stake_line], labels + ["Total wagered"],
                        loc="upper left", fontsize=8)
    fig.suptitle("F19 scientific replay, 2026 Weeks 1–5 | retrospective forecast", y=0.96)
    fig.text(0.5, 0.055, "ATS: assumed -110. Moneyline: frozen Week 4; other quoted odds have unverified timing. Fees and limits excluded.",
             ha="center", fontsize=8)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    forecasts = read_forecasts()
    raw, frozen = raw_quotes(), frozen_quotes()
    frame = pd.concat([
        ledger(forecasts, "ATS", "all_eligible", raw, frozen),
        ledger(forecasts, "moneyline", "quoted_odds", raw, frozen),
        ledger(forecasts, "moneyline", "frozen_pregame_subset", raw, frozen),
    ], ignore_index=True)
    summary, weekly = summarize(frame), summarize(frame, by_week=True)
    for name, data in [("bet_ledger.csv", frame), ("summary.csv", summary), ("weekly.csv", weekly)]:
        data.to_csv(OUTPUT / name, index=False, float_format="%.10g")
    plot_bankroll(frame, OUTPUT / "cumulative_net_profit.png")
    lines = [
        "# F19 $10 flat-stake betting replay, 2026 Weeks 1–5", "",
        "This is a retrospective, counterfactual settlement of the six F19 scientific models "
        "and the frozen equal F19 consensus. The F19 forecasts were produced after the games "
        "and the broader project had previously inspected 2026 outcomes. These totals are not "
        "a live betting track record or a forward profitability estimate.", "",
        "## Rules and coverage", "",
        "- Stake $10 on every eligible game, independently for each strategy and market. "
        "Each strategy starts at $0 net profit; a hypothetical $1,000 bankroll is also shown.",
        "- ATS: take the home side when predicted home margin plus the archived pregame home spread "
        "is positive; otherwise take the away side. A zero edge means no bet. A graded tie is "
        "a push. All ATS bets assume American odds of -110 because the archive lacks spread-side prices.",
        "- Moneyline: take home when its frozen forecast win probability exceeds 50%, otherwise "
        "away. An exact 50% probability means no bet. The selected side needs an actual quoted "
        "American moneyline; no quoted price means no bet. For negative odds, winning net profit "
        "is $10 × 100 / |odds|; for positive odds it is $10 × odds / 100. Losses cost $10.",
        "- Quoted-odds scenario: use the frozen Week 4 pregame moneyline snapshot; use the best "
        "available provider price in the raw CFBD archive for Weeks 1–3 and 5. Those 194 "
        "archive prices have unverified timing and may not have been available when a bet "
        "could have been placed. Prices from different providers are selected independently "
        "for each side; availability, limits, fees, and movement are not modeled.",
        "- Frozen-pregame subset: settle only the 56 Week 4 games with archived pregame "
        "moneyline prices. The prices are frozen pregame; the F19 predictions still are not.",
        "- There are 271 completed target games, 263 with F19 forecasts and pregame spreads, "
        "and eight unavailable F19 games. The quoted-odds scenario has 250 priced games "
        "(56 frozen Week 4, 194 with retrospective prices); 13 available-forecast games "
        "lack a selected-side moneyline.", "",
        "## Net results", "",
        "| Strategy | ATS bets | ATS net | ATS ROI | ML bets | ML net | ML ROI | Week 4 ML net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        ats = summary.loc[(summary.market == "ATS") & summary.strategy.eq(strategy)].iloc[0]
        ml = summary.loc[(summary.scenario == "quoted_odds") & summary.strategy.eq(strategy)].iloc[0]
        frozen_ml = summary.loc[(summary.scenario == "frozen_pregame_subset") & summary.strategy.eq(strategy)].iloc[0]
        lines.append(f"| {strategy} | {ats.bets} | {money(ats.net_profit_usd)} | {ats.roi:+.2%} "
                     f"| {ml.bets} | {money(ml.net_profit_usd)} | {ml.roi:+.2%} "
                     f"| {money(frozen_ml.net_profit_usd)} |")
    lines += ["", "The Week 4 subset contains 56 bets per strategy. Cumulative curves show "
              "net profit above or below the starting bankroll, not total account balance.", "",
              "## Files", "",
              "- `bet_ledger.csv`: every target game × strategy × scenario, including skipped bets "
              "and their reasons, price source, side, outcome, stake, and cumulative results.",
              "- `summary.csv`: strategy totals with win/loss/push and ROI.",
              "- `weekly.csv`: weekly totals and cumulative bankroll from an initial $1,000.",
              "- `cumulative_net_profit.png`: ATS and broad quoted-moneyline profit paths "
              "(left axes) plus the grey dotted cumulative amount wagered per strategy "
              "(right axes). All seven strategies stake the same total within each panel.",
              "- `manifest.json`: source and output SHA-256 hashes.", ""]
    (OUTPUT / "README.md").write_text("\n".join(lines))
    source_paths = [MODEL, CONSENSUS, MARKET, RAW_ODDS, FROZEN_ODDS, SOURCE_MANIFEST, Path(__file__)]
    manifest = {
        "schema": "tdnet-f19-flat-stake-study-v1", "season": 2026, "through_week": 5,
        "stake_per_bet_usd": STAKE, "ats_assumed_american_odds": ATS_ODDS,
        "forecast_is_retrospective_counterfactual": True,
        "moneyline_quote_timing": "Only Week 4 has a frozen pregame quote snapshot; Weeks 1–3 and 5 use a retrospective archive with unverified timing.",
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        "outputs_sha256": {p.name: sha256(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != "manifest.json"},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
