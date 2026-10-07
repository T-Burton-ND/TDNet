"""Relate F19 betting outcomes to prespecified confidence signals and cutoffs.

Run from the repository root with ``python3 scripts/publication/build_2026_f19_confidence_study.py``.
The ATS probability is an empirical historical-OOF residual estimate, not a
guaranteed calibrated cover probability. Every 2026 cutoff result is descriptive.
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
BASE = ROOT / "publication/2026/week_05/post_game/scientific/current_season_f0_f19/f19_flat_stake_betting"
OUTPUT = BASE / "confidence_study"
BASE_LEDGER = BASE / "bet_ledger.csv"
BASE_MANIFEST = BASE / "manifest.json"
FREEZE = Path("/groups/bsavoie2/tburton2/TDNet/f18_f19_constraint_free_v1/broad_search/final_fit/freeze")
FREEZE_MANIFEST = FREEZE / "freeze_manifest.json"
HISTORICAL_OOF = FREEZE / "historical_oof_and_ensembles.parquet"
STRATEGIES = ["M1", "M2", "M3", "M4", "M5", "M10", "F19 equal consensus"]
OOF_COLUMNS = {**{name: f"pred_margin_{name}" for name in STRATEGIES[:-1]},
               STRATEGIES[-1]: "pred_margin_equal"}
MIN_BETS = 30
THRESHOLDS = np.round(np.arange(0.50, 1.001, 0.01), 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signed_money(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_manifest = json.loads(BASE_MANIFEST.read_text())
    if sha256(BASE_LEDGER) != base_manifest["outputs_sha256"]["bet_ledger.csv"]:
        raise ValueError("Flat-stake ledger differs from its publication manifest.")
    freeze_manifest = json.loads(FREEZE_MANIFEST.read_text())
    if sha256(HISTORICAL_OOF) != freeze_manifest["historical_oof_sha256"]:
        raise ValueError("Historical OOF predictions differ from their freeze manifest.")
    ledger = pd.read_csv(BASE_LEDGER, dtype={"game_id": str})
    if len(ledger) != 7 * 271 * 3 or ledger.duplicated(
        ["strategy", "game_id", "market", "scenario"]
    ).any():
        raise ValueError("Unexpected base ledger shape or duplicate games.")
    oof = pd.read_parquet(HISTORICAL_OOF)
    oof = oof.loc[oof.tier.eq("F19")].copy()
    if len(oof) != 2226 or oof.duplicated("target_game_id").any():
        raise ValueError("Expected 2,226 unique historical F19 OOF games.")
    if not set(oof.season.unique()).issubset({2022, 2023, 2024, 2025}):
        raise ValueError("Historical residual source contains a nondevelopment season.")
    return ledger, oof


def add_confidence(ledger: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = ledger.copy()
    output["confidence"] = np.nan
    output["ats_model_agreement"] = np.nan
    output["moneyline_break_even_probability"] = np.nan
    output["moneyline_model_edge"] = np.nan
    residual_rows = oof[["target_game_id", "season"]].copy()
    ats = output.market.eq("ATS") & output.bet_placed
    # For a home ATS pick, actual cover occurs when residual > -predicted edge.
    # For an away pick, actual cover occurs when residual < -predicted edge.
    for strategy, column in OOF_COLUMNS.items():
        residual = (oof.actual_margin - oof[column]).to_numpy(dtype=float)
        if not np.isfinite(residual).all():
            raise ValueError(f"Nonfinite historical residual for {strategy}.")
        residual_rows["residual_" + strategy.replace(" ", "_")] = residual
        sorted_residual = np.sort(residual)
        mask = ats & output.strategy.eq(strategy)
        edge = (output.loc[mask, "pred_margin"] + output.loc[mask, "market_home_spread"]).to_numpy()
        cutoff = -edge
        left = np.searchsorted(sorted_residual, cutoff, side="left")
        right = np.searchsorted(sorted_residual, cutoff, side="right")
        confidence = np.where(edge > 0, (len(residual) - right) / len(residual), left / len(residual))
        output.loc[mask, "confidence"] = confidence
    # Six-model support is a separate agreement diagnostic, not a cover probability.
    model_ats = output.loc[ats & output.strategy.isin(STRATEGIES[:-1]),
                           ["game_id", "strategy", "bet_side"]]
    support = model_ats.pivot(index="game_id", columns="strategy", values="bet_side")
    if support.shape != (263, 6) or support.isna().any().any():
        raise ValueError("Expected six ATS model votes on each eligible F19 game.")
    for side in ["home", "away"]:
        share = support.eq(side).mean(axis=1)
        mask = ats & output.bet_side.eq(side)
        output.loc[mask, "ats_model_agreement"] = output.loc[mask, "game_id"].map(share)
    moneyline = output.market.eq("moneyline") & output.bet_placed
    chosen_probability = np.where(output.loc[moneyline, "bet_side"].eq("home"),
                                  output.loc[moneyline, "home_win_probability"],
                                  1 - output.loc[moneyline, "home_win_probability"])
    odds = output.loc[moneyline, "american_odds"].to_numpy()
    break_even = np.empty_like(odds)
    negative = odds < 0
    break_even[negative] = -odds[negative] / (-odds[negative] + 100)
    break_even[~negative] = 100 / (odds[~negative] + 100)
    output.loc[moneyline, "confidence"] = chosen_probability
    output.loc[moneyline, "moneyline_break_even_probability"] = break_even
    output.loc[moneyline, "moneyline_model_edge"] = chosen_probability - break_even
    if not output.loc[output.bet_placed, "confidence"].between(0, 1).all():
        raise ValueError("A placed bet is missing a valid confidence value.")
    return output, residual_rows


def aggregate(group: pd.DataFrame) -> dict[str, float | int]:
    bets = int(len(group))
    stake = float(group.stake_usd.sum())
    net = float(group.net_profit_usd.sum())
    return {"bets": bets, "wins": int(group.outcome.eq("win").sum()),
            "losses": int(group.outcome.eq("loss").sum()),
            "pushes": int(group.outcome.eq("push").sum()),
            "staked_usd": stake, "net_profit_usd": net,
            "roi": net / stake if stake else np.nan,
            "mean_confidence": float(group.confidence.mean()) if bets else np.nan,
            "observed_win_rate_ex_pushes": (
                float(group.outcome.eq("win").sum() / group.outcome.ne("push").sum())
                if group.outcome.ne("push").any() else np.nan),
            "frozen_price_bets": int(group.odds_source.eq("frozen_pregame_week4").sum()),
            "retrospective_price_bets": int(group.odds_source.eq(
                "retrospective_CFBD_quote_timing_unverified").sum())}


def threshold_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (market, scenario, strategy), group in frame.loc[frame.bet_placed].groupby(
        ["market", "scenario", "strategy"], sort=False
    ):
        baseline = float(group.net_profit_usd.sum())
        for cutoff in np.r_[0.0, THRESHOLDS]:
            selected = group.loc[group.confidence.ge(cutoff)]
            rows.append({"market": market, "scenario": scenario, "strategy": strategy,
                         "confidence_cutoff": cutoff, "all_bets_baseline_net_usd": baseline,
                         "net_change_vs_all_bets_usd": float(selected.net_profit_usd.sum() - baseline),
                         **aggregate(selected)})
    return pd.DataFrame(rows)


def confidence_bins(frame: pd.DataFrame) -> pd.DataFrame:
    bins = np.array([0, .5, .55, .60, .65, .70, .80, .90, 1.000001])
    rows = []
    for (market, scenario, strategy), group in frame.loc[frame.bet_placed].groupby(
        ["market", "scenario", "strategy"], sort=False
    ):
        for low, high in zip(bins[:-1], bins[1:]):
            selected = group.loc[group.confidence.ge(low) & group.confidence.lt(high)]
            rows.append({"market": market, "scenario": scenario, "strategy": strategy,
                         "confidence_band": f"{low:.0%}–{min(high, 1):.0%}",
                         "low_inclusive": low, "high_exclusive": min(high, 1),
                         **aggregate(selected)})
    return pd.DataFrame(rows)


def rank_correlation(first: pd.Series, second: pd.Series) -> float:
    if first.nunique() < 2 or second.nunique() < 2:
        return np.nan
    return float(first.rank().corr(second.rank()))


def correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (market, scenario, strategy), group in frame.loc[frame.bet_placed].groupby(
        ["market", "scenario", "strategy"], sort=False
    ):
        resolved = group.loc[group.outcome.ne("push")]
        rows.append({"market": market, "scenario": scenario, "strategy": strategy,
                     "bets": len(group),
                     "spearman_confidence_vs_net_profit_per_bet": rank_correlation(
                         group.confidence, group.net_profit_usd),
                     "spearman_confidence_vs_win": rank_correlation(
                         resolved.confidence, resolved.outcome.eq("win").astype(int)),
                     "spearman_model_edge_vs_net_profit_per_bet": (
                         rank_correlation(group.moneyline_model_edge, group.net_profit_usd)
                         if market == "moneyline" else np.nan)})
    return pd.DataFrame(rows)


def best_cutoffs(sweep: pd.DataFrame) -> pd.DataFrame:
    eligible = sweep.loc[sweep.confidence_cutoff.gt(0) & sweep.bets.ge(MIN_BETS)].copy()
    return (eligible.sort_values(["market", "scenario", "strategy", "net_profit_usd",
                                  "confidence_cutoff"], ascending=[True, True, True, False, True])
            .drop_duplicates(["market", "scenario", "strategy"])
            .assign(selection_warning="selected_on_same_2026_outcomes_not_forward_validated"))


def week_split_check(frame: pd.DataFrame) -> pd.DataFrame:
    """Select on Weeks 1–3 and settle Weeks 4–5 without reselecting the cutoff."""
    rows = []
    for (market, scenario, strategy), group in frame.loc[
        frame.bet_placed & frame.scenario.ne("frozen_pregame_subset")
    ].groupby(["market", "scenario", "strategy"], sort=False):
        train = group.loc[group.week.le(3)]
        check = group.loc[group.week.ge(4)]
        candidates = []
        for cutoff in np.r_[0.0, THRESHOLDS]:
            selected = train.loc[train.confidence.ge(cutoff)]
            if len(selected) >= MIN_BETS:
                candidates.append((float(selected.net_profit_usd.sum()), -cutoff,
                                   float(cutoff), len(selected)))
        if not candidates:
            raise ValueError(f"No training cutoff has {MIN_BETS} bets for {strategy}.")
        training_net, _, cutoff, training_bets = max(candidates)
        selected_check = check.loc[check.confidence.ge(cutoff)]
        rows.append({"market": market, "scenario": scenario, "strategy": strategy,
                     "selection_weeks": "1-3", "check_weeks": "4-5",
                     "selected_cutoff": cutoff, "selection_bets": training_bets,
                     "selection_net_usd": training_net, "check_all_bets": len(check),
                     "check_all_net_usd": float(check.net_profit_usd.sum()),
                     "check_selected_bets": len(selected_check),
                     "check_selected_net_usd": float(selected_check.net_profit_usd.sum()),
                     "check_net_change_vs_all_usd": float(
                         selected_check.net_profit_usd.sum() - check.net_profit_usd.sum()),
                     "warning": "threshold_split_only_forecasts_and_most_quotes_remain_retrospective"})
    return pd.DataFrame(rows)


def plot_thresholds(sweep: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.subplots_adjust(left=.07, right=.96, bottom=.12, top=.9, hspace=.3, wspace=.18)
    colors = plt.get_cmap("tab10").colors
    for column, (market, scenario, max_cutoff) in enumerate([
        ("ATS", "all_eligible", .75), ("moneyline", "quoted_odds", 1.0)
    ]):
        part = sweep.loc[(sweep.market == market) & (sweep.scenario == scenario) &
                         sweep.confidence_cutoff.between(.5, max_cutoff)]
        for index, strategy in enumerate(STRATEGIES):
            line = part.loc[part.strategy.eq(strategy)]
            axes[0, column].plot(line.confidence_cutoff * 100, line.net_profit_usd,
                                 color=colors[index], label=strategy, linewidth=2 if index == 6 else 1.3)
            supported = line.loc[line.bets.ge(MIN_BETS)]
            axes[1, column].plot(supported.confidence_cutoff * 100, supported.roi * 100,
                                 color=colors[index], linewidth=2 if index == 6 else 1.3)
        count = part.loc[part.strategy.eq(STRATEGIES[-1])]
        count_axis = axes[0, column].twinx()
        count_axis.plot(count.confidence_cutoff * 100, count.bets, color="grey", linestyle=":",
                        linewidth=2, label="Consensus bet count")
        count_axis.set_ylabel("Consensus bets", color="dimgray")
        count_axis.tick_params(axis="y", colors="dimgray")
        for row, label in [(0, "Net profit ($)"), (1, "ROI (%) where ≥30 bets")]:
            axis = axes[row, column]
            axis.axhline(0, color="black", linewidth=.8)
            axis.grid(alpha=.18)
            axis.set_ylabel(label)
            axis.set_xlabel("Minimum estimated confidence (%)")
        axes[0, column].set_title(f"{market.upper()} confidence cutoffs")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("F19 $10 betting replay: confidence cutoff sensitivity, 2026 Weeks 1–5")
    fig.text(.5, .035,
             "ATS confidence uses 2022–25 OOF residuals; ML uses picked-team win probability. Grey dotted = consensus bet count. These are in-sample 2026 curves.",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_readme(sweep: pd.DataFrame, best: pd.DataFrame,
                 correlations: pd.DataFrame, split: pd.DataFrame) -> None:
    lines = [
        "# F19 confidence and betting profit, 2026 Weeks 1–5", "",
        "This extends the $10 flat-stake F19 betting replay. It describes how realized profit "
        "changes when a wager is placed only above each confidence cutoff. The models, "
        "consensus, odds, and game outcomes are unchanged.", "",
        "## Meaning of a confidence percentage", "",
        "- ATS: empirical chance that the chosen side covers, estimated from each strategy's "
        "2,226 frozen 2022–2025 out-of-fold margin residuals. The residual CDF is evaluated "
        "at that game's predicted margin plus pregame spread. This is a historical error "
        "estimate, not a verified calibrated cover probability. The ledger also carries the "
        "fraction of six F19 models agreeing with the ATS side; that is agreement, not probability.",
        "- Moneyline: the frozen F19 probability assigned to the picked winner. The ledger "
        "also shows the selected quote's break-even probability and model-minus-price gap. "
        "A high win probability does not alone imply favorable odds.",
        "- Cutoffs run from 50% through 100% in 1-point steps. The 0% row is the original "
        "all-priced-bets baseline. Bets remain $10; skipped bets cost $0. ROI equals net "
        "profit divided by dollars actually staked.", "",
        "## Price and research limits", "",
        "ATS prices remain assumed -110. The broad moneyline scenario has 56 frozen "
        "pregame Week 4 prices and 194 retrospectively archived prices with unverified "
        "timing. A separate Week 4 moneyline scenario uses only its 56 frozen prices. "
        "F19 forecasts were produced after these games, and earlier project work had "
        "inspected 2026 results. Historical F19 market quote timing is also unverified. "
        "Cutoff rankings and correlations here reuse the same 2026 outcomes; they are "
        "not a tested forward betting rule. Many ATS probabilities cluster near 50%, "
        "so high cutoffs often leave very few games.", "",
        "## Observed relationship", "",
    ]
    consensus = "F19 equal consensus"
    ml_corr = correlations.loc[(correlations.scenario == "quoted_odds") &
                                correlations.strategy.eq(consensus)].iloc[0]
    ml_all = sweep.loc[(sweep.scenario == "quoted_odds") &
                       sweep.strategy.eq(consensus) & sweep.confidence_cutoff.eq(0)].iloc[0]
    ml_90 = sweep.loc[(sweep.scenario == "quoted_odds") &
                      sweep.strategy.eq(consensus) & sweep.confidence_cutoff.eq(.9)].iloc[0]
    ats_m3_all = sweep.loc[(sweep.market == "ATS") & sweep.strategy.eq("M3") &
                           sweep.confidence_cutoff.eq(0)].iloc[0]
    ats_m3_53 = sweep.loc[(sweep.market == "ATS") & sweep.strategy.eq("M3") &
                          sweep.confidence_cutoff.eq(.53)].iloc[0]
    lines += [
        f"- Consensus moneyline confidence had Spearman correlation "
        f"{ml_corr.spearman_confidence_vs_win:+.3f} with winning the game, but "
        f"{ml_corr.spearman_confidence_vs_net_profit_per_bet:+.3f} with dollars won per bet "
        f"on the 250 quoted-price games. The ≥90% cutoff kept {ml_90.bets} bets and "
        f"netted ${ml_90.net_profit_usd:,.2f}; all {ml_all.bets} quoted bets netted "
        f"${ml_all.net_profit_usd:,.2f}.",
        f"- ATS M3 at an estimated ≥53% cover cutoff kept {ats_m3_53.bets} bets and "
        f"netted ${ats_m3_53.net_profit_usd:,.2f}, versus ${ats_m3_all.net_profit_usd:,.2f} "
        f"on all {ats_m3_all.bets} ATS bets. That cutoff was found by examining these outcomes.",
        "- In the temporal cutoff check, Weeks 1–3 select the net-maximizing cutoff "
        "with at least 30 training bets (including an all-bets option); Weeks 4–5 are "
        "then settled without changing the cutoff. This isolates threshold selection "
        "within the replay, but the underlying F19 forecasts and most odds remain retrospective.",
        "", "## Weeks 1–3 selection, Weeks 4–5 check", "",
        "| Market | Strategy | Selected cutoff | Check bets | Check net | All-bets check net |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for market, strategy in [("ATS", "M3"), ("ATS", consensus),
                             ("moneyline", "M10"), ("moneyline", consensus)]:
        row = split.loc[split.market.eq(market) & split.strategy.eq(strategy)].iloc[0]
        lines.append(f"| {market} | {strategy} | "
                     f"{'all bets' if row.selected_cutoff == 0 else f'{row.selected_cutoff:.0%}'} "
                     f"| {row.check_selected_bets} | {signed_money(row.check_selected_net_usd)} "
                     f"| {signed_money(row.check_all_net_usd)} |")
    lines += ["",
        "## Selected examples", "",
        "The following rows maximize observed net profit among cutoffs retaining at least "
        "30 bets; the selection is entirely in-sample. Compare with the all-bets baseline "
        "rather than treating a selected cutoff as a recommendation.", "",
        "| Market | Scenario | Strategy | Cutoff | Bets | Net | Change vs all bets |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for market, scenario, strategy in [
        ("ATS", "all_eligible", "M3"),
        ("ATS", "all_eligible", "F19 equal consensus"),
        ("moneyline", "quoted_odds", "M10"),
        ("moneyline", "quoted_odds", "F19 equal consensus"),
        ("moneyline", "frozen_pregame_subset", "F19 equal consensus"),
    ]:
        row = best.loc[(best.market == market) & best.scenario.eq(scenario) &
                       best.strategy.eq(strategy)].iloc[0]
        lines.append(f"| {market} | {scenario} | {strategy} | {row.confidence_cutoff:.0%} "
                     f"| {row.bets} | {signed_money(row.net_profit_usd)} "
                     f"| {signed_money(row.net_change_vs_all_bets_usd)} |")
    lines += ["", "## Files", "", "- `confidence_ledger.csv`: game-level confidence, "
              "model agreement, quoted break-even probability, and unchanged outcomes.",
              "- `threshold_sweep.csv`: all strategies, scenarios, cutoffs, net, ROI, and bet counts.",
              "- `confidence_bins.csv`: disjoint confidence bands; these do not double-count games.",
              "- `correlations.csv`: descriptive Spearman rank correlations with win and net per bet.",
              "- `week_split_check.csv`: thresholds selected on Weeks 1–3, then checked on Weeks 4–5.",
              "- `best_cutoffs_in_sample.csv`: 30-bet-minimum observed net maxima, not recommended rules.",
              "- `historical_f19_oof_residuals.csv`: audited source for ATS percentages.",
              "- `confidence_thresholds.png`: profit, ROI, and surviving-bet views.",
              "- `manifest.json`: immutable input and output hashes.", ""]
    (OUTPUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ledger, oof = load_sources()
    confidence, residuals = add_confidence(ledger, oof)
    sweep = threshold_sweep(confidence)
    bins = confidence_bins(confidence)
    correlations = correlation_table(confidence)
    best = best_cutoffs(sweep)
    split = week_split_check(confidence)
    for name, frame in [
        ("confidence_ledger.csv", confidence), ("historical_f19_oof_residuals.csv", residuals),
        ("threshold_sweep.csv", sweep), ("confidence_bins.csv", bins),
        ("correlations.csv", correlations), ("best_cutoffs_in_sample.csv", best),
        ("week_split_check.csv", split),
    ]:
        frame.to_csv(OUTPUT / name, index=False, float_format="%.10g")
    plot_thresholds(sweep, OUTPUT / "confidence_thresholds.png")
    write_readme(sweep, best, correlations, split)
    manifest = {
        "schema": "tdnet-f19-betting-confidence-study-v1", "season": 2026,
        "through_week": 5, "stake_per_bet_usd": 10,
        "ats_confidence": "empirical chosen-side cover probability from 2226 F19 2022-2025 historical OOF residuals",
        "moneyline_confidence": "frozen picked-team win probability",
        "cutoffs": "0% all-priced baseline; 50%-100% in 1 percentage point steps",
        "min_bets_for_best_cutoff": MIN_BETS,
        "best_cutoff_is_in_sample": True,
        "input_sha256": {str(path): sha256(path) for path in
                         [BASE_LEDGER, BASE_MANIFEST, FREEZE_MANIFEST, HISTORICAL_OOF, Path(__file__)]},
        "outputs_sha256": {path.name: sha256(path) for path in OUTPUT.iterdir()
                           if path.is_file() and path.name != "manifest.json"},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(best[["market", "scenario", "strategy", "confidence_cutoff", "bets",
                "net_profit_usd", "net_change_vs_all_bets_usd"]].to_string(index=False))


if __name__ == "__main__":
    main()
