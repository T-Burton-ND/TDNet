"""Track flat-stake ATS and moneyline performance for TDNet consensuses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .figure_theme import TDNET_COLORS, apply_tdnet_theme

DEFAULT_STAKE = 10.0
DEFAULT_ATS_ODDS = -110.0
CONFIDENCE_FLOOR = 0.499
CONFIDENCE_MAX_STAKE = 25.0
THRESHOLD_SWEEP_STAKE = 10.0
THRESHOLD_SWEEP_MIN_BETS = 10

CONSENSUS_SOURCES = (
    (
        "Margin-wide consensus",
        "tables/margin_wide_prediction_vs_actual.csv",
        {
            "ats_pick": "ats_pick",
            "home_spread": "market_spread_close",
            "ats_result": "ats_result",
            "su_pick": "pred_winner",
            "actual_winner": "actual_winner",
        },
    ),
    (
        "F0–F6 scientific consensus",
        "scientific/scientific_consensus_game_results.csv",
        {
            "ats_pick": "consensus_against_spread_team",
            "home_spread": "consensus_home_team_market_spread",
            "ats_result": "consensus_ats_result",
            "su_pick": "consensus_straight_up_pick",
            "actual_winner": "actual_winner",
        },
    ),
    (
        "Full F0–F8 scientific consensus",
        "scientific/full_f0_f8/scientific_consensus_game_results.csv",
        {
            "ats_pick": "consensus_against_spread_team",
            "home_spread": "consensus_home_team_market_spread",
            "ats_result": "consensus_ats_result",
            "su_pick": "consensus_straight_up_pick",
            "actual_winner": "actual_winner",
        },
    ),
)

SERIES_COLORS = {
    "Margin-wide consensus": TDNET_COLORS["ion_blue"],
    "F0–F6 scientific consensus": TDNET_COLORS["electric_emerald"],
    "Full F0–F8 scientific consensus": TDNET_COLORS["edge_pink"],
}

MODEL_CONFIDENCE_SOURCES = {
    "Margin-wide consensus": (
        "tables/margin_wide_model_game_results.csv",
        "ats_pick",
        "pred_home_win_probability",
    ),
    "F0–F6 scientific consensus": (
        "scientific/scientific_model_game_results.csv",
        "model_against_spread_team",
        "consensus_predicted_home_win_probability",
    ),
    "Full F0–F8 scientific consensus": (
        "scientific/full_f0_f8/scientific_model_game_results.csv",
        "model_against_spread_team",
        "consensus_predicted_home_win_probability",
    ),
}


def american_odds_win_profit(stake: float, american_odds: float) -> float:
    """Return net winnings for a winning stake at American odds."""
    stake = float(stake)
    odds = float(american_odds)
    if stake < 0:
        raise ValueError("Stake cannot be negative.")
    if not np.isfinite(odds) or odds == 0:
        raise ValueError("American odds must be finite and non-zero.")
    return stake * (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def confidence_scaled_stake(
    confidence: object,
    *,
    floor: float = CONFIDENCE_FLOOR,
    maximum_stake: float = CONFIDENCE_MAX_STAKE,
) -> float:
    """Linearly scale confidence from ``floor -> $0`` to ``100% -> maximum``."""
    value = float(confidence)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(maximum_stake * (value - floor) / (1.0 - floor), 0, maximum_stake))


def settle_ats_bet(result: object, *, stake: float, american_odds: float) -> tuple[float, float]:
    """Return ``(net_profit, gross_return)`` for one ATS result."""
    normalized = str(result).strip().casefold()
    if normalized == "win":
        profit = american_odds_win_profit(stake, american_odds)
        return profit, float(stake) + profit
    if normalized == "loss":
        return -float(stake), 0.0
    if normalized == "push":
        return 0.0, float(stake)
    raise ValueError(f"Cannot settle ungraded ATS result {result!r}.")


def _read_consensus_source(
    path: Path,
    *,
    strategy: str,
    columns: dict[str, str],
    season: int,
    week: int,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "game_id",
        "home_team",
        "away_team",
        *columns.values(),
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path} is missing ATS bankroll columns {sorted(missing)}.")
    output = pd.DataFrame(
        {
            "season": int(season),
            "week": int(week),
            "game_id": frame["game_id"].astype(str),
            "away_team": frame["away_team"].astype(str),
            "home_team": frame["home_team"].astype(str),
            "strategy": strategy,
            "ats_pick": frame[columns["ats_pick"]],
            "home_spread": pd.to_numeric(frame[columns["home_spread"]], errors="coerce"),
            "ats_result": frame[columns["ats_result"]].astype(str).str.title(),
            "su_pick": frame[columns["su_pick"]],
            "actual_winner": frame[columns["actual_winner"]],
        }
    )
    return output.sort_values("game_id", kind="stable").reset_index(drop=True)


def _consensus_confidence_table(
    *, publication_root: str | Path, season: int, completed_week: int
) -> pd.DataFrame:
    """Return ATS model support and picked-team win probability per consensus."""
    root = Path(publication_root)
    rows: list[pd.DataFrame] = []
    source_by_strategy = {
        strategy: (relative_path, columns)
        for strategy, relative_path, columns in CONSENSUS_SOURCES
    }
    for week in range(int(completed_week) + 1):
        postgame = root / f"week_{week:02d}" / "post_game"
        for strategy, (model_relative, ats_model_pick, probability_column) in (
            MODEL_CONFIDENCE_SOURCES.items()
        ):
            consensus_relative, consensus_columns = source_by_strategy[strategy]
            consensus_path = postgame / consensus_relative
            model_path = postgame / model_relative
            if not consensus_path.exists() or not model_path.exists():
                continue
            consensus = _read_consensus_source(
                consensus_path,
                strategy=strategy,
                columns=consensus_columns,
                season=season,
                week=week,
            )
            models = pd.read_csv(model_path)
            required = {"game_id", ats_model_pick, probability_column}
            if missing := required - set(models):
                raise ValueError(
                    f"{model_path} is missing confidence columns {sorted(missing)}."
                )
            models = models.rename(
                columns={
                    ats_model_pick: "__model_ats_pick",
                    probability_column: "__model_home_win_probability",
                }
            )
            models = models[
                ["game_id", "__model_ats_pick", "__model_home_win_probability"]
            ].copy()
            models["game_id"] = models["game_id"].astype(str)
            merged = models.merge(
                consensus[
                    ["game_id", "home_team", "away_team", "ats_pick", "su_pick"]
                ],
                on="game_id",
                how="inner",
                validate="many_to_one",
            )
            merged["supports_ats_consensus"] = merged["__model_ats_pick"].eq(
                merged["ats_pick"]
            )
            merged["home_win_probability"] = pd.to_numeric(
                merged["__model_home_win_probability"], errors="coerce"
            )
            grouped = (
                merged.groupby("game_id", as_index=False, sort=False)
                .agg(
                    home_team=("home_team", "first"),
                    away_team=("away_team", "first"),
                    su_pick=("su_pick", "first"),
                    ats_confidence=("supports_ats_consensus", "mean"),
                    home_win_probability=("home_win_probability", "mean"),
                )
            )
            grouped["moneyline_confidence"] = grouped[
                "home_win_probability"
            ].where(
                grouped["su_pick"].eq(grouped["home_team"]),
                1.0 - grouped["home_win_probability"],
            )
            grouped["season"] = int(season)
            grouped["week"] = int(week)
            grouped["strategy"] = strategy
            rows.append(
                grouped[
                    [
                        "season",
                        "week",
                        "game_id",
                        "strategy",
                        "ats_confidence",
                        "moneyline_confidence",
                    ]
                ]
            )
    if not rows:
        raise ValueError("No model-level consensus confidence sources were found.")
    return pd.concat(rows, ignore_index=True)


def _apply_confidence_staking(
    ledger: pd.DataFrame,
    *,
    confidence_column: str,
    result_column: str,
    maximum_stake: float,
    confidence_floor: float,
) -> pd.DataFrame:
    """Re-settle an existing priced ledger using confidence-scaled stakes."""
    output = ledger.copy()
    output["base_bet_eligible"] = output["bet_placed"].astype(bool)
    output["confidence"] = pd.to_numeric(output[confidence_column], errors="coerce")
    output["stake"] = output["confidence"].map(
        lambda value: confidence_scaled_stake(
            value, floor=confidence_floor, maximum_stake=maximum_stake
        )
    )
    output.loc[~output["base_bet_eligible"], "stake"] = 0.0
    output["bet_placed"] = output["base_bet_eligible"] & output["stake"].gt(0)
    settlements = [
        settle_ats_bet(
            getattr(row, result_column),
            stake=row.stake,
            american_odds=row.american_odds,
        )
        if row.bet_placed
        else (0.0, 0.0)
        for row in output.itertuples(index=False)
    ]
    output[["net_profit", "gross_return"]] = pd.DataFrame(
        settlements, index=output.index
    )
    output["staking_policy"] = (
        f"linear_{confidence_floor:.3f}_to_1.000__usd_0_to_{maximum_stake:g}"
    )
    output = output.sort_values(["strategy", "week", "game_id"], kind="stable")
    output["cumulative_staked"] = output.groupby("strategy")["stake"].cumsum()
    output["cumulative_net_profit"] = output.groupby("strategy")["net_profit"].cumsum()
    output["cumulative_roi"] = (
        output["cumulative_net_profit"]
        / output["cumulative_staked"].replace(0, np.nan)
    )
    return output.reset_index(drop=True)


def build_consensus_ats_bet_ledger(
    *,
    publication_root: str | Path,
    season: int,
    completed_week: int,
    stake: float = DEFAULT_STAKE,
    fallback_american_odds: float = DEFAULT_ATS_ODDS,
) -> pd.DataFrame:
    """Build one flat-stake ledger row per consensus prediction and game.

    CFBD's betting-lines payload publishes the spread itself but not the price
    on each side of that spread.  Therefore the frozen published spread grades
    the bet while ``fallback_american_odds`` supplies a transparent price
    assumption.  Moneyline odds are intentionally not reused as ATS prices.
    """
    root = Path(publication_root)
    rows: list[pd.DataFrame] = []
    for week in range(int(completed_week) + 1):
        postgame = root / f"week_{week:02d}" / "post_game"
        for strategy, relative_path, columns in CONSENSUS_SOURCES:
            path = postgame / relative_path
            if not path.exists():
                continue
            rows.append(
                _read_consensus_source(
                    path,
                    strategy=strategy,
                    columns=columns,
                    season=season,
                    week=week,
                )
            )
    if not rows:
        raise ValueError("No published consensus ATS results were found.")

    ledger = pd.concat(rows, ignore_index=True)
    ledger["bet_placed"] = (
        ledger["home_spread"].notna()
        & ledger["ats_pick"].notna()
        & ledger["ats_result"].str.casefold().isin({"win", "loss", "push"})
    )
    ledger["stake"] = np.where(ledger["bet_placed"], float(stake), 0.0)
    ledger["american_odds"] = np.where(
        ledger["bet_placed"], float(fallback_american_odds), np.nan
    )
    ledger["odds_source"] = np.where(
        ledger["bet_placed"],
        "standard_-110_assumption_cfbd_has_no_spread_side_price",
        "no_bet_missing_or_ungraded_spread",
    )
    settlements = [
        settle_ats_bet(
            row.ats_result,
            stake=row.stake,
            american_odds=row.american_odds,
        )
        if row.bet_placed
        else (0.0, 0.0)
        for row in ledger.itertuples(index=False)
    ]
    ledger[["net_profit", "gross_return"]] = pd.DataFrame(
        settlements, index=ledger.index
    )
    strategy_order = {name: index for index, (name, _, _) in enumerate(CONSENSUS_SOURCES)}
    ledger["strategy_order"] = ledger["strategy"].map(strategy_order)
    ledger = ledger.sort_values(
        ["strategy_order", "week", "game_id"], kind="stable"
    ).reset_index(drop=True)
    ledger["cumulative_staked"] = ledger.groupby("strategy")["stake"].cumsum()
    ledger["cumulative_net_profit"] = ledger.groupby("strategy")["net_profit"].cumsum()
    ledger["cumulative_roi"] = (
        ledger["cumulative_net_profit"]
        / ledger["cumulative_staked"].replace(0, np.nan)
    )
    return ledger.drop(columns="strategy_order")


def _best_available_cfbd_moneylines(
    lines_path: str | Path,
) -> dict[str, dict[str, tuple[float, str] | None]]:
    """Return the best real home/away moneyline quote in each CFBD game payload."""
    path = Path(lines_path)
    if not path.exists():
        raise FileNotFoundError(f"CFBD moneyline snapshot does not exist: {path}")
    frame = pd.read_parquet(path)
    required = {"id", "lines"}
    if missing := required - set(frame):
        raise ValueError(f"CFBD line snapshot is missing {sorted(missing)}.")
    output: dict[str, dict[str, tuple[float, str] | None]] = {}
    for row in frame.itertuples(index=False):
        offers = row.lines.tolist() if hasattr(row.lines, "tolist") else row.lines
        sides: dict[str, tuple[float, str] | None] = {"home": None, "away": None}
        for side, key in (("home", "homeMoneyline"), ("away", "awayMoneyline")):
            quotes = []
            for offer in offers or []:
                value = offer.get(key)
                if value is None:
                    continue
                odds = float(value)
                if odds == 0 or not np.isfinite(odds):
                    continue
                quotes.append((odds, str(offer.get("provider") or "unknown provider")))
            if quotes:
                # Higher American odds always give the bettor a larger payout:
                # -105 is better than -110, and +125 is better than +115.
                sides[side] = max(quotes, key=lambda item: item[0])
        output[str(row.id)] = sides
    return output


def write_cfbd_moneyline_snapshot(
    *,
    lines_path: str | Path,
    game_ids: list[object] | pd.Series,
    output_path: str | Path,
    prediction_deadline_utc: str,
) -> Path:
    """Freeze best available CFBD moneyline quotes alongside a pregame package."""
    quotes = _best_available_cfbd_moneylines(lines_path)
    rows = []
    for game_id in dict.fromkeys(str(value) for value in game_ids):
        game = quotes.get(game_id, {})
        home = game.get("home")
        away = game.get("away")
        rows.append(
            {
                "game_id": game_id,
                "home_moneyline": home[0] if home else np.nan,
                "home_moneyline_provider": home[1] if home else pd.NA,
                "away_moneyline": away[0] if away else np.nan,
                "away_moneyline_provider": away[1] if away else pd.NA,
                "prediction_deadline_utc": prediction_deadline_utc,
                "source_lines_path": str(Path(lines_path)),
                "snapshot_timing": "pregame",
            }
        )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def _moneylines_from_frozen_snapshot(
    path: Path,
) -> dict[str, dict[str, tuple[float, str] | None]]:
    frame = pd.read_csv(path)
    output: dict[str, dict[str, tuple[float, str] | None]] = {}
    for row in frame.itertuples(index=False):
        home = (
            (float(row.home_moneyline), str(row.home_moneyline_provider))
            if pd.notna(row.home_moneyline)
            else None
        )
        away = (
            (float(row.away_moneyline), str(row.away_moneyline_provider))
            if pd.notna(row.away_moneyline)
            else None
        )
        output[str(row.game_id)] = {"home": home, "away": away}
    return output


def build_consensus_moneyline_bet_ledger(
    *,
    publication_root: str | Path,
    season: int,
    completed_week: int,
    lines_path: str | Path | None = None,
    stake: float = DEFAULT_STAKE,
) -> pd.DataFrame:
    """Build $10 straight-up ledgers using best available CFBD moneyline quotes."""
    root = Path(publication_root)
    source = (
        Path(lines_path)
        if lines_path is not None
        else root.parent.parent / "data" / "raw" / "cfbd" / "v2" / "lines" / f"{season}.parquet"
    )
    fallback_quotes = _best_available_cfbd_moneylines(source)
    rows: list[pd.DataFrame] = []
    week_quote_sources: dict[int, str] = {}
    week_quotes: dict[int, dict[str, dict[str, tuple[float, str] | None]]] = {}
    for week in range(int(completed_week) + 1):
        postgame = root / f"week_{week:02d}" / "post_game"
        frozen_moneylines = (
            root
            / f"week_{week:02d}"
            / "pre_game"
            / "metadata"
            / "cfbd_moneyline_snapshot.csv"
        )
        if frozen_moneylines.exists():
            week_quotes[week] = _moneylines_from_frozen_snapshot(frozen_moneylines)
            week_quote_sources[week] = "frozen_pregame_cfbd_moneyline_snapshot"
        else:
            week_quotes[week] = fallback_quotes
            week_quote_sources[week] = "retrospective_cfbd_snapshot_not_frozen_at_pick_time"
        for strategy, relative_path, columns in CONSENSUS_SOURCES:
            path = postgame / relative_path
            if path.exists():
                rows.append(
                    _read_consensus_source(
                        path,
                        strategy=strategy,
                        columns=columns,
                        season=season,
                        week=week,
                    )
                )
    if not rows:
        raise ValueError("No published consensus straight-up results were found.")
    ledger = pd.concat(rows, ignore_index=True)

    prices: list[float] = []
    providers: list[str | None] = []
    sides: list[str | None] = []
    for row in ledger.itertuples(index=False):
        side = "home" if row.su_pick == row.home_team else "away" if row.su_pick == row.away_team else None
        quote = week_quotes[int(row.week)].get(str(row.game_id), {}).get(side) if side else None
        sides.append(side)
        prices.append(float(quote[0]) if quote else np.nan)
        providers.append(str(quote[1]) if quote else None)
    ledger["bet_side"] = sides
    ledger["american_odds"] = prices
    ledger["odds_provider"] = providers
    ledger["moneyline_result"] = np.where(
        ledger["su_pick"].eq(ledger["actual_winner"]), "Win", "Loss"
    )
    ledger["bet_placed"] = ledger["american_odds"].notna() & ledger["bet_side"].notna()
    ledger["stake"] = np.where(ledger["bet_placed"], float(stake), 0.0)
    ledger["odds_source"] = ledger["week"].map(week_quote_sources)
    ledger.loc[~ledger["bet_placed"], "odds_source"] = "no_bet_missing_cfbd_moneyline"
    settlements = [
        settle_ats_bet(
            row.moneyline_result,
            stake=row.stake,
            american_odds=row.american_odds,
        )
        if row.bet_placed
        else (0.0, 0.0)
        for row in ledger.itertuples(index=False)
    ]
    ledger[["net_profit", "gross_return"]] = pd.DataFrame(
        settlements, index=ledger.index
    )
    strategy_order = {name: index for index, (name, _, _) in enumerate(CONSENSUS_SOURCES)}
    ledger["strategy_order"] = ledger["strategy"].map(strategy_order)
    ledger = ledger.sort_values(
        ["strategy_order", "week", "game_id"], kind="stable"
    ).reset_index(drop=True)
    ledger["cumulative_staked"] = ledger.groupby("strategy")["stake"].cumsum()
    ledger["cumulative_net_profit"] = ledger.groupby("strategy")["net_profit"].cumsum()
    ledger["cumulative_roi"] = (
        ledger["cumulative_net_profit"]
        / ledger["cumulative_staked"].replace(0, np.nan)
    )
    return ledger.drop(columns="strategy_order")


def build_confidence_scaled_ats_ledger(
    *,
    publication_root: str | Path,
    season: int,
    completed_week: int,
    fallback_american_odds: float = DEFAULT_ATS_ODDS,
    confidence_floor: float = CONFIDENCE_FLOOR,
    maximum_stake: float = CONFIDENCE_MAX_STAKE,
) -> pd.DataFrame:
    """Build an ATS ledger staked by model support for the consensus side."""
    base = build_consensus_ats_bet_ledger(
        publication_root=publication_root,
        season=season,
        completed_week=completed_week,
        stake=maximum_stake,
        fallback_american_odds=fallback_american_odds,
    )
    confidence = _consensus_confidence_table(
        publication_root=publication_root,
        season=season,
        completed_week=completed_week,
    )
    merged = base.merge(
        confidence,
        on=["season", "week", "game_id", "strategy"],
        how="left",
        validate="one_to_one",
    )
    return _apply_confidence_staking(
        merged,
        confidence_column="ats_confidence",
        result_column="ats_result",
        maximum_stake=maximum_stake,
        confidence_floor=confidence_floor,
    )


def build_confidence_scaled_moneyline_ledger(
    *,
    publication_root: str | Path,
    season: int,
    completed_week: int,
    lines_path: str | Path | None = None,
    confidence_floor: float = CONFIDENCE_FLOOR,
    maximum_stake: float = CONFIDENCE_MAX_STAKE,
) -> pd.DataFrame:
    """Build a moneyline ledger staked by picked-team win probability."""
    base = build_consensus_moneyline_bet_ledger(
        publication_root=publication_root,
        season=season,
        completed_week=completed_week,
        lines_path=lines_path,
        stake=maximum_stake,
    )
    confidence = _consensus_confidence_table(
        publication_root=publication_root,
        season=season,
        completed_week=completed_week,
    )
    merged = base.merge(
        confidence,
        on=["season", "week", "game_id", "strategy"],
        how="left",
        validate="one_to_one",
    )
    return _apply_confidence_staking(
        merged,
        confidence_column="moneyline_confidence",
        result_column="moneyline_result",
        maximum_stake=maximum_stake,
        confidence_floor=confidence_floor,
    )


def summarize_consensus_bankroll(
    ledger: pd.DataFrame, *, result_column: str = "ats_result"
) -> pd.DataFrame:
    """Aggregate a flat-stake ledger to weekly and cumulative bankroll results."""
    graded = ledger.loc[ledger["bet_placed"]].copy()
    graded["win"] = graded[result_column].eq("Win").astype(int)
    graded["loss"] = graded[result_column].eq("Loss").astype(int)
    graded["push"] = graded[result_column].eq("Push").astype(int)
    weekly = (
        graded.groupby(["season", "week", "strategy"], as_index=False, sort=False)
        .agg(
            bets=("game_id", "size"),
            wins=("win", "sum"),
            losses=("loss", "sum"),
            pushes=("push", "sum"),
            weekly_staked=("stake", "sum"),
            weekly_gross_return=("gross_return", "sum"),
            weekly_net_profit=("net_profit", "sum"),
        )
        .sort_values(["strategy", "week"], kind="stable")
    )
    grouped = weekly.groupby("strategy", sort=False)
    weekly["cumulative_bets"] = grouped["bets"].cumsum()
    weekly["cumulative_wins"] = grouped["wins"].cumsum()
    weekly["cumulative_losses"] = grouped["losses"].cumsum()
    weekly["cumulative_pushes"] = grouped["pushes"].cumsum()
    weekly["cumulative_staked"] = grouped["weekly_staked"].cumsum()
    weekly["cumulative_gross_return"] = grouped["weekly_gross_return"].cumsum()
    weekly["cumulative_net_profit"] = grouped["weekly_net_profit"].cumsum()
    weekly["cumulative_roi"] = (
        weekly["cumulative_net_profit"]
        / weekly["cumulative_staked"].replace(0, np.nan)
    )
    return weekly.reset_index(drop=True)


def summarize_consensus_ats_bankroll(ledger: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible ATS-specific summary wrapper."""
    return summarize_consensus_bankroll(ledger, result_column="ats_result")


def build_confidence_threshold_sweep(
    ledger: pd.DataFrame,
    *,
    result_column: str,
    stake: float = THRESHOLD_SWEEP_STAKE,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Backtest flat stakes only when confidence meets a minimum cutoff.

    This is an in-sample descriptive sweep. It deliberately uses a constant
    risk amount so the cutoff effect is not confounded with stake sizing.
    """
    cutoffs = (
        np.round(np.arange(0.50, 1.001, 0.01), 2)
        if thresholds is None
        else np.asarray(thresholds, dtype=float)
    )
    rows: list[dict[str, object]] = []
    for strategy, strategy_frame in ledger.groupby("strategy", sort=False):
        confidence = pd.to_numeric(strategy_frame["confidence"], errors="coerce")
        eligible = strategy_frame["base_bet_eligible"].astype(bool) & confidence.notna()
        for cutoff in cutoffs:
            selected = strategy_frame.loc[eligible & confidence.ge(float(cutoff))]
            results = selected[result_column].astype(str).str.casefold()
            wins = int(results.eq("win").sum())
            losses = int(results.eq("loss").sum())
            pushes = int(results.eq("push").sum())
            profits = [
                settle_ats_bet(result, stake=stake, american_odds=odds)[0]
                for result, odds in zip(results, selected["american_odds"], strict=True)
            ]
            staked = float(len(selected) * stake)
            net = float(np.sum(profits))
            rows.append(
                {
                    "strategy": strategy,
                    "confidence_cutoff": float(cutoff),
                    "bets": len(selected),
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "staked": staked,
                    "net_profit": net,
                    "roi": net / staked if staked else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_profitable_confidence_thresholds(
    sweep: pd.DataFrame, *, min_bets: int = THRESHOLD_SWEEP_MIN_BETS
) -> pd.DataFrame:
    """Identify the first profitable cutoff, with and without a sample floor."""
    rows = []
    for strategy, frame in sweep.groupby("strategy", sort=False):
        profitable = frame.loc[frame["net_profit"].gt(0)].sort_values(
            "confidence_cutoff", kind="stable"
        )
        supported = profitable.loc[profitable["bets"].ge(int(min_bets))]
        unrestricted = profitable.iloc[0] if not profitable.empty else None
        robust = supported.iloc[0] if not supported.empty else None
        rows.append(
            {
                "strategy": strategy,
                "minimum_positive_cutoff": (
                    float(unrestricted["confidence_cutoff"])
                    if unrestricted is not None
                    else np.nan
                ),
                "bets_at_minimum_positive_cutoff": (
                    int(unrestricted["bets"]) if unrestricted is not None else 0
                ),
                "net_at_minimum_positive_cutoff": (
                    float(unrestricted["net_profit"])
                    if unrestricted is not None
                    else np.nan
                ),
                "minimum_positive_cutoff_with_sample_floor": (
                    float(robust["confidence_cutoff"]) if robust is not None else np.nan
                ),
                "sample_floor_bets": int(min_bets),
                "bets_at_supported_cutoff": int(robust["bets"]) if robust is not None else 0,
                "net_at_supported_cutoff": (
                    float(robust["net_profit"]) if robust is not None else np.nan
                ),
                "roi_at_supported_cutoff": (
                    float(robust["roi"]) if robust is not None else np.nan
                ),
                "interpretation": "descriptive_in_sample_not_a_forward_guarantee",
            }
        )
    return pd.DataFrame(rows)


def _plot_confidence_threshold_sweep(
    sweep: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    market_label: str,
    min_bets: int = THRESHOLD_SWEEP_MIN_BETS,
) -> Path:
    apply_tdnet_theme()
    fig, (profit_axis, roi_axis) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    fig.patch.set_facecolor("#F7F4ED")
    for axis in (profit_axis, roi_axis):
        axis.set_facecolor("#FFFFFF")
        axis.spines[["top", "right"]].set_visible(False)
    for strategy, frame in sweep.groupby("strategy", sort=False):
        color = SERIES_COLORS[strategy]
        profit_axis.plot(frame["confidence_cutoff"], frame["net_profit"], lw=2.8, color=color, label=strategy)
        supported = frame["bets"].ge(min_bets)
        roi_axis.plot(
            frame.loc[supported, "confidence_cutoff"], frame.loc[supported, "roi"],
            lw=2.8, color=color, label=strategy,
        )
        first = frame.loc[frame["net_profit"].gt(0) & supported].head(1)
        if not first.empty:
            row = first.iloc[0]
            profit_axis.scatter(row["confidence_cutoff"], row["net_profit"], s=72, color=color, edgecolor="white", zorder=5)
            profit_axis.annotate(
                f"{row['confidence_cutoff']:.0%} (n={int(row['bets'])})",
                (row["confidence_cutoff"], row["net_profit"]), xytext=(5, 8),
                textcoords="offset points", fontsize=9.5, color=color, weight="bold",
            )
    profit_axis.axhline(0, color=TDNET_COLORS["slate"], lw=1.1, ls="--")
    roi_axis.axhline(0, color=TDNET_COLORS["slate"], lw=1.1, ls="--")
    profit_axis.set_ylabel("Net profit at $10 flat risk")
    profit_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    roi_axis.set_ylabel("ROI (shown where n ≥ 10)")
    roi_axis.set_xlabel("Bet only at or above confidence cutoff")
    roi_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    roi_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    profit_axis.legend(frameon=False, loc="best")
    fig.suptitle(
        f"TDNet {season} {market_label} Confidence Threshold Sweep",
        fontsize=22, weight="bold", color=TDNET_COLORS["midnight_gridiron"], y=0.98,
    )
    fig.text(
        0.5, 0.015,
        "Dots mark the lowest profitable cutoff with at least 10 bets. This is an in-sample season-to-date diagnostic, not a validated forward betting rule.",
        ha="center", fontsize=9.5, color=TDNET_COLORS["slate"],
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def _plot_consensus_bankroll(
    weekly: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    title: str,
    subtitle: str,
    footer: str,
) -> Path:
    """Render all three cumulative net-profit tracks in one figure."""
    apply_tdnet_theme()
    fig, axis = plt.subplots(figsize=(16, 9), facecolor="#F7F4ED")
    axis.set_facecolor("#FFFFFF")
    ordered_names = [name for name, _, _ in CONSENSUS_SOURCES]
    line_styles = ("-", "--", "-.")
    markers = ("o", "s", "D")
    label_offsets = (-16, 0, 16)
    maximum_week = int(pd.to_numeric(weekly["week"], errors="raise").max())
    for series_index, strategy in enumerate(ordered_names):
        frame = weekly.loc[weekly["strategy"].eq(strategy)].sort_values("week")
        if frame.empty:
            continue
        x = [-0.35, *frame["week"].astype(float).tolist()]
        y = [0.0, *frame["cumulative_net_profit"].astype(float).tolist()]
        color = SERIES_COLORS[strategy]
        axis.plot(
            x,
            y,
            marker=markers[series_index],
            ls=line_styles[series_index],
            lw=3.2,
            ms=7,
            color=color,
            label=strategy,
        )
        final = frame.iloc[-1]
        axis.annotate(
            f"{float(final['cumulative_net_profit']):+,.2f}  "
            f"({float(final['cumulative_roi']):+.1%})",
            (float(final["week"]), float(final["cumulative_net_profit"])),
            xytext=(9, label_offsets[series_index]),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=11,
            weight="bold",
        )
    axis.axhline(0, color=TDNET_COLORS["slate"], lw=1.2, ls="--")
    axis.set_xlim(-0.5, maximum_week + 0.75)
    axis.set_xticks(range(maximum_week + 1))
    axis.set_xlabel("Completed week")
    axis.set_ylabel("Cumulative net profit")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.3)
    axis.grid(axis="x", alpha=0.12)
    axis.legend(loc="best", frameon=False, fontsize=11)
    axis.set_title(
        f"TDNet {season} {title}\n{subtitle}",
        fontsize=22,
        weight="bold",
        color=TDNET_COLORS["midnight_gridiron"],
        pad=20,
    )
    fig.text(
        0.5,
        0.018,
        footer,
        ha="center",
        fontsize=9.5,
        color=TDNET_COLORS["slate"],
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def plot_consensus_ats_bankroll(
    weekly: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    stake: float = DEFAULT_STAKE,
    fallback_american_odds: float = DEFAULT_ATS_ODDS,
) -> Path:
    """Render all three cumulative ATS net-profit tracks in one figure."""
    return _plot_consensus_bankroll(
        weekly,
        path,
        season=season,
        title="Consensus ATS Bankroll",
        subtitle=f"${stake:g} risked on every graded pick · {fallback_american_odds:+g} ATS price",
        footer=(
            "Net profit includes the full stake on losses, odds-based winnings on wins, and returned stakes on pushes. "
            "CFBD publishes spreads but not spread-side juice; −110 is an explicit assumption and CFBD moneylines are not substituted."
        ),
    )


def plot_consensus_moneyline_bankroll(
    weekly: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    stake: float = DEFAULT_STAKE,
) -> Path:
    """Render all three cumulative moneyline net-profit tracks in one figure."""
    return _plot_consensus_bankroll(
        weekly,
        path,
        season=season,
        title="Consensus Moneyline Bankroll",
        subtitle=f"${stake:g} risked per priced pick · best available CFBD moneyline",
        footer=(
            "Net profit uses the best quoted CFBD moneyline across available providers for each selected team. "
            "Historical weeks without a frozen pregame moneyline file use the available retrospective CFBD snapshot; "
            "games without a quoted price are retained but no bet is placed."
        ),
    )


def write_consensus_betting_artifacts(
    *,
    publication_root: str | Path,
    output_root: str | Path,
    season: int,
    completed_week: int,
    lines_path: str | Path | None = None,
    stake: float = DEFAULT_STAKE,
    fallback_american_odds: float = DEFAULT_ATS_ODDS,
) -> dict[str, Path]:
    """Write auditable ATS and moneyline ledgers, summaries, and unified plots."""
    output = Path(output_root)
    tables = output / "tables"
    figures = output / "figures"
    metadata = output / "metadata"
    for directory in (tables, figures, metadata):
        directory.mkdir(parents=True, exist_ok=True)
    ats_ledger = build_consensus_ats_bet_ledger(
        publication_root=publication_root,
        season=season,
        completed_week=completed_week,
        stake=stake,
        fallback_american_odds=fallback_american_odds,
    )
    ats_weekly = summarize_consensus_ats_bankroll(ats_ledger)
    moneyline_ledger = build_consensus_moneyline_bet_ledger(
        publication_root=publication_root,
        season=season,
        completed_week=completed_week,
        lines_path=lines_path,
        stake=stake,
    )
    moneyline_weekly = summarize_consensus_bankroll(
        moneyline_ledger, result_column="moneyline_result"
    )
    confidence_ats_ledger = build_confidence_scaled_ats_ledger(
        publication_root=publication_root,
        season=season,
        completed_week=completed_week,
        fallback_american_odds=fallback_american_odds,
    )
    confidence_ats_weekly = summarize_consensus_ats_bankroll(confidence_ats_ledger)
    confidence_moneyline_ledger = build_confidence_scaled_moneyline_ledger(
        publication_root=publication_root,
        season=season,
        completed_week=completed_week,
        lines_path=lines_path,
    )
    confidence_moneyline_weekly = summarize_consensus_bankroll(
        confidence_moneyline_ledger, result_column="moneyline_result"
    )
    ats_threshold_sweep = build_confidence_threshold_sweep(
        confidence_ats_ledger, result_column="ats_result", stake=stake
    )
    moneyline_threshold_sweep = build_confidence_threshold_sweep(
        confidence_moneyline_ledger, result_column="moneyline_result", stake=stake
    )
    ats_threshold_summary = summarize_profitable_confidence_thresholds(
        ats_threshold_sweep
    )
    moneyline_threshold_summary = summarize_profitable_confidence_thresholds(
        moneyline_threshold_sweep
    )
    paths = {
        "ats_ledger_csv": tables / "consensus_ats_bankroll_bets.csv",
        "ats_weekly_csv": tables / "consensus_ats_bankroll_weekly.csv",
        "ats_figure_png": figures / "consensus_ats_bankroll.png",
        "moneyline_ledger_csv": tables / "consensus_moneyline_bankroll_bets.csv",
        "moneyline_weekly_csv": tables / "consensus_moneyline_bankroll_weekly.csv",
        "moneyline_figure_png": figures / "consensus_moneyline_bankroll.png",
        "confidence_ats_ledger_csv": tables
        / "consensus_confidence_scaled_ats_bankroll_bets.csv",
        "confidence_ats_weekly_csv": tables
        / "consensus_confidence_scaled_ats_bankroll_weekly.csv",
        "confidence_ats_figure_png": figures
        / "consensus_confidence_scaled_ats_bankroll.png",
        "confidence_moneyline_ledger_csv": tables
        / "consensus_confidence_scaled_moneyline_bankroll_bets.csv",
        "confidence_moneyline_weekly_csv": tables
        / "consensus_confidence_scaled_moneyline_bankroll_weekly.csv",
        "confidence_moneyline_figure_png": figures
        / "consensus_confidence_scaled_moneyline_bankroll.png",
        "ats_threshold_sweep_csv": tables / "consensus_ats_confidence_threshold_sweep.csv",
        "ats_threshold_summary_csv": tables / "consensus_ats_profitable_confidence_thresholds.csv",
        "ats_threshold_figure_png": figures / "consensus_ats_confidence_threshold_sweep.png",
        "moneyline_threshold_sweep_csv": tables / "consensus_moneyline_confidence_threshold_sweep.csv",
        "moneyline_threshold_summary_csv": tables / "consensus_moneyline_profitable_confidence_thresholds.csv",
        "moneyline_threshold_figure_png": figures / "consensus_moneyline_confidence_threshold_sweep.png",
        "methodology_json": metadata / "consensus_betting_methodology.json",
    }
    ats_ledger.to_csv(paths["ats_ledger_csv"], index=False)
    ats_weekly.to_csv(paths["ats_weekly_csv"], index=False)
    moneyline_ledger.to_csv(paths["moneyline_ledger_csv"], index=False)
    moneyline_weekly.to_csv(paths["moneyline_weekly_csv"], index=False)
    confidence_ats_ledger.to_csv(paths["confidence_ats_ledger_csv"], index=False)
    confidence_ats_weekly.to_csv(paths["confidence_ats_weekly_csv"], index=False)
    confidence_moneyline_ledger.to_csv(
        paths["confidence_moneyline_ledger_csv"], index=False
    )
    confidence_moneyline_weekly.to_csv(
        paths["confidence_moneyline_weekly_csv"], index=False
    )
    ats_threshold_sweep.to_csv(paths["ats_threshold_sweep_csv"], index=False)
    ats_threshold_summary.to_csv(paths["ats_threshold_summary_csv"], index=False)
    moneyline_threshold_sweep.to_csv(paths["moneyline_threshold_sweep_csv"], index=False)
    moneyline_threshold_summary.to_csv(paths["moneyline_threshold_summary_csv"], index=False)
    plot_consensus_ats_bankroll(
        ats_weekly,
        paths["ats_figure_png"],
        season=season,
        stake=stake,
        fallback_american_odds=fallback_american_odds,
    )
    plot_consensus_moneyline_bankroll(
        moneyline_weekly,
        paths["moneyline_figure_png"],
        season=season,
        stake=stake,
    )
    _plot_consensus_bankroll(
        confidence_ats_weekly,
        paths["confidence_ats_figure_png"],
        season=season,
        title="Confidence-Scaled ATS Bankroll",
        subtitle="49.9% = USD 0 · linear scaling · 100% = USD 25 · −110 ATS price",
        footer=(
            "ATS confidence is the share of underlying models supporting the published consensus ATS side. "
            "CFBD does not publish spread-side juice, so −110 remains an explicit assumption."
        ),
    )
    _plot_confidence_threshold_sweep(
        ats_threshold_sweep,
        paths["ats_threshold_figure_png"],
        season=season,
        market_label="ATS",
    )
    _plot_confidence_threshold_sweep(
        moneyline_threshold_sweep,
        paths["moneyline_threshold_figure_png"],
        season=season,
        market_label="Moneyline",
    )
    _plot_consensus_bankroll(
        confidence_moneyline_weekly,
        paths["confidence_moneyline_figure_png"],
        season=season,
        title="Confidence-Scaled Moneyline Bankroll",
        subtitle="49.9% = USD 0 · linear scaling · 100% = USD 25 · best available CFBD price",
        footer=(
            "Moneyline confidence is the published probability of the selected winner. Stakes scale linearly to USD 25; "
            "unpriced games and confidence at or below 49.9% receive no wager."
        ),
    )
    method = {
        "schema": "tdnet-consensus-betting-bankroll-v1",
        "season": int(season),
        "completed_week": int(completed_week),
        "strategies": [name for name, _, _ in CONSENSUS_SOURCES],
        "stake_per_graded_pick_usd": float(stake),
        "ats_american_odds": float(fallback_american_odds),
        "odds_source": "assumed_standard_price",
        "cfbd_limitation": (
            "CFBD betting-line records provide spread, opening spread, total, and "
            "moneylines, but no price for either side of the point spread."
        ),
        "settlement": {
            "win": "stake returned plus stake * 100 / abs(odds) for negative odds",
            "loss": "entire stake lost",
            "push": "stake returned; zero net profit",
        },
        "missing_line_policy": "No ATS bet is placed when the frozen spread or grade is missing.",
        "moneyline_policy": "CFBD moneyline odds are not used as ATS odds.",
        "moneyline_odds_source": "best available quoted price across CFBD providers",
        "moneyline_timing_policy": (
            "Use the frozen pregame CFBD moneyline snapshot when present; otherwise label "
            "the quote as retrospective and do not claim it was available at prediction time."
        ),
        "moneyline_missing_price_policy": "No moneyline bet is placed when CFBD has no quoted price for the selected team.",
        "confidence_staking": {
            "formula": "stake = clip(25 * (confidence - 0.499) / (1 - 0.499), 0, 25)",
            "confidence_floor": CONFIDENCE_FLOOR,
            "maximum_stake_usd": CONFIDENCE_MAX_STAKE,
            "ats_confidence": "share of underlying models supporting the published consensus ATS side",
            "moneyline_confidence": "published win probability assigned to the selected team",
        },
        "confidence_threshold_sweep": {
            "stake_usd": float(stake),
            "cutoffs": "0.50 through 1.00 in 0.01 increments",
            "selection_rule": "bet only when confidence >= cutoff",
            "supported_result_minimum_bets": THRESHOLD_SWEEP_MIN_BETS,
            "warning": "in-sample season-to-date diagnostic; selecting a cutoff on these outcomes is not forward validation",
        },
    }
    paths["methodology_json"].write_text(
        json.dumps(method, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def write_consensus_ats_bankroll_artifacts(**kwargs) -> dict[str, Path]:
    """Compatibility wrapper for callers created before moneyline tracking."""
    return write_consensus_betting_artifacts(**kwargs)
