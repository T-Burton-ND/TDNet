"""Game-level diagnostics beyond MAE, Brier, and winner accuracy.

All model-versus-market values use paired games. The default Week 7 snapshot is
retrospective; later cohorts need their own market-source timing audit.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import plot_2026_scientific_total_quality as style


ROOT = style.ROOT
SOURCE = style.SOURCE
OUTPUT = style.OUTPUT
MODEL_SOURCE = SOURCE / "common_120_model_game_results.parquet"
MARKET_SOURCE = SOURCE / "common_cohort_vegas_games.csv"
CONSENSUS_SOURCE = SOURCE / "common_120_consensus_game_results.csv"
PROBABILITY_CLIP = 1e-6


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    positive = int(y.sum())
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        raise ValueError("AUC requires both outcomes")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))


def _ece(y: np.ndarray, p: np.ndarray, game_id: np.ndarray,
         bins: int = 10) -> tuple[float, int]:
    """Equal-count expected calibration error, descriptive and bin-sensitive."""
    order = np.lexsort((game_id, p))
    groups = np.array_split(order, bins)
    terms = [len(group) / len(p) * abs(float(p[group].mean() - y[group].mean()))
             for group in groups]
    return float(sum(terms)), min(len(group) for group in groups)


def _selective(y: np.ndarray, p: np.ndarray, game_id: np.ndarray) -> tuple[float, float, np.ndarray]:
    confidence = np.maximum(p, 1 - p)
    correct = (p >= .5) == y
    order = np.lexsort((game_id, -confidence))
    errors = (~correct[order]).astype(float)
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    top_count = int(np.ceil(len(errors) * .25))
    return float(risk.mean()), float(1 - errors[:top_count].mean()), risk


def _probability_stats(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    frame = frame.sort_values("game_id")
    y = frame.actual_home_margin.gt(0).to_numpy(dtype=int)
    p = frame[column].to_numpy(dtype=float)
    game_id = frame.game_id.to_numpy()
    ece, minimum_bin = _ece(y, p, game_id)
    aurc, top_quartile_accuracy, _ = _selective(y, p, game_id)
    confident = np.maximum(p, 1 - p) >= .9
    return {
        "log_loss": float(_log_loss(y, p).mean()),
        "roc_auc": _auc(y, p),
        "ece_10_equal_count": ece,
        "ece_min_bin_games": minimum_bin,
        "mean_probability_bias_home": float((p - y).mean()),
        "mean_confidence_distance_from_half": float(np.abs(p - .5).mean()),
        "selective_error_aurc": aurc,
        "top_quartile_confidence_accuracy": top_quartile_accuracy,
        "top_quartile_games": int(np.ceil(len(p) * .25)),
        "high_confidence_games": int(confident.sum()),
        "high_confidence_misses": int(((p >= .5) != y)[confident].sum()),
    }


def _margin_stats(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    actual = frame.actual_home_margin.to_numpy(dtype=float)
    predicted = frame[prediction].to_numpy(dtype=float)
    errors = np.abs(predicted - actual)
    tail_count = int(np.ceil(len(errors) * .1))
    return {
        "median_absolute_margin_error": float(np.median(errors)),
        "p90_absolute_margin_error": float(np.quantile(errors, .9)),
        "worst_decile_mean_absolute_margin_error": float(np.sort(errors)[-tail_count:].mean()),
        "signed_home_margin_bias": float((predicted - actual).mean()),
    }


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = pd.read_parquet(MODEL_SOURCE)
    market = pd.read_csv(MARKET_SOURCE)
    consensus = pd.read_csv(CONSENSUS_SOURCE)
    if market.empty or len(model) != 120 * len(market) or len(consensus) != len(market):
        raise ValueError("Expected one market/consensus row and 120 model rows per game")
    if market.game_id.duplicated().any() or consensus.game_id.duplicated().any():
        raise ValueError("Duplicate market or consensus game IDs")
    if model.duplicated(["fingerprint", "model_id", "game_id"]).any():
        raise ValueError("Duplicate model/game prediction")
    if set(consensus.game_id) != set(market.game_id):
        raise ValueError("Consensus and market game IDs differ")
    if not np.allclose(consensus.actual_home_margin,
                       consensus.game_id.map(market.set_index("game_id").actual_home_margin)):
        raise ValueError("Consensus and market final margins disagree")
    joined = model.merge(
        market[["game_id", "market_home_spread", "market_home_probability"]],
        on="game_id", how="inner", validate="many_to_one"
    )
    if len(joined) != len(model):
        raise ValueError("Model-to-market game join is incomplete")
    if not np.allclose(joined.actual_home_margin,
                       joined.game_id.map(market.set_index("game_id").actual_home_margin)):
        raise ValueError("Model and market final margins disagree")
    spread = joined.loc[joined.market_home_spread.notna()].copy()
    probability = joined.loc[joined.market_home_probability.notna()].copy()
    market_spread = market.dropna(subset=["market_home_spread"]).copy()
    market_probability = market.dropna(subset=["market_home_probability"]).copy()
    if market_spread.empty or len(market_probability) < 10:
        raise ValueError("Need spread games and at least ten moneyline games for calibration bins")
    if spread.groupby(["fingerprint", "model_id"]).size().ne(len(market_spread)).any():
        raise ValueError("Incomplete model spread cohort")
    if probability.groupby(["fingerprint", "model_id"]).size().ne(len(market_probability)).any():
        raise ValueError("Incomplete model probability cohort")
    if not np.allclose(market_spread.margin_error, (
        market_spread.actual_home_margin + market_spread.market_home_spread
    ).abs()):
        raise ValueError("Archived market margin error is inconsistent")
    style.set_cohort(len(market_spread), len(market_probability), int(market.week.max()))
    return spread, probability, market_spread, market_probability


def analyze() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    spread, probability, market_spread, market_probability = load()
    spread_games = len(market_spread)
    probability_games = len(market_probability)
    market_spread["vegas_home_margin"] = -market_spread.market_home_spread
    vegas_margin = _margin_stats(market_spread, "vegas_home_margin")
    vegas_probability = _probability_stats(market_probability, "market_home_probability")
    market_y = market_probability.actual_home_margin.gt(0).to_numpy(dtype=float)
    market_p = market_probability.market_home_probability.to_numpy(dtype=float)
    market_brier = float(np.mean((market_p - market_y) ** 2))
    market_log_loss = vegas_probability["log_loss"]
    close = market_probability.market_home_probability.between(.4, .6)
    strong = market_probability.market_home_probability.le(.25) | market_probability.market_home_probability.ge(.75)
    close_games, strong_games = int(close.sum()), int(strong.sum())
    vegas = {
        "label": "Vegas", "fingerprint": "Vegas", "model_id": "Vegas", "kind": "market",
        "spread_games": spread_games, "probability_games": probability_games,
        **vegas_margin, **vegas_probability,
        "paired_margin_win_share_vs_vegas": .5,
        "paired_margin_advantage_points": 0.0,
        "weeks_margin_ahead_of_vegas": 0,
        "weeks_log_loss_ahead_of_vegas": 0,
        "close_brier_advantage": 0.0,
        "strong_brier_advantage": 0.0,
        "close_games": close_games,
        "strong_games": strong_games,
        "half_blend_brier_gain_vs_vegas": 0.0,
        "log_loss_gap_vs_vegas": 0.0,
    }
    rows = [vegas]
    game_rows = []
    market_margin_error = market_spread.set_index("game_id").margin_error
    for (fingerprint, model_id), margin in spread.groupby(["fingerprint", "model_id"], sort=True):
        probability_group = probability.loc[
            probability.fingerprint.eq(fingerprint) & probability.model_id.eq(model_id)
        ].copy().sort_values("game_id")
        margin = margin.sort_values("game_id").copy()
        market_error = margin.game_id.map(market_margin_error).to_numpy(dtype=float)
        model_error = np.abs(margin.pred_home_margin.to_numpy(dtype=float) - margin.actual_home_margin.to_numpy(dtype=float))
        lower = model_error < market_error - 1e-10
        tied = np.abs(model_error - market_error) <= 1e-10
        gamewise_share = float((lower.sum() + .5 * tied.sum()) / len(margin))
        p = probability_group.pred_home_win_probability.to_numpy(dtype=float)
        y = probability_group.actual_home_margin.gt(0).to_numpy(dtype=float)
        market_p = probability_group.market_home_probability.to_numpy(dtype=float)
        market_loss = _log_loss(y, market_p)
        model_loss = _log_loss(y, p)
        market_brier_values = (market_p - y) ** 2
        model_brier_values = (p - y) ** 2
        blend_brier = ((p + market_p) / 2 - y) ** 2
        close_group = (market_p >= .4) & (market_p <= .6)
        strong_group = (market_p <= .25) | (market_p >= .75)
        if close_group.sum() != close_games or strong_group.sum() != strong_games:
            raise ValueError("Model probability regimes differ from market")
        weekly_margin = margin.assign(
            market_error=market_error, model_error=model_error
        ).groupby("week")[["market_error", "model_error"]].mean()
        weekly_loss = probability_group.assign(
            market_log_loss=market_loss, model_log_loss=model_loss
        ).groupby("week")[["market_log_loss", "model_log_loss"]].mean()
        label = f"{fingerprint}/{model_id}"
        rows.append({
            "label": label, "fingerprint": fingerprint, "model_id": model_id, "kind": "model",
            "spread_games": len(margin), "probability_games": len(probability_group),
            **_margin_stats(margin, "pred_home_margin"),
            **_probability_stats(probability_group, "pred_home_win_probability"),
            "paired_margin_win_share_vs_vegas": gamewise_share,
            "paired_margin_advantage_points": float((market_error - model_error).mean()),
            "weeks_margin_ahead_of_vegas": int((weekly_margin.model_error < weekly_margin.market_error).sum()),
            "weeks_log_loss_ahead_of_vegas": int((weekly_loss.model_log_loss < weekly_loss.market_log_loss).sum()),
            "close_brier_advantage": float((market_brier_values[close_group] - model_brier_values[close_group]).mean()),
            "strong_brier_advantage": float((market_brier_values[strong_group] - model_brier_values[strong_group]).mean()),
            "close_games": int(close_group.sum()), "strong_games": int(strong_group.sum()),
            "half_blend_brier_gain_vs_vegas": float((market_brier_values - blend_brier).mean()),
            "log_loss_gap_vs_vegas": float((model_loss - market_loss).mean()),
        })
        game_rows.append(pd.DataFrame({
            "label": label, "game_id": probability_group.game_id.to_numpy(),
            "week": probability_group.week.to_numpy(),
            "market_probability": market_p, "model_probability": p,
            "home_won": y.astype(int),
            "market_log_loss": market_loss, "model_log_loss": model_loss,
            "market_brier": market_brier_values, "model_brier": model_brier_values,
            "half_blend_brier": blend_brier,
            "regime": np.where(close_group, "close", np.where(strong_group, "strong", "middle")),
        }))
    summary = pd.DataFrame(rows)
    expected_labels = {
        f"{'F17-market' if i == 17 else f'F{i}'}/{model_id}"
        for i in range(20) for model_id in style.MODELS
    }
    if (len(summary) != 121 or summary.label.duplicated().any()
            or set(summary.loc[summary.kind.eq("model"), "label"]) != expected_labels):
        raise ValueError("Expected 120 model rows plus Vegas")
    if not np.isclose(summary.loc[summary.kind.eq("market"), "log_loss"].iloc[0], market_log_loss):
        raise ValueError("Market log-loss mismatch")
    if not np.isclose(market_brier, market_probability.brier_error.mean()):
        raise ValueError("Archived market Brier is inconsistent")
    game = pd.concat(game_rows, ignore_index=True)
    if len(game) != 120 * probability_games:
        raise ValueError("Incomplete model probability game ledger")
    selected = {"Vegas": market_probability.copy()}
    selected["Vegas"]["probability"] = selected["Vegas"].market_home_probability
    for label in ("F19/M4", "F19/M1"):
        fingerprint, model_id = label.split("/")
        selected[label] = probability.loc[
            probability.fingerprint.eq(fingerprint) & probability.model_id.eq(model_id)
        ].copy()
        selected[label]["probability"] = selected[label].pred_home_win_probability
    consensus = pd.read_csv(CONSENSUS_SOURCE).merge(
        market_probability[["game_id"]], on="game_id", validate="one_to_one"
    )
    if len(consensus) != probability_games:
        raise ValueError("Consensus profile game cohort incomplete")
    consensus["probability"] = consensus.pred_home_win_probability
    selected["120-model consensus"] = consensus
    return summary, game, selected


def _scatter(frame: pd.DataFrame, title: str, subtitle: str, x: str, y: str,
             xlabel: str, ylabel: str, stem: str, reference_x: float | None = None,
             reference_y: float | None = None) -> str:
    fig, ax = style.figure(title, subtitle)
    style.plot_points(ax, frame, x, y, size=76)
    if reference_x is not None:
        ax.axvline(reference_x, color=style.GOLD, alpha=.45, lw=1.3, zorder=1)
    if reference_y is not None:
        ax.axhline(reference_y, color=style.GOLD, alpha=.45, lw=1.3, zorder=1)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    style.legend(ax)
    _save(fig, stem)
    return stem


def _save(fig, stem: str) -> None:
    style.save_pair(fig, stem)
    style.publicationize(fig, stem)
    style.save_pair(fig, stem + "_publication", dpi=300)
    plt.close(fig)


def _architecture_legend(ax, baseline: str, *, distribution=False) -> None:
    handles = [Line2D([0], [0], marker=style.MARKERS[model_id], linestyle="",
                      markerfacecolor=style.COLORS[model_id], markeredgecolor="white",
                      markersize=9, color="none", label=f"{model_id} · {style.NAMES[model_id]}")
               for model_id in style.MODELS]
    handles.append(Line2D([0], [0], color=style.GOLD, lw=2.3, label=baseline))
    if distribution:
        handles.extend([
            Line2D([0], [0], color=style.INK, lw=4.3, label="Interquartile range"),
            Line2D([0], [0], marker="|", linestyle="", color=style.INK,
                   markersize=14, markeredgewidth=2.4, label="Median"),
        ])
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False, fontsize=9, labelspacing=.9)


def _model_scatter(ax, frame: pd.DataFrame, x: str, y: str) -> None:
    for model_id in style.MODELS:
        group = frame.loc[frame.model_id.eq(model_id)]
        ax.scatter(group[x], group[y], marker=style.MARKERS[model_id],
                   color=style.COLORS[model_id], s=76, alpha=.85,
                   edgecolor="white", linewidth=.7, zorder=4)


def plot_reliability(selected: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fig, ax = style.figure("Probability reliability", f"Ten equal-count bins per forecaster, {style.COHORT['moneyline_games']} shared games. Dots show observed home-win frequency.")
    ax.plot([0, 1], [0, 1], color=style.MUTED, lw=1.2, alpha=.6)
    colors = {"Vegas": style.GOLD, "F19/M4": style.COLORS["M4"],
              "F19/M1": style.COLORS["M1"], "120-model consensus": style.MUTED}
    rows = []
    for label, frame in selected.items():
        frame = frame.sort_values(["probability", "game_id"])
        bins = np.array_split(np.arange(len(frame)), 10)
        x, y = [], []
        for index, part in enumerate(bins, start=1):
            group = frame.iloc[part]
            mean_p = float(group.probability.mean())
            observed = float(group.actual_home_margin.gt(0).mean())
            x.append(mean_p)
            y.append(observed)
            rows.append({"label": label, "bin": index, "games": len(group),
                         "mean_forecast_probability": mean_p, "observed_home_win_rate": observed})
        marker = style.MARKERS["M4"] if label == "F19/M4" else "o"
        ax.plot(x, y, marker=marker, ms=7, lw=2, color=colors[label], label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean forecast home-win probability",
           ylabel="Observed home-win frequency")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "17_probability_reliability")
    return pd.DataFrame(rows)


def plot_risk_coverage(selected: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fig, ax = style.figure("When confidence chooses the games", "Coverage is the fraction of most-confident picks retained. Higher accuracy is better.")
    colors = {"Vegas": style.GOLD, "F19/M4": style.COLORS["M4"],
              "F19/M1": style.COLORS["M1"], "120-model consensus": style.MUTED}
    rows = []
    lowest_plotted_accuracy = 1.0
    for label, frame in selected.items():
        frame = frame.sort_values("game_id")
        y = frame.actual_home_margin.gt(0).to_numpy(dtype=int)
        p = frame.probability.to_numpy(dtype=float)
        _, _, risk = _selective(y, p, frame.game_id.to_numpy())
        coverage = np.arange(1, len(frame) + 1) / len(frame)
        mask = coverage >= .05
        lowest_plotted_accuracy = min(lowest_plotted_accuracy, float((1 - risk[mask]).min()))
        ax.plot(coverage[mask], 1 - risk[mask], lw=2.2, color=colors[label], label=label)
        for fraction in (.25, .5, 1.0):
            index = int(np.ceil(len(frame) * fraction)) - 1
            rows.append({"label": label, "coverage": fraction, "games": index + 1,
                         "accuracy": float(1 - risk[index])})
    ax.set(xlim=(.05, 1), ylim=(.65, 1.0), xlabel="Coverage · most-confident games retained",
           ylabel="Winner accuracy among retained games")
    if SOURCE != style.DEFAULT_SOURCE:
        ax.set_ylim(max(0, lowest_plotted_accuracy - .03), 1.0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend(loc="lower left", frameon=False)
    _save(fig, "18_confidence_risk_coverage")
    return pd.DataFrame(rows)


def plot_weekly_brier_distribution(game: pd.DataFrame) -> pd.DataFrame:
    weekly = game.assign(brier_gain=game.market_brier - game.model_brier).groupby(
        ["label", "week"], as_index=False
    ).agg(brier_gain=("brier_gain", "mean"), games=("game_id", "size"))
    weekly["model_id"] = weekly.label.str.rsplit("/", n=1).str[-1]
    weeks = sorted(weekly.week.unique())
    if len(weekly) != 120 * len(weeks):
        raise ValueError("Weekly Brier distribution needs every model in each week")
    fig, ax = style.figure("Weekwise probability skill", "Each row shows all 120 related cells on the same moneyline games; thick bar is the interquartile range.", (16, 10))
    for position, week in enumerate(weeks):
        group = weekly.loc[weekly.week.eq(week)].sort_values("label").reset_index(drop=True)
        offsets = (((np.arange(len(group)) * 37) % len(group)) / (len(group) - 1) - .5) * .42
        for model_id in style.MODELS:
            mask = group.model_id.eq(model_id).to_numpy()
            ax.scatter(group.loc[mask, "brier_gain"], position + offsets[mask],
                       marker=style.MARKERS[model_id], color=style.COLORS[model_id],
                       s=38, alpha=.65, edgecolor="white", linewidth=.45, zorder=4)
        q1, median, q3 = group.brier_gain.quantile([.25, .5, .75])
        ax.plot([q1, q3], [position, position], color=style.INK, lw=4.3,
                solid_capstyle="round", zorder=6)
        ax.scatter([median], [position], marker="|", color=style.PANEL,
                   s=260, linewidth=2.4, zorder=7)
    style.vegas_line(ax)
    game_counts = game.groupby("week").game_id.nunique()
    ax.set_yticks(range(len(weeks)), [f"Week {week} · {game_counts.loc[week]} games" for week in weeks])
    ax.set_ylim(len(weeks) - .45, -.55)
    ax.set(xlabel="Mean Brier advantage versus Vegas ↑", ylabel="Moneyline-covered week")
    _architecture_legend(ax, "Vegas · zero advantage", distribution=True)
    _save(fig, "19_weekly_brier_distribution")
    return weekly


def plot_probability_duel(game: pd.DataFrame) -> pd.DataFrame:
    scored = game.assign(
        gain=game.market_brier - game.model_brier,
        lower=(game.model_brier < game.market_brier - 1e-12).astype(float),
        tie=np.isclose(game.model_brier, game.market_brier, atol=1e-12, rtol=0).astype(float),
    )
    duel = scored.groupby("label", as_index=False).agg(
        games=("game_id", "size"), mean_brier_gain=("gain", "mean"),
        lower_games=("lower", "sum"), tie_games=("tie", "sum"),
    )
    duel["gamewise_brier_win_share"] = (duel.lower_games + .5 * duel.tie_games) / duel.games
    duel["model_id"] = duel.label.str.rsplit("/", n=1).str[-1]
    if len(duel) != 120 or duel.games.nunique() != 1:
        raise ValueError("Probability duel needs 120 paired model rows")
    fig, ax = style.figure("Probability loss: frequency versus magnitude", "A model can win more games yet lose on mean Brier if a few misses are large.")
    _model_scatter(ax, duel, "gamewise_brier_win_share", "mean_brier_gain")
    ax.axvline(.5, color=style.GOLD, lw=2.3, alpha=.92, zorder=2)
    style.vegas_line(ax, horizontal=True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set(xlabel="Share of games with lower Brier loss than Vegas ↑",
           ylabel="Mean Brier advantage versus Vegas ↑")
    _architecture_legend(ax, "Vegas parity lines")
    _save(fig, "20_probability_gamewise_duel")
    return duel


def plot_brier_tail_risk(game: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in game.groupby("label"):
        losses = group.model_brier.to_numpy(dtype=float)
        tail = max(1, int(np.ceil(len(losses) * .1)))
        rows.append({"label": label, "model_id": label.rsplit("/", 1)[-1],
                     "games": len(losses), "mean_brier": float(losses.mean()),
                     "worst_decile_mean_brier": float(np.sort(losses)[-tail:].mean()),
                     "tail_games": tail})
    tail = pd.DataFrame(rows)
    market = game.drop_duplicates("game_id")
    market_losses = market.market_brier.to_numpy(dtype=float)
    market_tail_count = max(1, int(np.ceil(len(market_losses) * .1)))
    market_mean = float(market_losses.mean())
    market_tail = float(np.sort(market_losses)[-market_tail_count:].mean())
    if len(tail) != 120 or tail.games.nunique() != 1:
        raise ValueError("Brier tail-risk map needs 120 paired model rows")
    tail = pd.concat([tail, pd.DataFrame([{"label": "Vegas", "model_id": "Vegas",
        "games": len(market_losses), "mean_brier": market_mean,
        "worst_decile_mean_brier": market_tail, "tail_games": market_tail_count}])],
        ignore_index=True)
    fig, ax = style.figure("Average probability error versus worst misses", "Each tail is that forecaster's own worst 10% of games; lower on both axes is better.")
    _model_scatter(ax, tail.loc[tail.model_id.ne("Vegas")], "mean_brier", "worst_decile_mean_brier")
    ax.axvline(market_mean, color=style.GOLD, lw=2.3, alpha=.92, zorder=2)
    ax.axhline(market_tail, color=style.GOLD, lw=2.3, alpha=.92, zorder=2)
    ax.set(xlabel="Mean Brier loss ↓", ylabel="Worst-decile mean Brier loss ↓")
    _architecture_legend(ax, "Vegas thresholds")
    _save(fig, "21_brier_tail_risk")
    return tail


def _market_bin_heatmap(grouped: pd.DataFrame, counts: pd.Series, *, value: str,
                        title: str, subtitle: str, colorbar_label: str,
                        stem: str, reference: float = 0.0,
                        scale: float = 1.0, decimals: int = 1,
                        upper_endpoint: float | None = None,
                        lower_endpoint: float | None = None,
                        brier_ceiling: bool = False,
                        draw_reference: bool = True,
                        cell_count_value: str | None = None) -> None:
    grid = grouped.pivot(index="model_id", columns="bin_index", values=value)
    values = grid.reindex(index=style.MODELS, columns=range(10)).to_numpy(dtype=float) * scale
    if not np.isfinite(values).any():
        raise ValueError(f"{stem} needs at least one populated bin")
    finite = values[np.isfinite(values)]
    if brier_ceiling:
        # Keep zero at the blue endpoint; values above zero use an over-range
        # marker rather than silently changing the requested benchmark scale.
        floor = min(-.5, float(finite.min()))
        norm, cmap = Normalize(vmin=floor, vmax=0), "RdBu"
    elif upper_endpoint is not None:
        floor = (lower_endpoint if lower_endpoint is not None else
                 min(float(finite.min()), reference - (upper_endpoint - reference)))
        norm, cmap = TwoSlopeNorm(vmin=floor, vcenter=reference,
                                 vmax=upper_endpoint), "RdBu"
    else:
        spread = max(.5 if scale == 1000 else .01,
                     float(np.max(np.abs(finite - reference))))
        norm, cmap = TwoSlopeNorm(vmin=reference - spread, vcenter=reference,
                                 vmax=reference + spread), "RdBu"
    fig, ax = style.figure(title, subtitle, (16, 9.5))
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_over("#062b55")
    image = ax.imshow(values, aspect="auto", cmap=color_map, norm=norm)
    cell_counts = None
    if cell_count_value is not None:
        cell_counts = grouped.pivot(index="model_id", columns="bin_index",
                                    values=cell_count_value).reindex(
                                        index=style.MODELS, columns=range(10)).to_numpy()
    ax.set_yticks(range(len(style.MODELS)),
                  [f"{model_id} · {style.NAMES[model_id]}" for model_id in style.MODELS])
    ax.set_xticks(range(10), [f"{i*10}–{(i+1)*10}%\n(n={counts.loc[i]})" for i in range(10)])
    ax.set(xlabel="Vegas home-win probability bin", ylabel="Model architecture")
    ax.grid(False)
    for row in range(len(style.MODELS)):
        for col in range(10):
            value_at_cell = values[row, col]
            if np.isfinite(value_at_cell):
                red, green, blue, _ = image.cmap(image.norm(value_at_cell))
                luminance = .2126 * red + .7152 * green + .0722 * blue
                annotation = (f"{value_at_cell:+.{decimals}f}" if reference == 0
                              else f"{value_at_cell:.{decimals}f}")
                if cell_counts is not None:
                    annotation += f"\n({int(cell_counts[row, col])} calls)"
                ax.text(col, row, annotation,
                        ha="center", va="center", color=style.INK if luminance > .55 else "white",
                        fontsize=9 if cell_counts is not None else 10, weight="bold")
    extend = "max" if float(finite.max()) > float(norm.vmax) else "neither"
    colorbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.025,
                            label=colorbar_label, extend=extend)
    ticks = [tick for tick in colorbar.get_ticks()
             if norm.vmin <= tick <= norm.vmax]
    colorbar.set_ticks(sorted(set(ticks + [float(norm.vmax)])))
    if draw_reference:
        colorbar.ax.axhline(reference, color=style.GOLD, lw=2.3, alpha=.92, clip_on=False)
    fig.subplots_adjust(left=.15, right=.83, bottom=.19)
    _save(fig, stem)


def plot_market_probability_bins(game: pd.DataFrame) -> pd.DataFrame:
    scored = game.assign(
        bin_index=np.minimum((game.market_probability.to_numpy() * 10).astype(int), 9),
        brier_gain=game.market_brier - game.model_brier,
        model_id=game.label.str.rsplit("/", n=1).str[-1],
    )
    counts = scored.drop_duplicates("game_id").groupby("bin_index").size().reindex(range(10), fill_value=0)
    grouped = scored.groupby(["model_id", "bin_index"], as_index=False).agg(
        mean_brier_gain=("brier_gain", "mean"), model_game_rows=("game_id", "size")
    )
    record = grouped.assign(games=grouped.bin_index.map(counts),
                            lower_probability=grouped.bin_index / 10,
                            upper_probability=(grouped.bin_index + 1) / 10)
    _market_bin_heatmap(
        grouped, counts, value="mean_brier_gain",
        title="Brier advantage by market probability",
        subtitle="Cells average 20 related fingerprint models per architecture; columns are fixed Vegas home-win probability bins.",
        colorbar_label="Brier advantage versus Vegas · ×1000 ↑",
        stem="22_market_probability_bins", scale=1000, brier_ceiling=True)
    return record


def plot_other_market_probability_bins() -> dict[str, pd.DataFrame]:
    _, probability, market_spread, _ = load()
    shared = probability.loc[probability.market_home_spread.notna()].copy()
    market = market_spread.set_index("game_id")
    game_count = shared.game_id.nunique()
    if shared.market_home_probability.isna().any():
        raise ValueError("Shared market-bin cohort has missing moneyline probabilities")
    shared["bin_index"] = np.minimum((shared.market_home_probability * 10).astype(int), 9)
    shared["mae_advantage"] = shared.game_id.map(market.margin_error) - (
        shared.pred_home_margin - shared.actual_home_margin).abs()
    shared["winner_advantage_pp"] = 100 * (
        shared.winner_correct.astype(float) -
        shared.game_id.map(market.winner_correct).astype(float))
    # Use the project's existing ATS grading: a zero model edge selects away.
    cover = shared.actual_home_margin + shared.market_home_spread
    model_edge = shared.pred_home_margin + shared.market_home_spread
    shared["ats_push"] = cover.eq(0)
    shared["ats_correct"] = (model_edge.gt(0) == cover.gt(0)) & ~shared.ats_push
    counts = shared.drop_duplicates("game_id").groupby("bin_index").size().reindex(range(10), fill_value=0)
    resolved = shared.loc[~shared.ats_push]
    resolved_counts = resolved.drop_duplicates("game_id").groupby("bin_index").size().reindex(range(10), fill_value=0)
    if shared.groupby(["fingerprint", "model_id"]).game_id.nunique().ne(game_count).any():
        raise ValueError("Incomplete paired heatmap cohort")
    if resolved.groupby(["fingerprint", "model_id"]).game_id.nunique().ne(resolved.game_id.nunique()).any():
        raise ValueError("Incomplete ATS non-push cohort")
    output = {}
    specs = (
        ("mae", shared, "mae_advantage", "mean_mae_advantage_points",
         "MAE advantage by market probability",
         "Vegas absolute margin error minus model absolute margin error; 20 fingerprints per architecture.",
         "MAE advantage versus Vegas · points ↑", "24_market_probability_bin_mae_advantage", counts),
        ("winner", shared, "winner_advantage_pp", "mean_winner_advantage_pp",
         "Winner accuracy by market probability",
         "Model minus Vegas winner accuracy; percentage-point difference across 20 fingerprints per architecture.",
         "Winner accuracy advantage · percentage points ↑", "25_market_probability_bin_winner_accuracy", counts),
        ("ats", resolved, "ats_correct", "ats_accuracy_pct",
         "ATS accuracy by market probability",
         "Correct side against the spread; pushes excluded. Gold is 50%, not a Vegas ATS pick.",
         "ATS accuracy · % of non-push picks ↑", "26_market_probability_bin_ats_accuracy", resolved_counts),
    )
    for key, frame, source, result, title, subtitle, label, stem, bin_counts in specs:
        grouped = frame.groupby(["model_id", "bin_index"], as_index=False).agg(
            **{result: (source, "mean")}, model_game_rows=("game_id", "size"))
        if key == "ats":
            grouped[result] *= 100
        record = grouped.assign(games=grouped.bin_index.map(bin_counts),
                                lower_probability=grouped.bin_index / 10,
                                upper_probability=(grouped.bin_index + 1) / 10)
        if key == "ats":
            record["push_games"] = record.bin_index.map(counts - resolved_counts)
        _market_bin_heatmap(grouped, bin_counts, value=result, title=title,
                            subtitle=subtitle, colorbar_label=label, stem=stem,
                            reference=50.0 if key == "ats" else 0.0,
                            upper_endpoint=1.0 if key in ("mae", "winner") else None,
                            decimals=1)
        output[key] = record
    return output


def plot_market_probability_upsets() -> pd.DataFrame:
    _, probability, _, _ = load()
    shared = probability.loc[probability.market_home_spread.notna() &
                             probability.market_home_spread.ne(0)].copy()
    if shared.empty:
        raise ValueError("Upset heatmaps need nonzero market spreads")
    shared["bin_index"] = np.minimum((shared.market_home_probability * 10).astype(int), 9)
    favorite_home = shared.market_home_spread.lt(0)
    actual_home = shared.actual_home_margin.gt(0)
    picked_home = shared.pred_home_win_probability.ge(.5)
    shared["actual_upset"] = actual_home.ne(favorite_home)
    shared["upset_call"] = picked_home.ne(favorite_home)
    shared["correct_upset_call"] = shared.actual_upset & shared.upset_call
    unique = shared.drop_duplicates("game_id")
    counts = unique.groupby("bin_index").size().reindex(range(10), fill_value=0)
    upset_counts = unique.groupby("bin_index").actual_upset.sum().reindex(range(10), fill_value=0)
    grouped = shared.groupby(["model_id", "bin_index"], as_index=False).agg(
        model_game_rows=("game_id", "size"),
        actual_upset_model_rows=("actual_upset", "sum"),
        upset_calls=("upset_call", "sum"),
        correct_upset_calls=("correct_upset_call", "sum"))
    grouped["upset_recall_pct"] = 100 * grouped.correct_upset_calls.div(
        grouped.actual_upset_model_rows.replace(0, np.nan))
    grouped["upset_call_accuracy_pct"] = 100 * grouped.correct_upset_calls.div(
        grouped.upset_calls.replace(0, np.nan))
    record = grouped.assign(games=grouped.bin_index.map(counts),
                            actual_upset_games=grouped.bin_index.map(upset_counts),
                            lower_probability=grouped.bin_index / 10,
                            upper_probability=(grouped.bin_index + 1) / 10)
    _market_bin_heatmap(
        grouped, upset_counts, value="upset_recall_pct",
        title="Upset recall by market probability",
        subtitle="Correctly called underdog wins ÷ actual underdog wins; column n counts actual upset games.",
        colorbar_label="Upset recall · % of actual upsets ↑",
        stem="27_market_probability_bin_upset_recall", reference=50,
        upper_endpoint=100, lower_endpoint=0, draw_reference=False)
    _market_bin_heatmap(
        grouped, counts, value="upset_call_accuracy_pct",
        title="Accuracy of upset calls by market probability",
        subtitle="Correct upset calls ÷ underdog-win calls made; cells show their call counts.",
        colorbar_label="Correct upset calls · % of calls made ↑",
        stem="28_market_probability_bin_upset_call_accuracy", reference=50,
        upper_endpoint=100, lower_endpoint=0, draw_reference=False,
        cell_count_value="upset_calls")
    return record


def plot_weekly_margin_stability() -> tuple[pd.DataFrame, pd.DataFrame]:
    spread, _, market_spread, _ = load()
    market_error = market_spread.set_index("game_id").margin_error
    scored = spread.assign(
        label=spread.fingerprint + "/" + spread.model_id,
        margin_advantage=spread.game_id.map(market_error) -
        (spread.pred_home_margin - spread.actual_home_margin).abs(),
    )
    weekly = scored.groupby(["label", "model_id", "week"], as_index=False).agg(
        mean_margin_advantage=("margin_advantage", "mean"),
        games=("game_id", "size"),
    )
    stability = scored.groupby(["label", "model_id"], as_index=False).agg(
        mean_margin_advantage=("margin_advantage", "mean"), games=("game_id", "size"),
    )
    variation = weekly.groupby("label").mean_margin_advantage.agg(
        weekly_sd="std", weeks="count").reset_index()
    stability = stability.merge(variation, on="label", validate="one_to_one")
    if len(stability) != 120 or stability.weeks.min() < 2:
        raise ValueError("Margin stability needs 120 models with at least two scored weeks")
    fig, ax = style.figure("Mean margin edge versus weekly variation", "Each point is one cell; weekly SD is descriptive across schedule blocks, not a confidence interval.")
    _model_scatter(ax, stability, "mean_margin_advantage", "weekly_sd")
    style.vegas_line(ax)
    ax.set(xlabel="Mean MAE advantage versus Vegas · points ↑",
           ylabel="SD of weekly MAE advantage · points ↓")
    _architecture_legend(ax, "Vegas · zero mean advantage")
    _save(fig, "23_margin_weekly_stability")
    return weekly, stability


def main() -> None:
    style.configure(style.argument_parser().parse_args())
    global SOURCE, OUTPUT, MODEL_SOURCE, MARKET_SOURCE, CONSENSUS_SOURCE
    SOURCE, OUTPUT = style.SOURCE, style.OUTPUT
    MODEL_SOURCE = SOURCE / "common_120_model_game_results.parquet"
    MARKET_SOURCE = SOURCE / "common_cohort_vegas_games.csv"
    CONSENSUS_SOURCE = SOURCE / "common_120_consensus_game_results.csv"
    plt.style.use(ROOT / "docs/style/color_palettes/gridiron_light.mplstyle")
    plt.rcParams.update({"figure.autolayout": False,
                         "figure.facecolor": style.PAPER, "savefig.facecolor": style.PAPER,
                         "axes.facecolor": style.PANEL, "font.family": "DejaVu Sans",
                         "svg.hashsalt": "tdnet-2026-scientific-quality"})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary, game, selected = analyze()
    probability_games = style.COHORT["moneyline_games"]
    close_games = int(summary.loc[summary.kind.eq("market"), "close_games"].iloc[0])
    strong_games = int(summary.loc[summary.kind.eq("market"), "strong_games"].iloc[0])
    bin_min = probability_games // 10
    bin_max = int(np.ceil(probability_games / 10))
    top_count = int(np.ceil(probability_games * .25))
    summary.to_csv(OUTPUT / "diagnostic_quality.csv", index=False)
    game.to_parquet(OUTPUT / "diagnostic_model_game_loss.parquet", index=False)
    plot_specs = [
        ("Probability discrimination and penalty", "AUC tests ranking; log loss penalizes confident errors. Left and up are better.",
         "log_loss", "roc_auc", "Log loss ↓", "ROC AUC ↑", "10_probability_discrimination", None, None),
        ("Calibration versus sharpness", f"Calibration error uses ten equal-count bins ({bin_min}–{bin_max} games each); the binning choice still matters.",
         "ece_10_equal_count", "mean_confidence_distance_from_half", "Binned calibration error ↓", "Mean |p − 0.5| · sharpness ↑", "11_calibration_sharpness", None, None),
        ("Can confidence identify the safer picks?", f"AURC averages error as coverage rises; top-quartile accuracy uses the {top_count} most-confident games.",
         "selective_error_aurc", "top_quartile_confidence_accuracy", "Area under risk–coverage curve ↓", "Accuracy on top 25% confidence ↑", "12_selective_prediction", None, None),
        ("Typical miss versus costly miss", "Median absolute margin error versus mean error in each forecaster's worst 10% of games.",
         "median_absolute_margin_error", "worst_decile_mean_absolute_margin_error", "Median absolute margin error · points ↓", "Worst-decile mean absolute error · points ↓", "13_margin_tail_risk", None, None),
        ("How often is a model's margin closer than Vegas?", "One point per model: share of games with smaller absolute margin error versus the mean margin edge.",
         "paired_margin_win_share_vs_vegas", "paired_margin_advantage_points", "Gamewise margin-error win share vs Vegas ↑", "Mean margin-error advantage · points ↑", "14_gamewise_market_duel", .5, 0.0),
        ("Does the market regime change the answer?", f"Brier advantage versus Vegas in {close_games} near-tossups and {strong_games} strong-favorite games. Positive is better.",
         "close_brier_advantage", "strong_brier_advantage", "Brier advantage · close games ↑", "Brier advantage · strong-favorite games ↑", "15_market_regimes", 0.0, 0.0),
        ("Does blending add anything to Vegas?", "A prespecified 50/50 probability blend; positive vertical gain improves Vegas Brier. No weight tuning.",
         "log_loss_gap_vs_vegas", "half_blend_brier_gain_vs_vegas", "Model log-loss gap versus Vegas ↓", "50/50 blend Brier gain versus Vegas ↑", "16_market_blend_probe", 0.0, 0.0),
    ]
    for args in plot_specs:
        _scatter(summary, *args)
    plot_reliability(selected).to_csv(OUTPUT / "probability_reliability_bins.csv", index=False)
    plot_risk_coverage(selected).to_csv(OUTPUT / "confidence_coverage_checkpoints.csv", index=False)
    plot_weekly_brier_distribution(game).to_csv(OUTPUT / "weekly_brier_gain.csv", index=False)
    plot_probability_duel(game).to_csv(OUTPUT / "probability_gamewise_duel.csv", index=False)
    plot_brier_tail_risk(game).to_csv(OUTPUT / "brier_tail_risk.csv", index=False)
    plot_market_probability_bins(game).to_csv(OUTPUT / "market_probability_bin_brier_gain.csv", index=False)
    other_bins = plot_other_market_probability_bins()
    other_bins["mae"].to_csv(OUTPUT / "market_probability_bin_mae_advantage.csv", index=False)
    other_bins["winner"].to_csv(OUTPUT / "market_probability_bin_winner_accuracy.csv", index=False)
    other_bins["ats"].to_csv(OUTPUT / "market_probability_bin_ats_accuracy.csv", index=False)
    upset_bins = plot_market_probability_upsets()
    upset_bins.to_csv(OUTPUT / "market_probability_bin_upsets.csv", index=False)
    weekly_margin, margin_stability = plot_weekly_margin_stability()
    weekly_margin.to_csv(OUTPUT / "weekly_margin_advantage.csv", index=False)
    margin_stability.to_csv(OUTPUT / "margin_weekly_stability.csv", index=False)
    manifest = {
        "source": {path.name: style.sha256(path) for path in (MODEL_SOURCE, MARKET_SOURCE, CONSENSUS_SOURCE)},
        "source_directory": style.source_label(),
        "render_date": style.RENDER_DATE,
        "market_note": style.MARKET_NOTE,
        "cohort": {**style.COHORT, "model_points": 120,
                   "market_points": 1, "close_games": close_games,
                   "strong_favorite_games": strong_games,
                   "market_bin_games": int(other_bins["mae"].games.sum() / len(style.MODELS)),
                   "ats_nonpush_market_bin_games": int(other_bins["ats"].games.sum() / len(style.MODELS)),
                   "ats_push_market_bin_games": int(other_bins["ats"].push_games.sum() / len(style.MODELS)),
                   "market_bin_actual_upset_games": int(upset_bins.actual_upset_games.sum() / len(style.MODELS)),
                   "market_bin_upset_calls": int(upset_bins.upset_calls.sum())},
        "method": {"log_loss_clip": PROBABILITY_CLIP,
                   "calibration_error": f"ten equal-count probability bins of {bin_min}–{bin_max} games; game ID breaks probability ties",
                   "risk_coverage": "mean cumulative 0/1 error over all confidence-ranked coverages; game ID breaks confidence ties",
                   "blend": "arithmetic mean of model and Vegas home-win probabilities, fixed at 50/50",
                   "market_bin_ats": "model picks home iff predicted home margin plus Vegas home spread > 0; actual pushes excluded; 50% is a reference, not a Vegas pick",
                   "market_bin_upsets": "nonzero spread defines the favorite; a model calls an upset iff predicted winner differs from that favorite; recall denominator is actual upsets, call-accuracy denominator is upset calls made"},
        "outputs": {path.name: style.sha256(path) for path in sorted(OUTPUT.iterdir())
                    if path.name.startswith(tuple(f"{i:02d}_" for i in range(10, 29)))
                    or path.name in ("diagnostic_quality.csv", "diagnostic_model_game_loss.parquet",
                                     "probability_reliability_bins.csv", "confidence_coverage_checkpoints.csv",
                                     "weekly_brier_gain.csv", "probability_gamewise_duel.csv",
                                     "brier_tail_risk.csv", "market_probability_bin_brier_gain.csv",
                                     "weekly_margin_advantage.csv", "margin_weekly_stability.csv",
                                     "market_probability_bin_mae_advantage.csv",
                                     "market_probability_bin_winner_accuracy.csv",
                                     "market_probability_bin_ats_accuracy.csv",
                                     "market_probability_bin_upsets.csv")},
    }
    (OUTPUT / "diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote 19 diagnostic figures with publication variants and {len(summary)} score rows to {OUTPUT}")


if __name__ == "__main__":
    main()
