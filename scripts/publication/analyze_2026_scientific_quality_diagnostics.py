"""Game-level diagnostics beyond MAE, Brier, and winner accuracy.

All model-versus-market values use paired games. The default Week 7 snapshot is
retrospective; later cohorts need their own market-source timing audit.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    for extension in ("png", "svg"):
        path = OUTPUT / f"{stem}.{extension}"
        options = {"metadata": {"Date": style.RENDER_DATE}} if extension == "svg" else {}
        fig.savefig(path, dpi=200, facecolor=style.PAPER, **options)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def plot_reliability(selected: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fig, ax = style.figure("Probability reliability", f"Ten equal-count bins per forecaster, {style.COHORT['moneyline_games']} shared games. Dots show observed home-win frequency.")
    ax.plot([0, 1], [0, 1], color=style.MUTED, lw=1.2, alpha=.6)
    colors = {"Vegas": style.GOLD, "F19/M4": style.PINK,
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
        ax.plot(x, y, marker="o", ms=7, lw=2, color=colors[label], label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean forecast home-win probability",
           ylabel="Observed home-win frequency")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "17_probability_reliability")
    return pd.DataFrame(rows)


def plot_risk_coverage(selected: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fig, ax = style.figure("When confidence chooses the games", "Coverage is the fraction of most-confident picks retained. Higher accuracy is better.")
    colors = {"Vegas": style.GOLD, "F19/M4": style.PINK,
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
    manifest = {
        "source": {path.name: style.sha256(path) for path in (MODEL_SOURCE, MARKET_SOURCE, CONSENSUS_SOURCE)},
        "source_directory": style.source_label(),
        "render_date": style.RENDER_DATE,
        "market_note": style.MARKET_NOTE,
        "cohort": {**style.COHORT, "model_points": 120,
                   "market_points": 1, "close_games": close_games,
                   "strong_favorite_games": strong_games},
        "method": {"log_loss_clip": PROBABILITY_CLIP,
                   "calibration_error": f"ten equal-count probability bins of {bin_min}–{bin_max} games; game ID breaks probability ties",
                   "risk_coverage": "mean cumulative 0/1 error over all confidence-ranked coverages; game ID breaks confidence ties",
                   "blend": "arithmetic mean of model and Vegas home-win probabilities, fixed at 50/50"},
        "outputs": {path.name: style.sha256(path) for path in sorted(OUTPUT.iterdir())
                    if path.name.startswith(tuple(f"{i:02d}_" for i in range(10, 19)))
                    or path.name in ("diagnostic_quality.csv", "diagnostic_model_game_loss.parquet",
                                     "probability_reliability_bins.csv", "confidence_coverage_checkpoints.csv")},
    }
    (OUTPUT / "diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote 9 diagnostic figures and {len(summary)} score rows to {OUTPUT}")


if __name__ == "__main__":
    main()
