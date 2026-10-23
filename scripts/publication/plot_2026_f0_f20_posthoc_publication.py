#!/usr/bin/env python3
"""Publication views for the F0–F20 post-hoc scorecard on matched games."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

matplotlib.rcParams["svg.hashsalt"] = "tdnet-2026-f0-f20-posthoc"


ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "publication/2026/research/scientific_f0_f20_posthoc_scorecards_week01_07"
MARKET = ROOT / "publication/2026/week_07/post_game/retrospective_replay/cumulative/scientific_f0_f19/common_cohort_vegas_games.csv"
F20 = ROOT / "data/private/f20/replay_2026_week01_07_v2/f20_replay.parquet"
ARCHITECTURES = ("M1", "M2", "M3", "M4", "M5", "M10")
FINGERPRINTS = tuple(f"F{i}" for i in range(17)) + ("F17-market", "F18", "F19", "F20")
METRICS = {
    "brier": ("Brier score", "lower is better", "%.3f", "viridis_r"),
    "su_accuracy": ("Straight-up accuracy", "higher is better", "%.1f%%", "viridis"),
    "log_loss": ("Log loss", "lower is better", "%.3f", "viridis_r"),
    "margin_mae": ("Margin MAE · points", "lower is better", "%.1f", "viridis_r"),
    "ats_accuracy": ("ATS accuracy · pushes excluded", "higher is better", "%.1f%%", "viridis"),
    "upset_recall": ("Upset recall", "higher is better", "%.1f%%", "viridis"),
    "upset_call_accuracy": ("Upset-call accuracy", "higher is better", "%.1f%%", "viridis"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    paired = pd.read_csv(SCORES / "f20_paired_304_game_scorecard.csv")
    latest = pd.read_csv(SCORES / "latest_scorecard.csv")
    weekly = pd.read_csv(SCORES / "f20_paired_weekly_scorecard.csv")
    if (len(paired) != 126 or paired.duplicated(["fingerprint", "model_id"]).any()
            or set(zip(paired.fingerprint, paired.model_id)) != {
                (f, m) for f in FINGERPRINTS for m in ARCHITECTURES
            }):
        raise ValueError("Matched scorecard is not the complete 126-cell roster")
    if len(weekly) != 756 or set(weekly.week) != set(range(1, 7)):
        raise ValueError("Matched weekly scorecard is incomplete")
    if paired.games.nunique() != 1 or int(paired.games.iloc[0]) != 304:
        raise ValueError("Model cells are not on the same 304 games")
    if (len(latest) != 126 or latest.duplicated(["fingerprint", "model_id"]).any()
            or set(zip(latest.fingerprint, latest.model_id)) != set(zip(paired.fingerprint, paired.model_id))
            or not latest.loc[latest.fingerprint.ne("F20"), "games"].eq(380).all()
            or not latest.loc[latest.fingerprint.eq("F20"), "games"].eq(320).all()):
        raise ValueError("Latest-available scorecard is not the complete split-cohort roster")
    if paired.loc[paired.fingerprint.eq("F20"), ["margin_mae", "ats_accuracy"]].isna().any().any():
        raise ValueError("F20 margin and ATS must be scored on the paired games")
    f20 = pd.read_parquet(F20)
    market = pd.read_csv(MARKET)
    shared = market.loc[market.game_id.isin(f20.game_id) & market.market_home_probability.notna()].copy()
    if len(shared) != 304 or shared.market_home_spread.isna().any():
        raise ValueError("Market baseline is not on the 304 matched games")
    y = shared.actual_home_margin.gt(0)
    p = shared.market_home_probability.clip(1e-6, 1 - 1e-6)
    baseline = {
        "brier": float(shared.brier_error.mean()),
        "su_accuracy": float(shared.winner_correct.astype(float).mean()),
        "log_loss": float(-(y * np.log(p) + ~y * np.log1p(-p)).mean()),
        "margin_mae": float(shared.margin_error.mean()),
    }
    return paired, latest, weekly, baseline


def save(fig: plt.Figure, stem: str, publication: bool) -> dict[str, str]:
    suffix = "_publication" if publication else ""
    names = {}
    for ext in ("png", "svg"):
        path = SCORES / f"{stem}{suffix}.{ext}"
        kwargs = {"dpi": 300 if publication else 200, "bbox_inches": "tight",
                  "facecolor": fig.get_facecolor()}
        if ext == "svg":
            kwargs["metadata"] = {"Date": "2026-10-23"}
        fig.savefig(path, **kwargs)
        names[path.name] = sha256(path)
    plt.close(fig)
    return names


def display_value(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if metric in ("su_accuracy", "ats_accuracy", "upset_recall", "upset_call_accuracy"):
        return f"{100 * value:.1f}"
    if metric == "margin_mae":
        return f"{value:.1f}"
    return f"{value:.3f}"


def heatmap(frame: pd.DataFrame, metric: str, baseline: dict[str, float], publication: bool,
            cohort: str) -> plt.Figure:
    title, direction, _, cmap_name = METRICS[metric]
    grid = frame.pivot(index="fingerprint", columns="model_id", values=metric).loc[
        list(FINGERPRINTS), list(ARCHITECTURES)]
    values = grid.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    min_present = 110 if metric == "upset_call_accuracy" else 120
    if finite.size < min_present:
        raise ValueError(f"{metric} has missing F0–F19 paired values")
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#e9e8e4")
    fig, ax = plt.subplots(figsize=(11, 13.8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap,
                   norm=Normalize(vmin=float(finite.min()), vmax=float(finite.max())))
    ax.set_xticks(np.arange(6), ARCHITECTURES, fontsize=11, fontweight="bold")
    ax.set_yticks(np.arange(21), [f.replace("-market", " market") for f in FINGERPRINTS], fontsize=10)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 21, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(21):
        for j in range(6):
            value = values[i, j]
            if np.isnan(value):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="#656565")
                continue
            tone = (value - finite.min()) / max(float(finite.max() - finite.min()), 1e-12)
            # Viridis_r reverses the light/dark location of the readable text threshold.
            dark = tone < .55 if cmap_name == "viridis" else tone > .45
            ax.text(j, i, display_value(metric, value), ha="center", va="center",
                    fontsize=8.5, fontweight="normal",
                    color="white" if dark else "#111827")
    bar = fig.colorbar(im, ax=ax, fraction=.027, pad=.025)
    bar.ax.set_ylabel(f"{title} · {direction}", rotation=270, labelpad=22, fontsize=10)
    ax.set_ylabel("Fingerprint", fontsize=11, labelpad=10)
    if cohort == "paired":
        cohort_title = "304 paired games, Weeks 1–6"
        cohort_note = "All 126 cells on the same 304 games"
    else:
        cohort_title = "latest available season results"
        cohort_note = "F0–F19: 380 games, Weeks 1–7 · F20: 320 games, Weeks 1–6 · different cohorts"
    if not publication:
        fig.suptitle(f"F0–F20 scientific models · {title}", fontsize=17, fontweight="bold", y=.99)
        baseline_text = (f" · Vegas {display_value(metric, baseline[metric])}"
                         if cohort == "paired" and metric in baseline else "")
        fig.text(.5, .966, cohort_note + baseline_text,
                 ha="center", fontsize=10, color="#4b5563")
        fig.text(.5, .018, "Retrospective scores · expert source and market quote times unverified",
                 ha="center", fontsize=8.8, color="#4b5563")
    else:
        ax.set_title(f"F0–F20 · {title} · {cohort_title}", fontsize=14, pad=17)
        fig.text(.5, .022, cohort_note + " · retrospective",
                 ha="center", fontsize=8.5, color="#4b5563")
    fig.subplots_adjust(left=.14, right=.88, top=.94 if not publication else .95, bottom=.065)
    return fig


def weekly_plot(weekly: pd.DataFrame, metric: str, publication: bool) -> plt.Figure:
    title, direction, _, _ = METRICS[metric]
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    values = weekly.pivot(index="week", columns=["fingerprint", "model_id"], values=metric)
    if len(values.columns) != 126:
        raise ValueError("Expected all 126 F×M cells in matched weekly view")
    weeks = values.index.to_numpy()
    q1 = values.quantile(.25, axis=1).to_numpy(dtype=float)
    q3 = values.quantile(.75, axis=1).to_numpy(dtype=float)
    med = values.median(axis=1).to_numpy(dtype=float)
    scale = 100 if metric in ("su_accuracy", "ats_accuracy", "upset_recall", "upset_call_accuracy") else 1
    ax.fill_between(weeks, scale * q1, scale * q3, color="#d8e4ef", label="126-cell quartiles")
    ax.plot(weeks, scale * med, color="#45586c", linewidth=2.4, label="126-cell median")
    offsets = np.linspace(-.22, .22, len(values.columns))
    for index, week in enumerate(weeks):
        points = values.loc[week].to_numpy(dtype=float)
        finite = np.isfinite(points)
        ax.scatter(week + offsets[finite], scale * points[finite], s=15,
                   color="#0072b2", alpha=.28, linewidths=0,
                   label="Individual cells" if index == 0 else None, zorder=3)
    ax.set_xticks(range(1, 7))
    ax.set_xlim(.8, 6.2)
    ax.set_xlabel("CFBD week")
    ax.set_ylabel(title + (" · %" if scale == 100 else ""))
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncol=2, loc="best", fontsize=9, frameon=False)
    if publication:
        ax.set_title(f"F0–F20 weekly {title} · matched Weeks 1–6 games", fontsize=14, pad=13)
    else:
        fig.suptitle(f"Weekly {title} · {direction}", fontsize=17, fontweight="bold")
        fig.text(.5, .935, "All 126 F×M cells use the same matched games in each week.",
                 ha="center", fontsize=10, color="#4b5563")
    fig.text(.5, .015, "Retrospective scores · F20 has no CFBD Week 7 expert archive",
             ha="center", fontsize=8.5, color="#4b5563")
    fig.tight_layout(rect=(0, .04, 1, .9 if not publication else .96))
    return fig


def main() -> None:
    paired, latest, weekly, baseline = load()
    outputs = {}
    for metric in METRICS:
        for publication in (False, True):
            outputs.update(save(heatmap(paired, metric, baseline, publication, "paired"), f"paired_304_{metric}_matrix", publication))
            outputs.update(save(heatmap(latest, metric, baseline, publication, "latest"), f"latest_available_{metric}_matrix", publication))
            outputs.update(save(weekly_plot(weekly, metric, publication), f"paired_weekly_{metric}", publication))
    sources = (SCORES / "f20_paired_304_game_scorecard.csv", SCORES / "latest_scorecard.csv",
               SCORES / "f20_paired_weekly_scorecard.csv",
               MARKET, F20)
    manifest = {
        "scope": "2026 F0–F20 publication views on 304 shared games through CFBD Week 6",
        "model_cells": 126, "paired_games": 304, "source_timing_verified": False,
        "f20_margin_and_ats": "trained_margin_regressors_on_2017_2024",
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "outputs": outputs,
    }
    (SCORES / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print({"figures": 21, "publication_versions": 21, "files": len(outputs)})


if __name__ == "__main__":
    main()
