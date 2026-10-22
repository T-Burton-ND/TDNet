#!/usr/bin/env python3
"""Plot planned HPS search curves and M5 epoch losses from metrics-only ledgers."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCHITECTURES = ("M1", "M2", "M3", "M4", "M5", "M10")
FINGERPRINTS = tuple(f"F{i}" for i in range(20))
YEARS = tuple(range(2021, 2026))


def _axes():
    fig, axes = plt.subplots(5, 4, figsize=(17, 18), sharex=True)
    return fig, axes.flatten()


def plot_search(ledger: pd.DataFrame, output: Path) -> None:
    required = {"fingerprint", "architecture", "trial_id", "test_season", "margin_mae"}
    if not required.issubset(ledger):
        raise ValueError(f"Missing fold-metric columns: {sorted(required - set(ledger))}")
    if ledger.duplicated(list(required - {"margin_mae"})).any():
        raise ValueError("Duplicate cell/trial/fold result")
    if not np.isfinite(ledger.margin_mae.to_numpy(float)).all():
        raise ValueError("Nonfinite MAE in completed-trial ledger")
    expected = pd.MultiIndex.from_product([FINGERPRINTS, ARCHITECTURES, range(100), YEARS],
                                          names=["fingerprint", "architecture", "trial_id", "test_season"])
    observed = pd.MultiIndex.from_frame(ledger[["fingerprint", "architecture", "trial_id", "test_season"]])
    if len(ledger) != len(expected) or not observed.is_unique or len(expected.difference(observed)):
        raise ValueError("The 60,000 planned fold results are not complete")
    output.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for architecture in ARCHITECTURES:
        fig, axes = _axes()
        for ax, fingerprint in zip(axes, FINGERPRINTS):
            group = ledger.loc[ledger.fingerprint.eq(fingerprint) & ledger.architecture.eq(architecture)]
            table = group.pivot(index="trial_id", columns="test_season", values="margin_mae").sort_index()
            values = table.to_numpy(float)
            means = values.mean(axis=1)
            chosen = np.empty(100, dtype=int)
            best = 0
            for i in range(100):
                if (means[i], values[i].std(), i) < (means[best], values[best].std(), best):
                    best = i
                chosen[i] = best
            selected = values[chosen]
            x = np.arange(1, 101)
            for budget, trial, scores in zip(x, chosen, selected):
                summary_rows.append({"fingerprint": fingerprint, "architecture": architecture,
                                     "candidate_budget": int(budget), "selected_trial_id": int(trial),
                                     "mean_mae": float(scores.mean()), "q25_mae": float(np.quantile(scores, .25)),
                                     "q75_mae": float(np.quantile(scores, .75)), "n_folds": 5})
            ax.plot(x, selected.mean(axis=1), color="#225D76", linewidth=1.8)
            ax.fill_between(x, np.quantile(selected, .25, axis=1), np.quantile(selected, .75, axis=1),
                            color="#225D76", alpha=.22, linewidth=0)
            ax.set_title(fingerprint, fontsize=12)
            ax.grid(alpha=.15)
        fig.supxlabel("Candidate configurations evaluated")
        fig.supylabel("Five-fold validation MAE (points)")
        fig.suptitle(f"{architecture}: best mean validation MAE as search grows", fontsize=17)
        fig.text(.5, .015, "Line: selected candidate mean across 2021–25 folds. Shading: fold interquartile range. Selection curve is optimistic, not a held-out estimate.", ha="center", fontsize=9)
        fig.tight_layout(rect=(.02, .04, 1, .97))
        fig.savefig(output / f"{architecture.lower()}_search_progress.png", dpi=170)
        plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(output / "search_progress.csv", index=False)


def plot_m5_epochs(history: pd.DataFrame, ledger: pd.DataFrame, output: Path) -> None:
    required = {"fingerprint", "architecture", "trial_id", "test_season", "epoch", "train_loss", "validation_loss"}
    if not required.issubset(history):
        raise ValueError(f"Missing epoch-history columns: {sorted(required - set(history))}")
    history = history.loc[history.architecture.eq("M5")].copy()
    if history.empty:
        raise ValueError("No M5 epoch histories")
    if history.duplicated(["fingerprint", "trial_id", "test_season", "epoch"]).any():
        raise ValueError("Duplicate M5 epoch history row")
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = _axes()
    summary_rows = []
    for ax, fingerprint in zip(axes, FINGERPRINTS):
        group = history.loc[history.fingerprint.eq(fingerprint)]
        if group.empty:
            raise ValueError(f"Missing M5 histories for {fingerprint}")
        scores = ledger.loc[ledger.fingerprint.eq(fingerprint) & ledger.architecture.eq("M5")]
        trial_scores = scores.groupby("trial_id").margin_mae.agg(["mean", "std"])
        chosen = trial_scores.sort_values(["mean", "std"], kind="stable").index[0]
        group = group.loc[group.trial_id.eq(chosen)]
        if set(group.test_season) != set(YEARS):
            raise ValueError(f"Missing selected M5 fold histories for {fingerprint}")
        for column, color, label in (("train_loss", "#7893A0", "Training"),
                                     ("validation_loss", "#225D76", "Validation")):
            stats = group.groupby("epoch")[column].agg(["mean", "count"])
            quartiles = group.groupby("epoch")[column].quantile([.25, .75]).unstack()
            for epoch, row in stats.iterrows():
                summary_rows.append({"fingerprint": fingerprint, "selected_trial_id": int(chosen),
                                     "epoch": int(epoch), "split": label.lower(),
                                     "mean_loss": float(row["mean"]), "q25_loss": float(quartiles.loc[epoch, .25]),
                                     "q75_loss": float(quartiles.loc[epoch, .75]), "n_folds": int(row["count"])})
            ax.plot(stats.index, stats["mean"], color=color, label=label, linewidth=1.5)
            ax.fill_between(stats.index.to_numpy(), quartiles[.25].to_numpy(),
                            quartiles[.75].to_numpy(), color=color, alpha=.18, linewidth=0)
        ax.set_title(fingerprint, fontsize=12)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.supxlabel("Training epoch")
    fig.supylabel("Model training loss")
    fig.suptitle("M5: mean training and validation loss by fingerprint", fontsize=17)
    fig.text(.5, .015, "One MAE-selected configuration per cell. Line: available-fold mean; shading: fold interquartile range. Early stopping changes later-epoch counts.", ha="center", fontsize=9)
    fig.tight_layout(rect=(.02, .04, 1, .97))
    fig.savefig(output / "m5_epoch_loss.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(output / "m5_epoch_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-metrics", type=Path, required=True)
    parser.add_argument("--epoch-history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = pd.read_csv(args.fold_metrics)
    plot_search(ledger, args.output)
    if args.epoch_history:
        plot_m5_epochs(pd.read_csv(args.epoch_history), ledger, args.output)


if __name__ == "__main__":
    main()
