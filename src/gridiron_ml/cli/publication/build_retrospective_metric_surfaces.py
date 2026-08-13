"""Build model-family by fingerprint metric surfaces for the 2025 holdout.

These figures are retrospective descriptive summaries. They do not update the
2026 frozen roster and do not use prospective 2026 outcomes.
"""

from __future__ import annotations
from gridiron_ml.cli._paths import project_root

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = project_root()


METRICS = [
    ("winner_accuracy", "Winner accuracy", "Higher is better", "viridis"),
    ("brier_score", "Brier score", "Lower is better", "magma_r"),
    ("margin_mae", "Margin MAE", "Lower is better", "magma_r"),
    ("chalk_recall", "Chalk recall", "Higher is better", "viridis"),
    ("upset_recall", "Upset recall", "Higher is better", "viridis"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "outputs/publication/canonical_2025/canonical_2025_margin_fingerprint_model_results.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "docs/publication_2026/figures/retrospective_2025",
    )
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    required = {"model_family", "fingerprint_id", *(m[0] for m in METRICS)}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise SystemExit(f"Missing required metric columns: {missing}")

    table = results.groupby(["model_family", "fingerprint_id"], as_index=False)[
        [m[0] for m in METRICS]
    ].mean(numeric_only=True)
    families = sorted(table["model_family"].unique())
    fingerprints = sorted(table["fingerprint_id"].unique(), key=lambda x: int(str(x).lstrip("F")))
    indexed = table.set_index(["model_family", "fingerprint_id"])

    args.output_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.ravel()
    image_handles = []
    for ax, (column, title, direction, cmap) in zip(axes, METRICS):
        values = pd.DataFrame(
            [[indexed.loc[(family, fingerprint), column] for fingerprint in fingerprints] for family in families],
            index=families,
            columns=fingerprints,
        )
        image = ax.imshow(values.to_numpy(dtype=float), aspect="auto", cmap=cmap)
        image_handles.append(image)
        ax.set_title(f"{title}\n{direction}", fontsize=11)
        ax.set_xticks(range(len(fingerprints)), fingerprints)
        ax.set_yticks(range(len(families)), families)
        ax.set_xlabel("Fingerprint version")
        ax.set_ylabel("Model family")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                ax.text(col, row, f"{values.iat[row, col]:.3f}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    axes[-1].axis("off")
    fig.suptitle("TDNet 2025 retrospective model × fingerprint metric surfaces", fontsize=15)
    fig.text(0.5, 0.01, "Each cell is the 2025 holdout mean for one model family and fingerprint version; descriptive only.", ha="center", fontsize=9)
    stem = "figure_retrospective_2025_metric_surfaces"
    fig.savefig(args.output_root / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_root / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

    long = table.melt(["model_family", "fingerprint_id"], var_name="metric", value_name="value")
    long.to_csv(args.output_root / f"{stem}.csv", index=False)
    print(f"Wrote {stem} to {args.output_root}")


if __name__ == "__main__":
    main()
