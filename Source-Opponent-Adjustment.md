---
type: source-summary
up: "[[Fingerprint-and-Model-Decision-History]]"
source: ../docs/publication_2026/OPPONENT_ADJUSTMENT_COMPARISON.md
tags: [source, methods, model-selection]
---

# Opponent Adjustment Source

The opponent-adjustment comparison records seven method families and selects v1.4 Elo context using development-only evidence under the protocol’s ban on an averaged primary adjustment.

## Contribution

The source separates an already-consumed 2025 holdout table from the development selection used for the operational primary method.

## Methods and findings

The 2025 holdout uses train seasons 2010–2023, validation 2024, and test 2025. It cannot be reused to select the 2026 primary method. The matched development comparison trained on 2010–2021, validated on 2022, evaluated on 2023–2024, and excluded 2025 and 2026.

On the development comparison (train 2010–2021, validate 2022, evaluate 2023–2024), v1.7 ensemble average had raw MAE 12.9699, but an averaged primary adjustment was prohibited. Eligible non-averaged v1.4 Elo context was selected with MAE 12.9802, RMSE 16.4442, and winner accuracy 0.7043 over 1,488 games in two seasons. v1.4 MAE was 0.0103 points higher than v1.7, which remained supplemental. The separate 2025 holdout table reports v1.4 MAE 14.3523 and v1.7 MAE 14.5017; those consumed-test metrics are not selection evidence.

## Where it connects

The selection history is summarized in [Fingerprint and Model Decision History](Fingerprint-and-Model-Decision-History), alongside the distinct corrected-F6 decision. Do not use the 2025 holdout figures as new 2026 selection criteria.

## Source link

Source file: `../docs/publication_2026/OPPONENT_ADJUSTMENT_COMPARISON.md`.

## See also

[Fingerprint and Model Decision History](Fingerprint-and-Model-Decision-History) · [Deferred and Abandoned Directions](Deferred-and-Abandoned-Directions) · [TDNet Overview](TDNet-Overview)
