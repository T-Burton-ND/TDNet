# TDNet fingerprint-engineering confirmatory protocol

The machine-readable source of truth is
[`configs/publication/confirmatory_protocol.yaml`](../../configs/publication/confirmatory_protocol.yaml).
This document states the publication-facing interpretation.

## Scientific scope

The paper studies how nested, time-available football information interacts
with established model families and predictive performance. Margin prediction
is primary; winner metrics are secondary. Mixture-of-Experts, dynamic routing,
learned stacking weights, transformers, graph neural networks, and winner-only
operational models are outside the 2026 prospective pipeline.

2025 is consumed retrospective evidence. It is not an untouched test set.
2026 is the prospective season. F6 is the richest market-free TDNet
representation. F7 contains only declared market information, and F8 combines
F6 with F7 to measure incremental market value. F7/F8 are research comparators
and never enter official TDNet predictions, consensus, or polls.

## Canonical fingerprint ladder

| Tier | Meaning | Market status |
|---|---|---|
| F0 | minimal strength baseline: preseason roster talent and games played; no observed performance statistics | market-free |
| F1 | F0 plus raw box-score families | market-free |
| F2 | F1 plus efficiency/rate families | market-free |
| F3 | F2 plus the selected opponent-adjustment family | market-free |
| F4 | F3 plus situational, returning-production, and coaching context | market-free |
| F5 | F4 plus temporal dynamics | market-free |
| F6 | F5 plus schedule-graph features; complete market-free representation | market-free |
| F7 | declared market variables only | market-only |
| F8 | F6 plus F7 | market-aware |

There are no fingerprint aliases. Every concrete frame must materialize
exact feature names, count, family, source, availability rule, cutoff,
missingness rule, transformation, market/opponent flags, version, and schema
hash. Feature order is lexical and deterministic in the manifest.

## Models, baselines, and consensus

The confirmatory architecture matrix uses common temporal folds, games, target,
budgets, and three fixed seeds. Every M architecture is retained at every F0–F8
tier, producing 54 scientific margin cells. Only the 42 market-free F0–F6 cells
are eligible for prospective predictions and polls. The corrected-F6 wide
roster supplies 33 automated poll members plus one separate owner ballot.
Transparent baselines include random 50%, home/prior, season-to-date win rate,
raw and opponent-adjusted point differential, strict OOF historical-matchup
KNN, and the declared CFBD line. The human Top 25 ballot is independent, is
reported separately from the 33 automated ballots, and is not a model ballot or
metric input.

All-model consensus is equal-weight across valid eligible margin members for
that week. Compact consensus is a small, fixed, equal-weight set selected only
from development OOF predictions; it cannot use 2025 or 2026 outcomes. Failed
models are reported and excluded for that week rather than imputed or
retroactively retrained.

## Probability and inference

Raw margins remain preserved. The uncalibrated margin-to-probability link and a
calibrator fit only on nested development OOF predictions are both retained.
Evaluation reports Brier, log loss, accuracy, calibration intercept/slope, ECE
with declared bins, reliability counts, sharpness, margin MAE/RMSE, and sample
counts. Primary historical uncertainty uses a season-clustered paired
bootstrap; prospective 2026 uncertainty uses a paired week-block bootstrap.
Game-level paired bootstrap is sensitivity analysis. McNemar is reserved for
paired binary classifications. Holm controls the prespecified confirmatory
family; Benjamini–Hochberg at q=0.05 is limited to labeled exploratory families.
Practical-equivalence bounds are frozen before reading 2026 outcomes.

## Prospective timing and corrections

CFBD is the primary data provider. The weekly deadline is Thursday 23:59 in
`America/New_York`, with the equivalent UTC timestamp in every
bundle. Prediction bytes are immutable after the deadline. Corrections can
change future fingerprints and scoring records only through an append-only,
hash-linked amendment ledger; they never overwrite the original prediction
bundle. Publication requires one explicit user approval action.

## Claims we will not make without evidence

We will not claim that TDNet beats Vegas, that greater complexity universally
improves prediction, that feature importance is causal, that every
opponent-adjustment method helps, or that 2025 is untouched.
