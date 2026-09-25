---
type: reference
up: "[[TDNet-Overview]]"
tags: [protocol, 2026, research-design]
---

# Confirmatory Protocol 2026

The 2026 confirmatory protocol fixes TDNet’s study scope, feature tiers, model matrix, prospective eligibility, and inference rules.

## Study scope

- Population: FBS-versus-FBS regular-season games; FCS games and postseason are out of scope.
- Primary objective: game-margin prediction. Winner metrics are secondary.
- Development seasons: 2010–2024 as declared in protocol version 2.
- 2025: consumed retrospective evidence, not an untouched holdout.
- 2026: prospective season; 2026 outcomes are excluded from training and model selection for the frozen study.

## Controlled matrix

Under protocol version 2, the six architectures are M1 linear, M2 spline, M3 decision tree, M4 boosted trees, M5 neural network, and M10 K-nearest neighbors. Each is evaluated at F0–F8, yielding 54 scientific margin cells. The comparison uses common temporal splits, target, games, tuning budget, and three fixed seeds: 1701, 2718, and 3141.

The 42 protocol-version-2 F0–F6 market-free cells are eligible for official prospective predictions and polls. F7 (market-only) and F8 (F6 plus market) are research comparators and never enter official TDNet predictions, consensus, or polls. See [Fingerprint Ladder](Fingerprint-Ladder), [Model and Poll Surfaces](Model-and-Poll-Surfaces), [README Source](Source-README), and [Model Guide Source](Source-Model-Guide).

## Calibration and inference

Raw margins remain preserved. The uncalibrated margin-to-probability mapping and any calibrator fitted only on nested development out-of-fold predictions are both retained. Reported metrics include Brier score, log loss, accuracy, calibration intercept and slope, expected calibration error with declared bins, reliability counts, sharpness, margin MAE/RMSE, and sample counts.

Historical primary uncertainty is a season-clustered paired bootstrap; prospective 2026 primary uncertainty is a paired week-block bootstrap. Game-level paired bootstrap is sensitivity analysis. McNemar is for paired binary classifications. Holm controls the prespecified confirmatory family; Benjamini–Hochberg at q=0.05 is restricted to labeled exploratory families. Practical-equivalence bounds are fixed before 2026 outcomes are read.

## Prospective timing and correction

CFBD is the primary data provider. The deadline is Thursday 23:59 America/New_York, with an equivalent UTC timestamp recorded in each bundle. Prediction bytes are immutable after the deadline. Corrections may affect future fingerprints and scoring records only through an append-only, hash-linked amendment ledger; they do not overwrite the original prediction bundle. Publication requires one explicit owner approval.

## Claims boundary

The protocol does not authorize default claims that TDNet beats Vegas, complexity always improves prediction, feature importance is causal, every opponent-adjustment method helps, or 2025 was untouched. See the source summary [Confirmatory Protocol Source](Source-Confirmatory-Protocol) and the dated warning in [TDNet Master Plan Source](Source-TDNet-Master-Plan).

## Sources

- `../docs/publication_2026/CONFIRMATORY_PROTOCOL.md`
- `../configs/publication/confirmatory_protocol.yaml` (protocol version 2)
- [Confirmatory Protocol Source](Source-Confirmatory-Protocol)
- [TDNet Overview](TDNet-Overview)

## See also

[Fingerprint Ladder](Fingerprint-Ladder) · [Temporal Data Semantics](Temporal-Data-Semantics) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [Weekly Publication Workflow](Weekly-Publication-Workflow) · [README Source](Source-README) · [Model Guide Source](Source-Model-Guide) · [Weekly Operations Source](Source-Weekly-Operations) · [Project History and Directions](Project-History-and-Directions) · [Deferred and Abandoned Directions](Deferred-and-Abandoned-Directions)
