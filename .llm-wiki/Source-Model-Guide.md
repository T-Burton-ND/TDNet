---
type: source-summary
up: "[[Model-and-Poll-Surfaces]]"
source: source-archive/docs/MODEL_GUIDE.md.txt
tags: [source, models, implementation]
---

# Model Guide Source

The model guide documents the implemented TDNet model families, their shared prediction interface, practical uses, and limitations.

## Contribution

The guide connects model configuration to implementation in `src/gridiron_ml/models/`. It distinguishes light statistical baselines, linear estimators, tree models, and historical-matchup KNN while describing their common margin-oriented training and evaluation flow.

## Specific claims relevant to the wiki

- `TDKNN` uses pregame matchup fingerprints as a similarity space and regresses canonical home-team margin from prior matchups; it can emit neighbor-level audit details.
- The stat family normalizes selected football-stat columns and fits a linear margin model; documented variants include z-index, percentile, robust, and weighted forms.
- Linear models map configurations to actual sklearn estimators inside TDNet’s shared preprocessing/evaluation wrapper; they predict home margin and convert it to home-win probability.
- Tree models include random forest, extra trees, and classic sklearn gradient boosting. They can capture nonlinear structure but are harder to interpret and can overfit.
- These are game-level margin predictors, not play-level or drive-level models. Opponent adjustment and recursive season fingerprint evolution occur upstream.

## Where it connects

The guide explains model families used by [Package Architecture](Package-Architecture) and the roles of the scientific panel and operational poll roster in [Model and Poll Surfaces](Model-and-Poll-Surfaces). Its family descriptions should not be used to infer 2026 roster eligibility; use [Confirmatory Protocol 2026](Confirmatory-Protocol-2026) and [Confirmatory Protocol Source](Source-Confirmatory-Protocol) for that.

## Quotes worth keeping

> “TDNet models share the same broad interface.”

## Source link

Source file: `source-archive/docs/MODEL_GUIDE.md.txt`.

## See also

[Package Architecture](Package-Architecture) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [Confirmatory Protocol 2026](Confirmatory-Protocol-2026) · [Confirmatory Protocol Source](Source-Confirmatory-Protocol) · [TDNet Overview](TDNet-Overview)
