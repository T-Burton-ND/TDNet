---
type: synthesis
up: "[[Fingerprint-Ladder]]"
extends: "[[Fingerprint-Ladder]]"
tags: [fingerprints, next-generation, experiment-contract]
---

# Next-Generation Fingerprint Experiment

This page records the setup-only contract for exploratory, market-free F06→F09→F10→F11→F12 next-game fingerprints.

## Protected prediction boundary

Every dynamic team-week value is calculated using information known **before the target game**. The central target is the team's **next-game margin**. Same-game plots may explain a past game but do not show next-game usefulness. Targets are FBS-versus-FBS regular-season games; FCS opponents may supply prior context. Postseason numeric performance is excluded. 2010–2023 is base development, 2024 validation/design evidence, 2025 late-development evidence, and **2026 is the quarantined prospective season**. No 2026 row may influence discovery, formulas, scaling, missingness, correlations, pruning, SHAP, hyperparameters, lineage, recommendations, or design evaluation.

## Representations, lineages, and matching

[F06](Fingerprint-F06) is the 227-source-feature canonical baseline. [F09](Fingerprint-F09) adds game microstructure, [F10](Fingerprint-F10) roster/player information, [F11](Fingerprint-F11) derived coaching information, and [F12](Fingerprint-F12) unit states. F07/F08 remain historical market comparators outside this ancestry. For each generation, design `a` is atomic, `b` adds interpretable interactions, and `c` compresses into named football concepts with exact Excel equations and at most five source inputs. F06 has full/reduced variants; F09–F12 have full, late-reduced, and progressive-reduced variants within the same design letter.

Every source feature has a matchup counterpart; remove pairs atomically only when both members qualify. Final selected team-week features must work for M1/M2/M3/M4/M5/M10 without architecture-specific feature selection. Market, CFBD pregame win probability, and other probability/line-derived signals are forbidden as inputs; a market sidecar may score ATS/chalk/upset outcomes.

## Screening, reduction, and recommendation

Screen with M2 spline ridge and M4 histogram gradient boosting: ten fixed spaced setpoints per architecture, one seed, approximately 256 SHAP background games and 512 explanation games per cell. Attribute end-to-end permutation SHAP to source features and normalize within architecture. Use pair-closed, correlation-aware pruning: |r|≥0.995 duplicate candidate; 0.98–0.995 validated redundancy; 0.90–0.98 report only. A/B candidates must be weak under both architectures; C may trade average M2/M4 performance. Seek smallest reduced fingerprint within +0.25 MAE points of full while retaining ≥60 F06 concrete features and ≥10 from each added generation. If a family offers fewer than ten legitimate signals, disclose the contradiction before final reduction rather than retain junk. Final recommendation treats candidates within 0.5 MAE points as tied, then uses Brier, ATS, upset, chalk, and fewer features; report 2024 and 2025 separately.

## Acquisition, diagnostics, and operations

Keep large data under `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/` with ~100 GB soft budget. Reuse cached valid CFBD partitions, verify row-limit completeness, use at most three request attempts, and version incompatible schemas. CFBD's coverage starts at different dates: player usage 2013, returning production 2014, talent 2015, transfer portal 2021. Earlier absence is structural missingness, not a zero. Season aggregates must be reconstructed to the as-of week before use in current-season state. Do not bulk-fetch or submit training in setup.

The user reports a 30,000-call allowance for the new account key, while `configs/experiments/nextgen_fingerprints_v1.json` imposes a **hard 20,000-outbound-attempt ceiling for this experiment**. Every next-generation request, including retries and the key check, reserves one slot in `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/results/cfbd_api_call_budget.json` before sending. The ledger is shared across processes and fails closed at its cap or if removed. The key stays in TDNet's ignored, owner-only `.env` file; never commit or print it. The single-request check succeeded on 2026-09-25 (one request; 136 teams returned by CFBD `/teams/fbs?year=2025`). Do not rerun it casually.

Produce later diagnostics for at most 1,000 advanced features by consensus SHAP, with PNG and Markdown each. Default x is pregame feature, y is **next-game margin**; empirical next-game win frequency may accompany it. The Q4-minus-Q1 rushing offensive/defensive pair is mandatory. For garbage time, Q1/Q2/Q3 leads exceed 28/24/21; Q4 is garbage only when *every* qualifying play's absolute lead stays >16. No win-probability rule or free-text parsing.

The later SGE/UGE run caps all simultaneous experiment jobs at 50, trains only one generation at a time, and proceeds when at least three of ten setpoints succeed. Retry only below three, for at most three attempts, then record incomplete status. Keep one compact ultra-wide result Parquet with run and fingerprint×architecture summary rows, but no retained checkpoints or routine logs. F13–F15 are future design-only and have no assigned families; PCA is deferred.

## Durable sources

- `docs/nextgen_fingerprints/README.md` — full repository contract.
- `configs/experiments/nextgen_fingerprints_v1.json` — machine guardrails and lineage.
- `configs/experiments/nextgen_feature_manifest_schema_v1.json` — exact per-feature fields.
- `configs/experiments/nextgen_acquisition_v1.json` — endpoint inventory and partition plan.
- [CFBD reference](https://api.collegefootballdata.com/api/plays) and [availability](https://api.collegefootballdata.com/data-availability).

**Status:** setup/design only. No next-generation bulk acquisition, training, pruning results, or recommendation exists yet.

See also: [Fingerprint Ladder](Fingerprint-Ladder) · [F06](Fingerprint-F06) · [F09](Fingerprint-F09) · [F10](Fingerprint-F10) · [F11](Fingerprint-F11) · [F12](Fingerprint-F12) · [Temporal Data Semantics](Temporal-Data-Semantics)
