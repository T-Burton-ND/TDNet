---
type: synthesis
up: "[[Fingerprint-Ladder]]"
extends: "[[Fingerprint-Ladder]]"
tags: [fingerprints, next-generation, experiment-contract]
---

# Next-Generation Fingerprint Experiment

This page records the setup-only contract for exploratory, market-free F06→F09→F10→F11→F12 next-game fingerprints.

## Protected prediction boundary

Every dynamic team-week value is calculated using information known **before the target game**. The central target is the team's **next-game margin**. Same-game plots may explain a past game but do not show next-game usefulness. Targets are FBS-versus-FBS regular-season games; FCS opponents may supply prior context. Postseason numeric performance is excluded. 2010–2023 is base development, 2024 design-informed internal validation, 2025 design-informed late development, and **2026 is the quarantined prospective season**. The 2024/2025 metrics can influence design and recommendation, so they are development metrics rather than unbiased holdout estimates. No 2026 row may influence discovery, formulas, scaling, missingness, correlations, pruning, SHAP, hyperparameters, lineage, recommendations, or design evaluation.

## Representations, lineages, and matching

[F06](Fingerprint-F06) is the 227-source-feature canonical baseline. [F09](Fingerprint-F09) adds game microstructure, [F10](Fingerprint-F10) roster/player information, [F11](Fingerprint-F11) derived coaching information, and [F12](Fingerprint-F12) unit states. F07/F08 remain historical market comparators outside this ancestry. For each generation, design `a` is atomic, `b` adds interpretable interactions, and `c` compresses into named football concepts with exact Excel equations and at most five source inputs. F06 has full/reduced variants; F09–F12 have full, late-reduced, and progressive-reduced variants within the same design letter.

Every source feature has a matchup counterpart; remove pairs atomically only when both members qualify. Final selected team-week features must work for M1/M2/M3/M4/M5/M10 without architecture-specific feature selection. Market, CFBD pregame win probability, and other probability/line-derived signals are forbidden as inputs; a market sidecar may score ATS/chalk/upset outcomes.

## Screening, reduction, and recommendation

Screen with M2 spline ridge and M4 histogram gradient boosting: ten fixed spaced setpoints per architecture, one seed, approximately 256 SHAP background games and 512 explanation games per cell. Attribute end-to-end permutation SHAP to source features and normalize within architecture. Use pair-closed, correlation-aware pruning: |r|≥0.995 duplicate candidate; 0.98–0.995 validated redundancy; 0.90–0.98 report only. A/B candidates must be weak under both architectures; C may trade average M2/M4 performance. Seek smallest reduced fingerprint within +0.25 MAE points of full while retaining ≥60 F06 concrete features and ≥10 from each added generation. If a family offers fewer than ten legitimate signals, disclose the contradiction before final reduction rather than retain junk. Final recommendation treats candidates within 0.5 MAE points as tied, then uses Brier, ATS, upset, chalk, and fewer features; report 2024 and 2025 separately.

## Acquisition, diagnostics, and operations

Keep large data under `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/` with a 100 GB soft experiment budget from the nextgen config. Reuse verified cached partitions, use at most three total attempts per request, write Parquet atomically, and version incompatible schemas. The acquisition process is staged: A local inventory and checks, B broad cheap routes, C plays/drives, D player detail, and E gated `/plays/stats`. The separate request ledger records deterministic request identities, attempts, status, hashes, schemas, rows, bytes, and cache paths; the outbound-attempt budget ledger reserves before each network attempt. Both are under the experiment root. The live [CFBD OpenAPI 5.30.0](https://api.collegefootballdata.com/api/5.30.0/cfbd-openapi.json) audit classified its 85 GET routes in `configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json`. Cache-only ratings and retrospective season summaries require as-of validation before any use as predictors. CFBD's coverage starts at different dates: player usage 2013, returning production 2014, talent 2015, transfer portal 2021, according to the acquisition inventory and [CFBD availability](https://api.collegefootballdata.com/data-availability). Earlier absence is structural missingness, not a zero. Season aggregates must be reconstructed to the as-of week before use in current-season state.

The 2026-09-25 live CFBD `/info` preflight response confirmed a **30,000-call monthly account capacity**, reported 1 used and 29,999 remaining at that response, with reset at 2026-10-01 UTC; the provider's count may lag the local reservation ledger. `configs/experiments/nextgen_fingerprints_v1.json` now targets **at most 20,000 new calls**, has a **24,000-outbound-attempt hard experiment ceiling**, and preserves **at least 6,000 account calls**. The migrated shared budget ledger recorded 4 reserved attempts after the preflight key, quota, and play probes; this is a local accounting count, not the provider's usedCalls field. The key remains in ignored, owner-only TDNet `.env`.

The revised 2010–2025 default request manifest after one 2025 week-1 play smoke has **1,500 new planned requests and 257 verified reusable cached partitions** (`results/preflight/cfbd_plan_summary_v1.json`, 2026-09-26). Another **283 legacy cache files are candidates, not accepted reuses** without request-scope and coverage proof. The accepted `/games` cache seeds the local schedule and `/games/teams` is checked against it; provider-side full-season completeness is not independently established. The prior 1,224/533 manifest accepted too many files from legacy year caches. The play smoke returned 16,693 rows in a 648,159-byte compressed Parquet, and an immediate repeat consumed zero play calls. The **projected** total raw/canonical/intermediate footprint is 8,490,706,368 bytes, using the plan script's sampled-play and legacy-cache percentile multipliers; this is an untested storage estimate below the configured 100 GB soft limit. The filesystem had about 1.09 TB free at preflight. A single preflight command fitted all 10 M2 and 10 M4 frozen setpoints on synthetic data and checked the 42 lineage variants, pair closure, next-game alignment, postseason exclusion, 2026 exclusion, cache resume, and a synthetic row-level pre-target availability check. The `assert_temporal_feature_rows` guard requires provenance and rejects same-game, late, postseason, 2026, outcome, and forbidden input features. Future F09–F12 feature builders must invoke it at their fitting/ranking boundary; that end-to-end integration cannot be proven until those builders exist. Empty 200 responses and HTTP 400/401/403/404 now enter `needs_review` and block a stage instead of being treated as permanent structural absence. No bulk acquisition or experiment training occurred.

`/plays/stats` exposes player-to-play stat attribution, not complete snap participation. Cheaper `/plays`, `/games/players`, player PPA/success, usage, roster, and recruiting routes cover the registered F09–F12 features. Its 2012–2025 full game-partition scenario projects 11,509 extra requests before any cap subdivisions, so `configs/experiments/nextgen_plays_stats_audit_v1.json` classifies it `optional_if_budget` and excludes it from the default manifest. Stage E requires a demonstrated unique feature benefit, row-coverage check, recomputed call plan, and fresh live quota check.

Produce later diagnostics for at most 1,000 advanced features **total across generations**, prioritized by consensus architecture-normalized SHAP; duplicate formulas receive one plot unless materially different. Retain PNG, compact Markdown, and a lightweight index, with no duplicate per-feature observation tables. Default x is the pre-target-game feature; later y targets include **next-game margin**, win, points for/against, and time of possession where meaningful. Same-game association is descriptive only. The Q4-minus-Q1 rushing offensive/defensive pair is mandatory. For garbage time, Q1/Q2/Q3 leads exceed 28/24/21; Q4 is garbage only when *every* qualifying play's absolute lead stays >16. No win-probability rule or free-text parsing.

The later SGE/UGE run caps all simultaneous experiment jobs at 50, trains only one generation at a time, and proceeds when at least three of ten setpoints succeed. Retry only below three, for at most three attempts, then record incomplete status. Keep one compact ultra-wide result Parquet with run and fingerprint×architecture summary rows, including explicit sortable design-informed 2024/2025 MAE, RMSE, Brier, winner/ATS/chalk/upset columns, recommendation fields, and acquisition provenance. Do not retain screening checkpoints or routine logs. F13–F15 are future design-only and have no assigned families; PCA is deferred.

## Durable sources

- `docs/nextgen_fingerprints/README.md` — full repository contract.
- `configs/experiments/nextgen_fingerprints_v1.json` — machine guardrails and lineage.
- `configs/experiments/nextgen_feature_manifest_schema_v1.json` — exact per-feature fields.
- `configs/experiments/nextgen_acquisition_v1.json` — endpoint inventory and partition plan.
- `configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json` — complete classified OpenAPI route audit.
- `configs/experiments/nextgen_result_schema_v1.json` — ultra-wide result columns and provenance.
- `configs/experiments/nextgen_plays_stats_audit_v1.json` — expensive-endpoint value gate.
- `scripts/nextgen_preflight.py` — one-command contract, model, temporal, quota, and fetch smoke.
- [CFBD reference](https://api.collegefootballdata.com/api/plays) and [availability](https://api.collegefootballdata.com/data-availability).

**Status:** setup/design only. No next-generation bulk acquisition, training, pruning results, or recommendation exists yet.

See also: [Fingerprint Ladder](Fingerprint-Ladder) · [F06](Fingerprint-F06) · [F09](Fingerprint-F09) · [F10](Fingerprint-F10) · [F11](Fingerprint-F11) · [F12](Fingerprint-F12) · [Temporal Data Semantics](Temporal-Data-Semantics)
