---
type: synthesis
up: "[[Fingerprint-Ladder]]"
extends: "[[Fingerprint-Ladder]]"
tags: [fingerprints, next-generation, experiment-contract]
---

# Next-Generation Fingerprint Experiment

This page records the execution contract for exploratory, market-free F06→F09→F10→F11→F12 next-game fingerprints.

## Measured lessons — 2026-09-28

Empirical lessons now distinguish aggregate medians from matched configurations and verify common evaluation games. Observed F09 M4 gains, F06 compression tradeoffs, F10 A/M2 gains, and negative F11/F12 A comparisons are documented without attributing untested causes. See [empirical lessons and paired evidence](Nextgen-Empirical-Lessons-2026-09-28) for exact values, scope and limits. No new experiments were run.

## Current snapshot — 2026-09-28

Goal paused at user request. Verified existing outputs now include terminal F06 full/reduced and F09 full screening, F10 A M2, and F11/F12 A both architectures. F06 reduced A/B pass the numerical MAE tolerance but remain nonminimum first trials; C is the only accepted floor reduction. Existing F10/F11/F12 submissions remain active; no new experiment or acceptance was issued. See the [verified paused snapshot](Nextgen-Results-Snapshot-2026-09-28) for all metrics, evidence paths, source gaps and scheduler counts as of 12:52 UTC. This supersedes earlier live-status statements; the goal is paused and existing queued cells may still dispatch.

## Protected prediction boundary

Every dynamic team-week value is calculated using information known **before the target game**. The central target is the team's **next-game margin**. Same-game plots may explain a past game but do not show next-game usefulness. Targets are FBS-versus-FBS regular-season games; FCS opponents may supply prior context. Postseason numeric performance is excluded. 2010–2023 is base development, 2024 design-informed internal validation, 2025 design-informed late development, and **2026 is the quarantined prospective season**. The 2024/2025 metrics can influence design and recommendation, so they are development metrics rather than unbiased holdout estimates. No 2026 row may influence discovery, formulas, scaling, missingness, correlations, pruning, SHAP, hyperparameters, lineage, recommendations, or design evaluation.

## Representations, lineages, and matching

[F06](Fingerprint-F06) has a historical 227-source-feature canonical baseline; the exploratory a design uses 225 after the user-authorized coach SP exclusion below. [F09](Fingerprint-F09) adds game microstructure, [F10](Fingerprint-F10) roster/player information, [F11](Fingerprint-F11) derived coaching information, and [F12](Fingerprint-F12) unit states. F07/F08 remain historical market comparators outside this ancestry. For each generation, design `a` is atomic, `b` adds interpretable interactions, and `c` compresses into named football concepts with exact Excel equations and at most five source inputs. F06 has full/reduced variants; F09–F12 have full, late-reduced, and progressive-reduced variants within the same design letter.

Every source feature has a matchup counterpart; remove pairs atomically only when both members qualify. Final selected team-week features must work for M1/M2/M3/M4/M5/M10 without architecture-specific feature selection. Market, CFBD pregame win probability, and other probability/line-derived signals are forbidden as inputs; a market sidecar may score ATS/chalk/upset outcomes.

## Screening, reduction, and recommendation

Screen with M2 spline ridge and M4 histogram gradient boosting: ten fixed spaced setpoints per architecture, one seed, approximately 256 SHAP background games and 512 explanation games per cell. Attribute end-to-end permutation SHAP to source features and normalize within architecture. Use pair-closed, correlation-aware pruning: |r|≥0.995 duplicate candidate; 0.98–0.995 validated redundancy; 0.90–0.98 report only. A/B candidates must be weak under both architectures; C may trade average M2/M4 performance. Seek smallest reduced fingerprint within +0.25 MAE points of full while retaining ≥60 F06 concrete features and ≥10 from each added generation. If a family offers fewer than ten legitimate signals, disclose the contradiction before final reduction rather than retain junk. Final recommendation treats candidates within 0.5 MAE points as tied, then uses Brier, ATS, upset, chalk, and fewer features; report 2024 and 2025 separately.

## Acquisition, diagnostics, and operations

Keep large data under `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/` with a 100 GB soft experiment budget from the nextgen config. Reuse verified cached partitions, use at most three total attempts per request, write Parquet atomically, and version incompatible schemas. The acquisition process is staged: A fresh 2010–2025 `/games` schedules, B broad cheap routes, C plays/drives, D player detail, and E gated `/plays/stats`. The separate request ledger records deterministic request identities, attempts, status, hashes, schemas, rows, bytes, and cache paths; the outbound-attempt budget ledger reserves before each network attempt. Both are under the experiment root. The live [CFBD OpenAPI 5.30.0](https://api.collegefootballdata.com/api/5.30.0/cfbd-openapi.json) audit classified its 85 GET routes in `configs/experiments/nextgen_cfbd_endpoint_inventory_v1.json`. Cache-only ratings and retrospective season summaries require as-of validation before any use as predictors. CFBD's coverage starts at different dates: player usage 2013, returning production 2014, talent 2015, transfer portal 2021, according to the acquisition inventory and [CFBD availability](https://api.collegefootballdata.com/data-availability). Earlier absence is structural missingness, not a zero. Season aggregates must be reconstructed to the as-of week before use in current-season state.

The 2026-09-25 live CFBD `/info` preflight response confirmed a **30,000-call monthly account capacity**, reported 1 used and 29,999 remaining at that response, with reset at 2026-10-01 UTC; the provider's count may lag the local reservation ledger. `configs/experiments/nextgen_fingerprints_v1.json` now enforces a **20,000 actual-outbound-attempt hard ceiling**, including retries and quota checks, and preserves **at least 10,000 provider calls** from the 30,000-call allowance. The local atomic reservation ledger is authoritative for runaway protection and retained its 4 earlier reserved attempts when its limit was lowered. A live quota check must cover every remaining local budget slot plus the provider reserve; it does not require three times all pending requests to fit simultaneously. The local count is distinct from the provider's usedCalls field. The key remains in ignored, owner-only TDNet `.env`.

The 2026-09-26 **provisional pre-Stage-A** 2010–2025 manifest (`results/preflight/cfbd_plan_summary_v1.json`) has **1,756 planned requests, including 16 fresh `/games` calls, and 1 ledger-backed reusable `/plays` partition**. Legacy schedules only estimate request identities; they do not authorize team-game reuse. Stage A must fetch all 16 fresh schedules, then preflight must be rerun. The fresh ledger-backed completed regular-season schedules become the authority for later request planning and for expected game-ID checks on legacy `/games/teams`; later stages fail closed until this is complete. The earlier 1,500/257 plan counted 16 legacy schedules and 240 legacy team-game partitions before authoritative schedule proof. The play smoke returned 16,693 rows in a 648,159-byte compressed Parquet, and an immediate repeat consumed zero play calls. The **projected, untested** raw/canonical/intermediate footprint is 9,224,915,712 bytes, based on the sampled play partition, legacy-cache size percentile, and plan multipliers, below the configured 100 GB soft limit. No fresh schedules, bulk acquisition, or experiment training were run in this preparation pass.

The common `nextgen_artifacts.py` boundary makes every future F09–F12 builder use checked canonical feature-family writes and revalidates persisted rows before matchup, SHAP, or model fitting. Canonical rows require `team`, `target_game_id`, and `target_start_utc`. Matchup home/away keys are `(target_game_id, team)` and must match the fresh Stage-A schedule; DataFrame indexes have no matchup meaning. For each dynamic row the fresh ledger-backed `/games` schedule must confirm target-team participation and kickoff, source-team participation and kickoff, regular-season completed source status, and source before target. The row must satisfy source kickoff UTC < feature availability UTC < target kickoff UTC. `/games` completion status does not prove the precise time data became available, so future builders must document their data-availability cutoffs. The durable team-season Week-0 freeze is the first regular-season kickoff, exclusive; static rows need null source-game fields, documented prior availability, and the same frozen value across the season. Canonical artifact sidecars bind manifest, data, and schedule hashes. Manifest flags independently forbid market, target, pregame-win-probability, and disallowed temporal sources even with harmless-looking column names. The guard also rejects 2026 and postseason numeric history. Boundary tests pass; actual F09–F12 builders and model runners do not yet exist. Empty 200 responses and HTTP 400/401/403/404 remain `needs_review` and block a stage.

`/plays/stats` exposes player-to-play stat attribution, not complete snap participation. Stage E remains gated. The **uncommitted full game-partition scenario** on the provisional 2012–2025 pre-Stage-A schedule adds **11,509 estimated first attempts** before retries or cap subdivisions, bringing that hypothetical combined plan to **13,265 estimated first attempts**; after four earlier reservations, it would leave **6,731 local attempt slots** under the 20,000 cap for retries, quota checks, and subdivisions. Its **projected, untested** footprint is **89,771,158,608 bytes** against the 100 GiB soft limit. Sample first, verify stat meaning, coverage, and 2,000-row cap handling; identify concrete F10/F12 features uniquely enabled; then choose full, subset, or skip from incremental value, fresh-schedule calls, live quota, and storage. Absence of a PlayStat row cannot prove nonparticipation. No Stage E calls ran in preparation.

Produce later diagnostics for at most 1,000 advanced features **total across generations**, prioritized by consensus architecture-normalized SHAP; duplicate formulas receive one plot unless materially different. Retain PNG, compact Markdown, and a lightweight index, with no duplicate per-feature observation tables. Default x is the pre-target-game feature; later y targets include **next-game margin**, win, points for/against, and time of possession where meaningful. Same-game association is descriptive only. The Q4-minus-Q1 rushing offensive/defensive pair is mandatory. For garbage time, Q1/Q2/Q3 leads exceed 28/24/21; Q4 is garbage only when *every* qualifying play's absolute lead stays >16. No win-probability rule or free-text parsing.

The SGE/UGE run caps all simultaneous experiment jobs at 50 and proceeds when at least three of ten setpoints succeed. The original one-generation-at-a-time scheduling restriction is superseded by the user-authorized parallel policies below. Retry only below three, for at most three attempts, then record incomplete status. Keep one compact ultra-wide result Parquet with run and fingerprint×architecture summary rows, including explicit sortable design-informed 2024/2025 MAE, RMSE, Brier, winner/ATS/chalk/upset columns, recommendation fields, and acquisition provenance. Do not retain screening checkpoints or routine logs. F13–F15 are future design-only and have no assigned families; PCA is deferred.

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

## Execution launch — 2026-09-27

The user authorized execution through F12 and all requested metrics in `docs/nextgen_fingerprints/overnight_launch_prompt.md`, superseding setup-only restrictions. Starting branch `experiment/nextgen-fingerprints-v1` was clean at `cc3038f1bb293e127e69d18d00647bf6d37b1e5a`; a fresh fetch confirmed it current. The launch preflight fitted all ten frozen configurations for each of M2/M4 on smoke data, and the four targeted nextgen test modules passed 27 tests. These are engineering checks, not football model results.

Stage A completed all 16 fresh 2010–2025 schedule requests plus one quota call, advancing the experiment ledger from 4 to 21 reservations (`results/preflight/stage_A_latest.json`). Fresh-schedule preflight then verified 240 legacy team-game partitions, giving 257 total reusable partitions and 1,500 planned new requests. Its projected, untested storage estimate was 8,490,706,368 bytes. Those counts describe the original post-Stage-A inventory before launch coverage corrections.

The first Stage B invocation consumed 50 outbound attempts including quota and the failed request, leaving 71 total experiment reservations. It completed 48 requests and stopped on HTTP 400 from the documented legal `/coaches/tenures?year=2010` query. The reviewed failed request is retained in the ledger with its original error; this does not mean no coaches existed. The optional tenure route is deferred; tenure can be derived from the 16 successfully acquired `/coaches/seasons` partitions. Provider-documented coverage corrections exclude pre-2013 lines/ATS/pregame probabilities, pre-2016 CORE/kicker PAAR, pre-2013 player passing/rushing WEPA, pre-2012 player success, and expanded SRS in 2020. Evidence: [CFBD availability](https://api.collegefootballdata.com/data-availability), checked 2026-09-27. Missing eras are not zeros. Preflight after these changes had 1,376 remaining planned calls and 305 reusable partitions, with a projected, untested 7,755,064,152-byte footprint. Acquisition resumed from verified requests; no completed request was intentionally refetched.

**Status:** execution underway. Fresh schedules exist; F06–F12 screening, SHAP, reductions, diagnostics, and recommendations remain incomplete. See `docs/nextgen_fingerprints/execution_status.md` for the completion audit and artifact locations.

See also: [Fingerprint Ladder](Fingerprint-Ladder) · [F06](Fingerprint-F06) · [F09](Fingerprint-F09) · [F10](Fingerprint-F10) · [F11](Fingerprint-F11) · [F12](Fingerprint-F12) · [Temporal Data Semantics](Temporal-Data-Semantics)

## Frozen average-team references — 2026-09-27

The six assembled F06/F09 full fingerprints (a/b/c) each now have 16 frozen annual reference files for 2011–2026, totaling 96 files under `average_team_references/<fingerprint>/<season>.json` in the experiment artifact root. The build completed successfully; independent readback verified reference/source-season ordering, fingerprint identity, and feature/support key agreement for all files. This verifies artifact construction, not model or power-ranking performance.

Construction averages target states within each team-week, then gives equal weight to observed weeks within team-season, teams within season, and prior seasons. Each reference uses only seasons before its reference year. The 2026 reference uses available 2010–2025 states. Missing values are omitted separately at each level, with per-feature support counts; wholly unsupported values remain null. There is no 2010 reference because these fingerprints lack earlier state history. Existing reference payloads cannot silently change; a conflicting write requires explicit versioning.

These files retain the source-semantics audit requirement before prediction. Later full and reduced fingerprints still require their own exact-feature references after assembly. Implementation and focused weighting/freeze tests: `nextgen_references.py` and `tests/test_nextgen_references.py` (two tests passed).

## Stage E sample decision — 2026-09-27

The authorized four-game `/plays/stats` sample (2013, 2017, 2021, 2025) returned 171/195/233/194 rows, all below the 2,000-row cap. Every sampled attributed row joined to a canonical play; athlete IDs were present. Rushing-play recall was 100% in the first three games and 71/74 in the 2025 game. Attributed rushing yardage differed from canonical play yardage in 8/0/1/1 rows, respectively. Sources: `results/preflight/plays_stats_sample_audit.json`, with hashes of both sampled events and comparison play partitions.

The autonomous decision is **skip bulk acquisition**, recorded in `results/preflight/plays_stats_approval.json`. The sample offers possible quarter-specific player usage detail but lacks positions and participation denominators, has uneven event/yardage coverage, and does not resolve the outstanding unit-position gap. This decision preserves the sample and audit; it is not an assertion that the endpoint is universally unusable. Event absence must never imply nonparticipation. Further acquisition requires new incremental-value and coverage evidence. No bulk storage is allocated.

## Corrected game-detail query coverage — 2026-09-27

All four 2025 team/player passing/rushing game routes now have 16 successful weekly regular-season requests each (64 total). The earlier year-only HTTP 400s were query errors: OpenAPI descriptions require a team or week with the year. Corrected request identities preserve the original failures rather than relabeling them successful. Evidence: the current request manifest and request-ledger statuses for `/passing/teams/games`, `/rushing/teams/games`, `/passing/players/games`, and `/rushing/players/games`.

The subsequent inventory check still found five pending team-box partitions, 44 early player-PPA game weeks across 2010–2012, and 14 player-success game weeks in 2012. These unattempted requests are not proven structural absences. Team-box gap acquisition has started; no claim of complete acquisition or trained model results follows from the corrected routes.

## Canonical component coverage audit — 2026-09-27

`scripts/nextgen_feature_coverage.py` audited the 24 existing canonical component families against their recorded data and manifest hashes, writing `results/canonical_family_coverage.json` in the experiment artifact root. Across the 2010–2023 training window, no feature in these built families is entirely missing. This does not establish complete fingerprint coverage, temporal validity, or training authorization.

The year-by-year audit exposes left-truncation: observed-continuity design a has 1,282 target rows in 2010 but every feature value is missing; recruiting and previous-staff rows begin in 2011; offensive-room rows begin in 2013. For 2025, previous-staff design a covers 1,112 team-target rows, versus 1,524 in player usage, so eventual paired assembly must explicitly report its row exclusions. Missingness and absent rows remain distinct in the report. No model performance was measured.

## Advanced team-detail suitability check — 2026-09-27

Measured all 16 cached weekly partitions for each corrected team passing/rushing game route. Each contains 1,650 game-team rows, all in 2025, with no duplicate game-team keys. Thus these acquired routes provide no 2010–2023 training observations and cannot by themselves supply trainable F12 line/defensive states.

Passing location coverage: offense 20,579 of 50,403 eligible attempts; defense 20,513 of 50,223. Rushing direction coverage: offense 23,424 of 55,382 eligible attempts; defense 23,205 of 50,617. These are sums of provider availability counters over the cached 2025 corpus, not verified position attribution or predictive metrics. Zero-filled directional detail with no available attempts must not be interpreted as measured zero production. No new unit features were materialized from these routes.

## Authorized coach SP exclusion — 2026-09-27

The user resolved the source-policy conflict: “Remove those fields and relax exact preservation.” The exploratory baseline now excludes `coach_career_mean_sp_offense` and `coach_career_mean_sp_defense` before design formulas or metadata are constructed. The historical publication source and its 227-feature manifest remain unchanged. This decision supersedes the original exact-preservation requirement for the new experiment only.

F06 full a/b/c rebuilt with 225/235/153 features on 22,390 team-target rows. The source audit passed after removal. The model loader rejects either excluded field in columns or formula inputs. All 88 nextgen tests passed (five nonfatal warnings); this is implementation validation, not predictive-performance evidence. Pre-decision baseline-dependent artifacts and annual references were moved to `scratch/superseded_before_coach_sp_exclusion` within the experiment artifact root, so prior frozen references are not silently reused. F09 rebuilding and screening submission are tracked separately.

## Post-exclusion readback and acquisition completion — 2026-09-27

Independent readback verified all 96 regenerated annual references for six F06/F09 full fingerprints over 2011–2026: matching current data/manifest hashes, exact feature/support keys, strictly prior source seasons, and absence of both excluded coach SP fields. All three rebuilt F09 full fingerprints also passed the checked model loader, with 263/281/183 features and 11,151 paired target games each. No trained performance follows from these checks.

The current acquisition manifest now has 1,653 `success_complete`, 240 `skipped_existing_complete`, and 64 `failed_final` requests, with no pending or unresolved requests. All 60 early-history player-PPA/player-success requests in that manifest are reviewed failures, not proven structural absences. The bounded probe process exited successfully; it supplied no new early-history rows requiring offensive-room rebuilding.

The scientific baseline gate is resolved, but a fresh ticket check still returned 1 and the scheduler has no jobs. F06 submission awaits renewal of the missing Kerberos/AFS credentials cache. No credentials should be supplied in chat.

## F06 screening started — 2026-09-27

Cluster authentication was restored and SGE accepted the F06 full-screening array as job `1478702`, tasks 1–60, with a 48-task concurrency limit. The scheduler confirmed running tasks. The first observed failures are M2 setpoints `m2_09` and `m2_10` for designs a and b: SHAP exhausted task memory while allocating transformed arrays of approximately 3.02/3.30 GiB. Other tasks remain in progress; no final architecture success count or model recommendation is claimed. Preserve failed attempts and apply the below-three-success retry policy only after the frozen matrix is terminal.

`nextgen_reduced_artifacts.py` now constructs F06 R / later LR trials only from a complete terminal full-screening matrix and verified successful-run SHAP artifacts. It preserves target rows, removes deselected columns, enforces reciprocal pair closure and ancestry floors, refuses overwriting existing trials, and marks each proposal unaccepted pending its own screening and measured acceptance. Seven focused reduction tests passed. No real reduction proposal has been materialized yet; progressive ancestry remains separate work.

## First live consolidated results snapshot — 2026-09-27

`results/nextgen_results.parquet` now exists under the experiment artifact root, populated by the collector from actual screening records. The first snapshot contains 48 incomplete run rows, four failed run rows, and six incomplete architecture summaries. No architecture summary is screening-usable; queued tasks without progress artifacts have not been fabricated as completed runs. These counts describe this snapshot only and must be refreshed as the live array progresses. No recommendation or reduction acceptance is implied by the presence of the results file.

## Authorized parallel full screening — 2026-09-27

The user authorized running later generations and different data levels in parallel when feasible. This supersedes the original one-generation-at-a-time barrier for independent full fingerprints. Progressive reductions still require accepted prior survivors; reduction acceptance still requires terminal reference screening. The global 50-job limit and scientific protocol remain in force.

All three F09 full fingerprints passed checked-loader validation (11,151 paired games; 263/281/183 features for a/b/c). SGE accepted the 60-task F09 full array as `1479074.1-60:1`, with a 45-task cap and 24 GB per task. At submission, three F06 tasks remained, giving a combined maximum of 48 running jobs. Receipt: `experiments/F09/full_parallel_array.submission.json` under the experiment root. Submission is not a completed-training claim.

The separately versioned `nextgen_screening_parallel.py` preserves the original live F06 runner and its execution hashes. Its bindings include the parallel authorization; the results collector verifies each runner version separately. Four focused tests passed, including an AST comparison showing unchanged scientific computation apart from the scheduling gate and provenance binding.

## Independent reduction screening runner — 2026-09-27

The user's permission for parallel generations and different data levels also allows an independent late-reduction trial to run after its own full reference is terminal. `nextgen_screening_reduced_parallel.py` removes unrelated generation waits for R/LR trials, but requires matching immutable proposal evidence and the complete verified full matrix. PR continues to require the existing verified accepted-parent finalization gate. All trials still need their own measured acceptance. The global fifty-job limit remains unchanged.

A separate authorization file and runner preserve the hashes of every existing live job. New reduced-run bindings include the reduction and parent-verification helpers; do not modify those helpers once this runner has active jobs. The collector recognizes the third binding policy. Nine focused tests passed, including identical scientific-function ASTs, terminal evidence/proposal checks, PR parent gating, and result-verifier dispatch. No job has yet used this new runner; queued F06 A/B and running C retain their original runner.

Independent exact recomputation verified all 116 new annual references for reduced F06 A/B and full F11/F12, including source hashes and strictly prior season ordering. Evidence: `results/reduced_ab_f11_f12_reference_readback.json` under the artifact root. Together with previously verified sets, eighteen currently materialized fingerprints have 273 annual references; later LR/PR artifacts and their references remain outstanding.

## Source documentation and live capacity — 2026-09-27

`docs/nextgen_fingerprints/inherited_graph_documentation.json` now supplies source-reviewed upstream formulas, units, sign meanings, defaults, and film interpretations for all eight inherited F06 schedule-graph columns. Its implementation hash binds the reviewed `fingerprints/ladder.py`. The PageRank description includes the actual 0.25 reverse edge for margins of at most seven points and exactly fifty iterations; the older feature-matrix prose omitted the reverse edge. Final diagnostic rendering appends this supplement without changing the frozen trained manifests or ranking identity. Five diagnostic tests passed, including supplemental output and preservation of the original record.

The actual `results/feature_metadata_placeholder_audit.json` inventories eighteen materialized fingerprints and 253 unique definitions still requiring inherited metadata clarification after graph supplementation. This is a placeholder inventory, not proof that other fields are semantically correct. Full metadata completion and final plot generation remain outstanding.

Live full screening continues in F09–F12 alongside F06 reduced A/B. F06 reduced C has an accepted independent floor receipt (see its generation page), superseding the earlier all-generation-only parent gate. Two more completed F09 tasks enabled caps F09 36, F10 2, F11 5, F12 4, reduced A 1, reduced B 2, totaling fifty. Scheduler readback confirmed these running counts. Evidence: `results/concurrency_f12_expansion.json`; earlier caps are superseded.

## Inherited temporal documentation — 2026-09-27

`inherited_temporal_documentation.json` documents all 72 distinct temporal column names present across F06 full a/b/c: source-state reconstruction, last-one/last-three levels, half-life-three EWM, recent-minus-season difference, population volatility, and C-design equal-weight recent consensus. It binds both `fingerprints/ladder.py` and `nextgen_designs.py` by implementation hashes. Count jumps reconstruct one batch-average contribution, not individual missing games; bye rows add no observation; the one-contribution volatility default is zero. Provider raw-statistic definitions remain a separate scope of review.

The final diagnostic orchestrator loads both graph and temporal supplements, rejects changed source implementations or duplicate definitions, and appends source computation details without altering frozen manifests. Nine ladder/diagnostic tests passed. Actual readback verified all 72 feature names, formula/documentation fields, and both source hashes. Evidence: `results/feature_metadata_supplement_review.json` under the artifact root. After these supplements, 181 distinct definitions in the original placeholder inventory still lack supplementation; this does not imply that all other metadata is fully verified.

## Opponent-adjusted source documentation — 2026-09-27

`inherited_opponent_documentation.json` supplies source-reviewed v1.4 formulas, units, sign meanings, defaults, and rolling behavior for all 98 opponent-adjusted names present across full F06 designs. Source hashes bind the Elo-context implementation and C-consensus code. The documentation preserves actual historical sample-standard-deviation scaling, the home term, unchanged defensive residual sign, row-based last-three windows, and the EWM weight decay across missing rows. A six-row missing-value fixture independently matched the actual rolling function's EWM and last-three outputs; ten opponent-adjustment/diagnostic tests passed.

Readback verified the three disjoint source supplements against actual trained names and source hashes. They now cover 178 columns; 83 unique definitions from the original placeholder inventory remain unsupplemented. Evidence: `results/feature_metadata_opponent_supplement_review.json`. Provider raw-statistic definitions and other metadata semantics remain a separate review scope; full metadata completion is not claimed.

One newly completed F09 run allowed reduced A to expand to two concurrent tasks. Receipt: `results/concurrency_reduced_a_expansion.json`. Current caps are F09 35, F10 2, F11 5, F12 4, reduced A 2, reduced B 2, totaling fifty. Earlier capacity receipts are superseded.

## Canonical box-score and travel documentation — 2026-09-27

`inherited_boxscore_documentation.json` source-documents 35 inherited box-score, prior-scoring, support-count, and travel columns. It binds the actual builder, cleaners, and canonical rename map. The supplement records duplicate-key averaging, cumulative nonmissing row means, exact sparse-count zero-fill columns, per-game rate formation, score ownership, possession minutes, and travel transformations. Inherited travel distance is a dimensionless difference of log(1+kilometers), not the miles label in the older registry; it is averaged over source rows. Prior-score averages depend on earlier outcomes even though generic frozen manifest flags say target-derived false. This does not mean future labels enter these state features. Provider `stat_interceptions` event ownership remains explicitly unresolved rather than inferred from the canonical statDef prefix.

Eight row-semantics/diagnostic tests passed with existing pandas deprecation warnings. Readback verified all four disjoint supplements against actual trained names and source hashes: 213 columns now have supplements, and 48 unique definitions from the original placeholder inventory remain unsupplemented. Evidence: `results/feature_metadata_boxscore_supplement_review.json`. Provider definitions and other semantic review remain incomplete.

F10 was expanded to five active tasks after three F09 completions. Current caps and confirmed running counts are F09 32, F10 5, F11 5, F12 4, reduced A 2, reduced B 2, totaling fifty. Evidence: `results/concurrency_f10_expansion.json`. F09 C has fifteen successful runs (nine M2, six M4), with five still incomplete; reduced A has its first successful M2 run. These are partial screening results, not terminal reference or reduction acceptance.

## Inherited efficiency and prior documentation — 2026-09-27

The efficiency supplement adds source-reviewed transformations and provider terminology for forty offense/defense fields; the prior supplement adds eight talent, returning-production and coach-experience fields. Current provider terminology is cited to `https://apinext.collegefootballdata.com/metrics-and-definitions`, with explicit separation from unverified historical filtering behavior. Implementation hashes cover the actual builder, canonicalization, fetch, weekly merge and coach-history logic. The coach loader explicitly uses seasons strictly before the target season. The generic target/static flags are not substituted for this source review.

Readback verifies six disjoint supplements covering 261 inherited columns; no definition in the original placeholder inventory now lacks a supplement. This is not full all-generation semantic completion. Remaining caveats include provider event ownership for statDef_interceptions, historical filtering/publication timing and returning-PPA denominator reconstruction. In the actual 22,390-row F06_F_a artifact, passing returning-PPA share spans -129.5 to 9.5 and rushing returning-PPA share spans -45.5 to 76.667; these retained values are not probabilities and have not been clipped. Hash-bound evidence: `results/feature_metadata_all_inherited_supplement_review.json`. The separate forty-field readback is `feature_metadata_efficiency_supplement_review.json`.

The renderer now uses supplemental units and meanings in plot labels and primary Markdown while preserving the frozen scientific record. Five diagnostic tests passed. All-program real plots remain gated on terminal screening and generation finalizations.

New F09 completions enabled reduced A/B caps of three/four (`results/concurrency_reduced_ab_expansion.json`), then an F12 cap of five (`results/concurrency_f12_expansion_5.json`). Latest authorized caps are F09 28, F10 5, F11 5, F12 5, reduced A 3, reduced B 4: fifty total. These supersede prior capacity receipts.

## Interception ownership and verified documentation loader — 2026-09-27

A paired historical-source audit supports interpreting `statDef_interceptions` as interceptions thrown by the represented team, despite its canonical prefix. Across 27,403 finite paired 2010–2025 completed regular-season team-game rows, it equals the opponent's passes-intercepted field in 27,337 rows; among 18,731 rows that distinguish it from the team's own passes-intercepted field, 18,682 match the opponent. The 66 total disagreements remain explicit source-data discrepancies. Evidence: `results/inherited_interception_ownership_audit.json` binds all sixteen input CSVs. This is an empirical ownership inference, not a repair of source data or trained inputs.

The box-score supplement now gives the inferred offensive-turnover interpretation and lower-is-better direction, with the discrepancies preserved. The reusable diagnostic documentation loader verifies observational hashes as well as source implementations and rejects duplicate definitions. Six tests passed, including changed-observation, changed-code and duplicate-definition rejection. The actual loader independently accepted all 261 supplements; receipt: `results/feature_metadata_verified_loader_review.json`. Full metadata completion remains unclaimed because historical provider filtering/publication, returning-PPA denominators, all-generation semantics and final rendered diagnostics remain unfinished.

F09 C now has ten successful M2 and seven successful M4 results; three M4 cells remain active. Full C M2 median development MAE is 13.379143 / 12.920867, RMSE 16.943794 / 16.410657 and Brier 0.196414 / 0.187185 for 2024 / 2025. Evidence: `results/F09/full_c_m2_interim.json`. These design-informed results do not authorize the incomplete joint full reference or compare generations on different cohorts.

Latest verified caps are F09 24, F10 5, F11 5, F12 6, reduced A 4, reduced B 6: fifty total. Receipts: `results/concurrency_f12_expansion_6.json` and `results/concurrency_reduced_ab_expansion_2.json`. Earlier caps are superseded.
