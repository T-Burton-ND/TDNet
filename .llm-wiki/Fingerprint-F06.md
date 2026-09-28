---
type: concept
up: "[[Fingerprint-Ladder]]"
tags: [fingerprints, historical-ladder]
---

# Fingerprint F06

F06 is the historical market-free fingerprint tier: schedule graph.

## Current snapshot — 2026-09-28

All six full/reduced fingerprints have terminal matrices: 116 successes and four full A/B M2 memory failures. Reduced A (162 features) and B (170) now pass the measured MAE tolerance; their joint mean MAE deltas are −0.046495 and −0.072737 against their respective full references. Neither is a validated minimum. C (60) remains the only immutable accepted reduction. No smaller trials were launched. See the [verified paused snapshot](Nextgen-Results-Snapshot-2026-09-28) for all metrics, evidence paths, source gaps and scheduler counts as of 12:52 UTC. This supersedes earlier live-status statements; the goal is paused and existing queued cells may still dispatch.

## Information and ancestry

- **Parent:** F05.
- **Added information:** Adds schedule-graph information to F05, completing the canonical 227-source-feature market-free F6.
- **Status:** Historical 2026 scientific ladder; definition frozen. The new experiment does not rewrite its publication claims.
- **Timing:** Each dynamic state must be available before the target game; the target is the next game.
- **Matchups:** Existing team-week numeric sources pass through the reviewed matchup builder. Market input is excluded from the new F09–F12 research lineage.

## Exploratory bridge

The user authorized excluding `coach_career_mean_sp_offense` and `coach_career_mean_sp_defense` on 2026-09-27, relaxing exact preservation for the exploratory experiment. `F06_F_a` now contains the remaining 225 source features; the historical publication definition remains unchanged. `F06_F_b` adds interpretable re-expression and interactions; `F06_F_c` compresses the same information into Excel-reproducible concepts with at most five inputs. `F06_R_a/b/c` are independently reduced, with at least 60 concrete F06 survivors. F6-C/C25 is reference evidence, not a new lineage seed. The next market-free information generation is F09.

## Refreshed post-exclusion redundancy diagnostics

After the authorized two-field exclusion, refreshed Pearson/Spearman analysis of 22,390 aligned 2010–2025 team-target rows (225/235/153 features in a/b/c) found 12/12/10 feature-pair candidates at absolute correlation >=0.995 in F06_F_a/b/c. The 0.98–0.995 representative-validation bands contain 17/17/0 pairs; the 0.90–0.98 report-only bands contain 75/77/29 pairs. Current hash-verified evidence: `results/F06/F06_F_{a,b,c}/redundancy.parquet` and their provenance sidecars. These are correlated feature pairs, not removed features or pruning-pair counts. No pruning or performance conclusion is authorized by correlation alone.

## Source contract

See `configs/features/feature_ladders.yaml` and `docs/publication_2026/feature_manifests/F6.json` for exact features in the original numbering. F06's manifest has 227 names and schema hash `7679eb1eac141ce1fcc2e3690bee848b12a3a753903699c32826640af568d224`.

See also: [Fingerprint Ladder](Fingerprint-Ladder) · [Next-Generation Experiment](Next-Generation-Fingerprint-Experiment) · [Temporal Data Semantics](Temporal-Data-Semantics) · [F07](Fingerprint-F07) · [F09](Fingerprint-F09)

## First verified screening result — 2026-09-27

`F06_F_c__M4__m4_01` completed successfully in 910.37 seconds. For the revised 153-feature F06 c design, the separate development-year metrics are MAE 13.943517 / 13.638959 and Brier 0.202625 / 0.193761 for 2024 / 2025 respectively. Both years are design-informed development evidence, not unbiased holdouts. These values describe one frozen setpoint and must not be used as an architecture summary or lineage recommendation.

The run completed end-to-end source-coordinate permutation SHAP with 256 background games, 512 explanation games, seed 1701, and maximum absolute additivity error 1.0125e-13. The results collector revalidated current execution binding, fingerprint inputs, predictions, and SHAP hashes before adding the successful row to `results/nextgen_results.parquet`. Evidence: `experiments/F06/F06_F_c__M4__m4_01/result.json` and its output artifacts. At that first-result snapshot, no architecture summary was screening-usable; the full array remained active.

## First completed architecture — 2026-09-27

All ten frozen M2 configurations for the revised 153-feature `F06_F_c` completed successfully with verified SHAP and execution/input/output bindings. The collector now marks this architecture screening-usable with complete coverage. Across these ten configurations, median 2024 / 2025 development MAE is 13.386496 / 13.020017, RMSE is 17.097036 / 16.467372, and Brier is 0.197366 / 0.186576. These remain design-informed development results, not unbiased holdout estimates. Evidence: `results/nextgen_results.parquet`, row `F06_F_c__M2__summary`, backed by `experiments/F06/F06_F_c__M2__m2_01` through `m2_10` result artifacts under the experiment root.

The concurrent snapshot contains 26 successful full-screening runs and four failed runs; remaining jobs are active. M4 screening for c and both architectures for a/b are unfinished. No reduced fingerprint or lineage recommendation has been accepted.

## First real reduction trial — 2026-09-27

F06_F_c full screening is terminal with ten successful frozen configurations for each of M2 and M4. Its M4 median development MAE is 13.388646 / 12.843854 for 2024 / 2025, from `results/nextgen_results.parquet`; these are design-informed development results.

The first `F06_R_c` candidate retains exactly 60 of 153 source features, preserving reciprocal groups and all 22,390 team-target rows. Selection maximizes summed joint normalized SHAP importance over those groups at the minimum floor, using all twenty successful full-reference runs. Retained joint importance mass is 0.7689678844. This is a proposed aggressive C reduction, not an accepted accuracy tradeoff. Evidence: `fingerprints/F06_R_c/reduction_proposal.json` and hash-bound provenance under the experiment root.

SGE accepted its twenty-cell screening array as `1479077.1-20:1` with a two-task concurrency cap alongside the live F09 full array. Actual reduced-screening results and measured +0.25 MAE acceptance are pending. Receipt: `experiments/F06/reduced_c_array.submission.json`.

## Complete reduced-C M2 screening — 2026-09-27

All ten M2 configurations for the 60-feature `F06_R_c` completed successfully. On the same 22,390-row team-target corpus as full C, its median development MAE is 13.343549 / 12.937767 for 2024 / 2025, versus 13.386496 / 13.020017 for the 153-feature full C. Reduced-C RMSE is 17.056880 / 16.372434. Reduced-C Brier is 0.198063 / 0.186107: slightly worse in 2024 than full C's 0.197366, and slightly better in 2025 than 0.186576. These are actual ten-configuration architecture medians, not unbiased holdout results.

Evidence: `results/F06/reduced_c_m2_interim.json`, with hashes of all twenty compared M2 run-result files, and `results/nextgen_results.parquet` under the experiment root. M4 reduced screening remains unfinished; no joint C acceptance or lineage recommendation is claimed. Sixteen annual reduced-C references and forty-five F10 full references independently passed exact recomputation; evidence: `results/f06_reduced_c_f10_reference_readback.json`.

## First conservative A reduction trial — 2026-09-27

F06_F_a full screening is terminal with eight successful M2 configurations, two M2 memory failures, and ten successful M4 configurations. The measured M4 median development MAE is 13.301759 / 12.861404 for 2024 / 2025. Its missing M2 successes remain flagged; the at-least-three success rule does not authorize retries solely to reach ten.

The first F06_R_a proposal retains 162 of 225 features and all 22,390 team-target rows. Removed reciprocal-group members are each below uniform-share normalized importance in both architectures. The greedy group order uses conservative importance, capped at 10% cumulative removed importance in each model; measured removed importance is 0.098697948 for M2 and 0.088897415 for M4. Twenty-two preferred representatives of >=0.98 correlated pairs are protected by lower missingness, simpler inputs/equation, stronger joint importance, then earlier generation. Evidence: `results/F06/reduced_a_selection_trial_1.json` and `fingerprints/F06_R_a/reduction_proposal.json` under the artifact root.

Checked-loader validation passed on 11,195 paired games. SGE accepted twenty-cell trial array `1479192`, capped at twenty and held on F09 full array `1479074`; this reserves twenty of the forty-five slots released when F09 exits. Evidence: `experiments/F06/reduced_a_array.submission.json`. This is an unaccepted proposal, not the smallest validated representation. Screen both architectures, enforce +0.25 MAE separately in each year, and evaluate a smaller permitted candidate if this first trial passes.

Separately, one original F06 full task finished and its slot was assigned to F10, raising F10 array `1479168` from one to two concurrent tasks. The checked current bound remains fifty. F11 still reserves the two slots of reduced-C array `1479077` after that dependency completes; the queued A trial's reservation must also be counted in future capacity changes. Evidence: `experiments/F10/concurrency_update_2.json`.

## First conservative B reduction trial — 2026-09-27

F06_F_b is terminal with eight successful M2 configurations, two M2 memory failures, and ten successful M4 configurations. Its M4 median development MAE is 13.316360 / 12.883629 for 2024 / 2025. All original F06 full designs are now terminal; no failure was silently retried or discarded.

The first F06_R_b proposal retains 170 of 235 features on all 22,390 team-target rows. It applies the same pair-closed weak-under-both policy as A, protecting twenty-two preferred correlation representatives. Removed normalized importance is 0.098923071 in M2 and 0.091571615 in M4. Evidence: `results/F06/reduced_b_selection_trial_1.json` and `fingerprints/F06_R_b/reduction_proposal.json` under the experiment root. It remains unaccepted and is not claimed to be the minimum validated representation.

Checked-loader validation passed on 11,195 paired games. SGE accepted twenty-cell B trial array `1479244`, capped at twenty and held on F09 array `1479074`. Both held A/B arrays now reserve twenty slots each after F09 completes; with F12 cap one, F10 cap two, and the reduced-C/F11 two-slot dependency chain, the post-F09 maximum is forty-five. Future expansions must account for both A/B reservations. Receipt: `experiments/F06/reduced_b_array.submission.json`. Trial annual references are being built.

## Reduction arrays active and independent floor acceptance — 2026-09-27

All F09 full cells have started. Freed F09 slots were reassigned after lowering caps: reduced A `1479192` is confirmed running at cap one, and reduced B `1479244` is confirmed running at cap two. Their original dependency holds are superseded. Evidence: `results/concurrency_reduction_start.json` and `experiments/F06/reduced_b_concurrency_start.json`. F09's cap is now 42; F11 and final reduced C each have cap one. These changes keep the total authorized maximum at fifty.

Reduced C currently has ten successful M2 and nine successful M4 results, with its final cell still active. The new `nextgen_acceptance.accept_floor_reduction` can issue an immutable independent parent receipt only after a complete matrix passes measured acceptance and every concrete-generation floor is reached. The reader rechecks hashes and recomputes acceptance; non-floor passing trials cannot use this path. Eighteen focused tests passed. No real acceptance receipt exists yet, and A/B still require smaller-candidate exploration if their initial trials pass.

## Reduced C accepted at the feature floor — 2026-09-27

F06_R_c completed all ten M2 and ten M4 configurations successfully. The immutable `results/F06/F06_R_c/acceptance.json` verifies the full and reduced input/output evidence and accepts the 60-feature representation against the 153-feature full C reference. Development median MAE for 2024 / 2025 is 13.343549 / 12.937767 (M2) and 13.413353 / 12.823703 (M4). Relative to full C, the four changes are -0.042948, -0.082251, +0.024707, and -0.020151; joint mean change is -0.030161. The M4 2024 worsening is retained in the report. These are design-informed development results, not unbiased holdouts. All concrete feature floors are reached, permitting independent progressive-parent authorization; A/B and generation-wide recommendations remain unfinished.
