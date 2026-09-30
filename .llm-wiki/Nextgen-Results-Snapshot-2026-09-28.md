---
type: synthesis
up: "[[Next-Generation-Fingerprint-Experiment]]"
tags: [nextgen, experiments, results, paused]
---

# Nextgen Results Snapshot — 2026-09-28

Verified results available at 2026-09-28T12:52:56.970123+00:00 for the [next-generation experiment](Next-Generation-Fingerprint-Experiment), recorded when the user requested no new experiments and a wiki summary.

Later completion: [the 2026-09-29 results table](Nextgen-Completed-Model-Results-2026-09-29) records all terminal cells and verified prediction hashes. This page preserves the earlier frozen cutoff.

## Measured lessons — 2026-09-28

Subsequent read-only analysis of this frozen snapshot verified evaluation IDs and paired configurations. F06/F09/F10 A share their evaluation games; F11 A/F12 A also share theirs. On intersecting F10/F11 A games, F11 M2 is worse in both years. The earlier general coverage warning does not rule out these measured comparisons. See [empirical lessons and paired evidence](Nextgen-Empirical-Lessons-2026-09-28) for exact values, scope and limits. No new experiments were run.

## Pause and evidence boundary

The experiment goal is paused by explicit user instruction. This update refreshed existing result evidence and calculated descriptive comparisons; it launched no experiment, changed no scheduler setting, and issued no new acceptance receipt. Existing submitted arrays remain active and their queued cells can still dispatch. This dated snapshot supersedes earlier live-status paragraphs; it is not a promise that the scheduler remains unchanged afterward.

Artifact root: `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen`. Frozen snapshot: `results/snapshots/2026-09-28-paused/snapshot.json`, `results.parquet`, and `scheduler.xml`. The snapshot records source run paths/hashes, verified architecture summaries, MAE tolerance checks and scheduler readback. Frozen Parquet SHA-256: `8f8935f77bd39a6b185642b9bf20fca8f87293cd5387c0f44b3c9ffccee69dd4`. Main repository HEAD: `fd2f76a4ba4281be7681b0dff41098331e3bb08e`; the selector/progressive helper edits discussed below remain uncommitted and were not used to launch new trials.

The collector verified successful runs against input, manifest, execution and output bindings. Of 360 cells already submitted across 18 materialized fingerprints, 241 succeeded, 4 failed, 20 were running/incomplete and 95 were queued without result rows. No generation recommendation file exists. The only immutable reduction acceptance is F06_R_c.

| Existing array | Running | Queued | Cap last authorized |
|---|---:|---:|---:|
| F10 full, 1479168 | 5 | 38 | 5 |
| F11 full, 1479184 | 9 | 25 | 9 |
| F12 full, 1479229 | 6 | 32 | 6 |

F06 full/reduced and F09 full arrays have finished. Earlier caps including F09=20 and reduced A/B=4/6 are historical, not additional live capacity. No cap was increased during this summary.

## Protocol and interpretation

M2 is spline ridge; M4 is histogram gradient boosting. Each architecture has ten frozen configurations with seed 1701, training years 2010–2023 and design-informed development years 2024–2025. These development metrics are not unbiased holdout evidence. No 2026 data informed design. Source-coordinate SHAP uses 256 background and 512 explanation rows per relevant cell. Terminal screening is usable when all ten attempts are terminal and at least three succeeded; failed cells remain disclosed. The four F06 full A/B failures were the two largest M2 configurations per design, previously diagnosed as memory failures. Their summaries use eight successes, not ten.

The user authorized dropping only `coach_career_mean_sp_offense` and `coach_career_mean_sp_defense`, and relaxing exact preservation for exploratory artifacts. Historical 227-feature publication definitions remain separate. The attached launch document supplies experiment context; later direct user changes control the work.

## Full input coverage

Counts are actual assembled full fingerprints, with paired-game cohorts that shrink as source requirements increase. Do not subtract cross-generation metrics as a causal improvement estimate without a common-cohort evaluation.

| Generation | Features a / b / c | Team-target rows | Paired games | Terminal screening at snapshot |
|---|---|---:|---:|---|
| [F06](Fingerprint-F06) | 225 / 235 / 153 | 22,390 | 11,195 | All full and reduced designs |
| [F09](Fingerprint-F09) | 263 / 281 / 183 | 22,302 | 11,151 | All full designs |
| [F10](Fingerprint-F10) | 319 / 341 / 210 | 21,022 | 10,511 | Full A M2 only |
| [F11](Fingerprint-F11) | 326 / 349 / 215 | 17,018 | 8,509 | Full A both models |
| [F12](Fingerprint-F12) | 339 / 365 / 224 | 14,716 | 7,358 | Full A both models |

F10 has 17 successes and 5 running cells; F11 has 26 successes and 9 running; F12 has 22 successes and 6 running. Unstarted cells are excluded from the result table, not counted as failures. F09 has 60 successes. F06 has 116 successes and 4 failures across full and reduced trials.

## Terminal architecture results

All entries below are medians across successful frozen configurations after the full ten-cell architecture matrix became terminal. MAE and RMSE are margin points; Brier is a probability score. Each year pair is 2024 / 2025. These are per-fingerprint development cohorts as above. R variants use the same F06 paired cohort as their own full reference. No partial-architecture median is shown.

| Fingerprint | Model | Successes / 10 | MAE 2024 / 2025 | RMSE 2024 / 2025 | Brier 2024 / 2025 |
|---|---|---:|---|---|---|
| F06_F_a | M2 | 8 | 13.473046 / 13.080402 | 17.176109 / 16.507346 | 0.199074 / 0.187535 |
| F06_F_a | M4 | 10 | 13.301759 / 12.861404 | 16.856630 / 16.345939 | 0.197227 / 0.185040 |
| F06_F_b | M2 | 8 | 13.481029 / 13.127555 | 17.200706 / 16.621841 | 0.198890 / 0.188546 |
| F06_F_b | M4 | 10 | 13.316360 / 12.883629 | 16.855754 / 16.383259 | 0.197790 / 0.185364 |
| F06_F_c | M2 | 10 | 13.386496 / 13.020017 | 17.097036 / 16.467372 | 0.197366 / 0.186576 |
| F06_F_c | M4 | 10 | 13.388646 / 12.843854 | 16.905475 / 16.391149 | 0.199042 / 0.184871 |
| F06_R_a | M2 | 10 | 13.434764 / 12.985210 | 17.116033 / 16.465128 | 0.198570 / 0.187151 |
| F06_R_a | M4 | 10 | 13.310210 / 12.800448 | 16.863108 / 16.302223 | 0.196817 / 0.184644 |
| F06_R_b | M2 | 10 | 13.427415 / 13.011444 | 17.123448 / 16.519263 | 0.198330 / 0.187494 |
| F06_R_b | M4 | 10 | 13.272906 / 12.805861 | 16.810478 / 16.329111 | 0.197171 / 0.184465 |
| F06_R_c | M2 | 10 | 13.343549 / 12.937767 | 17.056880 / 16.372434 | 0.198063 / 0.186107 |
| F06_R_c | M4 | 10 | 13.413353 / 12.823703 | 17.001625 / 16.331589 | 0.199583 / 0.184639 |
| F09_F_a | M2 | 10 | 13.448118 / 13.037092 | 17.026788 / 16.503340 | 0.198052 / 0.188962 |
| F09_F_a | M4 | 10 | 13.133993 / 12.747902 | 16.633669 / 16.175480 | 0.194089 / 0.183826 |
| F09_F_b | M2 | 10 | 13.478394 / 13.184257 | 17.080554 / 16.677173 | 0.197692 / 0.191563 |
| F09_F_b | M4 | 10 | 13.122054 / 12.763479 | 16.619963 / 16.180306 | 0.193508 / 0.184828 |
| F09_F_c | M2 | 10 | 13.379143 / 12.920867 | 16.943794 / 16.410657 | 0.196414 / 0.187185 |
| F09_F_c | M4 | 10 | 13.214108 / 12.728329 | 16.708442 / 16.194766 | 0.195589 / 0.184597 |
| F10_F_a | M2 | 10 | 13.393119 / 12.896856 | 16.934777 / 16.415873 | 0.195496 / 0.186897 |
| F11_F_a | M2 | 10 | 13.706990 / 12.537048 | 17.255453 / 16.133479 | 0.199017 / 0.191761 |
| F11_F_a | M4 | 10 | 13.141755 / 12.206475 | 16.620010 / 15.700044 | 0.193172 / 0.184101 |
| F12_F_a | M2 | 10 | 13.823579 / 12.729218 | 17.404374 / 16.319428 | 0.201160 / 0.194312 |
| F12_F_a | M4 | 10 | 13.186888 / 12.393033 | 16.708435 / 15.873103 | 0.193836 / 0.186658 |

Classification medians below use each metric's eligible evaluation subset; they are fractions, not percentages. ATS/chalk/upset evaluation uses the separate market sidecar; market features are excluded from model inputs. They are secondary diagnostics, not finalized model-selection decisions.

| Fingerprint | Model | Winner accuracy 2024 / 2025 | ATS accuracy 2024 / 2025 | Chalk accuracy 2024 / 2025 | Upset accuracy 2024 / 2025 |
|---|---|---|---|---|---|
| F06_F_a | M2 | 0.691689 / 0.717305 | 0.493836 / 0.508772 | 0.877589 / 0.877224 | 0.223256 / 0.266667 |
| F06_F_a | M4 | 0.697721 / 0.714663 | 0.501370 / 0.513495 | 0.893597 / 0.885231 | 0.220930 / 0.220513 |
| F06_F_b | M2 | 0.694370 / 0.714663 | 0.491781 / 0.508772 | 0.879473 / 0.871886 | 0.225581 / 0.266667 |
| F06_F_b | M4 | 0.695710 / 0.712682 | 0.502740 / 0.516869 | 0.893597 / 0.879004 | 0.213953 / 0.217949 |
| F06_F_c | M2 | 0.694370 / 0.719287 | 0.494521 / 0.509447 | 0.880414 / 0.882562 | 0.223256 / 0.271795 |
| F06_F_c | M4 | 0.693700 / 0.718626 | 0.496575 / 0.515520 | 0.891714 / 0.887900 | 0.204651 / 0.228205 |
| F06_R_a | M2 | 0.693700 / 0.716645 | 0.493151 / 0.508097 | 0.874765 / 0.880783 | 0.234884 / 0.258974 |
| F06_R_a | M4 | 0.699732 / 0.714663 | 0.500685 / 0.521592 | 0.892655 / 0.887011 | 0.225581 / 0.215385 |
| F06_R_b | M2 | 0.689678 / 0.722589 | 0.493151 / 0.508097 | 0.873823 / 0.881673 | 0.232558 / 0.276923 |
| F06_R_b | M4 | 0.698391 / 0.716645 | 0.503425 / 0.519568 | 0.896422 / 0.887011 | 0.213953 / 0.210256 |
| F06_R_c | M2 | 0.694370 / 0.722589 | 0.486301 / 0.524291 | 0.884181 / 0.889680 | 0.218605 / 0.241026 |
| F06_R_c | M4 | 0.689678 / 0.718626 | 0.490411 / 0.511471 | 0.885122 / 0.891459 | 0.204651 / 0.220513 |
| F09_F_a | M2 | 0.678284 / 0.715324 | 0.484932 / 0.509447 | 0.865348 / 0.878114 | 0.218605 / 0.251282 |
| F09_F_a | M4 | 0.697721 / 0.711361 | 0.498630 / 0.521592 | 0.889831 / 0.881673 | 0.223256 / 0.215385 |
| F09_F_b | M2 | 0.680295 / 0.718626 | 0.491781 / 0.502024 | 0.864407 / 0.873665 | 0.227907 / 0.271795 |
| F09_F_b | M4 | 0.693029 / 0.714663 | 0.503425 / 0.523617 | 0.886064 / 0.887011 | 0.213953 / 0.220513 |
| F09_F_c | M2 | 0.684316 / 0.716645 | 0.489041 / 0.506073 | 0.872881 / 0.882562 | 0.218605 / 0.251282 |
| F09_F_c | M4 | 0.686327 / 0.715984 | 0.489726 / 0.520918 | 0.880414 / 0.887900 | 0.209302 / 0.220513 |
| F10_F_a | M2 | 0.681635 / 0.723910 | 0.497945 / 0.518219 | 0.868173 / 0.874555 | 0.234884 / 0.289744 |
| F11_F_a | M2 | 0.678914 / 0.709765 | 0.481209 / 0.533210 | 0.853604 / 0.875000 | 0.250000 / 0.278523 |
| F11_F_a | M4 | 0.694888 / 0.715190 | 0.491830 / 0.546125 | 0.886261 / 0.893564 | 0.239011 / 0.231544 |
| F12_F_a | M2 | 0.672524 / 0.698011 | 0.490196 / 0.531365 | 0.847973 / 0.860149 | 0.255495 / 0.261745 |
| F12_F_a | M4 | 0.691693 / 0.714286 | 0.488562 / 0.514760 | 0.872748 / 0.905941 | 0.252747 / 0.208054 |

## F06 reductions: measured pass versus minimum acceptance

The read-only `measured_acceptance` calculation confirms all three existing trials pass their MAE tolerance. Its `accepted: true` describes the numeric tolerance check; it does not create an immutable parent receipt or prove a minimum representation. A/B require each model/year change ≤+0.25 points; C requires the equal-model/year mean change ≤+0.25. Negative deltas improve MAE.

| Trial | Features full → trial | M2 Δ2024 / Δ2025 | M4 Δ2024 / Δ2025 | Joint mean delta | Formal state |
|---|---|---|---|---|---|
| F06_R_a | 225 → 162 | -0.038282 / -0.095192 | +0.008451 / -0.060956 | -0.046495 | Passing first trial; minimum unproven |
| F06_R_b | 235 → 170 | -0.053614 / -0.116111 | -0.043454 / -0.077768 | -0.072737 | Passing first trial; minimum unproven |
| F06_R_c | 153 → 60 | -0.042948 / -0.082251 | +0.024707 / -0.020151 | -0.030161 | Accepted at floor |

F06_R_c removes 93 of 153 features (60.8%) and reaches the required 60-feature F06 floor. Its immutable receipt is `results/F06/F06_R_c/acceptance.json`, SHA-256 `6d2a55a6855369a7b8599652578f225b4339968aec35680441c5c5add737a23c`. Reduced C worsens M4 2024 MAE by 0.024707 and Brier by 0.000541; it also worsens M2 2024 Brier by 0.000697. The small joint MAE improvement does not erase those negatives. Reduced A likewise worsens M4 2024 MAE by 0.008451. Reduced B improves all four MAE medians. A/B compare ten candidate successes against eight successful M2 reference configurations, with incomplete coverage explicitly flagged.

A/B proposals removed only pair-closed features weak under both models; removed normalized importance stayed below 0.10 per architecture and 22 preferred correlation representatives per design were protected. Proposal evidence: `results/F06/reduced_a_selection_trial_1.json` and `reduced_b_selection_trial_1.json`. No further candidates were created after the pause; minimum searches and generation finalization remain unfinished.

## Later-generation findings and limits

F09 full A/B/C all completed with ten successes per architecture. Within its shared full cohort, C has the lowest M2 MAE in both development years; M4 B has the lowest 2024 MAE and M4 C the lowest 2025 MAE. All three designs' within-architecture MAE differences are inside the protocol's 0.5-point practical-tie band. No final recommendation is justified without the remaining reduction and secondary-metric selection work.

F10 full A M2 and F11/F12 full A both architectures now have terminal summaries. F11/F12 smaller cohorts differ from earlier generations, so their lower 2025 M4 MAEs cannot establish the benefit of staff or unit features. B designs remain partial and C designs have not begun in F10–F12 at this snapshot.

The F09_PR_c artifact remains an **unpruned, unscreened pool**: 60 accepted F06 survivors plus all 30 new F09 features, 22,302 team-target rows. It is not one of the 18 submitted fingerprint artifacts. No later LR/PR trial or acceptance exists.

The earlier F09 C correlation audit completed: 43 reported pairs (33 report-only, 10 automatic-redundancy candidates), minimum 100 joint rows, on the 22,302-row full artifact. The summary independently rechecked its data, manifest and output hashes. Evidence: `results/F09/F09_F_c/redundancy.parquet` and `redundancy.provenance.json`. Every row retains `removal_authorized: false`; no correlation-based deletion was performed.

Before the pause, the new joint-importance floor selector exactly reproduced all 60 accepted F06 C survivors, retaining 0.7689678843558778 of normalized equal-M2/M4 source importance. Evidence: `results/F06/joint_floor_selector_reproduction.json`. It is a proposal generator, not an acceptance rule. Related progressive-helper changes allow an unchanged pool only when it is already a strict subset of its verified full reference. Twenty-one focused tests passed before the pause. These six implementation/test files remain uncommitted and no new job has used them.

## Data, references and documentation completed

- Eighteen materialized fingerprints have 273 independently recomputed, hash-verified annual average-team references. Each uses strictly earlier source seasons; references can target 2026 without using 2026 design data. Receipts include `results/f06_reduced_c_f10_reference_readback.json` and `results/reduced_ab_f11_f12_reference_readback.json`. Future LR/PR fingerprints need their own references.
- Acquisition ledger `results/preflight/cfbd_request_manifest_v1.jsonl` last audited at 1,653 successful-complete, 240 skipped-existing-complete and 64 failed-final records, with 1,858 reserved attempts against a 20,000 cap. Sixty early PPA/success failures with empty responses were not proven structural absences. Live provider quota was not refreshed for this summary; no acquisition was started.
- Six source-reviewed supplements cover all 261 inherited names in the original placeholder inventory. This closes that inventory, not the all-generation semantic audit. `results/feature_metadata_verified_loader_review.json` binds verified loading. Travel is a log(1+km) difference, not miles; prior score averages use prior outcomes; signed returning-PPA shares are not probabilities; temporal count jumps reconstruct batch averages, not individual games.
- The 2010–2025 interception ownership audit covered 27,403 finite paired rows: 27,337 matched the opponent's passes-intercepted field and 66 disagreed. This supports interpreting the canonical `statDef_interceptions` field as interceptions thrown, despite its name; disagreement rows and original model inputs were retained. Evidence: `results/inherited_interception_ownership_audit.json`.
- Required Q4-minus-Q1 offense/defense diagnostic plots already exist. Offense: 22,132 rows, Pearson −0.0803, Spearman −0.0763; defense: 22,099 rows, Pearson 0.1005, Spearman 0.0965, on their available diagnostic samples. These are descriptive correlations, not selected-model gains. Final consensus diagnostics are not generated; the renderer correctly refuses incomplete generation finalizations and preserves the existing index.

## Source gaps and unfinished deliverables

F10 lacks verified historical Week-0 roster membership/physical units, preseason returning-production/churn, transfer timing/current membership and defensive-production denominators. Observed usage overlap is not roster continuity; recruiting commitments are not current membership; observed experience is not complete career history. F11 uses prior-season staff and lacks verified current assignments, changes, interim/coordinator coverage. F12 covers observed offensive rooms and special teams, but lacks offensive-line/blocking and defensive-front/linebacker/secondary participation, plus explicit opposing-unit matchup coverage. These limitations are recorded in `results/f10_source_scope.json`, `f11_source_scope.json`, and `f12_source_scope.json`; supported subsets do not fulfill all requested source coverage.

Historical provider filters/publication timing and returning-PPA denominator reconstruction remain unresolved. The dynamic reporting lag is reconstructed as 48 hours after the latest league kickoff in a source week, not verified historical publication timestamps. The inherited baseline may omit multiple games for a team/week. Static roster/coach semantics are frozen under the user's source authorization, not proof of historical availability.

Unfinished: F10–F12 full matrices; minimum A/B reduction searches; all 24 later LR/PR fingerprints and their annual references; five generation finalizations/recommendations; full all-generation semantic audit; and global consensus diagnostics capped at 1,000 advanced identities, including both mandatory quarter differences. There is no final program winner. Resuming scientific work requires a later user instruction; this snapshot does not authorize continuation.

See also: [Experiment](Next-Generation-Fingerprint-Experiment) · [F06](Fingerprint-F06) · [F09](Fingerprint-F09) · [F10](Fingerprint-F10) · [F11](Fingerprint-F11) · [F12](Fingerprint-F12).
