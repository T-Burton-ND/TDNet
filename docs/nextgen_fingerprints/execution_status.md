# Next-generation execution status


## Authorized coach SP exclusion — 2026-09-27

The user resolved the source-policy conflict: “Remove those fields and relax exact preservation.” The exploratory baseline now excludes `coach_career_mean_sp_offense` and `coach_career_mean_sp_defense` before design formulas or metadata are constructed. The historical publication source and its 227-feature manifest remain unchanged. This decision supersedes the original exact-preservation requirement for the new experiment only.

F06 full a/b/c rebuilt with 225/235/153 features on 22,390 team-target rows. The source audit passed after removal. The model loader rejects either excluded field in columns or formula inputs. All 88 nextgen tests passed (five nonfatal warnings); this is implementation validation, not predictive-performance evidence. Pre-decision baseline-dependent artifacts and annual references were moved to `scratch/superseded_before_coach_sp_exclusion` within the experiment artifact root, so prior frozen references are not silently reused. F09 rebuilding and screening submission are tracked separately.

## Screening submission after exclusion

The 60-task F06 full-screening submission was attempted after successful checked-loader reads of all three revised designs. SGE rejected it: `job does not provide an AFS token`. The current session's `klist -s` returned 1, and `qstat` showed no jobs. No receipt or model result was created. The user has been asked to renew cluster Kerberos/AFS authentication through the normal login workflow; no password or token is requested in chat. This is a cluster authentication failure, not an automatic approval rejection. The scientific source-policy conflict is resolved.

## Launch provenance

- User authorized review and execution through F12 generation, training, and all requested results/metrics on 2026-09-27.
- Branch: `experiment/nextgen-fingerprints-v1`.
- Starting HEAD: `cc3038f1bb293e127e69d18d00647bf6d37b1e5a`; working tree clean; fetched origin and fast-forward check reported already up to date.
- Full user-supplied requirements: [overnight launch prompt](overnight_launch_prompt.md).
- Artifact root: `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/`.
- Python: `/users/tburton2/.conda/envs/gridiron/bin/python`.

## Verified milestones

- Launch preflight passed all 42 lineage definitions, 10 M2 and 10 M4 smoke fits, pair checks, temporal guards, and 2026 quarantine checks.
- Targeted nextgen tests: 27 passed in 5.38 seconds.
- Stage A: 16/16 fresh schedule requests completed for 2010–2025, plus one quota request. Shared ledger count advanced from 4 to 21 reserved outbound attempts. Evidence: `results/preflight/stage_A_latest.json` under the artifact root.
- Fresh schedule acquisition authenticated successfully without exposing the credential.
- Post-Stage-A preflight verified 240 legacy team-game partitions, 257 total reusable partitions, and 1,500 remaining planned requests before coverage corrections.
- Acquisition inventory corrected against CFBD's official availability page on 2026-09-27. Known era gaps are excluded; anomalous responses remain reviewed failures rather than complete/no-data claims.
- Optional `/coaches/tenures` and `/passing/teams/games` returned HTTP 400 for documented year queries. Their observed request records retain explicit failed dispositions; remaining endpoint requests are deferred. Coach-season records and structured plays/box scores supply alternatives.
- F06 exact-column source alignment produced 22,390 team rows / 11,195 paired FBS games through 2025 in `canonical/f06_aligned.parquet`. Its provenance sidecar records schedule/data/source/manifest hashes and year coverage. The 48-hour league-week reporting lag is a reconstruction assumption, not archived availability evidence. The source-semantics audit is still required before training.
- Four baseline tests cover canonical value preservation, fresh outcomes, paired identities, bye states, quarantine and league-wide reporting cutoffs. Five microstructure tests cover pre-play score reconstruction, Q4 garbage logic, middle eight, structured-only inputs, and rush/dropback reciprocal sufficient statistics.
- Combined acquisition/contract/baseline/microstructure check: 25 tests passed after implementation. Two pandas fragmentation warnings are nonfatal.
- Source audit observation: all ten inherited roster/coach columns are constant within team-season in the aligned 2010–2025 corpus. Comparing canonical score means to every fresh scheduled prior game found 646 disagreements among 21,176 dynamic rows. One inspected example is Hawai'i 2016: the fresh schedule has two Week-1 games, whereas the historical weekly fingerprint omits a game contribution. Preserve the exact baseline and investigate/document inherited coverage rather than silently claim fresh-schedule equivalence.
- Play score observation: game 401754373's field-goal row already reports the resulting three points. F09 therefore converts offense/defense scores to home/away scores, orders by structured drive/play numbers, and shifts the scoreboard to obtain pre-play state. No free-text parsing is used.
- Main commit `f5741ac` saves launch/code/tests/wiki progress locally. Push is pending explicit confirmation after automatic approval review rejected external publication twice despite the launch document's push instruction. Do not retry pushing until that confirmation arrives.

- Stage C completed 297/297 pending requests at 2026-09-27T17:06:35Z; the global ledger reached 850 reserved attempts. Source: `results/preflight/stage_C_latest.json`. Stage D acquisition subsequently started and remains incomplete. F09 materialization and full assembly are verified below.
- Current design/screening/F09/microstructure tests: 16 passed in 8.56 seconds. SHAP additivity tests use synthetic fixtures, not football performance results.
- F06 full a/b/c artifacts and the 60-task full-screening plan exist. No model array has been submitted: the inherited coaching SP+ postseason-policy conflict remains a training gate.

## Resume commands and current implementation boundary

Run Python from the path above, with `PYTHONPATH=src`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MPLCONFIGDIR=/tmp/tdnet-mpl` where needed. Group-storage writes, network and scheduler commands need the host execution permission context.

- `scripts/nextgen_preflight.py` refreshes the manifest after reviewed inventory changes.
- `scripts/nextgen_cfbd_acquire.py --execute-stage B` resumes only pending verified-manifest requests; continue C/D afterward.
- `scripts/nextgen_review_request.py` records explicit evidence-backed failure/unavailability dispositions without promoting incomplete data to complete.
- `python -m gridiron_ml.experiments.nextgen_baseline` rebuilds the F06 alignment artifact; this is not training.
- `nextgen_microstructure.py` implements past-game structured play flags and additive statistics. The F09 builder and F06 design/screening/SHAP runner are implemented. Full-fingerprint assembly now supports F09–F12 with inherited canonical-value verification. Reduction/finalization orchestration and complete F10–F12 feature coverage remain incomplete.

## Remaining completion audit

All unchecked items remain required, even if an intermediate run produces useful results.

- [ ] Fresh-schedule preflight, stages B/C/D, coverage and anomaly dispositions, Stage E sample audit and full/subset/skip decision.
- [ ] F06 full a/b/c, independent R a/b/c, exact canonical 227-column F06_F_a.
- [ ] F09 full/LR/PR a/b/c, required rushing pair and quarter slope investigation.
- [ ] F10 full/LR/PR a/b/c, frozen roster, production/usage/identity matching, matching coverage.
- [ ] F11 full/LR/PR a/b/c, new coaching history and documented thin-family exceptions if needed.
- [ ] F12 full/LR/PR a/b/c, all unit groups including special teams and reviewed opposing-unit pairings.
- [ ] Every generation barrier: 10 frozen setpoints per architecture, common end-to-end permutation SHAP, correlations, pair-closed reductions, lineage comparison and recommendation.
- [ ] Checked canonical artifacts and manifests with schedule, data, manifest hashes and documented availability cutoffs; no target/postseason/2026/market leakage.
- [ ] Ultra-wide results Parquet with explicit separate 2024/2025 metrics, run/summary/recommendation rows, failures and provenance.
- [ ] Up to 1,000 consensus-ranked advanced-feature PNG/Markdown diagnostics total, including polished Q4-minus-Q1 future margin and empirical win plots.
- [ ] Frozen annual average-team references, including 2026 reference derived only from completed states through 2025.
- [ ] Generation wiki designs, results, recommendations, index/log and commits.
- [ ] Final measured quota/storage/coverage report and artifact links; code/config commits pushed.
- [ ] Morning checklist for M1/M2/M3/M4/M5/M10 on recommended fingerprints.

F13–F15 design-only proposals are optional after F12 evaluation; no F13–F15 training is authorized.

## F09 artifact verification and F10 primitives

- F09 materialization completed: 22,640 rows for each new-family design, with 38/46/30 features. Independent `NextgenModelBoundary.from_canonical` reads passed. Mean missing-cell fractions: 0.0102/0.0183/0.0112. Full assembly is verified below; LR/PR assembly and training remain incomplete.
- F10/F12 structured box-score parser and cohort-constrained identity matcher added. Two tests passed; one acquired 55-game partition yielded 7,085 additive observations, no malformed selected values and no missing athlete IDs. This partition contained no tackle/sack category; do not infer zero defensive production from its absence.

## Player and coaching source preparation

- `scripts/nextgen_prepare_players.py` completed 245 partitions / 3,020,831 additive observations. These are not model-ready rows. Five player primitive tests pass.
- `scripts/nextgen_prepare_coaches.py` rebuilt 1,858 coach-team seasons across 348 coaches using only fresh regular-season scores; two coaching tests pass. Current-coach assignment availability remains unverified.
- Player-game PPA year-only acquisition failed HTTP 400. OpenAPI documents a week-or-team requirement; inventory now uses regular-season weekly partitions, preflight passed, and Stage D resumed.
- F12 position-resolved unit allocation is implemented with two passing tests; complete canonical F12 states and evaluation remain unfinished.

## F09 full assembly and sample gate

- F09_F_a/b/c assembled with 22,302 rows each and 265/283/185 features; 44 F06 parent target games lack paired F09 coverage. New values are checked against canonical family values at assembly and model loading.
- F10 usage a/b/c independent model-boundary reads passed: 22,646 rows each; average missing-cell fractions 0.0466/0.0439/0.0466.
- Stage E now has a bounded one-to-eight-game diagnostic sample mode; bulk acquisition still requires the separate audit/value/quota/storage gate. Eleven acquisition tests pass. Four Stage E sample game requests subsequently completed; bulk acquisition was skipped after the measured coverage/value audit. See `results/preflight/plays_stats_sample_audit.json` and `plays_stats_approval.json`.

## Current verified artifact scope — 2026-09-27

- F10 usage: 22,646 rows per design, with 16/17/8 features. F10 recruiting history: 21,368 rows per design for 2011–2025, with 16/17/8 features. These families passed independent canonical readback. Recruiting history excludes the target class and represents commitments, not current roster membership; historical availability is reconstructed, not archived.
- F10 observed experience: materialization completed with 22,646 rows per design and 8/9/3 features. Independent canonical readback passed for all three designs. Four focused tests passed. Counts describe acquired prior player-game records, not career participation or roster class year.
- F11 previous-staff history: 17,296 rows per design, with 7/8/5 features; all canonical readbacks passed. This does not establish current-coach assignment.
- F12 special teams: 22,646 rows per design, with 7/8/6 features; all canonical readbacks passed. Remaining position units and explicit opposing-unit comparisons are unfinished.
- Six F06/F09 full fingerprints each have 16 annual average-team references for 2011–2026 (96 files). Build completed; readback verified source/reference year ordering, identity, and support keys. Equal team-within-season then equal-season weighting is preserved. No 2010 reference is fabricated from absent earlier history. Later/reduced fingerprints need their own references.
- Required offensive/defensive Q4-minus-Q1 future-margin and empirical-win diagnostic PNG/Markdown artifacts exist. Remaining consensus-SHAP-ranked diagnostics await model runs.
- No real screening model has run. No measured performance/recommendation table exists. Exact F06 contains two inherited coach-career SP+ fields whose seasonal source includes postseason; training remains gated pending the user's precedence decision between exact preservation and the no-postseason invariant.
- Repository changes are committed locally. Push remains pending after the prior automatic approval rejection; no retry without the requested confirmation.

The checklist above remains unchecked where a whole requirement is incomplete. Partial families, successful synthetic tests, and reference construction do not establish F12 completion or predictive performance.

- F12 observed offensive rooms: 18,516 rows per design, with 6/8/3 features; all independent canonical readbacks passed. Remaining line/defensive units and opposing-unit matchup design are unfinished.
- Stage E audit completed for four games across 2013/2017/2021/2025. All returned below cap, but one game had incomplete rushing attribution and multiple samples had yardage disagreements. Autonomous decision: skip bulk; retain samples/audit, make no participation or position inference.

## Focused integration verification — 2026-09-27

- `pytest -q tests/test_nextgen*.py` completed with 86 passed and five warnings in 13.80 seconds. This checks implemented source/temporal/manifest, feature, assembly, screening, reduction, reporting, and ranking logic on test fixtures; it is not trained football-model performance evidence.
- Weekly player-success acquisition completed the requested 2013–2025 sequence. The annual endpoint's 2012 empty HTTP 200 remains a reviewed failed request, not complete coverage; its remaining 13 requests completed successfully. Further Stage D endpoints remain in progress.
- F10 observed-continuity materialization and independent canonical readback completed: 22,636 rows per design, 16/17/8 features. It measures dynamic observed-use overlap; preseason roster retention remains unverified.

## Passing/rushing game query correction — 2026-09-27

The cached official OpenAPI 5.30.0 parameter descriptions resolve the earlier HTTP 400s: player and team passing/rushing game endpoints require a team or week alongside the year. Their 2025 plans now use weekly regular-season requests. Original failed year-only identities remain preserved; new request identities require refreshed preflight before execution. Eleven acquisition tests passed after this inventory correction. These failures are query errors, not evidence of absent provider data.

- Corrected 2025 team/player passing/rushing game acquisition completed: 16 successful weekly requests per route, 64 total. Current manifest/ledger inspection found five pending team-box partitions (gap acquisition started), 44 unattempted early player-PPA weeks, and 14 unattempted 2012 player-success weeks. No unresolved needs-review/partial/retryable status was present at that inspection; planned requests remain incomplete.

## Canonical component coverage audit — 2026-09-27

`scripts/nextgen_feature_coverage.py` audited the 24 existing canonical component families against their recorded data and manifest hashes, writing `results/canonical_family_coverage.json` in the experiment artifact root. Across the 2010–2023 training window, no feature in these built families is entirely missing. This does not establish complete fingerprint coverage, temporal validity, or training authorization.

The year-by-year audit exposes left-truncation: observed-continuity design a has 1,282 target rows in 2010 but every feature value is missing; recruiting and previous-staff rows begin in 2011; offensive-room rows begin in 2013. For 2025, previous-staff design a covers 1,112 team-target rows, versus 1,524 in player usage, so eventual paired assembly must explicitly report its row exclusions. Missingness and absent rows remain distinct in the report. No model performance was measured.

## Advanced team-detail suitability check — 2026-09-27

Measured all 16 cached weekly partitions for each corrected team passing/rushing game route. Each contains 1,650 game-team rows, all in 2025, with no duplicate game-team keys. Thus these acquired routes provide no 2010–2023 training observations and cannot by themselves supply trainable F12 line/defensive states.

Passing location coverage: offense 20,579 of 50,403 eligible attempts; defense 20,513 of 50,223. Rushing direction coverage: offense 23,424 of 55,382 eligible attempts; defense 23,205 of 50,617. These are sums of provider availability counters over the cached 2025 corpus, not verified position attribution or predictive metrics. Zero-filled directional detail with no available attempts must not be interpreted as measured zero production. No new unit features were materialized from these routes.

## Post-exclusion readback and acquisition completion — 2026-09-27

Independent readback verified all 96 regenerated annual references for six F06/F09 full fingerprints over 2011–2026: matching current data/manifest hashes, exact feature/support keys, strictly prior source seasons, and absence of both excluded coach SP fields. All three rebuilt F09 full fingerprints also passed the checked model loader, with 263/281/183 features and 11,151 paired target games each. No trained performance follows from these checks.

The current acquisition manifest now has 1,653 `success_complete`, 240 `skipped_existing_complete`, and 64 `failed_final` requests, with no pending or unresolved requests. All 60 early-history player-PPA/player-success requests in that manifest are reviewed failures, not proven structural absences. The bounded probe process exited successfully; it supplied no new early-history rows requiring offensive-room rebuilding.

The scientific baseline gate is resolved, but a fresh ticket check still returned 1 and the scheduler has no jobs. F06 submission awaits renewal of the missing Kerberos/AFS credentials cache. No credentials should be supplied in chat.

## F06 screening started — 2026-09-27

Cluster authentication was restored and SGE accepted the F06 full-screening array as job `1478702`, tasks 1–60, with a 48-task concurrency limit. The scheduler confirmed running tasks. The first observed failures are M2 setpoints `m2_09` and `m2_10` for designs a and b: SHAP exhausted task memory while allocating transformed arrays of approximately 3.02/3.30 GiB. Other tasks remain in progress; no final architecture success count or model recommendation is claimed. Preserve failed attempts and apply the below-three-success retry policy only after the frozen matrix is terminal.

`nextgen_reduced_artifacts.py` now constructs F06 R / later LR trials only from a complete terminal full-screening matrix and verified successful-run SHAP artifacts. It preserves target rows, removes deselected columns, enforces reciprocal pair closure and ancestry floors, refuses overwriting existing trials, and marks each proposal unaccepted pending its own screening and measured acceptance. Seven focused reduction tests passed. No real reduction proposal has been materialized yet; progressive ancestry remains separate work.

## First live consolidated results snapshot — 2026-09-27

`results/nextgen_results.parquet` now exists under the experiment artifact root, populated by the collector from actual screening records. The first snapshot contains 48 incomplete run rows, four failed run rows, and six incomplete architecture summaries. No architecture summary is screening-usable; queued tasks without progress artifacts have not been fabricated as completed runs. These counts describe this snapshot only and must be refreshed as the live array progresses. No recommendation or reduction acceptance is implied by the presence of the results file.

## Revised F06 redundancy verification — 2026-09-27

All three current correlation reports passed source-data, manifest, and output hash readback after removal of the coach SP fields. On 22,390 team-target rows, near-duplicate pair counts remain 12/12/10 for a/b/c; representative-validation counts are 17/17/0 and report-only counts 75/77/29. Current files are under `results/F06/F06_F_{a,b,c}/redundancy.parquet`. These descriptive reports do not authorize pruning. The last scheduler observation still had 48 running tasks and a queued remainder, with only the four previously recorded M2 failures terminal.

## First verified screening result — 2026-09-27

`F06_F_c__M4__m4_01` completed successfully in 910.37 seconds. For the revised 153-feature F06 c design, the separate development-year metrics are MAE 13.943517 / 13.638959 and Brier 0.202625 / 0.193761 for 2024 / 2025 respectively. Both years are design-informed development evidence, not unbiased holdouts. These values describe one frozen setpoint and must not be used as an architecture summary or lineage recommendation.

The run completed end-to-end source-coordinate permutation SHAP with 256 background games, 512 explanation games, seed 1701, and maximum absolute additivity error 1.0125e-13. The results collector revalidated current execution binding, fingerprint inputs, predictions, and SHAP hashes before adding the successful row to `results/nextgen_results.parquet`. Evidence: `experiments/F06/F06_F_c__M4__m4_01/result.json` and its output artifacts. No architecture summary is screening-usable yet; the full array remains active.

## Progressive pool preparation — 2026-09-27

`nextgen_progressive.progressive_pool` prepares the prior same-design reduced survivors plus every new-generation feature from the accumulated full representation. It rejects changed inherited manifests, values or metadata, disallows resurrection of removed features to satisfy a new counterpart, and reports complete-game cohort exclusions. This is a pure preparation helper: parent finalization verification, persistence, screening, pruning, and measured acceptance still need orchestration. It has not produced an actual PR fingerprint.

The later F06 snapshot has all ten F06_F_c M2 configurations successful and screening-usable, superseding the first-result status above. Other full-screening cells remain active. The interim resource audit at `results/resource_audit_interim.json` records 1,858 reserved API attempts and 768,602,112 allocated bytes under the artifact root; provider quota was not refreshed by that read-only audit, and final resource verification remains required.

## Authorized parallel full screening — 2026-09-27

The user authorized running later generations and different data levels in parallel when feasible. This supersedes the original one-generation-at-a-time barrier for independent full fingerprints. Progressive reductions still require accepted prior survivors; reduction acceptance still requires terminal reference screening. The global 50-job limit and scientific protocol remain in force.

All three F09 full fingerprints passed checked-loader validation (11,151 paired games; 263/281/183 features for a/b/c). SGE accepted the 60-task F09 full array as `1479074.1-60:1`, with a 45-task cap and 24 GB per task. At submission, three F06 tasks remained, giving a combined maximum of 48 running jobs. Receipt: `experiments/F09/full_parallel_array.submission.json` under the experiment root. Submission is not a completed-training claim.

The separately versioned `nextgen_screening_parallel.py` preserves the original live F06 runner and its execution hashes. Its bindings include the parallel authorization; the results collector verifies each runner version separately. Four focused tests passed, including an AST comparison showing unchanged scientific computation apart from the scheduling gate and provenance binding.

## First real reduction trial — 2026-09-27

F06_F_c full screening is terminal with ten successful frozen configurations for each of M2 and M4. Its M4 median development MAE is 13.388646 / 12.843854 for 2024 / 2025, from `results/nextgen_results.parquet`; these are design-informed development results.

The first `F06_R_c` candidate retains exactly 60 of 153 source features, preserving reciprocal groups and all 22,390 team-target rows. Selection maximizes summed joint normalized SHAP importance over those groups at the minimum floor, using all twenty successful full-reference runs. Retained joint importance mass is 0.7689678844. This is a proposed aggressive C reduction, not an accepted accuracy tradeoff. Evidence: `fingerprints/F06_R_c/reduction_proposal.json` and hash-bound provenance under the experiment root.

SGE accepted its twenty-cell screening array as `1479077.1-20:1` with a two-task concurrency cap alongside the live F09 full array. Actual reduced-screening results and measured +0.25 MAE acceptance are pending. Receipt: `experiments/F06/reduced_c_array.submission.json`.

## Supported F10 full assembly — 2026-09-27

F10_F_a/b/c assembled from F09 full parents plus `player_usage`, `recruit_history`, `observed_experience`, and `observed_continuity` in the same design. Outputs contain 319/341/210 features respectively and 21,022 team-target rows (10,511 paired games) each. Canonical family joins excluded 640 parent games per design. Evidence: `fingerprints/F10_F_{a,b,c}/provenance.json` and their values/manifests under the experiment root. Independent checked-loader validation is in progress; no training or performance result is claimed.

`results/f10_source_scope.json` explicitly records incomplete requested source coverage. Historical Week-0 membership/position availability, transfer date/destination availability, and a complete returning-defensive-production denominator remain unverified and do not supply new features. Observed continuity means dynamic prior-use overlap; recruiting means prior commitments, not current membership; experience means acquired prior player-game records, not roster class or complete career history. Missing evidence is not zero production or injury. These gaps remain part of the final scope audit.

## F10 checked inputs and screening launch — 2026-09-27

All three supported F10 full fingerprints passed independent `checked_fingerprint` validation: 10,511 paired games each and 319/341/210 source features for a/b/c. `results/f10_full_readback.json` binds the verified data and manifest hashes; the explicit source gaps in `results/f10_source_scope.json` remain unresolved.

SGE accepted the sixty-cell F10 full screening array as `1479168.1-60:1`, initially limited to one task with 32 GB memory. Submission checked the reserved F09/F06-reduction caps plus remaining original F06 jobs and established a combined maximum of 50 running model jobs. Receipt: `experiments/F10/full_parallel_array.submission.json`. This is a launch record, not measured F10 performance or a completed generation.

## Supported F11 full assembly — 2026-09-27

All three F11 full fingerprints assembled from the same-design F10 full parent plus verified previous-staff history. Designs a/b/c contain 326/349/215 features and 17,018 team-target rows (8,509 paired games) each; prior-staff source coverage excludes 2,002 parent games per design. Evidence: `results/f11_full_materialization.json` under the experiment artifact root. Independent checked-loader readback is running; F11 training has not started.

`results/f11_source_scope.json` preserves the limits: previous-season staff history does not establish target-season employment, tenure, changes, interim assignments, or coordinators. The new family has 7/8/5 features; later reductions must document the legitimate-signal shortfall rather than invent features to meet ten.

F12 preparation will combine verified special teams and observed offensive rooms. `results/f12_source_scope.json` explicitly excludes unsupported historical OL/defensive assignments and their cross-unit matchups. This supported scope does not fulfill every requested unit; evaluation and an honest final coverage disposition remain required. The c design has nine legitimate new signals before pruning.

## Progressive parent evidence gate — 2026-09-27

`nextgen_progressive_artifacts.py` now prepares immutable unpruned pools under `progressive_pools`, separate from screening fingerprints. It verifies the previous same-design reduced parent against completed generation flags, measured acceptance, recommendation hashes, and unchanged evidence including all parent fingerprint inputs. Checked canonical reads and the pure progressive-pool helper then preserve prior survivors plus the complete new family, with explicit cohort losses and ancestry floors. A final source/evidence recheck precedes persistence.

Eight focused tests passed, covering both pool semantics and rejection of changed evidence, absent input bindings, wrong-parent acceptance, and incomplete finalization. No actual progressive pool or accepted PR fingerprint exists yet: parent reductions and finalizations remain pending. Candidate pruning, screening, and acceptance are still required after pool preparation.

## Validated F11 queued screening — 2026-09-27

All three F11 full fingerprints passed independent checked-loader validation: 8,509 paired games, with 326/349/215 features for a/b/c. The input hashes are recorded in `results/f11_full_readback.json` under the artifact root. SGE accepted sixty-cell array `1479184` at two-task concurrency and 32 GB per task, held on reduced-F06 array `1479077`. Its reserved two slots replace that dependency's two slots after completion; other arrays must account for this reservation before raising their caps. The checked combined maximum remains 50. Evidence: `experiments/F11/full_parallel_array.submission.json`.

This is a queued training submission, not completed F11 evaluation. Current-assignment source gaps remain unchanged. F12 assembly is active from the verified special-teams and offensive-room families.

## Progressive trial persistence — 2026-09-27

Progressive preparation now supports immutable candidate trials under `fingerprints`. It revalidates parent acceptance, rederives the pool from full and parent data, rejects changed definitions and cohort loss relative to the full screening reference, and restricts survivors to that pool. Every proposal requires the complete terminal full-screening matrix and verified successful source SHAP, then remains unaccepted until its own screening and measured acceptance. Eleven focused progressive and reduction tests passed. No actual PR trial has been generated because parent finalizations are still pending.

## Later-generation finalization implementation — 2026-09-27

`nextgen_finalize.finalize_generation` now checks F/LR/PR comparisons for F09–F12. It requires each terminal frozen matrix, verified source SHAP, unchanged projections on identical target cohorts, pair/ancestry floors and documented exceptions, and measured per-design reduction acceptance. Progressive ancestry is reconstructed from an accepted parent; current and transitive parent evidence are bound into recommendations/finalization. It does not create missing runs or authorize descendants from an unfinished comparison.

Fifteen focused projection, parent-evidence, ranking and reduction tests passed. This tests the component gates, not a completed real-data finalization. No later-generation recommendation has been issued.

## Complete reduced-C M2 screening — 2026-09-27

All ten M2 configurations for the 60-feature `F06_R_c` completed successfully. On the same 22,390-row team-target corpus as full C, its median development MAE is 13.343549 / 12.937767 for 2024 / 2025, versus 13.386496 / 13.020017 for the 153-feature full C. Reduced-C RMSE is 17.056880 / 16.372434. Reduced-C Brier is 0.198063 / 0.186107: slightly worse in 2024 than full C's 0.197366, and slightly better in 2025 than 0.186576. These are actual ten-configuration architecture medians, not unbiased holdout results.

Evidence: `results/F06/reduced_c_m2_interim.json`, with hashes of all twenty compared M2 run-result files, and `results/nextgen_results.parquet` under the experiment root. M4 reduced screening remains unfinished; no joint C acceptance or lineage recommendation is claimed. Sixteen annual reduced-C references and forty-five F10 full references independently passed exact recomputation; evidence: `results/f06_reduced_c_f10_reference_readback.json`.

## Supported F12 full assembly — 2026-09-27

All three F12 full designs assembled from the same-design F11 parent plus canonical special-teams and observed offensive-room families. The measured a/b/c feature counts are 339/365/224 on 14,716 team-target rows (7,358 paired games) each. Family coverage excludes 1,151 F11 parent games per design. Evidence: `results/f12_full_materialization.json` under the experiment artifact root. Independent checked-loader validation is active; F12 training has not started.

This is the supported source scope, not fulfillment of every requested unit: `results/f12_source_scope.json` records missing historical OL/blocking and defensive front/linebacker/secondary assignments, and consequently missing explicit offensive-room versus opposing-defensive-unit comparisons. The acquired enriched team detail only covers 2025, outside fitting years, and Stage E does not resolve historical position/participation attribution. Existing same-room comparisons must not be represented as those missing matchups.

The new special-teams plus offensive-room families add 13/16/9 features for a/b/c. Any c reduction must document the legitimate-signal shortfall and retain all nine unless separately evidenced independent-signal limitations justify fewer; no feature duplication is used to reach ten. Evaluation, reduced lineages, and final recommendations remain unfinished.

## First conservative A reduction trial — 2026-09-27

F06_F_a full screening is terminal with eight successful M2 configurations, two M2 memory failures, and ten successful M4 configurations. The measured M4 median development MAE is 13.301759 / 12.861404 for 2024 / 2025. Its missing M2 successes remain flagged; the at-least-three success rule does not authorize retries solely to reach ten.

The first F06_R_a proposal retains 162 of 225 features and all 22,390 team-target rows. Removed reciprocal-group members are each below uniform-share normalized importance in both architectures. The greedy group order uses conservative importance, capped at 10% cumulative removed importance in each model; measured removed importance is 0.098697948 for M2 and 0.088897415 for M4. Twenty-two preferred representatives of >=0.98 correlated pairs are protected by lower missingness, simpler inputs/equation, stronger joint importance, then earlier generation. Evidence: `results/F06/reduced_a_selection_trial_1.json` and `fingerprints/F06_R_a/reduction_proposal.json` under the artifact root.

Checked-loader validation passed on 11,195 paired games. SGE accepted twenty-cell trial array `1479192`, capped at twenty and held on F09 full array `1479074`; this reserves twenty of the forty-five slots released when F09 exits. Evidence: `experiments/F06/reduced_a_array.submission.json`. This is an unaccepted proposal, not the smallest validated representation. Screen both architectures, enforce +0.25 MAE separately in each year, and evaluate a smaller permitted candidate if this first trial passes.

Separately, one original F06 full task finished and its slot was assigned to F10, raising F10 array `1479168` from one to two concurrent tasks. The checked current bound remains fifty. F11 still reserves the two slots of reduced-C array `1479077` after that dependency completes; the queued A trial's reservation must also be counted in future capacity changes. Evidence: `experiments/F10/concurrency_update_2.json`.

## Global diagnostic primitives — 2026-09-27

`nextgen_diagnostics.py` now ranks supplied verified SHAP contexts within one global advanced-feature budget of at most 1,000. Identical scientific definitions can share a diagnostic across design labels, while different equations remain distinct. Consensus balances lineages within designs, designs within generations, generations containing a feature, and then M2/M4. Absence from a pruned representation is not treated as an observed zero effect. Direct atomic baseline fields are excluded under the documented predicate.

The renderer writes compact PNG/Markdown pairs for prior-state versus next-game margin and empirical win rate, retaining equations, provenance, sample/missingness rules, correlations, SHAP context, and supplied survival status. Four focused tests passed, including constant-feature plots, 2026 rejection, identity separation, and global weighting. These are tested primitives with fixture plots only; all-program verified-result collection, final global ranking, real diagnostic generation, and index integration remain outstanding.

## F12 full screening started — 2026-09-27

All three F12 full inputs passed independent checked-loader validation: 7,358 paired games and 339/365/224 features for a/b/c. Hash-bound readback: `results/f12_full_readback.json`. SGE accepted sixty-cell array `1479229` with 32 GB per task, initially held on F09 with a planned twenty-task cap (`experiments/F12/full_parallel_array.submission.json`).

The final original F06 full task then finished, freeing one current slot. F12's cap was lowered to one before its hold was changed. This cluster rejected an empty dependency list, so the hold was instead retargeted to the completed F06 array `1478702`; SGE confirmed no blocking jobs. The scheduler subsequently confirmed one running F12 task and fifty running tasks in total. Evidence: `experiments/F12/concurrency_start_1.json` and `resource_hold_release.json`. The actual current F12 cap is one, superseding the initial twenty-task submission cap; any later expansion needs a new capacity check including held A and F11 reservations.

F12 is actively screening, not evaluated or finalized. Unsupported line/defensive-unit attribution and cross-unit matchup requirements remain explicitly documented. Annual references are being built; the A design produced thirteen references for 2014–2026 from strictly prior observed seasons.

## First conservative B reduction trial — 2026-09-27

F06_F_b is terminal with eight successful M2 configurations, two M2 memory failures, and ten successful M4 configurations. Its M4 median development MAE is 13.316360 / 12.883629 for 2024 / 2025. All original F06 full designs are now terminal; no failure was silently retried or discarded.

The first F06_R_b proposal retains 170 of 235 features on all 22,390 team-target rows. It applies the same pair-closed weak-under-both policy as A, protecting twenty-two preferred correlation representatives. Removed normalized importance is 0.098923071 in M2 and 0.091571615 in M4. Evidence: `results/F06/reduced_b_selection_trial_1.json` and `fingerprints/F06_R_b/reduction_proposal.json` under the experiment root. It remains unaccepted and is not claimed to be the minimum validated representation.

Checked-loader validation passed on 11,195 paired games. SGE accepted twenty-cell B trial array `1479244`, capped at twenty and held on F09 array `1479074`. Both held A/B arrays now reserve twenty slots each after F09 completes; with F12 cap one, F10 cap two, and the reduced-C/F11 two-slot dependency chain, the post-F09 maximum is forty-five. Future expansions must account for both A/B reservations. Receipt: `experiments/F06/reduced_b_array.submission.json`. Trial annual references are being built.

## Independent reduction screening runner — 2026-09-27

The user's permission for parallel generations and different data levels also allows an independent late-reduction trial to run after its own full reference is terminal. `nextgen_screening_reduced_parallel.py` removes unrelated generation waits for R/LR trials, but requires matching immutable proposal evidence and the complete verified full matrix. PR continues to require the existing verified accepted-parent finalization gate. All trials still need their own measured acceptance. The global fifty-job limit remains unchanged.

A separate authorization file and runner preserve the hashes of every existing live job. New reduced-run bindings include the reduction and parent-verification helpers; do not modify those helpers once this runner has active jobs. The collector recognizes the third binding policy. Nine focused tests passed, including identical scientific-function ASTs, terminal evidence/proposal checks, PR parent gating, and result-verifier dispatch. No job has yet used this new runner; queued F06 A/B and running C retain their original runner.

Independent exact recomputation verified all 116 new annual references for reduced F06 A/B and full F11/F12, including source hashes and strictly prior season ordering. Evidence: `results/reduced_ab_f11_f12_reference_readback.json` under the artifact root. Together with previously verified sets, eighteen currently materialized fingerprints have 273 annual references; later LR/PR artifacts and their references remain outstanding.

## Independent floor acceptance and earlier parallel starts — 2026-09-27

`nextgen_acceptance.accept_floor_reduction` can finalize one R/PR parent independently when every concrete-generation feature floor is reached and its complete frozen M2/M4 matrix passes measured acceptance against full. It writes immutable evidence without changing trained input provenance. Non-floor first trials cannot use this path. The parent reader rechecks input, execution and successful-output hashes and recomputes the full/candidate MAE acceptance; complete generation finalization remains the fallback for other parents. Later finalizers bind either authorization form transitively. Eighteen focused tests passed. No real floor-acceptance receipt has been issued yet: reduced C has nineteen successful results and one active cell.

All F09 cells have started, allowing permanent freed slots to be reassigned. Its cap was lowered from 45 to 44 when A's cap was reduced from 20 to one and A's hold was retargeted to completed F06; A was confirmed running. Evidence: `results/concurrency_reduction_start.json`. Reduced C and F11 were then assigned one slot each, and F11's resource hold was released; F11 was confirmed running. Evidence: `experiments/F11/concurrency_start.json`.

Two further F09 completions allowed its cap to drop to 42 and B's held twenty-task cap to become two before releasing B. Evidence: `experiments/F06/reduced_b_concurrency_start.json`. Current authorized maxima are F09 42, F10 2, F11 1, F12 1, reduced A 1, reduced B 2, and reduced C 1: fifty total. These current caps supersede initial submission reservations; future changes must recheck scheduler state.

## Reduced C accepted at the feature floor — 2026-09-27

F06_R_c completed all ten M2 and ten M4 configurations successfully. The immutable `results/F06/F06_R_c/acceptance.json` verifies the full and reduced input/output evidence and accepts the 60-feature representation against the 153-feature full C reference. Development median MAE for 2024 / 2025 is 13.343549 / 12.937767 (M2) and 13.413353 / 12.823703 (M4). Relative to full C, the four changes are -0.042948, -0.082251, +0.024707, and -0.020151; joint mean change is -0.030161. The M4 2024 worsening is retained in the report. These are design-informed development results, not unbiased holdouts. All concrete feature floors are reached, permitting independent progressive-parent authorization; A/B and generation-wide recommendations remain unfinished.

After reduced C completed and additional F09 tasks ended, capacity was reassigned within the fifty-job bound. Current caps: F09 38, F10 2, F11 5, F12 2, reduced A 1, reduced B 2. Receipts: `experiments/F12/concurrency_update_2.json` and `results/concurrency_after_reduced_c.json`. These supersede earlier caps.

The final diagnostic orchestrator now verifies all generation finalizations, all 42 screening contexts, and recommendation/input evidence before global selection and rendering. Mandatory offense/defense Q4-minus-Q1 rushing plots share the global 1,000-feature limit and retain their true global ranks. Five primitive tests passed; an actual incomplete-program check refused missing F06 finalization and preserved the existing index. Final real diagnostic generation remains outstanding.

## First authorized progressive pool — 2026-09-27

The accepted floor-sized F06 reduced C now authorizes the F09_PR_c pool. Actual `progressive_pools/F09_PR_c/provenance.json` records 60 inherited F06 features plus all 30 new F09 features on 22,302 team-target rows (11,151 games), with no excluded full-reference games. Parent receipt SHA-256 is `6d2a55a6855369a7b8599652578f225b4339968aec35680441c5c5add737a23c`. This is an unpruned pool, not an accepted or screened progressive fingerprint. Selection and screening require terminal F09 full-reference evidence.

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
