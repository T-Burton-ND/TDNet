---
type: concept
up: "[[Fingerprint-Ladder]]"
extends: "[[Fingerprint-F10]]"
tags: [fingerprints, next-generation, design]
---

# Fingerprint F11

F11 is the planned market-free information generation for coaching history; supported full fingerprints are validated and screening is active, while evaluation remains unfinished.

## Measured lessons — 2026-09-28

Re-scoring existing F10 A predictions on F11 A evaluation games reverses the apparent 2025 M2 gain: F10 median 12.446026 versus F11 12.537048 on 553 games. F11 also worsens 2024 M2 on 626 shared games. Training coverage remains different, so this is a pipeline comparison, not a coaching-feature ablation. See [empirical lessons and paired evidence](Nextgen-Empirical-Lessons-2026-09-28) for exact values, scope and limits. No new experiments were run.

## Current snapshot — 2026-09-28

Full A has ten successes per architecture. Median development MAE 2024 / 2025 is 13.706990 / 12.537048 (M2), 13.141755 / 12.206475 (M4). B remains partial. Overall 26 successful, nine running and 25 queued cells. Its smaller cohort prevents interpreting cross-generation differences as measured feature gains. See the [verified paused snapshot](Nextgen-Results-Snapshot-2026-09-28) for all metrics, evidence paths, source gaps and scheduler counts as of 12:52 UTC. This supersedes earlier live-status statements; the goal is paused and existing queued cells may still dispatch.

## Information added and sources

- **Parent:** F10, same a/b/c design only. F07/F08 are excluded.
- **New family:** Derived head-coach tenure, prior-team history, change/interim indicators, and trajectory beyond existing F06 coaching context.
- **Candidate CFBD sources:** `/coaches, /coaches/seasons, /coaches/tenures`. Availability and as-of coverage must be audited before materialization.
- **Family rule:** No literal coach-identity memorization and no fabricated coordinator data. A thin or null gain is informative; the hard ten-feature ancestry floor must be reconciled if fewer than ten legitimate signals exist.

## Representations and lineages

For each design `a`, `b`, and `c`, build `F11_F_*`, `F11_LR_*`, and `F11_PR_*`. `a` is relatively atomic; `b` adds football-informed interpretable interactions; `c` compresses into film-intuitive Excel-reproducible equations with at most five source inputs. Full inherits previous full. Late reduction prunes the current full version. Progressive reduction inherits previous reduced version, adds this generation's complete new family, then prunes; `F11_PR_*` never crossbreeds design tracks.

## Temporal, matchup, and reduction semantics

All dynamic quantities use only completed regular-season performance before the **next target game**. FBS–FBS regular-season games are targets; FCS games can be historical context. Postseason numeric performance and 2026 design evidence are excluded. Market and pregame win-probability inputs are forbidden. Team-week sources remain ordinary numeric features with an explicit counterpart and comparison equation; a pruning pair is atomic and can be removed only when both members qualify. A reduced fingerprint retains at least 60 concrete F06 features and ten from each added generation, subject to explicit shortfall disposition if a family cannot support ten legitimate signals.

Every derived feature's manifest records endpoints/columns, exact formula and units, availability and cutoff, aggregation/sample/missingness, garbage handling, counterpart and matchup formula, provenance, code path, and version. Fit scaling or C weights only on pre-2026 development evidence. Screen with the same M2/M4 setpoints and architecture-normalized source-level permutation SHAP; evaluate next-game margin.

## Previous-staff canonical family

On 2026-09-27, the `prior_staff_a/b/c` families materialized 17,296 team-target rows each, with 7/8/5 features. All three passed independent `NextgenModelBoundary` reads. Evidence: `results/f11_prior_staff_materialization.json` and canonical family provenance sidecars. These features summarize the previous season's unambiguous staff using only earlier regular-season scores; they do not assert that this coach remains employed in the target season. The family is deliberately thin. Supported F11 assembly is recorded below; reduction signal-floor disposition and evaluation remain outstanding.


## Supported F11 full assembly — 2026-09-27

All three F11 full fingerprints assembled from the same-design F10 full parent plus verified previous-staff history. Designs a/b/c contain 326/349/215 features and 17,018 team-target rows (8,509 paired games) each; prior-staff source coverage excludes 2,002 parent games per design. Evidence: `results/f11_full_materialization.json` under the experiment artifact root. Independent checked-loader readback subsequently passed; the queued submission is recorded below.

`results/f11_source_scope.json` preserves the limits: previous-season staff history does not establish target-season employment, tenure, changes, interim assignments, or coordinators. The new family has 7/8/5 features; later reductions must document the legitimate-signal shortfall rather than invent features to meet ten.


## Validated F11 queued screening — 2026-09-27

All three F11 full fingerprints passed independent checked-loader validation: 8,509 paired games, with 326/349/215 features for a/b/c. The input hashes are recorded in `results/f11_full_readback.json` under the artifact root. SGE accepted sixty-cell array `1479184` at two-task concurrency and 32 GB per task, held on reduced-F06 array `1479077`. Its reserved two slots replace that dependency's two slots after completion; other arrays must account for this reservation before raising their caps. The checked combined maximum remains 50. Evidence: `experiments/F11/full_parallel_array.submission.json`.

This is a queued training submission, not completed F11 evaluation. Current-assignment source gaps remain unchanged. F12 assembly is active from the verified special-teams and offensive-room families.

## Screening active after resource release — 2026-09-27

SGE confirmed F11 array `1479184` running after its cap was reduced to one and its resource dependency was retargeted to completed original F06 array `1478702`. The final reduced-C task also has a one-task cap, so the two previously reserved slots are shared concurrently. Evidence: `experiments/F11/concurrency_start.json` under the experiment root. This is active training, not completed F11 evaluation. The initial held two-task submission described above is superseded by this current one-task cap.

## Status

Regular-only coaching-history preparation completed on 2026-09-27: 1,858 coach-team seasons across 348 coaches (`results/coach_regular_history_preparation.json`). Scores are rebuilt from fresh regular-game schedules, excluding ambiguous split-coach attribution and ignoring provider season SP+/record aggregates. Two coaching tests pass. Current-coach assignment availability and evaluation remain unfinished. The inherited F06 SP+ issue was resolved by the user-authorized exclusion of both coach-career SP+ fields from the exploratory baseline.

 F09 and F10 full screening are active; F11 screening is active, and recommendations remain pending. See the [experiment contract](Next-Generation-Fingerprint-Experiment) and repository `docs/nextgen_fingerprints/README.md`.

See also: [Fingerprint Ladder](Fingerprint-Ladder) · [Next-Generation Experiment](Next-Generation-Fingerprint-Experiment) · [Fingerprint F10](Fingerprint-F10) · [Fingerprint F12](Fingerprint-F12)
