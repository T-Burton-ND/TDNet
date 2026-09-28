---
type: concept
up: "[[Fingerprint-Ladder]]"
extends: "[[Fingerprint-F09]]"
tags: [fingerprints, next-generation, design]
---

# Fingerprint F10

F10 is the planned market-free information generation for roster and player state; source preparation is implemented, while complete features and evaluation remain unfinished.

## Current snapshot — 2026-09-28

Full A M2 is terminal with ten successes: median development MAE 13.393119 / 12.896856 (2024 / 2025). Full A M4 has seven successes and three running cells; B M2 has two running cells. Overall 17 successful, five running and 38 queued cells. No generation recommendation exists. See the [verified paused snapshot](Nextgen-Results-Snapshot-2026-09-28) for all metrics, evidence paths, source gaps and scheduler counts as of 12:52 UTC. This supersedes earlier live-status statements; the goal is paused and existing queued cells may still dispatch.

## Information added and sources

- **Parent:** F09, same a/b/c design only. F07/F08 are excluded.
- **New family:** Week-0 roster, recruiting, transfers, documented use and production, continuity, concentration, and position-appropriate physical summaries.
- **Candidate CFBD sources:** `/roster, /recruiting/players, /player/portal, /player/usage, /games/players, /stats/player/season`. Availability and as-of coverage must be audited before materialization.
- **Family rule:** Freeze roster at Week 0. Prefer actual use, then prior production; match by stable ID, exact normalized name, then conservative constrained fuzzy match. Do not infer injury or assign production to an unplayed freshman/transfer. Transfer-era absence is structural missingness.

## Representations and lineages

For each design `a`, `b`, and `c`, build `F10_F_*`, `F10_LR_*`, and `F10_PR_*`. `a` is relatively atomic; `b` adds football-informed interpretable interactions; `c` compresses into film-intuitive Excel-reproducible equations with at most five source inputs. Full inherits previous full. Late reduction prunes the current full version. Progressive reduction inherits previous reduced version, adds this generation's complete new family, then prunes; `F10_PR_*` never crossbreeds design tracks.

## Temporal, matchup, and reduction semantics

All dynamic quantities use only completed regular-season performance before the **next target game**. FBS–FBS regular-season games are targets; FCS games can be historical context. Postseason numeric performance and 2026 design evidence are excluded. Market and pregame win-probability inputs are forbidden. Team-week sources remain ordinary numeric features with an explicit counterpart and comparison equation; a pruning pair is atomic and can be removed only when both members qualify. A reduced fingerprint retains at least 60 concrete F06 features and ten from each added generation, subject to explicit shortfall disposition if a family cannot support ten legitimate signals.

Every derived feature's manifest records endpoints/columns, exact formula and units, availability and cutoff, aggregation/sample/missingness, garbage handling, counterpart and matchup formula, provenance, code path, and version. Fit scaling or C weights only on pre-2026 development evidence. Screen with the same M2/M4 setpoints and architecture-normalized source-level permutation SHAP; evaluate next-game margin.

## Prior-usage family materialization

On 2026-09-27, the observed-player usage family materialized 22,646 team-target rows per design for the 2010–2025 schedule, with 16/17/8 features for a/b/c (`results/f10_usage_materialization.json`). These passed canonical write validation and independent model-boundary readback for all three designs; mean missing-cell fractions are 0.0466/0.0439/0.0466. These artifacts cover prior usage concentration only. They do not establish frozen roster availability, returning production, recruiting or transfers, and are not complete F10 fingerprints.

## Recruiting-history family materialization

On 2026-09-27, preceding high-school recruiting cohorts produced 21,368 team-target rows per design on the 2011–2025 target schedule, with 16/17/8 features for a/b/c. Evidence: `results/f10_recruit_history_materialization.json` under the experiment artifact root. All three designs passed independent canonical model-boundary readback; two focused recruiting tests passed. No 2010 target rows are present because the acquired cohort history begins in 2010 and excludes each target season's class.

These position-unit ratings and blue-chip shares describe up to four preceding recruiting classes, not current roster talent, enrollment, or retention. Availability is reconstructed at January 1 following the latest contributing class; it is not verified against an archived provider snapshot. Current-class recruits and unknown position assignments are excluded; unsupported unit statistics remain missing. This source assumption and missing early history must remain explicit in downstream reporting.

## Observed-experience family materialization

On 2026-09-27, acquired regular-game player records produced 22,646 team-target rows per design on the 2010–2025 schedule, with 8/9/3 experience features for a/b/c. Evidence: `results/f10_experience_materialization.json` under the experiment artifact root. All three canonical families passed independent model-boundary readback; four focused experience tests passed.

Experience counts distinct acquired games with a finite player-box record, weighted by actual prior workload over the latest 12 available team games. Records from prior teams may contribute to the count, but every count is frozen through the latest contributing team-game cutoff and remains strictly before the target. This is observed record history, not career games played, class year, proof of snaps, or future availability. Unknown identities and unsupported workloads remain missing. Early history is left-truncated by acquisition coverage. The compact design averages offense, defense, and special-teams workload groups only when every component exists.

## Observed-continuity family materialization

On 2026-09-27, the acquired regular player boxes produced 22,636 team-target rows per observed-continuity design on the 2010–2025 schedule, with 16/17/8 features for a/b/c. Evidence: `results/f10_continuity_materialization.json`. Every design passed independent canonical model-boundary readback; two focused continuity tests passed.

The family compares same-team current-season observed users with the immediately preceding season's users. It measures current workload belonging to overlapping users and the share of prior production attributable to those users. Values remain missing before current-season usage exists or when either workload lacks valid identities/counts. This is dynamic observed-use overlap, not frozen-roster retention, a departure flag, or an injury/availability inference. It does not resolve the remaining preseason roster/transfer requirements.

## Roster source limitation

The 2026-09-27 roster audit covers 16 acquired 2010–2025 partitions and 305,541 rows. In 122,086 rows, `year` equals the requested season rather than a plausible class number; other rows contain class-like values. This field is not usable directly as weighted experience. Evidence: `results/roster_semantics_audit.json` and `scripts/nextgen_audit_rosters.py`, including per-source hashes. The audit does not establish historical Week-0 membership or position availability. Observed prior regular-game appearances are a defensible experience source once joined with pre-target timing; no class year should be invented.

## Status

Player-source preparation completed on 2026-09-27: 245 acquired 2010–2025 game-player partitions produced 3,020,831 additive observations (`results/player_observation_preparation.json`). These are intermediate source observations, explicitly not model-ready features. Implemented primitives cover constrained identity matching, prior-game usage concentration, and returning production requiring source-backed frozen-roster availability. Five player tests pass; complete F10 designs, remaining canonical feature families and evaluation remain unfinished.

 At that earlier snapshot, no F09–F12 training or recommendation had been run; the later parallel F09 launch is recorded in the experiment page. See the [experiment contract](Next-Generation-Fingerprint-Experiment) and repository `docs/nextgen_fingerprints/README.md`.

See also: [Fingerprint Ladder](Fingerprint-Ladder) · [Next-Generation Experiment](Next-Generation-Fingerprint-Experiment) · [Fingerprint F09](Fingerprint-F09) · [Fingerprint F11](Fingerprint-F11)

## Supported F10 full assembly — 2026-09-27

F10_F_a/b/c assembled from F09 full parents plus `player_usage`, `recruit_history`, `observed_experience`, and `observed_continuity` in the same design. Outputs contain 319/341/210 features respectively and 21,022 team-target rows (10,511 paired games) each. Canonical family joins excluded 640 parent games per design. Evidence: `fingerprints/F10_F_{a,b,c}/provenance.json` and their values/manifests under the experiment root. Independent checked-loader validation is in progress; no training or performance result is claimed.

`results/f10_source_scope.json` explicitly records incomplete requested source coverage. Historical Week-0 membership/position availability, transfer date/destination availability, and a complete returning-defensive-production denominator remain unverified and do not supply new features. Observed continuity means dynamic prior-use overlap; recruiting means prior commitments, not current membership; experience means acquired prior player-game records, not roster class or complete career history. Missing evidence is not zero production or injury. These gaps remain part of the final scope audit.

## F10 checked inputs and screening launch — 2026-09-27

All three supported F10 full fingerprints passed independent `checked_fingerprint` validation: 10,511 paired games each and 319/341/210 source features for a/b/c. `results/f10_full_readback.json` binds the verified data and manifest hashes; the explicit source gaps in `results/f10_source_scope.json` remain unresolved.

SGE accepted the sixty-cell F10 full screening array as `1479168.1-60:1`, initially limited to one task with 32 GB memory. Submission checked the reserved F09/F06-reduction caps plus remaining original F06 jobs and established a combined maximum of 50 running model jobs. Receipt: `experiments/F10/full_parallel_array.submission.json`. This is a launch record, not measured F10 performance or a completed generation.
