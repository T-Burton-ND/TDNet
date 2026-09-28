---
type: concept
up: "[[Fingerprint-Ladder]]"
extends: "[[Fingerprint-F11]]"
tags: [fingerprints, next-generation, design]
---

# Fingerprint F12

F12 is the market-free information generation for unit-level team states. Its supported full fingerprints are assembled from special teams and observed offensive rooms; full screening is active, while evaluation and missing unit coverage remain unfinished.

## Measured lessons — 2026-09-28

Full A worsens MAE relative to F11 A on identical evaluation IDs: all ten M2 configurations in each year, and seven of ten M4 configurations in each year. Brier medians worsen for both models in both years. Different historical training coverage does not erase this measured negative evaluation result. See [empirical lessons and paired evidence](Nextgen-Empirical-Lessons-2026-09-28) for exact values, scope and limits. No new experiments were run.

## Current snapshot — 2026-09-28

Full A has ten successes per architecture. Median development MAE 2024 / 2025 is 13.823579 / 12.729218 (M2), 13.186888 / 12.393033 (M4). B remains partial. Overall 22 successful, six running and 32 queued cells. Supported source coverage and smaller-cohort caveats remain; no generation recommendation exists. See the [verified paused snapshot](Nextgen-Results-Snapshot-2026-09-28) for all metrics, evidence paths, source gaps and scheduler counts as of 12:52 UTC. This supersedes earlier live-status statements; the goal is paused and existing queued cells may still dispatch.

## Information added and sources

- **Parent:** F11, same a/b/c design only. F07/F08 are excluded.
- **New family:** Opponent-independent team-week OL, QB, RB, pass-catcher, defensive line, linebacker, secondary, and special-teams states.
- **Candidate CFBD sources:** `F06/F09/F10/F11 canonical team-week families`. Availability and as-of coverage must be audited before materialization.
- **Family rule:** The matchup builder compares home offense with away defense, such as OL↔DL and WR/TE↔DB, and repeats in the opposite direction. Do not bake the next opponent into the fingerprint.

## Representations and lineages

For each design `a`, `b`, and `c`, build `F12_F_*`, `F12_LR_*`, and `F12_PR_*`. `a` is relatively atomic; `b` adds football-informed interpretable interactions; `c` compresses into film-intuitive Excel-reproducible equations with at most five source inputs. Full inherits previous full. Late reduction prunes the current full version. Progressive reduction inherits previous reduced version, adds this generation's complete new family, then prunes; `F12_PR_*` never crossbreeds design tracks.

## Temporal, matchup, and reduction semantics

All dynamic quantities use only completed regular-season performance before the **next target game**. FBS–FBS regular-season games are targets; FCS games can be historical context. Postseason numeric performance and 2026 design evidence are excluded. Market and pregame win-probability inputs are forbidden. Team-week sources remain ordinary numeric features with an explicit counterpart and comparison equation; a pruning pair is atomic and can be removed only when both members qualify. A reduced fingerprint retains at least 60 concrete F06 features and ten from each added generation, subject to explicit shortfall disposition if a family cannot support ten legitimate signals.

Every derived feature's manifest records endpoints/columns, exact formula and units, availability and cutoff, aggregation/sample/missingness, garbage handling, counterpart and matchup formula, provenance, code path, and version. Fit scaling or C weights only on pre-2026 development evidence. Screen with the same M2/M4 setpoints and architecture-normalized source-level permutation SHAP; evaluate next-game margin.

## Special-teams family materialization

On 2026-09-27, the acquired 2010–2025 regular-game player boxes produced 22,646 team-target rows per design, with 7/8/6 special-teams features for a/b/c. Evidence: `results/f12_special_teams_materialization.json` under the experiment artifact root. All three canonical families passed independent `NextgenModelBoundary.from_canonical` readback, checking manifest/data hashes, schedule pairing, and temporal provenance.

The family uses the latest 12 available regular-game box observations with a reconstructed 48-hour reporting lag. Signals cover field-goal/extra-point conversion, punting, and kick/punt returns; rate numerators and denominators require joint observation and minimum support. This is a completed special-teams component, not a complete F12 fingerprint or evidence of predictive value. Other unit families and their explicit cross-unit matchup comparisons remain unfinished.

## Offensive-room source preparation

The 2026-09-27 audit of 200 acquired player-game PPA partitions found 258,553 raw rows. Removing 3,002 exact duplicates and excluding 44,731 unmatched or ambiguous schedule rows yielded 210,820 matched records in `canonical/player_ppa_games.parquet`. Matched records have observed QB, RB/FB, or WR/TE positions; none lacks a stable player identity or mapped unit. Evidence: `results/player_ppa_game_binding.json`. Six conflicting raw season/week/team/opponent/player keys were observed during diagnosis, but none survived schedule matching; the retained-corpus conflicting-key exclusion count is zero.

These are historical source observations, not canonical F12 feature families or a frozen current roster. The provider lacks attempt counts here, so the implemented room-state primitive uses equal-player-game PPA summaries with at least three observed prior games and a reconstructed 48-hour lag. A focused temporal/support test passes. Canonical a/b/c room materialization is verified below; defensive/line units remain unfinished.

## Offensive-room family materialization

On 2026-09-27, the verified historical player-PPA corpus produced 18,516 team-target rows per offensive-room design, with 6/8/3 features for a/b/c. Evidence: `results/f12_offensive_rooms_materialization.json`. All three designs passed independent canonical model-boundary readback. The a design retains mean and dispersion for QB passing, RB/FB rushing, and WR/TE receiving; b adds phase comparisons; c uses mean minus dispersion within each room. Two focused tests passed. These components do not complete the remaining line, defensive-unit, or cross-unit matchup requirements.


## Supported F12 full assembly — 2026-09-27

All three F12 full designs assembled from the same-design F11 parent plus canonical special-teams and observed offensive-room families. The measured a/b/c feature counts are 339/365/224 on 14,716 team-target rows (7,358 paired games) each. Family coverage excludes 1,151 F11 parent games per design. Evidence: `results/f12_full_materialization.json` under the experiment artifact root. Independent checked-loader validation subsequently passed; active screening is recorded below.

This is the supported source scope, not fulfillment of every requested unit: `results/f12_source_scope.json` records missing historical OL/blocking and defensive front/linebacker/secondary assignments, and consequently missing explicit offensive-room versus opposing-defensive-unit comparisons. The acquired enriched team detail only covers 2025, outside fitting years, and Stage E does not resolve historical position/participation attribution. Existing same-room comparisons must not be represented as those missing matchups.

The new special-teams plus offensive-room families add 13/16/9 features for a/b/c. Any c reduction must document the legitimate-signal shortfall and retain all nine unless separately evidenced independent-signal limitations justify fewer; no feature duplication is used to reach ten. Evaluation, reduced lineages, and final recommendations remain unfinished.


## F12 full screening started — 2026-09-27

All three F12 full inputs passed independent checked-loader validation: 7,358 paired games and 339/365/224 features for a/b/c. Hash-bound readback: `results/f12_full_readback.json`. SGE accepted sixty-cell array `1479229` with 32 GB per task, initially held on F09 with a planned twenty-task cap (`experiments/F12/full_parallel_array.submission.json`).

The final original F06 full task then finished, freeing one current slot. F12's cap was lowered to one before its hold was changed. This cluster rejected an empty dependency list, so the hold was instead retargeted to the completed F06 array `1478702`; SGE confirmed no blocking jobs. The scheduler subsequently confirmed one running F12 task and fifty running tasks in total. Evidence: `experiments/F12/concurrency_start_1.json` and `resource_hold_release.json`. The actual current F12 cap is one, superseding the initial twenty-task submission cap; any later expansion needs a new capacity check including held A and F11 reservations.

F12 is actively screening, not evaluated or finalized. Unsupported line/defensive-unit attribution and cross-unit matchup requirements remain explicitly documented. Annual references are being built; the A design produced thirteen references for 2014–2026 from strictly prior observed seasons.

## Status

**Supported full screening active; evaluation pending.** F09/F10/F12 screening is active and F11 is queued. Reduced lineages and final recommendations remain pending. See the [experiment contract](Next-Generation-Fingerprint-Experiment) and repository `docs/nextgen_fingerprints/README.md`.

See also: [Fingerprint Ladder](Fingerprint-Ladder) · [Next-Generation Experiment](Next-Generation-Fingerprint-Experiment) · [Fingerprint F11](Fingerprint-F11)
