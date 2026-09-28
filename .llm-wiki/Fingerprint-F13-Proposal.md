---
type: synthesis
up: "[[Next-Generation-Fingerprint-Experiment]]"
derived_from: ["[[Nextgen-Empirical-Lessons-2026-09-28]]", "[[Fingerprint-F09]]"]
tags: [fingerprints, proposal, untested, play-by-play]
---

# Fingerprint F13 Proposal

Proposed future research: context-adjusted play performance and outcome distributions, motivated by measured F09 gains; no F13 experiment, acquisition, lineage change or implementation is authorized or completed.

## Evidence motivating the proposal

The [empirical lessons](Nextgen-Empirical-Lessons-2026-09-28) show F09 improves M4 MAE in all ten shared configurations in 2024 for each A/B/C design, with less consistent gains in 2025. They also show the current F11 A/M2 and F12 A pipelines can worsen same-game performance. Thus there is evidence to investigate play-derived information further, but no evidence yet that this proposed F13 will improve predictions or explain the F09 gain.

Current [F09](Fingerprint-F09) summarizes quarter/half rushing and success differences, pressure/location success rates, rushing yardage components, and drive efficiency/pace. Its builder aggregates sufficient statistics across the latest twelve qualifying regular-season games. It does not explicitly fit a play-level contextual baseline or expose a full conditional outcome distribution. These are source-code observations from `src/gridiron_ml/experiments/nextgen_f09.py` and `nextgen_microstructure.py`, not new model results.

## Proposed information

1. **Performance relative to play context:** offensive and defensive success/yardage residuals conditional on down, distance, field position, pre-play score/time and play type. Estimate any fitted contextual baselines only from information available before the relevant prediction cutoff; use out-of-time construction for historical training features.
2. **Opponent adjustment at the play level:** estimate offense and defense contributions while accounting for context and prior opponent strength, avoiding the assumption that the existing aggregate F06 adjustments already normalize the new play contexts. This is a hypothesis to test, not a demonstrated deficiency.
3. **Outcome shape and reliability:** separate ordinary-play efficiency from explosive-play dependence, negative-play/sack frequency and lower-tail outcomes. Track sample support and shrink sparse rates toward an available prior baseline. Test incremental value against F09's existing success and rushing-component features rather than treating renamed equivalents as new information.
4. **Conditional response profiles:** differences between neutral, passing and short-yardage contexts, with offense and defense kept separate. Store opponent-independent team states; any future matchup comparison belongs in the matchup builder. Prefer a small supported set over a large sparse context grid.

A concrete question: does a team's apparent rushing strength come from repeatable gains in ordinary situations, a few long gains, or easier down/distance/opponent contexts? The archive may allow these descriptions; whether they improve next-game forecasts is unknown.

## Proposed comparison and lineage

Use identical training and evaluation game cohorts to compare a reference pipeline against the same pipeline plus the proposed family, with shared frozen configurations. Report differences of medians, median paired differences, and per-configuration improvement counts, alongside Brier and actual-upset recognition. This addresses the interpretation failures found in the completed results.

Scientifically, F09 and the supported F10 A pipeline are useful references. The existing experiment's accepted-parent lineage rules remain unchanged. A future F13 research fork from an earlier generation would be a proposed protocol change, not an automatic authorization to bypass F12 or accepted-parent gates. If F13 must extend F12, retain an earlier-generation reference comparison so inherited deterioration is visible.

## What play-by-play is already available

A read-only aggregation of `results/f09_play_coverage.json` under `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen`, inspected 2026-09-28, reports:

| Scope | Measured inventory |
|---|---:|
| Seasons represented | 2010–2025 |
| Weekly partitions | 245 |
| Raw play rows summed across partitions | 2,351,115 |
| Expected game entries | 13,059 |
| Observed game entries | 12,964 |
| Distinct missing game IDs | 95 |
| Derived source team-game rows | 25,928 |

The request ledger identifies FBS, regular-season, year/week partitions. This is a substantial existing archive, not proof of complete play-level coverage within every represented game. Some games versus FCS opponents may be included by the FBS filter. The 95 absent game IDs warrant coverage investigation, not an assumption they are recoverable by another request. These play-coverage counts differ in scope from the paired next-game training cohort.

## When broader acquisition is justified

A complete, cached archive of **in-scope** plays is justified for fitting contextual baselines, rare-event estimates and reproducible alternative aggregations. The present archive already provides most of that starting point. Inventory and reuse it first; backfill only verified missing or insufficient partitions. Expanding to older seasons, other divisions or postseason needs a stated target and coverage/domain validation, and would change the current experiment scope. Keep 2026 quarantined from development.

Player-play associations could support a separate participation study, but should be sampled for historical coverage and attribution completeness before bulk acquisition. Ordinary play-by-play cannot be assumed to provide complete eleven-player participation, blocking assignments or defensive alignments. CFBD's current `/plays` schema exposes structured game state and outcomes; `/plays/stats` returns player/stat associations with a documented 2,000-record limit, which requires explicit completeness handling. See [official Plays API](https://api.collegefootballdata.com/api/plays), consulted 2026-09-28. Current documentation is not proof of uniform historical field availability.

Raw outcomes and context support internally fitted, time-safe baselines. A provider's present-day PPA/adjusted rating should not be assumed historically available or trained without later data. The proposal does not authorize importing pregame win probabilities or market inputs.

## Status

Untested proposal only. No expected improvement size is asserted. No new model, data pull, scheduler action or F13 artifact was created. The scientific goal remains paused.

See also: [Measured lessons](Nextgen-Empirical-Lessons-2026-09-28) · [F09](Fingerprint-F09) · [Experiment contract](Next-Generation-Fingerprint-Experiment).
