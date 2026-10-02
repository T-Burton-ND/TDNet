---
type: synthesis
up: "[[Next-Generation-Fingerprint-Experiment]]"
derived_from: ["[[Nextgen-Empirical-Lessons-2026-09-28]]", "[[Fingerprint-F09]]"]
tags: [fingerprints, proposal, untested, play-by-play]
---

# Fingerprint F13 Proposal

This page originally proposed context-adjusted play performance and outcome distributions after measured F09 gains. The later [F13–F17 A screen](Nextgen-F13-F17-A-Screen-2026-10-02) implemented a context-only F13 A candidate after a separate source correction and found worse median paired MAE in both design-informed development years for M2 and M4; opponent-strength adjustment, B/C designs, reduction and lineage acceptance remain untested.

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

## Acquisition is complete; validation is now the priority

The approved 2013–2025 attribution pass and best-effort recovery are complete: 10,639 of 10,704 approved games have successful caches and 65 returned empty responses. See [archive completion](Nextgen-Historical-Archive-Completion) for exact scope, missingness and budget. The [full archive review](Nextgen-Full-Archive-Review-2026-09-30) verifies file integrity, structured field support, historical label omissions and actor semantics. The [F13–F17 roadmap](Nextgen-F13-F17-Roadmap) supersedes the earlier sample-first acquisition advice with a concrete sequence of research proposals.

No further pull is needed to start these candidate audits. More seasons, divisions or postseason would change scope and require a stated research need. Existing player associations support observed roles, not complete eleven-player participation or blocking assignments. Provider PPA/adjusted ratings remain temporally unverified; market inputs and provider pregame win probabilities remain excluded. Keep 2026 quarantined.

Before F13, separate a versioned taxonomy/order repair from new information. The audit found historical pass labels excluded from the saved F09 classifier; the local repair and validation are recorded in the review. Existing model scores remain the outputs of their original source version. Any later corrected-reference comparison must precede a claim that F13 adds predictive value.

## Status

The original proposal/review created no model or F13 artifact at its September 2026 cutoff. The later common-cohort A experiment is measured and negative; see the [screen results](Nextgen-F13-F17-A-Screen-2026-10-02). The broader opponent-strength and representation proposals here remain untested, and no F13 lineage was accepted.

See also: [Measured lessons](Nextgen-Empirical-Lessons-2026-09-28) · [F09](Fingerprint-F09) · [Experiment contract](Next-Generation-Fingerprint-Experiment) · [F13–F17 A screen](Nextgen-F13-F17-A-Screen-2026-10-02).

## Authorized archive acquisition — 2026-09-28

The user subsequently approved filling useful historical data within the existing budget. See [historical archive completion](Nextgen-Historical-Archive-Completion) for verified cache reuse, sampled coverage limits and progress. This authorizes raw acquisition only; model experiments remain paused and F13 remains untested.
