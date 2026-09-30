---
type: synthesis
up: "[[Next-Generation-Fingerprint-Experiment]]"
derived_from: ["[[Nextgen-Full-Archive-Review-2026-09-30]]", "[[Nextgen-Empirical-Lessons-2026-09-28]]", "[[Fingerprint-F13-Proposal]]"]
tags: [fingerprints, roadmap, proposal, untested, F13, F14, F15, F16, F17]
---

# Proposed F13–F17 Research Roadmap

Five research generations are feasible from the saved archive, with taxonomy repair first and all predictive benefits untested.

## Scope and evidence

The user requested plans, documentation and a Markdown-only public push. The user subsequently authorized necessary repairs/additions; the local classifier repair is complete and documented in the review. New experiments, data requests, canonical rebuilds, lineage changes and model acceptance were not performed. The [full archive review](Nextgen-Full-Archive-Review-2026-09-30) provides measured coverage, field checks, hashes and limitations. The scientific goal remains paused. Generation names below are proposed reservations, not completed or formally accepted fingerprints.

The archive already contains the main raw material. In the authoritative 2010–2025 regular-season scope there are 2,351,115 plays, every one joining to a drive; 2013–2025 attribution has 10,639 successful approved game caches. Additional API spending is not a prerequisite for any direction below. However, the original saved F09 classifier omitted historical `Pass Completion` / `Pass Interception` labels, and actor coverage is incomplete and uneven. The source repair has not been propagated into preserved canonical/model artifacts. Source volume is not sufficient validation.

The [empirical lessons](Nextgen-Empirical-Lessons-2026-09-28) motivate this order: F09 M4 gains were more consistent than M2 gains; F10 A/M2 improved in most shared configurations; F11 A/M2 and F12 A comparisons worsened on common games. These observations support careful incremental experiments, not an assumption that later or wider generations will win. The later completed-results table adds runs but does not replace paired comparisons with raw-median rankings.

| Proposed generation | Research question | New information beyond existing generations | Data readiness |
|---|---|---|---|
| F13 | How well does a team perform relative to the situations it faces? | Internally fitted context residuals and outcome-tail shape | Strong raw foundation; taxonomy and as-of fitting must be validated |
| F14 | How does an offense sustain or lose a possession? | Ordered series transitions and conditional drive failure/finishing paths | Plays and drives join throughout; sequence ambiguities and missing events need gates |
| F15 | Which observed players carry particular situational roles? | Context-specific actor concentration and redistribution | Feasible for observed offensive roles in 2013–2025; incomplete attribution limits claims |
| F16 | Is a team's adjusted performance stable or changing? | Changes and uncertainty in newly corrected, context-adjusted states | No new source needed; depends on validated F13/F14 states and adequate history |
| F17 | Against which opponent styles does a team's performance transfer? | Prior-style-conditioned response profiles rather than strength adjustment alone | Sufficient underlying games; team-level support and stability remain unmeasured |

## Prerequisite: separate a source correction from a scientific addition

The audit found 118,823 `Pass Completion` and 5,816 `Pass Interception` rows excluded by the original saved F09 pass allowlist; its SHA matches saved F09 partition metadata. The concentration in 2010–2013 and the reappearance of completion labels in 2025 make a whole-archive taxonomy map necessary. Only 36 literal Sack rows occur in 2013, so a two-label patch alone is not a complete semantics audit.

Before F13, propose a versioned source-normalization step: reconcile structured type definitions with box-score aggregates where available, retain original labels, distinguish administrative/penalty/no-play events, and audit down/distance transitions. Resolve or exclude the 28 repeated sequence positions before using chronology. Preserve original artifacts and results. A corrected-reference-only comparison must be distinct from corrected reference plus new features; otherwise a “F13 improvement” could simply reflect repairing inherited inputs. The authorized local repair now recognizes the two catalog-confirmed pass labels and handles period ordering/ambiguous score context, with twelve focused tests passing. An offline full-archive check recognizes 124,639 additional dropbacks. The broader reconciliation and correction-only model comparison remain unexecuted; original canonical/model artifacts remain intact.

## F13 — context-adjusted performance and distribution shape

**Question.** Does an apparently efficient offense create repeatable gains after accounting for down, distance, field position, pre-play score/time, play class and previously known opponent strength? Can a defense limit ordinary gains even if a few explosive plays dominate its mean?

**Candidate state.** Separate offensive and defensive mean yardage/success residuals, residual lower-tail frequency, and concentration of positive production in unusually large gains. Compare a compact set of supported situations, such as early downs and obvious passing situations. A residual is observed outcome minus a baseline expected outcome for that context. Estimate success and yardage baselines separately; their units differ. Residualized rates and distribution summaries add information beyond F09's raw success/YPA, rushing components and coarse situational averages. Do not merely rename those existing rates.

**Construction.** Use only completed prior regular-season plays under the documented reporting lag. Raw score fields are post-play; reconstruct pre-play score before conditioning. Fit contextual baselines and opponent effects only from data earlier than the feature cutoff; construct training examples out of time as well. Cross-fitted or rolling historical baselines must not see later outcomes. Begin with transparent pooled/shrunk estimates; compare context-only and context-plus-opponent versions separately. Provider PPA or present-day ratings are not substitutes for this temporal proof.

**Readiness/gate.** Rich context exists, but the reported 99.68% complete-context share is among plays recognized by the old classifier, not proof of post-correction coverage. Recompute feature support after normalization and distinguish overtime, garbage-time rules and genuine missingness. Freeze context bins, shrinkage and minimum support from training data. Reject a feature if its apparent effect is confined to taxonomy eras or changing missingness. The [F13 proposal](Fingerprint-F13-Proposal) retains the original rationale and lineage discussion.

## F14 — series sustainability and possession failure paths

**Question.** Do teams repeatedly move the chains, recover after a setback, and finish drives from comparable starting situations, or depend on a single explosive play?

**Candidate state.** Conditional first-down conversion after an unsuccessful early down; distribution of consecutive successful series within a drive; probability a drive stalls after reaching a specified field zone; and scoring versus empty possessions conditional on comparable starting field position. Measure offense and defense separately. A compact finishing residual can compare observed drive outcome to a past-only baseline for starting field position and situation.

**Increment.** F09 already has mean drive start, net points per drive, points per opportunity, opportunity/quality-drive rates and seconds per play. F14 adds sequence and transition structure, not those averages under new names. Proposed recovery-after-setback rates are descriptive; they do not establish play-calling causality or a psychological “resilience” trait.

**Readiness/gate.** All archived plays join to drive identity, and sequence positions are present, but identity joins do not prove every event was recorded. Before extracting a series, reconcile down resets, accepted penalties, touchdowns, turnovers, halftime, kneels and drive outcomes using structured fields. Retain an unknown category or exclude an ambiguous sequence rather than inventing a chain. Report excluded drives by year and team. Require supported denominators and compare incrementally against the corrected F09 reference and F13. No drive parser or feature materialization was built for this plan.

## F15 — observed situational player roles and redistribution

**Question.** Is production broadly distributed, or concentrated in particular observed rushers/receivers in particular contexts? Does the distribution of documented contributors change before the next game?

**Candidate state.** Context-specific rusher or receiver concentration, overlap of documented contributors between recent and longer prior windows, and change in the share of observed short-yardage rushes or receptions assigned to leading prior contributors. Use play context from `/plays` and identities/stat labels from successful `/plays/stats` game caches. Deduplicate validated actor/play/stat associations and keep source-quality support alongside each denominator.

**Increment.** F10 already includes aggregate usage, observed experience and season-to-season continuity; F12 includes offensive rooms. The addition is linking a documented actor to an individual play's situation. Focus first on rush/reception roles; “Target” rows are not yet a validated complete denominator and cannot simply be treated as all intended targets. Do not claim routes, snap shares, blocking, defensive assignments, injury absence, or current lineup knowledge. Absence of attribution is not nonparticipation.

**Readiness/gate.** Restrict initial candidate eligibility to supported 2013–2025 history, preserving early-season cold starts. The two sparse 2012 probes are not a usable historical season. Rush association coverage among successful game caches varies substantially: 91.45% in 2013, 89.43% in 2021, 97.45% in 2025. Canonical outcomes and athlete-attribution yardage disagree in 13,341 rows in the full audit; conflicting records need explicit quarantine or a documented source authority. Validate rates by team, year and context before choosing support cutoffs. Keep both unknown-event share and included-event count as diagnostics, not automatically model inputs. Test complete-support subsets to expose selection sensitivity. No model may turn provider gaps into zeros or use future roster snapshots to infer a player was available.

## F16 — change and uncertainty in adjusted team state

**Question.** After normalizing situations, is recent performance meaningfully different from a team's established level, or simply noisy small-sample variation?

**Candidate state.** A proposed short-versus-long comparison of F13 residual means (for example, last three available games versus last twelve), changes in lower-tail risk, and dispersion of F14 possession outcomes across games. These window lengths are starting proposals, not measured optimal choices. Carry counts/effective support and shrink differences when evidence is weak. Distinguish a change in situations faced from a change in performance conditional on those situations.

**Increment.** F06 already contains five-game volatility fields, F09 has within-game quarter/half trends, and F10 has season continuity. F16 is justified only if it measures changes in the newly normalized contextual distributions, with an explicit noise/support model. Raw moving averages, generic volatility or renamed existing trends are not a new generation.

**Readiness/gate.** Available chronology supports construction without acquisition, but team-season support has not been measured for these proposed windows. Avoid season-boundary claims based on a single prior game; declare how history crosses seasons and how low-support states remain missing. Hold window/shrinkage choices fixed before evaluating development years. Compare against F13/F14 plus the existing volatility fields. If differences are entirely explained by those fields or missing-data artifacts, prune/defer the family rather than force a larger fingerprint.

## F17 — opponent-style response and transferability

**Question.** Does a team's adjusted performance transfer across different opponent tendencies, or is it concentrated against a narrow set of styles?

**Candidate state.** Past-only response slopes or pooled conditional residual contrasts against opponents with different rush/dropback mix, tempo or explosive-play tendencies. Keep a small set of interpretable axes; candidate axes themselves must be estimated from each opponent's earlier games. Separate offensive and defensive response profiles. A subsequent matchup builder can combine one team's stored response profile with the opponent's pregame style state; the canonical team state remains independent of the identity or outcome of the target opponent.

**Increment.** F13 adjusts for situation and opponent strength. F17 tests whether the *relationship* between team performance and a previously known style varies, which is distinct from a single strength correction or generic offense-minus-defense difference. Avoid claiming causal schematic advantages: these are conditional associations and may reflect schedule selection.

**Readiness/gate.** Existing plays, drives and schedule supply the ingredients; no scheme/formation labels are assumed. Sparse or unbalanced opponent exposure is the main unresolved risk. Use pooled shrinkage rather than an unrestricted opponent-by-team lookup. Validate range of exposure and stability on earlier time blocks, then compare against F13 alone and the preceding accepted parent on the same games. If there is insufficient within-team style variation, retain a pooled diagnostic or defer the generation. This fifth direction is more conditional than F13/F14; its readiness is not established by the total number of plays.

## Reserve directions and sources that remain unsuitable

Rest/travel/environment conditioning is a possible substitute if F17 lacks support. The archive has 12,735 schedule-matched weather records with temperature and wind and 12,943 geocoded game venues, but actual weather is retrospective and current venue metadata does not prove historical attributes. Only prior-game observed weather may describe past performance; target-game actual weather cannot serve as a pregame forecast. F06 already includes travel differences, so a reserve family must establish incremental content. Full defensive participation, line blocking and historically certified current rosters remain unsupported; spending remaining API budget on the same empty responses would not solve those semantics.

## Shared implementation and evaluation contract for a future authorized phase

1. **Freeze evidence and repair separately.** Preserve raw caches, taxonomy versions, source hashes and original results. Produce the correction-only reference before interpreting any new family's effect. No additional acquisition is required by this plan.
2. **Preserve temporal boundaries.** Use authoritative completed regular-season source games; maintain source kickoff < declared availability < target kickoff. Existing builders use documented reconstructed lags, not archived provider publication timestamps. Verify each family's actual rule; F09's trailing state uses kickoff + 48 hours, while inherited weekly sources have their own conservative weekly lag. Week-0 static claims need prior availability proof. Fit transformations, baselines, imputation and shrinkage without future observations. Keep 2026 and postseason outside this development archive.
3. **Measure candidate coverage before training.** Report source games, plays, actor support, missingness, excluded sequences, and target-team pairs by year/team. A cached endpoint is not a feature manifest. Freeze minimum sample rules and missingness policy on training-era evidence; do not invent cutoffs because they improve 2024/2025 scores.
4. **Preserve design and lineage rules.** Propose atomic A states first, interpretable B interactions and C composites only where justified. C equations remain Excel-reproducible with at most five source inputs; paired offense/defense counterparts and provenance remain explicit. Full and reduced tracks retain same-design accepted-parent rules. Using F09/F10 as an earlier reference is scientifically useful; making either an actual new parent is a protocol decision, not silently bypassing F12. Use one-family additions/ablations so a cumulative ladder cannot hide an unhelpful addition.
5. **Control both training and evaluation cohorts.** Compare reference and addition with identical training game IDs, evaluation game IDs, preprocessing rules, frozen configurations and seeds within each architecture. For restricted F15 eras, refit its comparator on that same restricted training population in a later authorized experiment. Matching only test games cannot isolate a family when training coverage changes. Report common-cohort results and operational coverage separately.
6. **Report tradeoffs, not just a winning median.** Include per-configuration paired MAE changes, median paired change, difference of medians and improvement counts, plus Brier, winner/upset recognition and ATS diagnostics. Market data remain evaluation-only. Configuration outcomes are correlated, not independent statistical replications. No acceptance based solely on one design-informed year or raw cross-cohort medians. Keep M2 and M4 separate; previous results do not establish a universal winner.
7. **Stop when evidence fails.** Defer a family that lacks meaningful support, duplicates an existing formula, leaks future information, or fails the predeclared acceptance gate. Five proposed generations do not obligate five accepted generations. Plans may be implemented independently where feasible, but no parallel experiments were started for this review.

See also: [Full archive review](Nextgen-Full-Archive-Review-2026-09-30) · [F13 proposal](Fingerprint-F13-Proposal) · [Empirical lessons](Nextgen-Empirical-Lessons-2026-09-28) · [Experiment contract](Next-Generation-Fingerprint-Experiment).
