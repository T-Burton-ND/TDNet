---
type: concept
up: "[[Fingerprint-Ladder]]"
extends: "[[Fingerprint-F09]]"
tags: [fingerprints, next-generation, design]
---

# Fingerprint F10

F10 is the planned market-free information generation for roster and player state; its features are not yet implemented or evaluated.

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

## Status

**Setup/design only.** No F09–F12 training or recommendation has been run. See the [experiment contract](Next-Generation-Fingerprint-Experiment) and repository `docs/nextgen_fingerprints/README.md`.

See also: [Fingerprint Ladder](Fingerprint-Ladder) · [Next-Generation Experiment](Next-Generation-Fingerprint-Experiment) · [Fingerprint F09](Fingerprint-F09) · [Fingerprint F11](Fingerprint-F11)
