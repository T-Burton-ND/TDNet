---
type: concept
up: "[[Fingerprint-Ladder]]"
extends: "[[Fingerprint-F10]]"
tags: [fingerprints, next-generation, design]
---

# Fingerprint F11

F11 is the planned market-free information generation for coaching history; its features are not yet implemented or evaluated.

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

## Status

**Setup/design only.** No F09–F12 training or recommendation has been run. See the [experiment contract](Next-Generation-Fingerprint-Experiment) and repository `docs/nextgen_fingerprints/README.md`.

See also: [Fingerprint Ladder](Fingerprint-Ladder) · [Next-Generation Experiment](Next-Generation-Fingerprint-Experiment) · [Fingerprint F10](Fingerprint-F10) · [Fingerprint F12](Fingerprint-F12)
