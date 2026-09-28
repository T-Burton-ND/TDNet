---
type: concept
up: "[[Fingerprint-Ladder]]"
extends: "[[Fingerprint-F06]]"
tags: [fingerprints, next-generation, design]
---

# Fingerprint F09

F09 is the planned market-free information generation for game microstructure; its structured-play builder is implemented, with canonical artifacts verified; evaluation remains incomplete.

## Current snapshot — 2026-09-28

Full A/B/C screening is complete: 60 successful cells. C M4 median development MAE is 13.214108 / 12.728329 for 2024 / 2025. The 90-feature progressive C pool remains unscreened. The completed C correlation audit reports 43 pairs and authorizes no removal. No LR/PR trial was started. See the [verified paused snapshot](Nextgen-Results-Snapshot-2026-09-28) for all metrics, evidence paths, source gaps and scheduler counts as of 12:52 UTC. This supersedes earlier live-status statements; the goal is paused and existing queued cells may still dispatch.

## Information added and sources

- **Parent:** F06, same a/b/c design only. F07/F08 are excluded.
- **New family:** Structured numeric plays and drives: whole game, halves, quarters, middle eight, field position, pace, scoring opportunities, efficiency, and situational success. No free-text parsing.
- **Candidate CFBD sources:** `/plays, /drives, /games/teams, /stats/game/advanced`. Availability and as-of coverage must be audited before materialization.
- **Family rule:** Include offensive and defensive Q4-minus-Q1 rushing YPA, with quarter-trend investigation. Q4 garbage only when every qualifying Q4 play has absolute lead >16; Q1–Q3 thresholds are 28/24/21. Garbage overrides clock leverage.

## Representations and lineages

For each design `a`, `b`, and `c`, build `F09_F_*`, `F09_LR_*`, and `F09_PR_*`. `a` is relatively atomic; `b` adds football-informed interpretable interactions; `c` compresses into film-intuitive Excel-reproducible equations with at most five source inputs. Full inherits previous full. Late reduction prunes the current full version. Progressive reduction inherits previous reduced version, adds this generation's complete new family, then prunes; `F09_PR_*` never crossbreeds design tracks.

## Temporal, matchup, and reduction semantics

All dynamic quantities use only completed regular-season performance before the **next target game**. FBS–FBS regular-season games are targets; FCS games can be historical context. Postseason numeric performance and 2026 design evidence are excluded. Market and pregame win-probability inputs are forbidden. Team-week sources remain ordinary numeric features with an explicit counterpart and comparison equation; a pruning pair is atomic and can be removed only when both members qualify. A reduced fingerprint retains at least 60 concrete F06 features and ten from each added generation, subject to explicit shortfall disposition if a family cannot support ten legitimate signals.

Every derived feature's manifest records endpoints/columns, exact formula and units, availability and cutoff, aggregation/sample/missingness, garbage handling, counterpart and matchup formula, provenance, code path, and version. Fit scaling or C weights only on pre-2026 development evidence. Screen with the same M2/M4 setpoints and architecture-normalized source-level permutation SHAP; evaluate next-game margin.

## Required rushing diagnostics

The 2026-09-27 diagnostic run plots prior Q4-minus-Q1 rushing YPA against next-game margin and empirical win rate on the assembled 2010–2025 F09_F_a corpus. Offense has 22,132 finite team-target observations, Pearson −0.0803 and Spearman −0.0763 with next-game margin. Defense (rushing allowed) has 22,099 observations, Pearson 0.1005 and Spearman 0.0965. These weak associations do not establish incremental model value or justify removal. The signs should not be rewritten as evidence for a simple fatigue explanation. Repeated teams and opposing rows are dependent.

PNG/Markdown pairs and a lightweight index are under `feature_diagnostics/`; source script: `scripts/nextgen_rushing_diagnostics.py`. Both plots were visually checked. SHAP and survival/reduction status are explicitly pending; these mandatory plots are not claimed to be consensus-ranked final diagnostics.

## Status

**Initial implementation snapshot; current screening status appears below.** The 2010–2025 Stage-C acquisition invocation completed all 297 pending requests (`results/preflight/stage_C_latest.json`, 2026-09-27T17:06:35Z), bringing the experiment ledger to 850 reserved outbound attempts. Request completion does not prove exhaustive provider game coverage.

`nextgen_f09.py` implements a trailing 12-regular-game state with a reconstructed 48-hour reporting lag, reciprocal quarter rushing differences/slopes, and final writes through `NextgenFeatureBuilder`. Materialization completed for the 2010–2025 schedule: each design has 22,640 team-target rows, with 38/46/30 features for a/b/c. Independent `NextgenModelBoundary.from_canonical` reads passed all three artifacts; mean cell missingness is 0.0102/0.0183/0.0112 respectively. Evidence: `results/f09_materialization.json` and the boundary verification command on 2026-09-27. These are new-family artifacts. Subsequent full-fingerprint assembly produced 22,302 team-target rows in each F09_F_a/b/c with 265/283/185 features, excluding 44 F06 parent games lacking paired new-family coverage. Assembly preserves parent values and checks new values against canonical families (`fingerprints/F09_F_*/provenance.json`). Reduced LR/PR fingerprints and all model evaluation remain unfinished. The joint design/screening/F09/microstructure tests passed 16 checks on 2026-09-27; synthetic model checks are not football evaluation evidence. That initial assembly was superseded after the user authorized removal of the two coaching SP+ fields: current F09 a/b/c counts are 263/281/183, with the same 22,302 rows. The source audit now authorizes training.

 F09–F12 full screening is now running; final recommendations remain unfinished. See the [experiment contract](Next-Generation-Fingerprint-Experiment) and repository `docs/nextgen_fingerprints/README.md`.

See also: [Fingerprint Ladder](Fingerprint-Ladder) · [Next-Generation Experiment](Next-Generation-Fingerprint-Experiment) · [Fingerprint F06](Fingerprint-F06) · [Fingerprint F10](Fingerprint-F10)

## First authorized progressive pool — 2026-09-27

The accepted floor-sized [F06 reduced C](Fingerprint-F06) now authorizes the F09_PR_c pool. Actual `progressive_pools/F09_PR_c/provenance.json` records 60 inherited F06 features plus all 30 new F09 features on 22,302 team-target rows (11,151 games), with no excluded full-reference games. Parent receipt SHA-256 is `6d2a55a6855369a7b8599652578f225b4339968aec35680441c5c5add737a23c`. This is an unpruned pool, not an accepted or screened progressive fingerprint. Selection and screening require terminal F09 full-reference evidence.

## Full C M2 screening complete — 2026-09-27

F09 C now has ten successful M2 and seven successful M4 results; three M4 cells remain active. Full C M2 median development MAE is 13.379143 / 12.920867, RMSE 16.943794 / 16.410657 and Brier 0.196414 / 0.187185 for 2024 / 2025. Evidence: `results/F09/full_c_m2_interim.json`. These design-informed results do not authorize the incomplete joint full reference or compare generations on different cohorts.
