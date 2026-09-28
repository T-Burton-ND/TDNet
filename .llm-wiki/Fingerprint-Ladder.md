---
type: concept
up: "[[TDNet-Overview]]"
tags: [fingerprints, features, research-design]
---

# Fingerprint Ladder

The fingerprint ladder preserves the historical F00–F08 study and adds a separate market-free next-game research path F06 → F09 → F10 → F11 → F12.

**Prediction invariant:** a dynamic team-week fingerprint uses only information available before the target game, and its central target is the team's **next-game margin**. Same-game association is descriptive, not next-game predictive evidence.

## Historical 2026 ladder

| Tier | Added information or role | Historical meaning |
|---|---|---|
| [F00](Fingerprint-F00) | Preseason roster talent and games played | Market-free baseline |
| [F01](Fingerprint-F01) | Raw box scores | Market-free |
| [F02](Fingerprint-F02) | Efficiency and rates | Market-free |
| [F03](Fingerprint-F03) | Selected opponent adjustment | Market-free |
| [F04](Fingerprint-F04) | Situational, returning, coaching context | Market-free |
| [F05](Fingerprint-F05) | Temporal dynamics | Market-free |
| [F06](Fingerprint-F06) | Schedule graph; complete 227-feature F6 | Primary market-free baseline |
| [F07](Fingerprint-F07) | Market variables only | Historical market-only comparator |
| [F08](Fingerprint-F08) | F06 plus F07 | Historical market-aware comparator |

F07/F08 retain their original meanings and are **not ancestors** of F09–F12. F00–F06 alone remain eligible for the frozen official 2026 protocol. This new research does not change frozen predictions or historical F0–F8 claims.

## New market-free information generations

| Generation | New information | Status |
|---|---|---|
| [F09](Fingerprint-F09) | Structured game microstructure | Design/setup |
| [F10](Fingerprint-F10) | Week-0 roster, recruiting, transfer, player use and production | Design/setup |
| [F11](Fingerprint-F11) | New derived coaching history | Design/setup |
| [F12](Fingerprint-F12) | Team-driven unit states | Design/setup |

F13–F15 may receive design work after F12 evaluation; their information families are intentionally unassigned. PCA would be a representation suffix rather than a new information number.

## Naming and ancestry

Use zero-padded generation, lineage, and design: `F09_PR_b`. The three designs are `a` atomic/canonical, `b` expanded football-informed interactions, and `c` compact interpretable Excel-reproducible composition with no more than five source inputs. They inherit only within the same letter.

F06 has `F06_F_a/b/c` and independently reduced `F06_R_a/b/c`. F06_F_a reproduces the canonical 227-feature baseline. Every F09–F12 generation has `F`, `LR`, and `PR` for each of a/b/c. `F` inherits the previous full representation. `LR` prunes the full current representation. `PR` inherits the preceding reduced PR (F06_R at F09), adds the complete current information family, then prunes. LR never inherits prior LR. The old F6-C/C25 study is reference evidence only.

Source features declare an explicit counterpart and matchup formula; pairs are atomic in pruning. Each final team-week fingerprint is architecture-independent. Market and pregame win-probability features cannot enter F09–F12. The 2026 season is quarantined from all new design operations, even if cached.

## Sources

- `configs/features/feature_ladders.yaml` and `docs/publication_2026/feature_manifests/F6.json` for the historical ladder.
- `configs/experiments/nextgen_fingerprints_v1.json` and `docs/nextgen_fingerprints/README.md` for the exploratory contract.
- [CFBD data availability](https://api.collegefootballdata.com/data-availability) for provider coverage, subject to actual partition audits.

See also: [Next-Generation Experiment](Next-Generation-Fingerprint-Experiment) · [Temporal Data Semantics](Temporal-Data-Semantics) · [Confirmatory Protocol 2026](Confirmatory-Protocol-2026) · [TDNet Overview](TDNet-Overview)
