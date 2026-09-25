---
type: concept
up: "[[TDNet-Overview]]"
tags: [temporal-data, leakage, fingerprints]
---

# Temporal Data Semantics

TDNet’s game-row and feature-cutoff rules determine which information is safe to use for a pregame prediction.

## v0 row meanings

The README defines v0 fingerprints with `state_after_week` semantics:

- `keys_week=N` includes games completed through week N.
- Week 0 is the preseason/bootstrap state.
- `y_next_margin` is the safe default training target.
- `y_margin_this_week` describes the completed game on that same row and is unsafe as a target when paired with postgame/current-week features.

For pregame modeling, a feature must be available by the target game’s cutoff, not merely present in a row. The 2026 protocol requires explicit availability rules and cutoffs in feature manifests. See [Fingerprint Ladder](Fingerprint-Ladder) and [Confirmatory Protocol 2026](Confirmatory-Protocol-2026).

## Market data boundary

Market/Vegas columns are evaluation context by default. Training rejects `market_*` columns unless a configuration explicitly opts into market features. Within the confirmatory study, F7 is market-only and F8 combines F6 with market information; both are comparison-only and excluded from official prospective predictions, consensus, and polls.

## Experimental time-adjusted features

The separate time-adjusted fingerprint experiment builds on opponent-adjusted frames and uses only prior seasons as the reference population for a row’s season. Its documented variants are t2.1 same-week z-scores, t2.2 season-phase z-scores, and t2.3 recency-weighted same-week z-scores. These are experiment variants, not aliases for the canonical F0–F8 ladder.

## Sources

- `../README.md`
- `source-archive/docs/time_adjusted_fingerprints.md.txt`
- `source-archive/docs/publication_2026/CONFIRMATORY_PROTOCOL.md.txt`
- `../configs/publication/confirmatory_protocol.yaml`
- [README Source](Source-README)
- [Confirmatory Protocol Source](Source-Confirmatory-Protocol)

## See also

[TDNet Overview](TDNet-Overview) · [Package Architecture](Package-Architecture) · [Fingerprint Ladder](Fingerprint-Ladder) · [Confirmatory Protocol 2026](Confirmatory-Protocol-2026)
