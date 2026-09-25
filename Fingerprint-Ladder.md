---
type: concept
up: "[[TDNet-Overview]]"
tags: [fingerprints, features, research-design]
---

# Fingerprint Ladder

The 2026 protocol defines nine named feature representations, F0–F8, with a fixed market-free path through F6 and separate market comparison tiers.

| Tier | Contents | Role in 2026 protocol |
|---|---|---|
| F0 | Preseason roster talent and games played; no observed performance statistics | Market-free baseline |
| F1 | F0 plus raw box-score families | Market-free |
| F2 | F1 plus efficiency and rate families | Market-free |
| F3 | F2 plus the selected opponent-adjustment family | Market-free |
| F4 | F3 plus situational, returning-production, and coaching context | Market-free |
| F5 | F4 plus temporal dynamics | Market-free |
| F6 | F5 plus schedule-graph features; complete market-free representation | Primary market-free representation |
| F7 | Declared market variables only | Market-only research comparator |
| F8 | F6 plus F7 | Market-aware research comparator |

Only F0–F6 are eligible for official 2026 predictions, consensus, or polls. F7 and F8 are research comparisons and must not be described as operational prediction members. These roles are established in the canonical protocol, not inferred from the older tiers in the master plan.

## Feature contract

The protocol requires each materialized frame to record exact feature names, count, family, source, availability rule, cutoff, missingness rule, transformation, market/opponent flags, version, and schema hash. Feature order is lexical and deterministic in the manifest. Fingerprint tiers have no aliases.

## Relationship to temporal semantics

The tier answers which feature families are present; it does not by itself guarantee correct timing. Every input still needs an as-of availability rule and cutoff. See [Temporal Data Semantics](Temporal-Data-Semantics) and [Package Architecture](Package-Architecture).

## Sources

- `../docs/publication_2026/CONFIRMATORY_PROTOCOL.md`
- `../configs/publication/confirmatory_protocol.yaml`
- `../configs/features/feature_ladders.yaml`
- [Confirmatory Protocol Source](Source-Confirmatory-Protocol)
- [README Source](Source-README)
- [TDNet Master Plan Source](Source-TDNet-Master-Plan) for historical feature-tier proposals only.

## See also

[TDNet Overview](TDNet-Overview) · [Temporal Data Semantics](Temporal-Data-Semantics) · [Confirmatory Protocol 2026](Confirmatory-Protocol-2026) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [README Source](Source-README)
