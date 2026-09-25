---
type: source-summary
up: "[[Confirmatory-Protocol-2026]]"
source: source-archive/docs/publication_2026/CONFIRMATORY_PROTOCOL.md.txt and ../configs/publication/confirmatory_protocol.yaml
tags: [source, protocol, 2026]
---

# Confirmatory Protocol Source

The publication protocol and its version-2 YAML configuration define TDNet’s current 2026 confirmatory study and machine-readable eligibility rules.

## Contribution

The Markdown file explains the publication-facing interpretation; the YAML declares exact study seasons, eligible tiers, model levels, seeds, poll rules, calibration, inference, prospective timing, and release constraints. The YAML is the machine-readable source of truth.

## Specific claims relevant to the wiki

- The study uses development seasons 2010–2024, treats 2025 as consumed retrospective evidence, and keeps 2026 outcomes out of training/model selection for the prospective evaluation.
- Scope is FBS-versus-FBS regular-season games; FCS and postseason are out of scope.
- The architecture levels are M1, M2, M3, M4, M5, and M10; seeds are 1701, 2718, and 3141; the full F0–F8 matrix has 54 margin cells.
- The official prospective surface is F0–F6 (42 market-free cells). F7 and F8 are research comparators only.
- Market-dependent results are not silently merged into the football-only poll. Manual ballot is separate from model ballots and metrics.
- Prediction bytes freeze at Thursday 23:59 America/New_York; corrections go through an append-only, hash-linked amendment ledger; publication requires explicit user approval.
- Inference uses season-clustered paired bootstrap historically and week-blocked paired bootstrap prospectively; Holm controls confirmatory multiplicity and Benjamini–Hochberg q=0.05 is restricted to exploratory families.

## Where it connects

These rules are expanded in [Confirmatory Protocol 2026](Confirmatory-Protocol-2026), [Fingerprint Ladder](Fingerprint-Ladder), [Temporal Data Semantics](Temporal-Data-Semantics), [Model and Poll Surfaces](Model-and-Poll-Surfaces), and [Weekly Publication Workflow](Weekly-Publication-Workflow). The repository scope is introduced in [TDNet Overview](TDNet-Overview). See [Model Guide Source](Source-Model-Guide) and [TDNet Master Plan Source](Source-TDNet-Master-Plan) for implementation context and dated older language that should not override this configuration.

## Quotes worth keeping

> “Prediction bytes are immutable after the deadline.”

## Source links

- `source-archive/docs/publication_2026/CONFIRMATORY_PROTOCOL.md.txt`
- `../configs/publication/confirmatory_protocol.yaml`

## See also

[Confirmatory Protocol 2026](Confirmatory-Protocol-2026) · [Fingerprint Ladder](Fingerprint-Ladder) · [Temporal Data Semantics](Temporal-Data-Semantics) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [Weekly Publication Workflow](Weekly-Publication-Workflow) · [TDNet Overview](TDNet-Overview) · [Model Guide Source](Source-Model-Guide) · [TDNet Master Plan Source](Source-TDNet-Master-Plan) · [Deferred Research Source](Source-Deferred-Research)
