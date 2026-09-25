---
type: index
up: "[[SCHEMA_TDNet]]"
tags: [conventions, vocabulary, kg]
---

# Edge Types

Canonical list of typed-edge predicates available in this wiki. Pages
reference these by inline link so readers can navigate to a definition:
`[*partOf*](Edge-Types#partOf)`, `[*supports*](Edge-Types#supports)`,
etc.

Forward predicates are author-facing: assert them in YAML frontmatter
or as inline Variant 1 annotations on body links. Inverse predicates
are materialised by the KG build pipeline; do not assert them in
source documents.

The full Variant 1 inline annotation form and the rules for choosing
when to use frontmatter versus inline are in
[SCHEMA_TDNet](SCHEMA_TDNet).

## up

Hierarchy parent. Every page has one. Frontmatter only (every page).

## source

Cites this external literature item (paper, book, web page) as a
source for claims on this page.

## extends

Builds on the intellectual content of the target. Intellectual
lineage. Use for "this concept generalises / refines / continues that
one." Inverse: `extendedBy`.

## supports

Provides evidence or argument FOR the target's claim. Inverse:
`supportedBy`. Use when this page strengthens the case the target
makes.

## criticizes

Argues against the target's claim, identifies a flaw, or proposes a
counter. Inverse: `criticizedBy`. Use when this page weakens or
challenges the target.

## concept

This page is about (discusses) the target concept. Inverse:
`conceptOf`. Use for literature or implementation pages pointing at
the concept they discuss.

## partOf

This page is a structural component or stage of the target page (a
larger composite). Inverse: `hasPart`.

## dependsOn

This page's design or claim depends on the target page being in place
(architectural, logical, or argumentative dependency). Inverse:
`prerequisiteOf`. Distinct from `extends` (which is about intellectual
lineage, not dependency).

## defines

This page is the canonical definition source for the target. Inverse:
`definedBy`. Distinct from `concept` (which is "discusses" rather than
"is the definition of").

## resolvedBy

An open question, gap, or concern raised by this page is answered or
addressed by the target page. Inverse: `resolves`. Useful for design
docs that explicitly mark which downstream work resolved them.

## incorporatedInto

Content from this page (a passage, a recommendation, a methodology
section) appears in or has been adopted by the target page. Inverse:
`incorporates`. The staging-document pattern: a working document feeds
the formal deliverable.

## outOfScopeFor

This page's content is explicitly NOT to appear in the target page.
Inverse: `excludes`. Useful for keeping speculative or exploratory
material cleanly separated from formal deliverables (proposals,
specifications).

## precedes

This page precedes the target in a sequence (version history, stage
progression, phase order). Inverse: `precededBy`.

## feedsInto

Content, signals, or data from this page flow into the target (data
pipeline or system component relationship). Inverse: `informedBy`.
Distinct from `supports` (evidence for a claim) and `dependsOn`
(architectural prerequisite).

## related

Fallback lateral connection when none of the above fits. Prefer
something more specific if it applies. Heavy use of `related` is a
signal that the more specific vocabulary should be extended.

## mentions

Body cross-reference; emitted automatically by the KG extractor for
any plain link in the page body. Not asserted manually.

## See also

- [SCHEMA_TDNet](SCHEMA_TDNet) — wiki conventions, including the Variant 1 inline annotation form that uses the predicates on this page
