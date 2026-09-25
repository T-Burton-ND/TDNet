---
type: source-summary
up: "[[Weekly-Publication-Workflow]]"
source: source-archive/docs/publication_2026/WEEKLY_OPERATIONS.md.txt and source-archive/docs/publication_2026/THREE_NOTEBOOK_WORKFLOW.md.txt
tags: [source, operations, publication]
---

# Weekly Operations Source

The weekly operations runbook and notebook guide define TDNet’s recurring 2026 data refresh, review, prediction, scoring, and owner-ballot workflow.

## Contribution

The runbook identifies Monday refresh/inspection, Tuesday frozen-bundle preparation, and Sunday retrospective scoring. The notebook guide assigns distinct responsibilities to three recurring publication notebooks and one reader-facing reproduction notebook.

## Specific claims relevant to the wiki

- Monday performs the scheduled CFBD refresh and certifies endpoint completeness; an owner must approve the inspection report before Tuesday proceeds.
- Tuesday produces one deadline bundle with locked-roster predictions and poll material. It does not send an X post.
- Sunday scores the immutable prediction bundle from cached outcomes; it is network-free and does not issue a second CFBD request.
- The three recurring notebooks are weekly predictions, manual Top-25 poll, and Sunday results; reproduction is a separate reader workflow using the reader’s own API key.
- Do not install cron before run times are chosen and `CFBD_API_KEY` is configured. Social posting remains draft-only unless separately configured and approved.

## Where it connects

The sequence and safeguards are summarized in [Weekly Publication Workflow](Weekly-Publication-Workflow). Roster boundaries are in [Model and Poll Surfaces](Model-and-Poll-Surfaces); package-level responsibilities are mapped in [Package Architecture](Package-Architecture) and the project-wide boundaries in [TDNet Overview](TDNet-Overview).

## Quotes worth keeping

> “This is the only scheduled CFBD refresh of the week.”

## Source links

- `source-archive/docs/publication_2026/WEEKLY_OPERATIONS.md.txt`
- `source-archive/docs/publication_2026/THREE_NOTEBOOK_WORKFLOW.md.txt`

## See also

[Weekly Publication Workflow](Weekly-Publication-Workflow) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [Package Architecture](Package-Architecture) · [Confirmatory Protocol 2026](Confirmatory-Protocol-2026) · [TDNet Overview](TDNet-Overview)
