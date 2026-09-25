---
type: source-summary
up: "[[TDNet-Overview]]"
source: ../README.md
tags: [source, project-overview]
---

# README Source

The repository README describes TDNet’s purpose, current 2026 boundaries, common workflows, package entry points, and public-data handling.

## Contribution

The README presents TDNet as a research-oriented Python package that fetches CFBD data, builds time-dependent team fingerprints, trains matchup models, compares predictions with Vegas baselines, produces weekly poll/report artifacts, and runs TD Sim. It points readers to the 2026 confirmatory protocol and model guide.

## Specific claims relevant to the wiki

- 2025 is consumed retrospective evidence, not an untouched test set; 2026 is FBS-versus-FBS regular season only.
- The six-model scientific panel spans F0–F8 (54 margin cells); only market-free F0–F6 (42 cells) are eligible for prospective predictions and polls.
- The corrected-F6 wide-margin bundle has a broader operational roster, with 33 automated poll members plus one separate owner ballot.
- v0 row semantics use `state_after_week`; `keys_week=N` includes games through N, week 0 is bootstrap, `y_next_margin` is the safe default target, and `y_margin_this_week` is unsafe with same-row postgame/current-week features.
- Raw CFBD data are not redistributed; readers need their own API credentials for reproduction.

## Where it connects

The protocol statements ground [Confirmatory Protocol 2026](Confirmatory-Protocol-2026), [Fingerprint Ladder](Fingerprint-Ladder), [Temporal Data Semantics](Temporal-Data-Semantics), and [Model and Poll Surfaces](Model-and-Poll-Surfaces). The paths and entry points ground [Package Architecture](Package-Architecture); the recurring notebooks lead to [Weekly Publication Workflow](Weekly-Publication-Workflow).

## Quotes worth keeping

> “The 2025 season is consumed retrospective evidence, not an untouched test set.”

## Source link

Source file: `../README.md`.

## See also

[TDNet Overview](TDNet-Overview) · [Confirmatory Protocol 2026](Confirmatory-Protocol-2026) · [Temporal Data Semantics](Temporal-Data-Semantics) · [Fingerprint Ladder](Fingerprint-Ladder) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [Weekly Publication Workflow](Weekly-Publication-Workflow) · [Package Architecture](Package-Architecture)
