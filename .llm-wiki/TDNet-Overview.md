---
type: synthesis
up: "[[Home_TDNet]]"
tags: [project, research, operations]
---

# TDNet Overview

TDNet is a Python research package for time-dependent college-football team fingerprints, matchup prediction, weekly publication, and season simulation.

## Project flow

TDNet fetches authorized data from CollegeFootballData (CFBD), builds team-game tables and time-available fingerprints, forms matchup features, fits game-margin models, and creates evaluation and publication artifacts. The package also uses approved model outputs in TD Sim season simulations. See [Package Architecture](Package-Architecture) for the main code paths.

## Research boundary

The 2026 confirmatory study is FBS-versus-FBS regular season only. Margin is the primary objective and winner metrics are secondary. The 2025 season is retrospective evidence already consumed by development; it is not an untouched test set. The 2026 season is prospective, and its outcomes must not enter model selection or retraining for the frozen confirmatory system.

The canonical 2026 protocol defines 54 scientific cells across six model architectures and F0–F8. Only the 42 market-free F0–F6 cells are eligible for prospective predictions and polls. F7 and F8 are market comparators. See [Fingerprint Ladder](Fingerprint-Ladder), [Confirmatory Protocol 2026](Confirmatory-Protocol-2026), and [Model and Poll Surfaces](Model-and-Poll-Surfaces).

## Operating boundary

Weekly prediction bytes are frozen at the declared deadline. Corrections use a hash-linked append-only amendment ledger; they do not overwrite the original bundle. Publication requires explicit owner approval. The project does not claim that TDNet beats Vegas, that greater model complexity always helps, or that feature importance is causal without evidence. See [Weekly Publication Workflow](Weekly-Publication-Workflow).

## Source authority

Use `configs/publication/confirmatory_protocol.yaml` as the machine-readable source of truth for the 2026 confirmatory design and [Confirmatory Protocol Source](Source-Confirmatory-Protocol) for its publication-facing interpretation. The README is the current high-level project map. The preserved [TDNet Master Plan Source](Source-TDNet-Master-Plan) contains valuable design history and older protocol language, with superseded claims flagged.

## Source pages

- [README Source](Source-README) — project purpose, workflows, package entry points, and public-data boundaries.
- [Confirmatory Protocol Source](Source-Confirmatory-Protocol) — canonical study design and invariants.
- [Model Guide Source](Source-Model-Guide) — current implementation families and caveats.
- [Weekly Operations Source](Source-Weekly-Operations) — recurring runbook and notebook responsibilities.
- [TDNet Master Plan Source](Source-TDNet-Master-Plan) — dated design record with explicit stale-section warnings.

## Project history

The checked-out Git history begins at a public-release commit in August 2026 and is not a complete origin story. Dated planning, run-inventory, decision, and release records establish a partial project timeline and distinguish later completion from earlier interrupted work. See [Project History and Directions](Project-History-and-Directions) and [Deferred and Abandoned Directions](Deferred-and-Abandoned-Directions).

## See also

[Package Architecture](Package-Architecture) · [Fingerprint Ladder](Fingerprint-Ladder) · [Temporal Data Semantics](Temporal-Data-Semantics) · [Confirmatory Protocol 2026](Confirmatory-Protocol-2026) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [Weekly Publication Workflow](Weekly-Publication-Workflow) · [Project History and Directions](Project-History-and-Directions) · [Deferred and Abandoned Directions](Deferred-and-Abandoned-Directions) · [Opponent Adjustment Source](Source-Opponent-Adjustment) · [Next-Generation Fingerprint Experiment](Next-Generation-Fingerprint-Experiment)
