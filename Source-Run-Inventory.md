---
type: source-summary
up: "[[Experiment-Program-Recovery]]"
source: ../docs/technical/RUN_INVENTORY_2026-07-20.md
tags: [source, experiments, historical]
---

# Run Inventory Source

The 2026-07-20 read-only SGE inventory records the storage-limited state of the then-active TDNet experiment arrays and must be treated as a dated snapshot.

## Contribution

The inventory preserves per-array completions, incomplete task counts, stale statuses, and the audit conditions after jobs disappeared from `qstat`; it explicitly says no jobs were submitted during the audit.

## Specific claims

At the 2026-07-20 audit, `/groups` was reported as 20T total, 20T used, 119G available, and 100% utilization. After excluding the 22-task `legacy_balanced_recovery` array dropped by owner decision, 33,540 tasks remained in the then-current scope. The source does not document why that array was dropped.

Fourteen stale `publication_matrix` failure records came from the killed pre-fix attempt where `meta_df` lacked matchup-pairing fields. The corrected run had passed that old failure zone before storage became limiting. Task status files marked `running` despite no matching jobs were considered stale unless backed by valid completion artifacts.

## Where it connects

The later dated [release status](Source-Release-and-Operations-History) records corrected HPS and scientific refit completion, so the July remaining-task total must not be read as current. [Experiment Program Recovery](Experiment-Program-Recovery) preserves this sequence and the evidence boundary.

## Source link

Source file: `../docs/technical/RUN_INVENTORY_2026-07-20.md`.

## See also

[Experiment Program Recovery](Experiment-Program-Recovery) · [Deferred and Abandoned Directions](Deferred-and-Abandoned-Directions) · [Project History and Directions](Project-History-and-Directions) · [Release and Operations History Source](Source-Release-and-Operations-History)
