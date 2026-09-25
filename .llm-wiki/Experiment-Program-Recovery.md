---
type: synthesis
up: "[[Project-History-and-Directions]]"
tags: [experiments, operations, history]
derived_from: ["[[Source-Run-Inventory]]", "[[Source-Release-and-Operations-History]]", "[[Fingerprint-and-Model-Decision-History]]"]
---

# Experiment Program Recovery

This page records the transition from a July 2026 storage-limited experiment snapshot to later documented completion and publication artifacts.

## July 20, 2026: interrupted run state

A read-only audit was made after TDNet SGE jobs disappeared from `qstat`. `/groups` reported 20T total, 20T used, 119G available, and 100% utilization; recent accounting records were not yet available. Several task `status.json` files still said `running` despite no matching jobs, so were treated as stale unless a valid completion artifact existed. No jobs were submitted during the audit.

The then-current incomplete scope had 33,540 tasks remaining after excluding the 22-task `legacy_balanced_recovery` array dropped by owner decision. Fourteen stale failure records belonged to a killed pre-fix attempt whose `meta_df` lacked matchup-pairing fields; the corrected run had passed that failure zone before storage became limiting. The detailed array counts are preserved in [Run Inventory Source](Source-Run-Inventory). These counts are a historical snapshot, not a current backlog.

## Later documented completion

The release status dated 2026-08-17 records corrected F5/F6/F8 HPS completion at 13,920/13,920, full F0–F8 scientific refit at 54/54 cells, and a corrected-F6 wide bundle with 34 learned estimators plus two equal-weight ensembles. These used data through 2025 with 2026 excluded from fitting. The true 2025 holdout regeneration instead used checkpoints through 2024.

All four 2025 regeneration packages later completed: through-2025 operational dry runs and through-2024 true-holdout packages for scientific and wide rosters. Each package had 17 poll weeks, 16 prediction weeks, and zero poll failures. The dry run checks pipeline behavior; it is not a 2025 holdout performance estimate. The through-2024 package supplies retrospective 2025 evaluation.

## Publication became a weekly record

Visible repository history after the 2026-08-13 release commit shows an operational publication sequence: scientific immutable pipeline and Week 0 package on 2026-08-28; Week 0 postgame figures on 08-30; Week 1 prediction graphics and Thursday games on 09-03; prediction and postgame archives on 09-10; Week 2 results/Week 3 predictions and cumulative performance plus Vegas baseline on 09-13; Week 3 results/Week 4 predictions and margin/scientific figures on 09-21; expanded weekly and scientific tracking on 09-23. The checked-out Git history does not expose the full pre-August project origin.

## See also

[Project History and Directions](Project-History-and-Directions) · [Deferred and Abandoned Directions](Deferred-and-Abandoned-Directions) · [Weekly Publication Workflow](Weekly-Publication-Workflow) · [Run Inventory Source](Source-Run-Inventory) · [Release and Operations History Source](Source-Release-and-Operations-History) · [Fingerprint and Model Decision History](Fingerprint-and-Model-Decision-History) · [TDNet Master Plan Source](Source-TDNet-Master-Plan)
