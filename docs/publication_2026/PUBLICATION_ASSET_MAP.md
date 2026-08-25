# Publication asset map

Use the labels below when moving between retrospective paper evidence and
prospective 2026 production. The directories are intentionally separate so a
2025 holdout artifact cannot be mistaken for a 2026 frozen checkpoint.

| Label | Location | Role | Fit cutoff |
|---|---|---|---|
| 2026 corrected-F6 wide roster | `EXTERNAL_DURABLE_ROOT/publication_artifacts/corrected_f6_wide_margin_roster/through_2025_v1/` | 34 learned margin estimators plus two equal-weight ensembles | 2010–2025 |
| 2026 learned weekly inventory | `docs/publication_2026/weekly_learned_model_inventory.csv` | 33 poll members from corrected F6; four distinct KNN variants; owner ballot separate | 2010–2025 |
| 2026 scientific F0-F8 bundle | `EXTERNAL_DURABLE_ROOT/publication_artifacts/scientific_roster_refits/f0_f8_margin_through_2025_v1/` | 54 calibrated margin cells; F0-F6 eligible for predictions/polls, F7/F8 comparative-only | 2010–2025 |
| 2025 holdout / trained through 2024 | `EXTERNAL_DURABLE_ROOT/publication_artifacts/2025_roster_regenerations/holdout_2025/` | Verified scientific and wide-roster holdout outputs | 2010–2024 |

Weekly products use one operational roster and two time-scoped directories:
`publication/<season>/week_<NN>/pre_game/` and
`publication/<season>/week_<NN>/post_game/`, plus a sibling `analysis/`
directory for requested descriptive figures. Top-25 outputs share each
package's `figures/` and `tables/`; there is no separate Top-25 directory.
Scientific weekly rehearsals, if
needed for internal checks, live under `data/publication/` and never mix with
the public weekly package.
Retrospective weeks also contain the postgame prediction PNG and scorecard. The
scientific retrospective Top-25 poll is an implied ranking view of the same
held-out prediction margins; it is not a second preregistered objective.

The wide-margin roster is operational. The scientific directory is emitted by
the same poll and prediction code paths when a runtime-ready scientific
inventory is supplied. The scientific inventory contains all 54 cells. Weekly
products use the 42 market-free F0–F6 cells; all 12 F7/F8 cells are withheld. Winner-objective,
Top-1, Top-3, duplicate private/public, and separate manual-poll output trees
are not publication products.
