# 2026 weekly and paper model roster

The weekly/paper roster is materialized by
`src/gridiron_ml/cli/publication/build_weekly_learned_model_inventory.py` from the
2026 full-roster freeze bundle. It includes every currently available learned
model role and all KNN variants. The source contains 36 rows: 34 learned F6
estimators and two equal-weight ensembles. The current operational surface is
33 automated prediction/poll members after excluding the three declared
invalid poll-ordering surfaces. The owner contributes one separate manual
Top-25 ballot.

The inventory/audit count is not the live ballot count. The roster contains the distinct
`margin_tree_random_forest`, `margin_boosted_hist_gradient_boosted`,
`margin_tree_gradient_boosted`, `margin_temporal_temporal_random_forest`, and
`margin_temporal_temporal_hist_gradient_boosted` roles.

The older 2025 roster remains separate retrospective evidence; its checkpoints
were trained through 2024 and are not relabeled as 2026 frozen models.

The scientific F0–F8 × M roster is separate. Its F0–F6 cells are eligible for
public predictions and polls; F7/F8 are comparison-only.
