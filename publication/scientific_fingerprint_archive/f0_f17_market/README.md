# Scientific fingerprint archive: F0–F17 market

This package brings the six scientific model families together across the historical F0–F8 ladder, the separate F06 broad-A screen, and the candidate F09–F17 market research stages. It is a retrospective evaluation archive, not a current-week forecast.

## Figures

- [MAE heat map](figures/scientific_fingerprint_margin_mae_heatmap.png)
- [Winner accuracy heat map](figures/scientific_fingerprint_winner_accuracy_heatmap.png)
- [ATS accuracy heat map](figures/scientific_fingerprint_ats_accuracy_heatmap.png)
- F0–F8 historical cumulative curves for the full roster: [MAE](figures/historical_f0_f8_cumulative_margin_mae.png) · [upset recall](figures/historical_f0_f8_cumulative_upset_recall.png) · [winner accuracy](figures/historical_f0_f8_cumulative_winner_accuracy.png) · [Brier](figures/historical_f0_f8_cumulative_brier_score.png) · [ATS accuracy](figures/historical_f0_f8_cumulative_ats_accuracy.png)
- F09–F17 full-roster model-mean cumulative curves, 2024: [MAE](figures/nextgen_full_roster_cumulative_margin_mae_2024.png) · [upset recall](figures/nextgen_full_roster_cumulative_upset_recall_2024.png) · [winner accuracy](figures/nextgen_full_roster_cumulative_winner_accuracy_2024.png) · [Brier](figures/nextgen_full_roster_cumulative_brier_score_2024.png) · [ATS accuracy](figures/nextgen_full_roster_cumulative_ats_accuracy_2024.png)
- F09–F17 full-roster model-mean cumulative curves, 2025: [MAE](figures/nextgen_full_roster_cumulative_margin_mae_2025.png) · [upset recall](figures/nextgen_full_roster_cumulative_upset_recall_2025.png) · [winner accuracy](figures/nextgen_full_roster_cumulative_winner_accuracy_2025.png) · [Brier](figures/nextgen_full_roster_cumulative_brier_score_2025.png) · [ATS accuracy](figures/nextgen_full_roster_cumulative_ats_accuracy_2025.png)
- F09–F17 equal-weight six-model consensus cumulative curves, 2024: [MAE](figures/nextgen_consensus_cumulative_margin_mae_2024.png) · [upset recall](figures/nextgen_consensus_cumulative_upset_recall_2024.png) · [winner accuracy](figures/nextgen_consensus_cumulative_winner_accuracy_2024.png) · [Brier](figures/nextgen_consensus_cumulative_brier_score_2024.png) · [ATS accuracy](figures/nextgen_consensus_cumulative_ats_accuracy_2024.png)
- F09–F17 equal-weight six-model consensus cumulative curves, 2025: [MAE](figures/nextgen_consensus_cumulative_margin_mae_2025.png) · [upset recall](figures/nextgen_consensus_cumulative_upset_recall_2025.png) · [winner accuracy](figures/nextgen_consensus_cumulative_winner_accuracy_2025.png) · [Brier](figures/nextgen_consensus_cumulative_brier_score_2025.png) · [ATS accuracy](figures/nextgen_consensus_cumulative_ats_accuracy_2025.png)

## Unified all-generation curves

These five figures deliberately put all available model/fingerprint cumulative summaries on one F0–F17 axis: [MAE](figures/unified_all_models_cumulative_margin_mae.png) · [upset recall](figures/unified_all_models_cumulative_upset_recall.png) · [winner accuracy](figures/unified_all_models_cumulative_winner_accuracy.png) · [Brier](figures/unified_all_models_cumulative_brier_score.png) · [ATS accuracy](figures/unified_all_models_cumulative_ats_accuracy.png). The [underlying point table](performance/unified_cumulative_fingerprint_performance.csv) includes the cohort and game count for each point.

These unified figures are an intentionally non-equivalent landscape view. F0–F8 points pool held-out 2015–2024 fold metrics; F06 broad is a separate 2025 cohort available only for M2/M4; F09–F17 pool each model architecture's mean forecasts over the 2024 and 2025 development games. The line gaps and background bands mark those cohort changes. Consensus is shown only for F09–F17 because historical game-level predictions for F0–F8 were not retained. Read vertical differences across those boundaries descriptively, not as paired improvements.

The heat maps contain all six model families: M1, M2, M3, M4, M5 and M10. The Vegas row is the line-only baseline on each matching evaluation cohort for MAE and winner accuracy. A line-only baseline has no measured ATS pick accuracy, so its ATS cell is intentionally blank. F06 broad is only available for M2 and M4. Blank cells indicate unmeasured results, not zero performance.

## Performance and prediction tables

- `performance/all_fingerprint_scorecard.csv` has one summary for every measured model–stage pair, including all F0–F8 models and the later scientific runs.
- `performance/historical_annual_fold_performance.csv` preserves each F0–F8 model's held-out season metrics from the selected 2015–2024 rolling-origin folds. F7 and F8 remain separate market-bearing historical branches.
- `performance/historical_cumulative_performance.csv` adds held-out seasons in order. Margin MAE, winner accuracy and Brier are pooled by game count; ATS is pooled by non-push decision count; upset recall is an unweighted mean of annual fold recalls.
- `predictions/nextgen_game_predictions_2024_2025.csv.gz` contains game-level out-of-sample predictions for each saved fit/seed and game in the F09–F17 market stages. It includes the derived ATS pick/result but does not expose the underlying market line.
- `predictions/nextgen_roster_model_mean_predictions_2024_2025.csv.gz` averages the saved fit/seed forecasts within each model architecture before scoring.
- `predictions/nextgen_scientific_consensus_predictions_2024_2025.csv.gz` averages those six architecture-level forecasts equally for each game and fingerprint.
- `performance/nextgen_weekly_performance.csv` reports per-week results for those 2024 and 2025 game-level predictions. Each value is the median across the saved fits/seeds for that model and fingerprint.
- `performance/nextgen_season_to_date_performance.csv` accumulates each season's weekly results through each week, with the same replicate-median convention.
- `performance/nextgen_year_scorecards.csv` reports season-end metric medians across fits/seeds.
- `performance/nextgen_model_mean_cumulative_performance.csv` and `performance/nextgen_consensus_cumulative_performance.csv` score the architecture-mean forecasts and six-model consensus cumulatively by week, including MAE, upset recall, winner accuracy, Brier and ATS accuracy.
- `performance/vegas_cumulative_performance_2024_2025.csv` supplies the season-matched cumulative Vegas margin, winner, upset-recall and Brier baselines used by the curves.
- `performance/unified_cumulative_fingerprint_performance.csv` supplies the points used in the intentionally cross-cohort all-generation curve set.
- `provenance/nextgen_run_receipts.csv` records every consumed F09–F17 run and prediction hash.

## Coverage and interpretation

The historical F0–F8 selected-fold artifact retains model-level metrics by held-out season, not the individual game prediction rows. Those models are included in the annual/cumulative scorecards and all three heat maps; their game-level weekly predictions cannot be reconstructed from the retained result artifact without rerunning those experiments. The separately measured F06 broad-A cohort also has score summaries for M2/M4, but no weekly prediction export in this package.

F09–F16 are candidate A representations and have not been accepted into the prospective fingerprint lineage. F17 market uses archived historical lines without quote timestamps and is retrospective research only. Evaluation periods differ: F0–F8 summarize ten rolling test seasons, F06 broad uses its separate 2025 cohort, and F09–F17 use the common 2024/2025 development games. Cross-cohort differences are descriptive, not paired improvements or prospective evidence.

All figure backgrounds use the supplied warm parchment palette. The [manifest](manifest.json) records row counts and SHA-256 hashes for package files. Source experiment predictions, checkpoints and raw CFBD records remain outside this publication directory.

The tables and figures are built by [`build_scientific_fingerprint_archive.py`](../../../scripts/build_scientific_fingerprint_archive.py) from the preserved experiment receipts, the selected historical-fold results, and the evaluation-only game-week/market sidecars.
