---
type: synthesis
up: "[[Project-History-and-Directions]]"
tags: [history, models, decisions]
derived_from: ["[[Source-Wide-Margin-Decisions]]", "[[Source-Opponent-Adjustment]]", "[[Model-and-Poll-Surfaces]]"]
---

# Fingerprint and Model Decision History

This page records the documented selection and retirement decisions that shaped TDNet’s 2026 scientific comparison and weekly model roster.

## Wide-margin fingerprint: full F6 selected

The corrected F5/F6/F8 hyperparameter search completed 13,920/13,920 trials with zero failed, missing, or duplicate rows and ten rolling-origin folds per configuration. Full scientific evaluation completed 1,080/1,080 fits across F0–F8, two objectives, six model types, and ten folds. F6-C compression completed 1,080/1,080 candidate fits and 108/108 leakage-safe selected-path evaluations across nine eligible folds.

F6 lowered mean margin MAE versus F5 for all six scientific model types, by 0.049 points on average. This was a small, consistent primary-metric improvement, not a claim of improvement on every metric. Across shared outer folds 1–9 and six model types, full F6 versus F6-C was:

| Metric | Full F6 | F6-C | F6-C minus F6 |
|---|---:|---:|---:|
| Margin MAE (points) | 13.446 | 13.550 | +0.104 |
| Winner accuracy | 71.424% | 71.435% | +0.011 pp |
| Chalk recall | 88.893% | 88.720% | -0.173 pp |
| Upset recall | 21.984% | 22.478% | +0.494 pp |
| Brier score | 0.19176 | 0.19268 | +0.00092 |
| ATS accuracy | 51.050% | 50.506% | -0.544 pp |

F6-C improved MAE in one of six model types. Since the operational wide roster uses a common fingerprint and margin MAE is primary, full corrected F6 was selected; F6-C remains supplemental, not a new tier or operational replacement. F6-C target feature counts for margin were M1=35, M2=15, M3=15, M4=10, M5=15, and M10=10; counterpart matchup features were selected atomically, so realized fold counts could vary by one.

## Roster cleanup and presentation boundaries

The installed corrected-F6 wide roster has 34 learned estimators plus two equal-weight ensembles. Five exact implementations reused corrected-F6 scientific HPS selections; the remaining 29 learned roles received margin-only gap searches. Its fixed representation has exactly 681 matchup coordinates, including 24 schedule-graph coordinates. Legacy temporal wrappers that added 117 undeclared coordinates were disabled. KNN `compact` and `full_fingerprint` labels were retired because they described feature subsets that would be duplicate or misleading when all members use F6; four KNN ballots instead represent uniform/distance weighting crossed with Euclidean/Manhattan distance.

Three statistical margin estimators remain in prediction and ensembles but are excluded from Top-25 voting after invalid poll-ordering signatures in 2025 rehearsal (Air Force first in Week 15 for `margin_stat_z_index`; Week 16 signatures for robust and weighted variants). The exclusion affects poll presentation only, not checkpoints or game predictions. The broader operational surface differs from the [scientific panel](Model-and-Poll-Surfaces).

Pre-correction F5-only and 48-cell F0–F7 candidates are explicitly superseded because F4/F5 were duplicated. They must not be reused, relabeled, or published as current results.

## Opponent adjustment selection

Seven methods were compared. The 2025 holdout used train 2010–2023, validation 2024, test 2025; it is consumed and cannot select the 2026 primary method. Development selection instead used train 2010–2021, validation 2022, evaluation 2023–2024, excluding 2025 and 2026. v1.7 ensemble averaging had the lowest raw development MAE (12.9699), but an averaged primary adjustment was disallowed. Among eligible non-averaged candidates, v1.4 Elo context was selected: MAE 12.9802, RMSE 16.4442, winner accuracy 0.7043, across 1,488 games in two seasons. v1.7 remained a supplemental comparator; its MAE was only 0.0103 points higher. Do not conflate this development result with the separate 2025 holdout table.

## See also

[Project History and Directions](Project-History-and-Directions) · [Experiment Program Recovery](Experiment-Program-Recovery) · [Deferred and Abandoned Directions](Deferred-and-Abandoned-Directions) · [Model and Poll Surfaces](Model-and-Poll-Surfaces) · [Fingerprint Ladder](Fingerprint-Ladder) · [Wide Margin Decisions Source](Source-Wide-Margin-Decisions) · [Opponent Adjustment Source](Source-Opponent-Adjustment)
