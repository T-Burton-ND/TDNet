# F0–F20 scientific post-hoc scorecards, 2026

This is the current **126-cell** scientific scorecard: F0–F20, each with M1–M5 and M10. It grades all saved completed games through CFBD Week 7 for F0–F19 (380 games per cell), and all archived completed games through CFBD Week 6 for F20 (320 games per cell). The original expert archive has no CFBD Week 7 F20 rows. F20 now has separately trained margin regressors alongside its frozen win-probability models; both outputs are scored here.

## Score tables

| File | Content |
| --- | --- |
| [latest_scorecard.csv](latest_scorecard.csv) | One row per 126 F×M cells at its latest available week. It states `through_week` and `games` on every row. |
| [weekly_scorecard.csv](weekly_scorecard.csv) | Every available cell-week: 120 cells for Weeks 1–7 and six F20 cells for Weeks 1–6 (876 rows). |
| [cumulative_scorecard.csv](cumulative_scorecard.csv) | The same 876 cell-week positions, cumulatively scored. No carried-forward F20 Week 7 row. |
| [f20_paired_304_game_scorecard.csv](f20_paired_304_game_scorecard.csv) | All 126 cells on the **same 304 games** with F20 and archived market probability, for direct comparison. |
| [f20_paired_weekly_scorecard.csv](f20_paired_weekly_scorecard.csv) | Per-week scores for those same matched games, Weeks 1–6. |

Straight-up (SU) uses predicted home-win probability ≥ 0.5. Brier and log loss use the home-win outcome. Margin MAE is mean absolute home-margin error. ATS uses the archived home spread: predicted home cover occurs when predicted home margin plus home spread is positive. A predicted cover exactly zero is `ats_no_pick`; an actual cover exactly zero is a push, excluded from ATS accuracy. Missing lines are counted separately as `ats_no_line`. The full F0–F19 cohort has 379 line-covered games per cell; F20 has 311 line-covered games and nine without an archived spread. An upset requires a nonzero archived spread to define the favorite; `upset_recall` is correct upset calls divided by actual upsets, and `upset_call_accuracy` divides by model upset calls.

## Publication figures

The paired [Brier](paired_304_brier_matrix_publication.png), [SU](paired_304_su_accuracy_matrix_publication.png), [log loss](paired_304_log_loss_matrix_publication.png), [margin MAE](paired_304_margin_mae_matrix_publication.png), [ATS](paired_304_ats_accuracy_matrix_publication.png), [upset recall](paired_304_upset_recall_matrix_publication.png), and [upset-call accuracy](paired_304_upset_call_accuracy_matrix_publication.png) matrices show every F×M cell on the same 304 Weeks 1–6 games. A metric is N/A only where its denominator is zero, such as a model making no upset calls. The latest-available matrices use the same seven metric stems with `latest_available_` in place of `paired_304_`; they show F0–F19 through Week 7 and F20 through Week 6, with different counts printed on each figure. Use the paired matrices for comparisons across fingerprints.

Weekly figures show the quartiles, median, and individual points for all 126 cells on the same games for [Brier](paired_weekly_brier_publication.png), [SU](paired_weekly_su_accuracy_publication.png), [log loss](paired_weekly_log_loss_publication.png), [margin MAE](paired_weekly_margin_mae_publication.png), [ATS](paired_weekly_ats_accuracy_publication.png), [upset recall](paired_weekly_upset_recall_publication.png), and [upset-call accuracy](paired_weekly_upset_call_accuracy_publication.png). Every figure also has a full-size PNG, a full-size SVG, and a `_publication.svg` print version. The 21 figure designs and hashes are listed in [figure_manifest.json](figure_manifest.json). The updated [cumulative Week 1–7 view](../scientific_f0_f20_cumulative_week01_07_v3_margin/README.md) also includes F20 margin curves and a matched margin companion.

The [rotating cumulative 3D views](../scientific_f0_f20_cumulative_3d_week01_07/README.md) place one final cumulative marker for each of the 126 cells in SU–ATS–Brier space. One shows each cell's latest available week; the matched 304-game view supports direct F20 comparisons.

The [F20-inclusive architecture and fingerprint violin plots](../scientific_f0_f20_violin_week01_06/README.md) reproduce the earlier balanced relative skill display on the same 304-game cohort for all 126 cells. Full and publication PNG/SVG versions are available.

These are **retrospective descriptive** scores. F20 models were fitted after the scored 2026 games; its weekly expert source publication times are assumed pregame but unverified. Week 7 market quote times for F0–F19 are unverified, and some market-fed fingerprints used ex-post source cache values. These results do not measure prospective performance or betting returns. The older [F0–F19 quality gallery](../scientific_total_quality/README.md) remains the frozen 120-cell Week 7 analysis with its original cohort and manifests; this F0–F20 set is the publication-facing expansion.

Rebuild from the repository root:

```bash
python3 scripts/publication/build_2026_f0_f20_posthoc_scorecards.py
MPLCONFIGDIR=/tmp/tdnet-mpl python3 scripts/publication/plot_2026_f0_f20_posthoc_publication.py
```

The builders validate roster completeness, game IDs, final margins, saved grading fields, F20's six probabilities and margins, paired cohort size, and ATS denominators. [manifest.json](manifest.json) and [figure_manifest.json](figure_manifest.json) record source and output SHA-256 hashes.
