# Wide-margin fingerprint decision

## Decision

Train the single-fingerprint operational wide-margin roster on **full F6**.
Retain F6-C as a supplemental compression analysis, not as the operational
fingerprint. F7 and F8 remain market-bearing comparisons and are ineligible for
the independent TDNet poll roster.

## Completed evidence

- Corrected F5/F6/F8 HPS: 13,920/13,920 successful trials; zero failed,
  missing, or duplicate rows; ten rolling-origin folds per configuration.
- Full scientific heat-map evaluation: 1,080/1,080 successful fits covering
  F0--F8, two objectives, six model types, and ten rolling-origin folds.
- Sequential F6-C study: 1,080/1,080 successful candidate fits and 108/108
  leakage-safe selected-path evaluations across the nine eligible folds.
- Six metric figures were rendered in PNG, SVG, and PDF: MAE, winner accuracy,
  chalk recall, upset recall, Brier score, and ATS accuracy.

## F6 versus F5

Using the HPS selection metric, F6 lowered mean margin MAE relative to F5 in
all six scientific model types. The mean improvement across types was 0.049
points. The gain is consistent but small; it does not imply uniform improvement
in every secondary metric. F6 primarily improves margin error and chalk recall,
while F5 retains somewhat higher upset recall.

## F6-C versus full F6

The fair comparison uses outer folds 1--9, which are shared by the sequential
F6-C path and full F6. Averaged equally across the six margin-model types:

| Metric | Full F6 | F6-C | F6-C minus F6 |
|---|---:|---:|---:|
| MAE (points) | 13.446 | 13.550 | +0.104 |
| Winner accuracy | 71.424% | 71.435% | +0.011 pp |
| Chalk recall | 88.893% | 88.720% | -0.173 pp |
| Upset recall | 21.984% | 22.478% | +0.494 pp |
| Brier score | 0.19176 | 0.19268 | +0.00092 |
| ATS accuracy | 51.050% | 50.506% | -0.544 pp |

Lower is better for MAE and Brier score. F6-C improved MAE in only one of six
model types. Its small upset-recall gain does not offset the primary margin-MAE
loss for an operational margin roster.

The final F6-C deployment target counts selected by the prespecified one-SE
rule were M1=35, M2=15, M3=15, M4=10, M5=15, and M10=10 source features for
the margin objective. Target and realized counts may differ by one in individual
folds because reviewed matchup counterparts are selected atomically.

## Interpretation

This is a representation decision, not a claim that F6 dominates every metric.
The wide roster keeps one common fingerprint, so full F6 is selected using its
primary margin-error criterion. The balanced scientific roster continues to
retain one representative of every model type at every information tier so the
accuracy, calibration, chalk/upset, and market-comparison tradeoffs remain
visible.

## Operational refit design

The former wide roster did not have uniform corrected-F6 search coverage.
Five exact implementations (ridge, spline ridge, random forest, histogram
gradient boosting, and MLP) reuse their completed corrected-F6 scientific HPS
selections. The other 29 learned roles receive a margin-only corrected-F6 gap
search with the same ten rolling-origin folds. This produces 788 candidate
hyperparameter configurations and 7,880 fold-level fits.

The installed roster is defined as 34 learned estimators plus two equal-weight
ensembles. The four KNN ballots are now actual algorithmic variants—uniform or
distance weighting crossed with Euclidean or Manhattan distance. The legacy
`compact` and `full_fingerprint` KNN labels are retired because they denoted
feature subsets and would become duplicate or misleading labels when every
learned member is fitted on F6.

Legacy temporal wrappers formerly appended 117 undeclared coordinates to the
fixed F6 matrix. That expansion is disabled for this roster: every learned
member is validated against exactly 681 corrected-F6 matchup coordinates,
including 24 schedule-graph coordinates. The operational refit uses seasons
2010--2025. A separate, otherwise identical roster uses seasons 2010--2024 and
holds all of 2025 out for the checked-in retrospective examples.

The 2025 holdout replay identified one presentation-specific eligibility
failure: `margin_stat_z_index` ranked Air Force first in its Week 15 Top-25
ballot. The model remains in the prespecified margin-prediction roster and
equal-weight ensembles; removing it based on holdout accuracy would be
post-selection leakage. It is excluded only from poll voting under the existing
invalid-ballot sanity rule (`known_invalid_poll_ordering_surface`).

The complete through-2025 rehearsal subsequently found the same invalid
ordering signature at Week 16 for `margin_stat_robust` and
`margin_stat_weighted`. All three statistical estimators remain members of the
prespecified margin prediction roster and ensembles, but are excluded from
TDNet Top-25 ballots only. This changes neither fitted checkpoints nor game
predictions; it prevents scalar margin estimators with unsuitable team-ordering
surfaces from being presented as poll models.

## Machine-readable evidence

- Heat-map grid: `scientific_roster_metric_grid.parquet`
- F6-C selected path: `sequential_selected_results.parquet`
- F6-C deployment counts: `deployment_feature_counts.parquet`
- Coverage reports: `summary/heatmaps/report.json` and
  `summary/compression_coverage.json`
