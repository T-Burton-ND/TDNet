# Scientific-roster F6 SHAP study

## Purpose

This study explains the tuned F6 scientific roster without expanding the model
scope beyond M1, M2, M3, M4, M5, and M10. It is an interpretability analysis,
not evidence that a feature is causal and not an automatic feature-selection
procedure. All 227 declared F6 source features must appear in the supplemental
figure set, including features with negligible estimated importance.

The protocol was specified before the new-fingerprint HPS completed. The HPS
selects model configurations; it does not use SHAP results.

## Analysis unit and leakage boundary

For each objective, model level, and rolling-origin outer fold, the selected HPS
configuration is refit using only that fold's training and validation seasons.
The SHAP background is sampled only from the outer training seasons. SHAP values
are evaluated only on the held-out outer test season. Neither 2025 nor 2026 is
used for ranking features or selecting an explanation method.

This produces 120 explanation cells:

\[
2\ \text{objectives}\times 6\ \text{models}\times 10\ \text{outer folds}=120.
\]

The final corrected study uses 128 background games and 256 held-out
explanation games per cell with the corrected F6 HPS finalists. Both samples
are deterministic and remain inside their declared fold partitions. The older
32/16 F5-seeded run is retained only as a superseded provisional artifact.

## SHAP methods

The margin models are explained on the predicted-points scale. The winner
models are explained on the predicted home-win-probability scale. The primary
worker uses the same end-to-end permutation SHAP estimator for every
architecture. Its callable starts with the concatenated home and away F6 source
vectors, constructs the reviewed unit-matchup representation, and then invokes
the fitted model. This common input and output contract makes architecture
comparisons interpretable without relying on model-specific transformed spaces.

The repository's existing generic SHAP helper cannot be used unchanged for
this study: code inspection shows that it currently handles linear and a subset
of tree pipelines, but not the spline, histogram-boosted, MLP, or KNN roster as
a complete publication analysis. The dedicated study worker must explain the
public prediction callable so the margin objective is in points and the winner
objective is home-win probability, and it must emit the source-aggregation
tables defined below. The figure renderer is intentionally downstream of that
strict table contract.

Raw SHAP magnitudes are not compared across architectures. Each model-fold is
normalized so its absolute feature importance sums to one before consensus or
concordance calculations.

## From source coordinates to the 227 F6 features

The explanation matrix contains 454 columns: home and away copies of each of
the 227 declared source fields. Unit-matchup construction occurs inside the
prediction callable, including reviewed rates and offense-versus-defense
pairings. Absolute home and away SHAP magnitudes are summed for each source.
This avoids post-hoc division of one engineered-coordinate attribution between
multiple raw fields and captures contributions through derived rates.

The implementation audit additionally requires exactly 681 fitted F6 matchup
coordinates. This guard catches a source field that survives the registry but
is silently lost before fitting.

## Prespecified summaries

For feature (j), model (m), and fold (k), let

\[
I_{jmk}=\frac{\operatorname{mean}_i|\phi_{ijmk}|}
{\sum_l\operatorname{mean}_i|\phi_{ilmk}|}.
\]

The main feature score is the median \(I_{jmk}\) across valid folds and models.
Uncertainty is the fold-clustered 95% bootstrap interval. Stability includes
the median rank, rank interquartile range, and frequency of appearing in the
top 10, 25, 50, and 100. Model agreement is measured with Spearman rank
correlation. Direction, when identifiable, is the held-out Spearman correlation
between the feature value and its SHAP value.

Missing explanations are never converted to zeros. Each model must have at
least eight valid outer folds. The renderer fails if any of the 227 F6 features
is absent from the importance atlas.

## Figure suite

1. **Family allocation:** normalized importance share by feature family,
   objective, and model.
2. **Architecture concordance:** pairwise Spearman rank agreement among the six
   scientific models, separately for margin and winner.
3. **Complete feature atlas:** paginated heatmaps containing every F6 feature,
   with six model columns, consensus importance, rank, and fold stability.
4. **Rank-stability atlas:** median rank and fold IQR for every feature; no
   top-feature truncation.
5. **Direction atlas:** source-level direction estimates for every feature,
   with indeterminate interaction-derived directions visibly distinguished.
6. **Dependence supplement:** twelve features per page, showing held-out feature
   value versus SHAP contribution and a binned median curve. Every eligible
   feature receives a panel.
7. **F5/F6 block focus:** the 60 temporal and eight graph features shown in
   context against inherited F0--F4 information.
8. **Explainer audit:** completion, sample sizes, additivity error, runtime, and
   native-versus-common-permutation rank agreement for every model-fold.

PNG and SVG files support manuscript assembly; the complete atlases and
dependence panels are also combined into multipage PDF supplements. A figure
manifest records the exact features and input hashes represented on every page.

## Relationship to a compressed F6

These figures may motivate a separately named compressed representation such as
F6-C25, but SHAP rank alone will not select it. Any choice of feature count must
be evaluated in an additional nested rolling-origin study in which ranking and
the choice of (N) occur strictly inside each outer training fold. F6 remains
the information tier; F6-C\(N\) is a derived compression experiment.

Machine-readable settings are in
`configs/publication/scientific_roster_shap_study.yaml`.

## Renderer input contract

The importance input is one row per objective, model, outer fold, and aggregated
F6 source feature. Required columns are `objective`, `model_level`,
`outer_fold`, `source_feature`, and `mean_abs_shap`. Optional audit columns are
`explainer_method`, `n_explained`, `runtime_seconds`, and `additivity_error`.
The optional `common_permutation_rank_rho` field records each cell's Spearman
agreement between its native explainer and the common permutation sensitivity.
Duplicate source rows are rejected because they indicate that matchup
coordinates have not yet been aggregated.

The optional effect input is long form, with `objective`, `model_level`,
`outer_fold`, `source_feature`, `feature_value_z`, and `shap_value`. It drives
the direction and dependence supplements. Missing source-level signed effects
remain blank rather than being imputed as zero. The importance input must still
contain all 227 features in every supplied fold; each model needs at least eight
complete folds.

The executable renderer is
`python -m gridiron_ml.cli.publication.build_scientific_roster_shap_figures`.
It writes machine-readable coverage, importance, explainer-audit, page-manifest,
and input-hash tables alongside the figures.
