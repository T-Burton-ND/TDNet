# TDNet Current Model Catalog

Last updated: 2026-07-16

This catalog lists the concrete model types currently implemented in TDNet. A
model type may be trained once for the **winner** objective and once for the
**margin** objective; those are two fitted checkpoints, not two different kinds
of mathematics.

| Family | Model type | Human explanation of the math | Current role |
|---|---|---|---|
| Naive baseline | `majority` | Ignores the matchup features and always predicts the side that won most often in the training data, along with that historical win rate. | Sanity/negative-control baseline |
| Naive baseline | `constant_margin` | Ignores the matchup features and predicts the same average training-game margin for every game. | Sanity/negative-control baseline |
| Naive baseline | `home_team` | Always selects the home team and uses a fixed positive home margin. It measures whether a real model adds value beyond blindly assuming home-field advantage. | Sanity/negative-control baseline |
| Linear | `ols` | Fits one coefficient per feature so that a weighted sum of the features minimizes squared prediction errors. There is no shrinkage penalty. | Legacy trained family |
| Linear | `ridge` | OLS plus an L2 penalty that discourages large coefficients. It spreads credit across correlated football statistics and usually produces a more stable fit. | Legacy trained family |
| Linear | `lasso` | Linear regression with an L1 penalty. The penalty can drive unhelpful coefficients exactly to zero, so it performs automatic feature selection. | Legacy trained family |
| Linear | `elastic_net` | Combines L1 and L2 penalties: it can remove weak features like lasso while sharing weight among correlated features like ridge. | Legacy trained family |
| Linear | `huber` | Fits a line with squared-error treatment for ordinary misses but switches to roughly absolute-error treatment for very large misses, reducing the influence of blowouts and bad outliers. | Legacy trained family |
| Linear | `bayesian` | Bayesian ridge regression places probability distributions over the coefficients and their amount of shrinkage, then uses the posterior estimate for predictions. | Legacy trained family |
| Linear | `ard` | Automatic Relevance Determination gives each coefficient its own learned shrinkage strength. Features judged irrelevant are pushed very close to zero. | Legacy trained family |
| Linear | `ransac` | Repeatedly fits a base linear/ridge model to candidate inlier subsets, keeps the fit supported by the largest consistent set, and refits without strong outliers. | Legacy trained family; recovery variant uses ridge base |
| Linear | `orthogonal_matching_pursuit` | Builds a sparse equation greedily: at each step it adds the feature most aligned with the remaining error, then refits the selected features. | Legacy trained family |
| Linear | `sgd` | Learns linear coefficients through many small gradient updates instead of solving the whole regression at once; regularization limits coefficient size. | Legacy trained family |
| Linear | `passive_aggressive` | Leaves the coefficients alone when a prediction is close enough, but makes an aggressive corrective update when the error exceeds a tolerance. | Legacy trained family |
| Statistical index | `z_index` | Converts features to standard-deviation units, corrects their football direction, and uses ridge regression to combine the standardized advantages into a margin/index. | Legacy trained family |
| Statistical index | `percentile` | Replaces raw values with their positions in the historical feature distribution, then ridge-combines those relative rankings. This reduces sensitivity to scale and extreme values. | Legacy trained family |
| Statistical index | `robust` | Centers features on the median and scales them with MAD/IQR instead of mean and standard deviation, clips extremes, then combines them with ridge regression. | Legacy trained family |
| Statistical index | `weighted` | Applies declared football-domain weights to groups such as offense, defense, efficiency, turnovers, and talent before a regularized statistical fit. | Legacy trained family |
| Spline additive | `spline_ridge` | Expands each numeric feature into smooth piecewise-polynomial curves, allowing effects to bend or plateau, then ridge-combines those curves. Interactions are still controlled and limited. | New publication-search family |
| Classical tree ensemble | `random_forest` | Fits many decision trees to bootstrapped samples and random feature subsets, then averages them. Each tree learns nonlinear thresholds and interactions. | Legacy trained family |
| Classical tree ensemble | `extra_trees` | Fits many trees using unusually randomized split thresholds and feature choices, then averages them. More randomization can reduce variance and overfitting. | Legacy trained family |
| Classical tree ensemble | `gradient_boosted` | Builds shallow trees sequentially; each new tree focuses on correcting the errors left by the current ensemble. The final prediction is the sum of many small corrections. | Legacy trained family |
| Boosted tree | `hist_gradient_boosted` | Bins continuous features into histograms and sequentially adds trees that follow the loss gradient. Binning makes large searches faster and regularization controls leaf complexity. | New publication-search family |
| Neural | `mlp` | Passes standardized features through stacked dense layers. Each layer forms weighted combinations and nonlinear GELU/ReLU transformations, letting the network learn smooth interactions; dropout, weight decay, and early stopping regularize it. | New publication-search family |
| Structured neural | `structured_mlp` | For every paired home/away statistic, explicitly supplies the home value, away value, difference, and product to an MLP. The network can therefore learn both matchup gaps and “both teams are strong/weak” interactions. | New exploratory publication-search family |
| Frozen ensemble | `mean_probability` | Takes already-frozen member models and computes a nonnegative weighted average of their home-win probabilities and predicted margins. Equal weights are used unless declared otherwise. | Composite prediction output; tune after out-of-fold predictions exist |
| Frozen ensemble | `median_margin` | Uses the median member-model margin, which is resistant to one extreme model, while win probabilities remain a nonnegative weighted average. | Robust composite prediction output |
| Kernel | `rbf_kernel_ridge` | Measures similarity with a Gaussian/RBF kernel, allowing every training matchup to contribute a smooth nonlinear bump; ridge regularization controls the combined influence. | New publication-search family |
| Kernel | `rbf_svr` | Fits a smooth RBF-kernel surface while ignoring errors inside an epsilon-sized tube and penalizing larger misses. Only influential support-vector games define the surface. | New publication-search family |
| Kernel | `gaussian_process` | Places a Gaussian-process probability distribution over smooth matchup functions. Similar games have correlated predictions and a noise term represents unexplained score variation; training is deterministically capped for cubic cost. | New publication-search family |
| Kernel | `nystroem_ridge` | Approximates a large RBF kernel using a sampled low-dimensional basis, then fits ridge regression in that nonlinear basis. It trades a little exactness for much better scaling. | New publication-search family |
| Temporal | `decay_ridge` | Builds prior-week states and exponentially decayed histories, then ridge-combines them. Recent form receives more weight according to a tuned half-life. | New publication-search family |
| Temporal | `trend_elastic_net` | Combines lagged state, decayed form, and multiweek change features with sparse elastic-net regression, selecting the most useful momentum signals. | New publication-search family |
| Temporal | `temporal_random_forest` | Feeds lag, decay, and trend fingerprints into many randomized trees so recent form can interact nonlinearly with matchup context. | New publication-search family |
| Temporal | `temporal_hist_gradient_boosted` | Sequentially learns histogram-tree corrections from time-dependent fingerprints, capturing nonlinear hot/cold streak thresholds and interactions efficiently. | New publication-search family |

## Inventory summary

- **35 concrete types** across 11 named families when baselines and the two
  ensemble aggregation rules are included.
- **18 legacy trained types:** 11 linear, 4 statistical-index, and 3 classical
  tree models.
- **12 new trainable publication-search types:** spline ridge, histogram gradient
  boosting, tabular MLP, structured MLP, four kernel models, and four temporal models.
- **3 naive baselines** and **2 frozen ensemble rules** complete the catalog.
- Balanced-objective checkpoints are out of scope. Trainable types are being
  considered for separate winner- and margin-objective checkpoints.

## Implemented TDStat backends not presently separate catalog models

The `TDStat` class also contains lower-level `corr_linear`, `ols`, `ridge`, and
`bayes_bootstrap_ridge` modes. They currently have no dedicated model YAML in
`configs/models/stat/` and are not separate entries in the publication search
roster; the four configured statistical models above are the active TDStat
variants.

Sources of truth: `src/gridiron_ml/models/`, `configs/models/`, and
`configs/models/publication_model_registry.yaml`.
