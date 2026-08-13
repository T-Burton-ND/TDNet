# TDNet Model Guide

This guide explains the current model families in TDNet, how each implementation works, and the practical strengths and weaknesses of each architecture. The main README covers workflows; this file is meant to help decide what to train, compare, and trust.

TDNet models share the same broad interface:

1. `TD_Eval` loads fingerprint rows and builds matchup features with `MatchupBuilder`.
2. A model trains on matchup features and a margin target where positive margin means home team win.
3. The model emits `pred_margin`, `pred_proba_home_win`, and `pred_pick_home`.
4. Evaluation compares model output to actual margins, winner accuracy, and market baselines.
5. TD Sim consumes standardized game-level prediction outputs rather than model internals.

## Historical-matchup KNN Family

Implementation: [src/gridiron_ml/models/td_knn.py](src/gridiron_ml/models/td_knn.py)

`TDKNN` treats the pregame matchup fingerprint as a similarity space and
regresses the canonical home-team margin from prior matchups only. The
`knn_uniform` and `knn_distance` roles use training-fold median imputation and
standardization inside a serialized sklearn pipeline. The distance-weighted
role is the publication primary; compact/full-fingerprint roles remain
available for feature-complexity experiments.

Every prediction includes selected `k`, metric and weighting, neighbor margin
summaries, distance concentration, effective neighbor count, weighted outcome
fraction, and JSON-encoded neighbor IDs/seasons/weeks/teams/margins/distances/
weights. `TDEval.save_outputs()` also writes the long-form
`predictions/neighbor_audit.csv`. When a validation split is supplied, the
probability calibrator is fitted only on validation predictions; otherwise the
standard TDNet margin-to-probability link is used.

## Stat Family

Implementation: [src/gridiron_ml/models/td_stat.py](src/gridiron_ml/models/td_stat.py)

The stat family is the lightest-weight model family. It focuses on selected football stat columns, normalizes them, then fits a simple linear margin model. By default it selects columns with prefixes like `statOff_`, `statDef_`, `statGen_`, and `statSpe_`; the newer variants also allow `offense_`, `defense_`, and sometimes `roster_` features depending on config.

All current stat variants:

- Coerce inputs to numeric pandas dataframes.
- Fill missing values with training medians.
- Select stat-like feature columns.
- Normalize selected columns according to the variant.
- Fit a linear model to home-margin target values.
- Convert predicted margins to home win probabilities using a logistic transform and a learned residual scale.

### `stat_z_index`

Config: [configs/models/stat/config_z_index.yaml](configs/models/stat/config_z_index.yaml)

This is the baseline TDStat model. It standardizes each selected feature with the usual z-score formula:

```text
z = (x - mean) / standard_deviation
```

It then fits ridge regression with `z_ridge_alpha`, currently `10.0`.

Pros:

- Simple and easy to reason about.
- Fast to train and cheap to run.
- Coefficients are interpretable because each feature is on a common scale.
- Good baseline for checking whether richer models are adding real signal.

Cons:

- Sensitive to outliers because means and standard deviations move when one team has an extreme value.
- Early-season distributions can be unstable.
- Linear feature effects can miss threshold effects and interactions.
- Equal scaling does not encode any football prior about which feature groups matter more.

Best use:

- Baseline model.
- Sanity check for new feature builds.
- Quick comparison against the market or against more complex architectures.

### `stat_percentile`

Config: [configs/models/stat/config_percentile.yaml](configs/models/stat/config_percentile.yaml)

This variant converts each feature value into a percentile rank relative to the training population. If `percentile_center` is true, percentiles are centered to roughly `[-1, 1]`:

```text
centered_percentile = (percentile - 0.5) * 2
```

The implementation also infers feature direction. For features where lower is better, such as turnovers or penalties, it flips the percentile so better values remain higher transformed values.

After percentile transformation, the model fits ridge regression with `percentile_ridge_alpha`.

Pros:

- Much less sensitive to extreme values than z-score scaling.
- Works well when the order of teams matters more than the raw distance between teams.
- Handles weird feature distributions better than standard z-scores.
- Easy to compare to poll/ranking intuition.

Cons:

- Loses magnitude information. A team barely above average and a team far above average only differ by rank spacing.
- Percentile resolution depends on the size and quality of the reference population.
- Direction inference is pattern-based unless explicitly overridden in config.
- Linear weights on percentiles can still miss interactions.

Best use:

- Early outlier-resistant stat baseline.
- Historical KNN-style thinking where relative position matters.
- Comparing whether rank-order signal beats raw magnitude signal.

### `stat_robust`

Config: [configs/models/stat/config_robust.yaml](configs/models/stat/config_robust.yaml)

This variant standardizes features around the median rather than the mean. It supports robust scale estimates:

```text
robust_z = (x - median) / robust_scale
```

The current config uses MAD scaling:

```text
robust_scale = MAD * 1.4826
```

The implementation can also use IQR scaling. After scaling, values can be clipped with `robust_clip`, currently `5.0`, to prevent extreme teams from dominating the linear fit.

Pros:

- Keeps the familiar z-score shape but reduces outlier influence.
- More stable than standard z-score when feature distributions are skewed.
- Clipping protects training from rare or noisy values.
- Usually a clean upgrade path from `stat_z_index`.

Cons:

- Still assumes monotonic linear effects after transformation.
- MAD/IQR can be noisy for small samples.
- Clipping can hide legitimate extreme team quality if set too aggressively.
- Does not add explicit football feature priors by itself.

Best use:

- Safer replacement candidate for `stat_z_index`.
- Seasons or weeks where stat distributions include unstable extremes.
- Baseline before adding opponent adjustments.

### `stat_weighted`

Config: [configs/models/stat/config_weighted.yaml](configs/models/stat/config_weighted.yaml)

This variant applies explicit football-family weights to normalized features before fitting ridge regression. The default config uses robust scaling, then multiplies features by family and modifier weights.

Broad family keys include:

- `offense`
- `defense`
- `special_teams`
- `general`
- `roster`

Modifier groups include:

- `efficiency`
- `explosiveness`
- `field_position`
- `finishing_drives`
- `turnovers`
- `penalties`
- `talent`

For example, a feature whose name indicates both offense and efficiency gets the offense weight and the efficiency modifier.

Pros:

- Lets football intuition enter the model without building a complex architecture.
- Easy to tune by editing YAML.
- Can test whether hand-weighted feature families beat equal weighting.
- Keeps the final model linear and inspectable.

Cons:

- Weight choices are subjective until tuned against validation seasons.
- Feature family detection is name-pattern based.
- Multiplicative weights can overemphasize a feature if both family and modifier weights are high.
- Less flexible than learned nonlinear interactions.

Best use:

- Testing football-prior hypotheses.
- Fast iteration on which feature families matter.
- A bridge between pure stat baselines and richer tree models.

## Linear Family

Implementation: [src/gridiron_ml/models/td_linear.py](src/gridiron_ml/models/td_linear.py)

The linear family now dispatches to real sklearn linear-model algorithms. Each config maps to a concrete estimator class, wrapped in a shared TDNet pipeline that keeps training, checkpointing, evaluation, and TD Sim behavior consistent.

The basic training flow:

1. Convert all features to numeric values.
2. Fill missing values with training medians.
3. Median-impute missing values inside an sklearn `Pipeline`.
4. Standardize features when `training.standardize` is true.
5. Fit the selected sklearn regressor to home-margin targets.
6. Save train/validation metric rows and coefficient-based feature importance when available.

The prediction target remains:

```text
pred_margin = model.predict(matchup_features)
```

The model then converts margin to probability with:

```text
prob_home_win = sigmoid(pred_margin / margin_temperature)
```

### Shared Linear Metrics

Configs can still set `loss_function` to:

- `RMSE`
- `MAE`
- `Composite`

For sklearn estimators this setting controls TDNet's training-history/evaluation summary, not the estimator's internal fitting objective unless the estimator itself exposes a matching objective. Most configs use `RMSE` as the reporting objective. The composite metric combines:

- Smooth margin loss.
- Win probability binary cross entropy.
- Confidence-weighted favorite correctness loss.
- Calibration loss.

### `ols`

Config: [configs/models/linear/config_ols.yaml](configs/models/linear/config_ols.yaml)

Estimator: `sklearn.linear_model.LinearRegression`

This is ordinary least squares linear regression.

Pros:

- Cleanest linear baseline.
- Easy to interpret.
- Useful for seeing whether regularization is actually helping.

Cons:

- Can overfit when feature count is high.
- Coefficients can become unstable with collinear features.
- Sensitive to leverage points and outliers.

Best use:

- Debugging and baseline comparisons.

### `ridge`

Config: [configs/models/linear/config_ridge.yaml](configs/models/linear/config_ridge.yaml)

Estimator: `sklearn.linear_model.Ridge`

This is true ridge regression with L2 regularization. Current config sets `alpha: 10.0`.

Pros:

- Reduces coefficient instability from correlated features.
- Usually strong for wide fingerprint tables.
- Keeps all features available rather than forcing sparse selection.

Cons:

- Does not perform feature selection.
- Can still underfit if penalty is too high.
- Linear relationships only.

Best use:

- Default linear benchmark.
- Strong comparison point against trees and TDStat.

### `lasso`

Config: [configs/models/linear/config_lasso.yaml](configs/models/linear/config_lasso.yaml)

Estimator: `sklearn.linear_model.Lasso`

This is true Lasso regression with L1 regularization. Current config sets `alpha: 0.01`.

Pros:

- Encourages sparse weights.
- Can help identify a smaller set of useful features.
- Useful when many features are noisy or redundant.

Cons:

- Can arbitrarily choose one feature among correlated groups.
- May underuse weak but collectively useful signals.
- Needs scaling and alpha tuning to avoid zeroing too much.

Best use:

- Feature selection pressure.
- Model comparison when you suspect the fingerprint table is too wide.

### `elastic_net`

Config: [configs/models/linear/config_elastic_net.yaml](configs/models/linear/config_elastic_net.yaml)

Estimator: `sklearn.linear_model.ElasticNet`

This combines L1 and L2 penalties. Current config sets `alpha: 0.01` and `l1_ratio: 0.5`.

Pros:

- Balances sparsity and coefficient stability.
- More forgiving than pure Lasso when features are correlated.
- Good practical middle ground.

Cons:

- Two regularization knobs instead of one.
- Still linear.
- Needs validation discipline to avoid tuning to noise.

Best use:

- Default candidate when both feature selection and stability matter.

### `huber`

Config: [configs/models/linear/config_huber.yaml](configs/models/linear/config_huber.yaml)

Estimator: `sklearn.linear_model.HuberRegressor`

This is true Huber regression. It behaves like squared-error regression for moderate residuals and absolute-error regression for large residuals, which makes it less sensitive to outlier margins.

Pros:

- Robust to unusual blowouts and bad feature rows.
- Still produces linear coefficients.
- Good stress test against OLS/Ridge.

Cons:

- More sensitive to convergence settings than OLS/Ridge.
- `epsilon` controls how aggressively residuals are treated as outliers.
- Can underweight real blowout signal if too robust.

Best use:

- Margin robustness tests.

### `bayesian`

Config: [configs/models/linear/config_bayesian.yaml](configs/models/linear/config_bayesian.yaml)

Estimator: `sklearn.linear_model.BayesianRidge`

This is true Bayesian ridge regression. It estimates regularization strength from the data and exposes a probabilistic linear-regression framing, though TDNet currently uses only the point margin prediction.

Pros:

- Regularization is data-adaptive.
- Useful comparison to manually tuned Ridge.
- Handles collinearity better than OLS.

Cons:

- TDNet does not yet consume predictive uncertainty from it.
- Linear assumptions still apply.
- Priors can be too generic without tuning.

Best use:

- Data-adaptive regularized linear benchmark.

### `ard`

Config: [configs/models/linear/config_ard.yaml](configs/models/linear/config_ard.yaml)

Estimator: `sklearn.linear_model.ARDRegression`

Automatic Relevance Determination is a sparse Bayesian linear model. It can shrink irrelevant coefficients heavily while keeping useful predictors active.

Pros:

- Bayesian-style sparsity without manually choosing an L1 penalty.
- Can identify feature groups that contribute little.
- Good contrast with Lasso and Bayesian Ridge.

Cons:

- Can be slower and less stable than Ridge.
- Sparse decisions may vary with correlated features.
- TDNet currently uses point predictions only.

Best use:

- Sparse Bayesian linear comparison.

### `ransac`

Config: [configs/models/linear/config_ransac.yaml](configs/models/linear/config_ransac.yaml)

Estimator: `sklearn.linear_model.RANSACRegressor`

RANSAC repeatedly fits a linear model on candidate inlier subsets and rejects outlier rows. The wrapped base estimator is `LinearRegression`.

Pros:

- Explicitly robust to outlier games or noisy rows.
- Tests whether a few strange observations are distorting OLS-style fits.
- Still simple and interpretable through the final inlier model.

Cons:

- Randomized and more variable than deterministic linear models.
- Can throw away legitimate rare football outcomes.
- Needs enough rows to identify stable inlier subsets.

Best use:

- Outlier-robust linear comparison.

### `orthogonal_matching_pursuit`

Config: [configs/models/linear/config_orthogonal_matching_pursuit.yaml](configs/models/linear/config_orthogonal_matching_pursuit.yaml)

Estimator: `sklearn.linear_model.OrthogonalMatchingPursuit`

Orthogonal Matching Pursuit greedily selects features that best explain residual variation.

Pros:

- Produces sparse, interpretable linear models.
- Very different feature-selection behavior from Lasso.
- Useful when only a small number of fingerprint features should matter.

Cons:

- Greedy feature selection can miss better feature combinations.
- Sensitive to correlated predictors.
- Can underfit if the signal is broad and distributed.

Best use:

- Sparse feature-selection comparison.

### `sgd`

Config: [configs/models/linear/config_sgd.yaml](configs/models/linear/config_sgd.yaml)

Estimator: `sklearn.linear_model.SGDRegressor`

This is stochastic-gradient linear regression. The current config uses squared-error loss with L2 penalty.

Pros:

- Scales well to larger feature tables.
- Can emulate several linear objectives through config.
- Very different optimization path from closed-form and coordinate-descent models.

Cons:

- More sensitive to scaling and learning-rate settings.
- Can be noisier than deterministic solvers.
- Usually needs tuning to beat Ridge on small or medium tabular data.

Best use:

- Scalable linear baseline and optimizer-diversity check.

### `passive_aggressive`

Config: [configs/models/linear/config_passive_aggressive.yaml](configs/models/linear/config_passive_aggressive.yaml)

Estimator: `sklearn.linear_model.PassiveAggressiveRegressor`

Passive-aggressive regression updates aggressively when predictions miss by more than the epsilon-insensitive zone and stays passive otherwise.

Pros:

- Different online-learning style objective.
- Can react strongly to badly missed examples.
- Cheap and fast.

Cons:

- Sensitive to `C`, scaling, and data ordering.
- Not naturally calibrated for margin magnitude.
- Can be volatile compared with Ridge.

Best use:

- Online-style linear baseline and architecture diversity.

## Tree Family

Implementation: [src/gridiron_ml/models/td_tree.py](src/gridiron_ml/models/td_tree.py)

The tree family wraps sklearn regressors behind the same TDNet interface. It predicts margin directly and reuses the probability, loss breakdown, and ranking helpers inherited from `TDLinear`.

Training flow:

1. Convert all features to numeric values.
2. Optionally pull a sample-weight column if configured.
3. Build an sklearn pipeline with median imputation.
4. Optionally add `StandardScaler`, although current tree configs leave standardization off.
5. Fit the selected tree regressor.
6. Save model-native feature importances when available.

Tree models are currently:

- `random_forest`
- `extra_trees`
- `gradient_boosted`

### `random_forest`

Config: [configs/models/tree/config_random_forest.yaml](configs/models/tree/config_random_forest.yaml)

This uses `RandomForestRegressor` with 500 estimators, `max_features: sqrt`, bootstrap sampling, and parallel jobs.

Pros:

- Strong nonlinear baseline.
- Handles feature interactions without manually specifying them.
- Bootstrap aggregation reduces variance.
- Provides feature importances.

Cons:

- Less interpretable than linear or stat models.
- Can smooth away sharp signals.
- Larger checkpoint and slower inference than stat models.
- Feature importance can be biased toward high-cardinality or high-variance features.

Best use:

- General-purpose nonlinear benchmark.
- Comparing whether feature interactions matter.

### `extra_trees`

Config: [configs/models/tree/config_extra_trees.yaml](configs/models/tree/config_extra_trees.yaml)

This uses `ExtraTreesRegressor` with 500 estimators and randomized split thresholds. Bootstrap is off in the default config.

Pros:

- Often faster and more variance-reducing than random forest.
- Randomized thresholds can regularize noisy features.
- Strong at discovering broad nonlinear structure.

Cons:

- Can be less precise when exact split points matter.
- Interpretability is still limited.
- May underfit subtle margin relationships if randomness is too aggressive.

Best use:

- Robust nonlinear baseline.
- Situations where random forest looks too variance-heavy.

### `gradient_boosted`

Config: [configs/models/tree/config_gradient_boosted.yaml](configs/models/tree/config_gradient_boosted.yaml)

This uses `GradientBoostingRegressor` with shallow trees, learning rate `0.03`, 500 estimators, minimum leaf size 10, and subsampling.

Pros:

- Sequentially corrects residual errors.
- Often strong on tabular data.
- Captures nonlinear effects and interactions.
- Shallow trees can produce useful structured corrections.

Cons:

- More sensitive to hyperparameters than forest models.
- Can overfit if estimators, depth, or learning rate are poorly set.
- Slower to tune.
- Current implementation uses classic sklearn gradient boosting, not histogram boosting or XGBoost.

Best use:

- Strong tabular candidate after baseline models are stable.
- Testing whether residual structure remains after linear/stat models.

## Practical Comparison

The families are intentionally complementary:

| Family | What It Tests | Main Strength | Main Risk |
| --- | --- | --- | --- |
| Stat | Can football stats alone produce a useful rating? | Fast, interpretable, stable baseline | Misses interactions and may depend on scaling assumptions |
| Linear | Can the full fingerprint table add signal linearly? | Transparent weights, good with regularization | Cannot model nonlinear interactions |
| Tree | Do nonlinear splits and interactions improve prediction? | Captures interactions without manual feature crosses | Harder to interpret and easier to overfit |

Recommended evaluation order:

1. Train `stat_z_index`, `stat_percentile`, `stat_robust`, and `stat_weighted`.
2. Compare those to `ols`, `ridge`, `lasso`, `elastic_net`, `huber`, `bayesian`, and `ard`.
3. Add `ransac`, `orthogonal_matching_pursuit`, `sgd`, and `passive_aggressive` for robustness, sparsity, and optimizer diversity.
4. Use `random_forest` and `extra_trees` as nonlinear baselines.
5. Use `gradient_boosted` when you are ready to tune.
6. Judge models on multiple seasons and compare against Vegas, not just raw RMSE.

## Current Caveats

- Linear-model names now map to actual sklearn estimators, but each estimator still uses TDNet's shared imputation, scaling, probability, and evaluation wrapper.
- The stat feature direction logic is pattern-based unless `feature_directions` is configured explicitly.
- Current models are game-level margin predictors, not play-level or drive-level models.
- Market data should remain evaluation context unless a future model is explicitly intended to use betting-market features.
- Opponent adjustment and recursive season fingerprint evolution belong upstream of these models and should be evaluated separately.
