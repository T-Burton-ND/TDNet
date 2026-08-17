# TDNet 2026 Fingerprint Engineering and Prospective Publication Plan

**Prepared:** 2026-07-15  
**Purpose:** Codex implementation specification  
**Primary scientific goal:** Produce a publication-quality fingerprint-engineering study showing how predictive performance, calibration, robustness, and interpretability change as increasingly rich football information is supplied to increasingly flexible statistical and machine-learning models.

## Master status ledger (2026-07-29)

This is the single remaining TDNet planning file. Older planning documents were
absorbed here or converted into technical documentation under `docs/technical/`.
Historical proposals below are reference material; only the checklist and the
active protocol govern current work.

### Executed

- [x] Establish the F0–F7 fingerprint ladder and the six confirmatory model
  candidates.
- [x] Complete the leakage-safe 2025 replay/refit design and freeze boundary.
- [x] Add public College Football Database attribution and provenance language.
- [x] Build the weekly publication/poll workflow and release gates.
- [x] Move Python entry points into `src/gridiron_ml/cli/`, leaving shell/SGE
  launchers under `scripts/`.
- [x] Submit the all-fingerprint scientific refit chain (`1316334`,
  `1316335.1-48`, `1316336`).
- [x] Consolidate repository plans, notebooks, style assets, and operational
  logs into the publication documentation layout.

### Remaining

- [x] Complete the corrected F0–F8 refit and validate all 54 frozen cells.
- [x] Complete cross-fitted calibration through 2025 and prepare the scientific
  and wide-roster archives and checksums.
- [ ] Publish the prepared model archives and verify fresh downloads.
- [ ] Run the first live weekly poll workflow against the locked 2026 roster.
- [ ] Complete the final manuscript figures, tables, and prospective-season
  results as data become available.

### Absorbed or superseded plans

- The proposed full hyperparameter-search rerun is superseded by the completed
  search results and the bounded F0–F7 refit array; no new search is authorized
  without an explicit owner decision.
- The preseason-to-postseason readiness checklist and publication question log
  are represented by the executed/remaining ledger above.
- The 17 June artifact-reduction prompt is implemented in the current output
  and publication workflows.

Technical notes retained outside this plan:

- [Model catalog](technical/MODEL_CATALOG.md)
- [Architecture designs](technical/MODEL_ARCHITECTURES.md)
- [Opponent-adjusted statistics note](technical/OPPONENT_ADJUSTED_STATS.md)
- [2026 run inventory](technical/RUN_INVENTORY_2026-07-20.md)
- [Unit-matchup feature pairing](technical/unit_matchup_feature_pairing_reviewed.xlsx)

## Active 2026 publication protocol (2026-07-26)

This supersedes stale planning language below where it conflicts with the
current owner decisions. No new hyperparameter search is submitted. The
existing search results are refit through 2025 using football-only features;
market-bearing rows and F7/F8 are excluded. Winner and margin/MAE are separate
frozen model sets. KNN, constant-margin, home-team, and majority models are
comparative baselines only and are excluded from weekly predictions, consensus,
and TDNet ballots. The canonical weekly poll uses uncapped raw fitted scores
with deterministic ties. Vegas is evaluation-only, split into winner-accuracy
and margin-MAE views, each with full-roster and top-five plots. Raw CFBD-derived
tables remain private; public releases contain code, hashes, compact metadata,
and explicitly selected aggregate figures.

The older F7/F8 market sections and large search-array counts below are
historical design notes only. They are not part of the executed 2026 protocol,
are not confirmatory evidence, and must not be presented as active model or
job plans in a public methods release.

---

## 1. Scientific framing

The intended paper is not simply:

> “We searched many models and found one that predicted winners accurately.”

The intended paper is:

> “We systematically engineered increasingly information-rich representations of a football team, evaluated those representations across a controlled ladder of model complexity, froze the complete preseason modeling system, and prospectively evaluated it throughout the 2026 season.”

The central experiment is a **feature-complexity × model-complexity matrix**.

At one extreme:

- simple box-score features;
- a transparent linear or logistic regression model;
- minimal tuning;
- easily interpretable coefficients.

At the other extreme:

- efficiency, opponent-adjusted, temporal, graph/fingerprint, interaction, and historical-context features;
- nonlinear models such as random forests, gradient boosting, multilayer neural networks, and optional embedding-based models;
- complete calibration and robustness evaluation.

The study must distinguish:

1. gains caused by richer information;
2. gains caused by more flexible model classes;
3. gains caused by interactions between rich information and flexible models;
4. apparent gains caused by overfitting, leakage, unstable tuning, or market-derived information;
5. performance that survives a prospective, preseason-frozen 2026 evaluation.

---

## 2. Primary paper questions

### 2.1 Representation question

How much predictive value is added as the representation progresses from raw box-score information to engineered efficiency statistics, opponent-adjusted features, temporal summaries, and full team fingerprints?

### 2.2 Model-complexity question

How much value is added as the model progresses from linear statistical methods to tree ensembles and neural networks?

### 2.3 Interaction question

Do complex models benefit meaningfully from complex fingerprints, or do simpler models extract most of the available signal?

### 2.4 Prospective generalization question

Do the historically selected models retain their performance, calibration, and relative ranking during the untouched 2026 season?

### 2.5 Incremental-information question

Do fingerprint models add predictive information beyond:

- simple team statistics;
- standard efficiency features;
- public rankings;
- conventional team-strength ratings;
- market spreads or implied probabilities?

### 2.6 Engineering question

Which feature families are stable, useful, redundant, or harmful across seasons, folds, seeds, model classes, and prediction objectives?

---

## 3. Non-negotiable experimental principles

Codex must preserve the following principles throughout implementation.

### 3.1 The 2026 confirmatory models are frozen preseason

No model promoted into the confirmatory 2026 evaluation may be:

- retrained;
- retuned;
- recalibrated;
- reweighted;
- supplied with a newly added feature;
- changed because of observed 2026 performance.

Any post-freeze experiment must receive a new model identity and be labeled exploratory.

### 3.2 Feature and model complexity must be varied independently

Do not compare only:

- simple features with a simple model;
- rich features with a complex model.

That confounds feature complexity with model complexity.

Every feasible feature tier must be evaluated across every core model class. This factorial structure is the paper’s main scientific design.

### 3.3 Historical evaluation must be temporal and grouped

Primary historical validation must not rely on random game-level splits.

Use:

- rolling-origin season splits;
- leave-one-season-out evaluation where appropriate;
- grouped folds that prevent the same target period from leaking across train and validation;
- strict feature timestamps.

### 3.4 Finalist game-level predictions must be retained

Metric-only output is insufficient for publication analysis.

For every finalist configuration, retain out-of-fold game-level predictions so paired comparisons, calibration, confidence intervals, subgroup analyses, and error audits can be performed.

### 3.5 Publication outputs must be reproducible from tables

Every publication or blog figure must be generated from canonical tables by scripts. Do not rely on manually edited spreadsheets, notebooks with hidden state, or one-off plotting code.

### 3.6 SGE is the default heavy-compute backend

All expensive experiment families must support:

- manifest generation;
- chunked SGE arrays;
- local smoke tests;
- resumability;
- deterministic seeds;
- compact worker outputs;
- post-run merge;
- status and failure summaries;
- controlled disk usage.

---

# Part I — Controlled feature engineering ladder

## 4. Build a canonical feature registry

Create:

```text
configs/features/feature_registry.yaml
```

Each feature must include:

```yaml
feature_name:
  family: box_score
  tier: F1
  description: Human-readable definition
  source: source_name
  units: yards_per_play
  direction: higher_is_better
  availability_rule: available_after_game_final
  temporal_lag: 1_game
  aggregation_window: season_to_date
  opponent_adjusted: false
  market_derived: false
  target_derived: false
  allowed_preseason: false
  allowed_weekly: true
  missing_policy: median_by_season
  code_path: package.module:function
  version: 1
```

The registry must be machine-readable and used to generate:

- feature manifests;
- feature-family lists;
- data-availability audits;
- ablation configs;
- blog descriptions;
- manuscript methods tables.

No production feature should exist only as an undocumented column name.

---

## 5. Define feature tiers

The exact available features can be adapted to the existing repository, but the conceptual tiers must remain fixed.

## F0 — Minimal strength baseline

Purpose: establish the smallest informative team-strength baseline without
using observed performance statistics.

Canonical team-state inputs:

- preseason `roster_talent`, which supplies the team-strength prior;
- `games_played`, which supplies season-to-date sample-size context.

Rules:

- F0 contains no observed box-score, efficiency, opponent-adjusted, temporal,
  schedule-network, or market statistic.
- F1 is the first tier that introduces observed game performance.
- Team identity and team fixed effects are not F0 inputs; any identity-only
  diagnostic must be reported separately because it may memorize historically
  strong programs.

## F1 — Raw box-score level information

Purpose: represent the information a traditional box score provides with minimal transformation.

Candidate offense and defense summaries:

- points;
- total yards;
- passing yards;
- rushing yards;
- attempts;
- completions;
- turnovers;
- sacks;
- first downs;
- penalties;
- possession;
- plays;
- third-down conversions;
- red-zone attempts and scores where available.

Aggregation variants:

- season-to-date mean;
- previous-N-game mean;
- exponentially weighted mean;
- opponent-minus-team differences.

Avoid opponent adjustment in this tier.

## F2 — Efficiency and rate statistics

Purpose: test whether normalization and football-aware rates add value beyond raw totals.

Candidate features:

- yards per play;
- yards per pass attempt;
- yards per rush;
- completion rate;
- success rate;
- points per drive;
- points per scoring opportunity;
- explosive-play rate;
- havoc rate;
- sack rate;
- turnover rate;
- third-down conversion rate;
- red-zone efficiency;
- pace;
- plays per drive;
- field-position summaries;
- offensive and defensive efficiency differentials.

F2 should be engineered from information already present in F1 wherever possible.

## F3 — Opponent-adjusted strength features

Purpose: estimate performance after accounting for opponent quality and schedule.

Candidate methods:

- iterative opponent adjustment;
- ridge or least-squares offense/defense ratings;
- adjusted efficiency;
- schedule-strength correction;
- Elo-like strength;
- Massey/Colley-style components where appropriate;
- graph-based strength metrics;
- residual performance above opponent expectation.

Every opponent-adjusted feature must be computed using data available strictly before the target game.

Store:

- method name;
- fit cutoff;
- convergence status;
- number of prior games;
- uncertainty or sample-size indicator where feasible.

## F4 — Temporal, situational, and trajectory features

Purpose: represent how a team is changing, not only its average level.

Candidate features:

- recent-versus-season performance;
- trend slopes;
- exponentially weighted form;
- volatility;
- consistency;
- offensive/defensive imbalance;
- returning production or preseason priors;
- coaching transition;
- travel distance;
- rest advantage;
- bye-week indicator;
- home-field interaction;
- early-season uncertainty;
- conference/nonconference context;
- ranked/unranked context only when ranks were already public.

Injuries should not enter the confirmatory system unless their source, timestamp, and historical reproducibility are reliable.

## F5 — Core fingerprint families

Purpose: encode a team as a structured, multivariate performance fingerprint.

Use the project’s established fingerprint definitions. Every family must be registered separately so it can be:

- included alone;
- removed alone;
- cumulatively added;
- shuffled;
- stability-tested;
- compared across model classes.

Examples may include:

- distributional fingerprints;
- offense/defense profile fingerprints;
- play-style fingerprints;
- opponent-response fingerprints;
- schedule-network fingerprints;
- matchup fingerprints;
- temporal fingerprints;
- rank/order fingerprints;
- interaction fingerprints.

Codex must not treat “all fingerprint columns” as one opaque block.

## F6 — Full non-market football representation

Purpose: combine all legal football-derived information.

Include:

- F1;
- F2;
- F3;
- F4;
- F5.

Exclude all betting-market features.

This is the primary representation for claims about football information independent of Vegas.

## F7 — Market-only representation

Purpose: establish a difficult information benchmark.

Candidate inputs:

- spread at model freeze time;
- moneyline-implied probability after vig adjustment;
- total;
- opening-to-freeze line movement;
- market availability flags.

This tier must not include team fingerprints.

## F8 — Football plus market representation

Purpose: test whether fingerprints add incremental value beyond market information.

Include:

- F6;
- F7.

Claims must distinguish clearly between:

- market-free performance;
- market-aware performance;
- incremental value over market-only predictions.

---

## 6. Cumulative feature paths

At minimum, evaluate:

```text
F0
F1
F1+F2
F1+F2+F3
F1+F2+F3+F4
F1+F2+F3+F4+F5 = F6
F7
F6+F7 = F8
```

Also evaluate selected isolated families:

```text
F2 only
F3 only
F5 family A only
F5 family B only
...
all F5 fingerprints without conventional stats
```

The canonical cumulative order must be declared before evaluating 2026.

Create:

```text
configs/features/feature_ladders.yaml
```

---

# Part II — Controlled model-complexity ladder

## 7. Required model families

Every model must expose a standard interface:

```python
fit(X_train, y_train, sample_weight=None)
predict(X)
predict_proba(X)        # classification where applicable
save(path)
load(path)
get_metadata()
```

All preprocessing must be bundled into the fitted pipeline or checkpoint.

## M0 — Constant and naive models

Required:

- majority winner;
- constant mean home margin;
- home-team winner;
- previous-season strength heuristic;
- higher-ranked team where available;
- market favorite for market comparisons.

## M1 — Linear statistical models

Required:

### Winner

- logistic regression;
- regularized logistic regression with ridge;
- optional elastic-net logistic regression.

### Margin

- ordinary least squares;
- ridge regression;
- elastic net;
- optional robust regression as a sensitivity model.

Implementation requirements:

- scaling inside pipeline;
- imputation inside pipeline;
- coefficients exported;
- coefficient confidence/stability summaries where feasible;
- no preprocessing fit on validation or test data.

## M2 — Generalized additive or controlled nonlinear statistical model

Recommended:

- spline/GAM model;
- or a polynomial/interactions model with strict regularization.

Purpose:

- provide an intermediate step between linear regression and black-box ensembles;
- determine whether smooth nonlinear effects explain most gains.

Treat this as recommended rather than blocking if package support is problematic.

## M3 — Random forest

Required for:

- winner;
- balanced winner objective if retained;
- margin.

Required controls:

- number of trees;
- maximum depth;
- minimum leaf size;
- maximum feature fraction;
- class weighting;
- bootstrap settings;
- deterministic seeds;
- `n_jobs=1` inside SGE workers unless a task is deliberately assigned multiple internal cores.

Export:

- permutation importance;
- impurity importance only as a secondary diagnostic;
- tree depth and leaf summaries;
- calibration results;
- feature-family importance aggregation.

## M4 — Gradient-boosted trees

Required unless there is a strong repository constraint.

Preferred implementation:

- XGBoost, LightGBM, or HistGradientBoosting;
- choose one primary implementation and document its version.

Required controls:

- learning rate;
- depth or leaves;
- number of rounds;
- regularization;
- row/column sampling;
- early stopping inside training folds only.

This is an important state-of-practice tabular baseline and should not be omitted while adding neural networks.

## M5 — Multilayer perceptron for tabular data

Required new model family.

Implement a practical tabular MLP rather than an unnecessarily exotic architecture.

Suggested initial architecture search:

- 1–4 hidden layers;
- widths such as 32, 64, 128, 256;
- ReLU or GELU;
- dropout 0.0–0.5;
- batch normalization or layer normalization as an option;
- weight decay;
- learning-rate schedule;
- early stopping;
- 3–5 deterministic seeds for finalists.

Winner output:

- one logit;
- binary cross-entropy or weighted BCE;
- optional label smoothing only as a secondary experiment.

Margin output:

- one continuous value;
- MSE, Huber, or MAE-compatible loss comparison;
- optional dual-head mean and uncertainty model as an exploratory extension.

Required preprocessing:

- training-fold-only imputation;
- standardization;
- optional quantile transformation as a declared hyperparameter;
- all transformations serialized with the checkpoint.

Required exports:

- training and validation loss curves;
- epoch selected;
- seed;
- architecture;
- parameter count;
- optimizer settings;
- early-stopping reason;
- raw and calibrated probabilities;
- checkpoint hash.

Do not use GPU dependence as a requirement. The model must have a CPU SGE path.

## M6 — Optional structured or embedding neural model

Exploratory, not required for the initial preseason freeze.

Possible forms:

- separate offense and defense encoders;
- feature-family tower network;
- team fingerprint encoder followed by matchup head;
- shared Siamese-style team encoder;
- autoencoder-compressed fingerprint followed by prediction head;
- transformer-like tabular architecture only if justified.

A useful structured matchup model could compute:

```text
home_embedding = encoder(home_fingerprint)
away_embedding = encoder(away_fingerprint)
matchup_input = [
    home_embedding,
    away_embedding,
    home_embedding - away_embedding,
    home_embedding * away_embedding,
    context_features
]
prediction = matchup_head(matchup_input)
```

This should not delay the core MLP or preseason freezing.

## M7 — Ensemble

Required only after individual models are frozen and assessed.

Candidate ensemble methods:

- unweighted probability average;
- nonnegative weights learned only from historical out-of-fold predictions;
- median margin ensemble;
- stacking with an explicitly nested historical procedure.

Freeze ensemble membership and weights preseason.

Do not optimize ensemble weights on 2026.

---

# Part III — Main factorial experiment

## 8. Required comparison matrix

Run every feasible pairing:

| Feature tier | Linear | GAM/intermediate | Random forest | Boosted trees | MLP |
|---|---:|---:|---:|---:|---:|
| F0 | Yes | Optional | Yes | Yes | Yes |
| F1 | Yes | Recommended | Yes | Yes | Yes |
| F1+F2 | Yes | Recommended | Yes | Yes | Yes |
| F1+F2+F3 | Yes | Recommended | Yes | Yes | Yes |
| F1+F2+F3+F4 | Yes | Recommended | Yes | Yes | Yes |
| F6 full football | Yes | Recommended | Yes | Yes | Yes |
| F7 market only | Yes | Optional | Yes | Yes | Yes |
| F8 football+market | Yes | Recommended | Yes | Yes | Yes |

This matrix is the center of the paper.

The analysis must quantify:

- marginal gain from richer features holding model fixed;
- marginal gain from a richer model holding features fixed;
- interaction between feature tier and model family;
- robustness of each gain across season folds and seeds;
- whether complexity improves mean performance at the cost of calibration or stability.

---

## 9. Prediction objectives

Maintain separate tasks.

### Winner probability

Primary outputs:

- predicted home-win probability;
- predicted winner.

Metrics:

- Brier score;
- log loss;
- accuracy;
- balanced accuracy;
- ROC AUC as secondary;
- upset recall;
- calibration intercept/slope;
- expected calibration error;
- reliability diagrams.

### Margin

Primary output:

- predicted home margin.

Metrics:

- MAE;
- RMSE;
- mean signed error;
- median absolute error;
- fraction within 3, 7, 10, and 14 points;
- residual calibration by predicted-margin bin.

### Balanced or upset-aware objective

Keep only if it has a clearly defined use.

Do not allow a vaguely named “balanced” objective to remain in the paper without specifying:

- target;
- loss;
- weighting;
- decision threshold;
- scientific interpretation.

### Rankings

Treat ranking generation as a related downstream use of team-strength predictions, not as an interchangeable game-level objective.

Evaluate rankings using future outcomes, not only agreement with human polls.

---

# Part IV — Historical validation and model selection

## 10. Split design

Implement canonical split definitions in:

```text
configs/splits/
  rolling_origin.yaml
  leave_one_season_out.yaml
  final_historical_holdout.yaml
```

Recommended hierarchy:

1. **Development seasons:** feature engineering and broad model search.
2. **Inner temporal folds:** hyperparameter tuning.
3. **Outer season folds:** honest historical model comparison.
4. **Final historical holdout season:** final rehearsal only.
5. **2026 season:** untouched prospective evaluation.

Every prediction row must store:

- train seasons;
- validation season;
- outer fold;
- inner-fold definition;
- seed;
- feature cutoff;
- model/config ID.

## 11. Reconsider the 743,400-row exhaustive rerun

Do not launch the entire robustness expansion automatically.

First create a finalist-selection stage from the completed search.

### Candidate reduction rules

For each objective, model family, and feature tier:

1. remove failed or incomplete configurations;
2. remove configurations dominated across all major metrics;
3. identify the top region, not only rank 1;
4. retain configurations within a predefined practical threshold or one-standard-error band;
5. retain at least one simple configuration;
6. retain representative neighboring hyperparameters;
7. retain configurations selected across multiple folds;
8. cap finalists per family/tier unless diversity requires more.

Suggested target:

- 5–15 finalists per feature-tier/model-family cell;
- 3 seeds for all finalists;
- 5 seeds for final champions if affordable;
- season-based outer folds.

The full rerun may remain available as an optional exhaustive mode, but the default publication run should prioritize defensible nested validation and prediction retention.

---

## 12. Game-level finalist prediction storage

Create canonical schema:

```text
prediction_id
experiment_id
objective
feature_tier
feature_manifest_sha256
model_family
model_config_id
model_seed
outer_fold
train_seasons
test_season
game_id
game_start_time_utc
created_at_utc
home_team
away_team
neutral_site
pred_home_win_probability
pred_home_margin
pred_winner
actual_home_win
actual_home_margin
market_spread_at_cutoff
market_home_probability_at_cutoff
status
```

Store finalists in Parquet.

Do not store hundreds of thousands of redundant model objects. Store:

- compact metrics for all trials;
- game-level predictions for finalists;
- checkpoints only for finalists and champions;
- complete metadata and hashes for frozen models.

---

# Part V — SGE implementation requirements

## 13. Standard experiment workflow

Every expensive experiment family must implement this sequence:

```text
config
  -> manifest builder
  -> chunk manifest
  -> smoke test
  -> SGE submit wrapper
  -> worker
  -> compact fragment output
  -> merge
  -> validation
  -> summary tables
  -> publication/blog report
```

## 14. Required shared scripts

Create or standardize:

```text
src/gridiron_ml/cli/experiments/build_experiment_manifest.py
src/gridiron_ml/cli/experiments/build_chunk_manifest.py
src/gridiron_ml/cli/experiments/run_experiment_chunk.py
src/gridiron_ml/cli/experiments/merge_experiment_chunks.py
src/gridiron_ml/cli/experiments/validate_experiment_output.py
src/gridiron_ml/cli/experiments/summarize_experiment.py
src/gridiron_ml/cli/experiments/finalize_selected_models.py
```

SGE wrappers:

```text
scripts/sge/experiment_chunk_task.sge
scripts/sge/submit_experiment_chunks.sh
scripts/sge/submit_feature_model_matrix.sh
scripts/sge/submit_neural_network_search.sh
scripts/sge/submit_robustness_suite.sh
scripts/sge/submit_ablation_suite.sh
scripts/sge/submit_finalist_refits.sh
```

Do not duplicate nearly identical SGE files for each experiment when one parameterized wrapper is sufficient.

## 15. SGE manifest requirements

Each task row must specify:

```text
task_id
chunk_id
experiment_id
objective
feature_config
model_config
split_config
seed
output_path
estimated_memory_gb
estimated_runtime
```

The worker must select work by `SGE_TASK_ID`.

## 16. Resource and threading rules

Defaults:

```bash
NSLOTS=4
CHUNK_SIZE=4
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
```

Inside each chunk:

- run up to four independent single-threaded trials;
- avoid nested parallelism;
- set tree model `n_jobs=1`;
- set PyTorch CPU threads deliberately;
- record actual host, slots, and environment.

Neural-network trials may use one trial per task if memory or runtime requires it. Make chunk size configurable per experiment family.

## 17. Smoke-test requirement

Every submit wrapper must support:

```bash
--smoke-test
--max-chunks 5
--max-trials 20
```

Smoke tests must verify:

- imports;
- config parsing;
- data loading;
- feature construction;
- one successful fit for every model family;
- checkpoint saving;
- prediction output;
- metrics output;
- merge behavior;
- disk estimates;
- failure propagation.

No full submission should proceed when smoke-test validation fails.

## 18. Resume and idempotency

Workers must:

- skip completed valid trial rows;
- rerun missing or corrupt rows;
- write temporary files and atomically rename;
- not overwrite a valid result silently;
- emit structured failure records;
- permit resubmission of failed task IDs only.

Create:

```text
summary/status/completed_trials.parquet
summary/status/failed_trials.parquet
summary/status/missing_trials.parquet
summary/status/duplicate_trials.parquet
summary/status/merge_report.json
```

## 19. Disk guardrails

Before submission:

- calculate expected fragment count;
- estimate retained size;
- inspect available space;
- abort below configurable free-space threshold;
- print cleanup candidates;
- default to Parquet;
- use CSV.GZ only for portable public summaries.

Large intermediate predictions should be retained only for finalist runs.

---

# Part VI — Neural-network implementation work

## 20. Create neural-network package

Suggested layout:

```text
src/tdnet/models/neural/
  __init__.py
  datasets.py
  preprocessing.py
  mlp.py
  structured_matchup.py
  losses.py
  training.py
  calibration.py
  checkpoint.py
  inference.py
  diagnostics.py
```

## 21. Determinism metadata

Every neural run must record:

- Python version;
- PyTorch version;
- platform;
- device;
- seed;
- deterministic-algorithm setting;
- thread settings;
- architecture;
- parameter count;
- optimizer;
- scheduler;
- batch size;
- maximum epochs;
- early-stopping patience;
- best epoch;
- training duration;
- preprocessing hashes;
- checkpoint SHA-256.

Perfect bitwise reproducibility across all hardware is not guaranteed, so record sufficient metadata and evaluate multiple seeds.

## 22. Neural hyperparameter search

Start with a bounded, scientifically interpretable search.

Suggested dimensions:

```text
hidden_layers: [1, 2, 3, 4]
hidden_width: [32, 64, 128, 256]
activation: [relu, gelu]
dropout: [0.0, 0.1, 0.25, 0.4]
weight_decay: [0, 1e-6, 1e-5, 1e-4, 1e-3]
learning_rate: [1e-4, 3e-4, 1e-3, 3e-3]
batch_size: [32, 64, 128, 256]
normalization: [none, batch_norm, layer_norm]
loss: objective-specific
```

Do not run a full Cartesian product unless justified.

Preferred search sequence:

1. coarse random or quasi-random search;
2. retain robust regions;
3. seed expansion on finalists;
4. nested temporal evaluation;
5. calibration;
6. preseason final fit.

## 23. Neural calibration

Evaluate:

- uncalibrated MLP;
- temperature scaling;
- Platt/logistic calibration;
- isotonic regression only when enough calibration data exist.

Calibration must be trained only on historical held-out predictions.

Freeze the calibration object with the model.

## 24. Neural learning curves

Produce:

- performance versus training-season count;
- performance versus number of games;
- train and validation loss by epoch;
- calibration versus training size;
- feature-tier comparisons at matched architecture.

These are important for determining whether the neural model is data-limited.

---

# Part VII — Fingerprint engineering experiments

## 25. Feature-family ablation

For every major finalist model:

- full features;
- remove each family one at a time;
- each family alone;
- conventional-only;
- fingerprint-only;
- full football features.

Output paired game-level predictions.

Required tables:

```text
feature_family_ablation_metrics.parquet
feature_family_ablation_deltas.parquet
feature_family_ablation_predictions.parquet
```

## 26. Cumulative feature performance

For each core model family, show performance as feature tiers are cumulatively added.

Required figures:

- Brier score versus feature tier;
- accuracy versus feature tier;
- margin MAE versus feature tier;
- calibration slope versus feature tier;
- training cost versus feature tier;
- feature count versus performance;
- marginal improvement per added feature family.

## 27. Negative controls

Required:

- shuffled target;
- random Gaussian features matched to fingerprint dimension;
- shuffled fingerprint columns;
- within-season fingerprint shuffle;
- team-identity-only model;
- season/week-only model;
- duplicated/noise columns;
- target-game leakage injection that the validator must reject.

The leakage-injection test should be an automated test, not a production experiment.

## 28. Feature importance and stability

For tree models:

- grouped permutation importance;
- seed stability;
- fold stability;
- season stability;
- family-level importance;
- optional SHAP on a bounded finalist sample.

For linear models:

- standardized coefficients;
- sign stability;
- magnitude stability;
- regularization path.

For neural models:

- permutation importance;
- grouped occlusion;
- integrated gradients only if implemented carefully;
- do not rely on raw first-layer weights.

Produce rank-correlation matrices and top-K overlap across folds/seeds.

## 29. Redundancy and uniqueness

Analyze whether fingerprints merely reconstruct simpler ratings.

Required experiments:

1. predict fingerprint components from F1–F4;
2. predict conventional ratings from F5;
3. compute correlations and mutual-information summaries;
4. compare F1–F4 against F1–F5;
5. add fingerprints last and measure paired out-of-fold gain;
6. conditional permutation of fingerprint families where feasible.

---

# Part VIII — Final preseason model selection

## 30. Freeze a compact model roster

Do not freeze dozens of nearly indistinguishable public models.

Recommended confirmatory roster:

### Winner probability

- simple champion: F1 + logistic regression;
- engineered statistical champion: F1–F4 + regularized logistic/GAM;
- fingerprint tree champion: F6 + random forest or boosted trees;
- fingerprint neural champion: F6 + MLP;
- market-only comparator: F7;
- market-aware fingerprint champion: F8;
- one preseason-frozen ensemble.

### Margin

- simple champion: F1 + ridge regression;
- engineered statistical champion;
- fingerprint tree champion;
- fingerprint neural champion;
- market spread comparator;
- market-aware champion;
- one ensemble.

This roster directly supports the narrative from simple information/simple model to rich information/complex model.

## 31. Champion selection rule

The selection script must apply a predeclared rule such as:

1. identify models within the practical-equivalence or one-standard-error region;
2. reject models with unacceptable calibration or instability;
3. prefer the simpler model when performance is practically equivalent;
4. select one champion per planned role;
5. write a machine-readable selection report.

Create:

```text
configs/publication/final_model_selection.yaml
docs/publication_2026/preseason/model_selection_report.md
docs/publication_2026/preseason/model_selection_table.csv
docs/publication_2026/preseason/final_model_inventory.csv
```

Do not manually select the final champion without a recorded rule and explanation.

---

# Part IX — State-of-the-art preseason freeze and proof

## 32. Evidence standard

A SHA-256 hash stored only on the same machine is not enough.

A strong modern proof uses several independent layers:

1. **content-addressed model and data manifests;**
2. **signed source-control state;**
3. **immutable public release;**
4. **independent archival timestamp and persistent identifier;**
5. **public transparency-log signature or attestation;**
6. **repeatable verification command;**
7. **prospective weekly prediction bundles derived from the frozen inventory.**

The objective is to prove:

- exactly which code existed;
- exactly which model bytes existed;
- exactly which preprocessing and feature definitions existed;
- exactly which data snapshot or data hashes were used;
- exactly when the bundle was published;
- that later files differ if even one byte changes;
- that weekly predictions point back to the preseason-frozen models.

## 33. Recommended preseason freeze stack

### Layer A — Deterministic release bundle

Create:

```text
docs/publication_2026/preseason/freeze_bundle/
  final_model_inventory.csv
  model_selection_report.md
  model_cards/
  checkpoints/
  preprocessing/
  calibration/
  feature_registry.yaml
  feature_ladders.yaml
  split_definitions/
  environment/
  code_provenance.json
  data_snapshot_manifest.json
  freeze_manifest.json
  SHA256SUMS
  README.md
```

Where legal or practical constraints prevent public checkpoint release, include:

- checkpoint SHA-256;
- byte size;
- model metadata;
- storage location class;
- verification procedure;
- reason the checkpoint is private.

### Layer B — Canonical manifest

`freeze_manifest.json` should contain:

```json
{
  "freeze_version": "2026-preseason-v1",
  "created_at_utc": "...",
  "git_commit": "...",
  "git_tree": "...",
  "git_dirty": false,
  "models": [],
  "feature_manifest_sha256": "...",
  "data_snapshot_sha256": "...",
  "schedule_snapshot_sha256": "...",
  "environment_lock_sha256": "...",
  "container_sha256": "...",
  "files": {
    "relative/path": {
      "sha256": "...",
      "size_bytes": 0
    }
  }
}
```

Create a canonical serialization before hashing the manifest.

### Layer C — Signed Git commit and signed annotated tag

Require:

- clean worktree;
- signed commit;
- signed annotated tag such as `tdnet-2026-preseason-freeze-v1`;
- protected tag or release workflow.

The tag must point to the exact source tree used to build the freeze bundle.

### Layer D — GitHub immutable release

Publish:

- the small public-safe freeze bundle;
- `SHA256SUMS`;
- manifest;
- signature/attestation bundles;
- verification script;
- release notes.

Enable GitHub immutable releases if available. GitHub documents immutable releases as preventing changes to release assets and the associated tag after publication.

### Layer E — Zenodo archive and DOI

Connect the repository or upload the release bundle to Zenodo.

Archive the exact preseason release and record:

- version-specific DOI;
- concept DOI where applicable;
- deposit timestamp;
- GitHub release tag;
- manifest hash.

Zenodo’s GitHub integration can archive releases and provide a persistent DOI-backed research record.

### Layer F — Sigstore/Cosign signing and transparency log

Use keyless Sigstore signing for the manifest or release archive:

```bash
cosign sign-blob \
  --yes \
  --bundle freeze_manifest.sigstore.json \
  freeze_manifest.json
```

The resulting bundle should contain the signature, certificate, timestamp information, and transparency-log inclusion proof.

Verify with a repository script, recording the expected identity and issuer.

Sigstore’s Rekor transparency log provides an externally auditable record of the signing event.

### Layer G — Optional formal timestamp authority

For additional redundancy, timestamp the manifest hash using one of:

- an RFC 3161 timestamp authority;
- an institutional timestamping service;
- OpenTimestamps;
- another public append-only timestamp mechanism.

This is optional when GitHub, Zenodo, and Sigstore are all used, but it provides another independent proof.

### Layer H — SLSA/in-toto-style provenance

Generate an attestation describing:

- source commit;
- builder identity;
- build command;
- config inputs;
- data-manifest hash;
- produced checkpoint hashes;
- environment/container digest.

Use an in-toto-style predicate or SLSA-inspired provenance file even if the project does not claim formal SLSA certification.

The key idea is to attest not only that the files existed, but how they were produced.

## 34. Strong recommended combination

The recommended practical stack is:

> **Signed commit/tag + GitHub immutable release + Zenodo DOI archive + Sigstore transparency-log signature + SHA-256 manifest + reproducible verification script.**

This is stronger than screenshots, emailed files, ordinary Git tags, or a lone hash.

A third party should be able to run:

```bash
python src/gridiron_ml/cli/publication/verify_preseason_freeze.py \
  --bundle docs/publication_2026/preseason/freeze_bundle \
  --expected-tag tdnet-2026-preseason-freeze-v1 \
  --expected-doi-file docs/publication_2026/preseason/zenodo_record.json \
  --sigstore-bundle docs/publication_2026/preseason/freeze_manifest.sigstore.json
```

And receive:

```text
PASS: all file hashes match
PASS: manifest self-hash matches
PASS: git commit and tag match
PASS: release assets match local files
PASS: Sigstore signature and transparency inclusion verify
PASS: Zenodo record references the expected release
PASS: all model checkpoints are represented in inventory
PASS: worktree was clean at freeze
PASS: no model was created after the declared freeze timestamp
```

## 35. Freeze script suite

Create:

```text
src/gridiron_ml/cli/publication/build_preseason_freeze_bundle.py
src/gridiron_ml/cli/publication/hash_freeze_bundle.py
src/gridiron_ml/cli/publication/generate_provenance_attestation.py
scripts/publication/sign_freeze_manifest.sh
src/gridiron_ml/cli/publication/verify_preseason_freeze.py
src/gridiron_ml/cli/publication/build_model_cards.py
src/gridiron_ml/cli/publication/render_freeze_readme.py
scripts/publication/check_release_asset_integrity.sh
```

The build script must fail when:

- the worktree is dirty;
- a final checkpoint is missing;
- preprocessing is missing;
- calibration is missing;
- a model has no game-level historical evaluation;
- a checkpoint hash differs from inventory;
- a selected model references an unregistered feature;
- a feature has no availability rule;
- the environment lock is missing;
- the final model selection report is absent.

## 36. Model identity

Assign each frozen model a stable ID:

```text
TDNET26-WIN-F1-LOGREG-v1
TDNET26-WIN-F6-RF-v1
TDNET26-WIN-F6-MLP-v1
TDNET26-WIN-F8-GBM-v1
TDNET26-MAR-F1-RIDGE-v1
TDNET26-MAR-F6-RF-v1
TDNET26-MAR-F6-MLP-v1
```

Every weekly prediction row must contain:

- model ID;
- checkpoint hash;
- preprocessing hash;
- calibration hash;
- preseason freeze manifest hash;
- preseason release tag;
- code commit used for inference;
- weekly data snapshot hash.

The weekly inference code may receive bug fixes only through an amendment process. Any change must be explicit and must not alter the frozen checkpoint or feature semantics without creating a new exploratory model ID.

---

# Part X — Weekly prospective prediction and blog pipeline

## 37. Use the existing publication-readiness plan

Retain the previously specified principles:

- one immutable prediction row per model/game before kickoff;
- required game identity;
- provenance fields;
- frozen schedule, market, and poll snapshots;
- a public prediction bundle;
- separate post-game scoring;
- a cumulative season ledger;
- no mutation of original predictions.

Implement those requirements as part of the same publication package rather than as a disconnected workflow.

## 38. Weekly directory layout

```text
publication/2026/
  preseason/
    freeze_bundle/
    historical_evaluation/
    figures/
    tables/
    blog/
  weekly_predictions/
    week_00/
      public/
      private/
      verification/
    week_01/
  weekly_scores/
    week_00/
      tables/
      figures/
      blog/
    week_01/
  season_summary/
    tables/
    figures/
    blog/
    manuscript_inputs/
  amendments/
```

## 39. Weekly workflow

Implement one command:

```bash
python src/gridiron_ml/cli/publication/run_weekly_publication_pipeline.py \
  --season 2026 \
  --week 0 \
  --deadline-utc "..." \
  --freeze-manifest docs/publication_2026/preseason/freeze_bundle/freeze_manifest.json
```

Stages:

1. pull or load source snapshots;
2. verify all feature timestamps;
3. build weekly as-of feature matrix;
4. run every frozen model;
5. validate row completeness;
6. reject post-kickoff predictions;
7. create public-safe prediction table;
8. create manifest and hashes;
9. sign the weekly manifest;
10. render README and blog tables;
11. optionally prepare GitHub release assets;
12. write season-ledger entry.

Post-game:

```bash
python src/gridiron_ml/cli/publication/score_weekly_publication_bundle.py ...
```

This must never mutate the original prediction table.

## 40. Blog-ready output standard

Every experiment and weekly report should generate:

```text
tables/
  *.parquet
  *.csv
figures/
  *.png
  *.svg
blog/
  summary.md
  figure_captions.md
  key_findings.json
  alt_text.md
metadata/
  report_manifest.json
```

Blog Markdown should be ready to paste into the user’s site with minimal editing.

It should include:

- concise plain-language framing;
- model and feature-tier labels;
- exact sample size;
- frozen-model version;
- freeze release tag and DOI;
- figure captions;
- caveats;
- links/placeholders to public artifacts;
- no unsupported “beat Vegas” claim;
- no statement based on exploratory models presented as confirmatory.

## 41. Required preseason blog outputs

Generate:

1. **What is a football fingerprint?**
2. **The feature ladder: box scores to full fingerprints**
3. **The model ladder: regression to neural networks**
4. **How the 2026 models were selected**
5. **How the preseason freeze can be independently verified**
6. **What counts as a fair test against Vegas**
7. **2026 prediction protocol and amendment policy**

## 42. Required experiment figures

Core paper/blog figures:

1. feature-complexity × model-complexity heatmap;
2. cumulative winner performance versus feature tier;
3. cumulative margin performance versus feature tier;
4. model complexity versus performance for each feature tier;
5. calibration curves by finalist;
6. learning curves;
7. feature-family ablation deltas;
8. feature importance stability;
9. seed/fold/season performance distributions;
10. random/shuffled-feature controls;
11. market-only versus football-only versus combined;
12. performance versus training cost and inference cost;
13. model disagreement matrix;
14. historical season-by-season performance;
15. prospective cumulative 2026 performance.

Do not create every conceivable PNG during worker execution. Plot only after canonical merge.

---

# Part XI — Statistical analysis

## 43. Paired comparisons

All major comparisons must use predictions on the same games.

Winner models:

- paired Brier-score difference;
- paired log-loss difference;
- paired accuracy difference;
- McNemar test for hard predictions;
- bootstrap confidence intervals.

Margin models:

- paired MAE difference;
- paired RMSE difference;
- paired signed-error difference;
- bootstrap confidence intervals.

Report uncertainty for differences, not only uncertainty around each model.

## 44. Dependence-aware sensitivity

Because teams appear repeatedly, include sensitivity analyses that resample or cluster by:

- week;
- season for historical analysis;
- team where feasible;
- game as the simplest comparison.

Avoid presenting game-level independence as unquestionably true.

## 45. Complexity effects

Create a tidy analysis table with:

```text
feature_complexity_level
model_complexity_level
objective
metric
fold
season
seed
value
```

Analyze:

- main effect of feature complexity;
- main effect of model complexity;
- interaction;
- monotonicity;
- diminishing returns;
- stability penalty;
- compute cost.

A mixed-effects or hierarchical descriptive analysis may be used, but do not make the paper depend on an unnecessarily fragile inferential model.

## 46. Practical significance

Before the season, define practical thresholds for:

- Brier improvement;
- accuracy improvement;
- margin MAE improvement;
- calibration improvement;
- incremental gain beyond market.

Use historical variability to justify thresholds.

Classify comparisons as:

- meaningfully superior;
- practically equivalent;
- inconclusive;
- meaningfully inferior.

---

# Part XII — Data leakage and as-of audits

## 47. Feature availability audit

Generate:

```text
docs/publication_2026/preseason/tables/feature_availability_audit.csv
```

Fields:

```text
feature_name
source
event_date
release_timestamp
retrieval_timestamp
effective_cutoff
lag_rule
revision_risk
historical_asof_reconstructable
allowed_in_confirmatory_model
notes
```

## 48. Automated guardrails

Add tests that reject:

- target-game box-score values;
- future opponent games in opponent adjustment;
- polls before their publication timestamp;
- retrospectively corrected data without versioning;
- closing odds masquerading as pre-freeze odds;
- schedule changes after bundle generation;
- missing kickoff times;
- duplicate game/model predictions;
- model checkpoints absent from the preseason inventory.

---

# Part XIII — Manuscript-ready outputs

## 49. Canonical publication tables

Generate:

```text
table_01_data_summary.csv
table_02_feature_tiers.csv
table_03_model_families.csv
table_04_historical_performance.csv
table_05_feature_model_matrix.csv
table_06_calibration.csv
table_07_ablation.csv
table_08_negative_controls.csv
table_09_market_incremental_value.csv
table_10_prospective_2026_performance.csv
table_11_subgroup_sensitivity.csv
table_12_compute_cost.csv
```

## 50. Manuscript shell

Create:

```text
publication/2026/manuscript/
  manuscript_outline.md
  methods_preseason_frozen.md
  results_placeholders.md
  limitations_preseason_frozen.md
  reproducibility_appendix.md
  statistical_analysis_plan.md
```

Fill the Methods and statistical plan before Week 0. Leave prospective result values blank.

---

# Part XIV — Implementation phases and acceptance criteria

## Phase 0 — Repository audit

Tasks:

- locate existing feature generation;
- locate model registry;
- locate completed hyperparameter outputs;
- locate SGE scripts;
- locate prediction/report scripts;
- map duplicated logic;
- document current gaps.

Output:

```text
docs/publication_2026/current_state_audit.md
```

Acceptance:

- every existing relevant script and artifact is indexed;
- no new duplicate workflow is started unnecessarily.

## Phase 1 — Canonical registries and schemas

Tasks:

- feature registry;
- feature ladders;
- model registry;
- split configs;
- prediction schemas;
- experiment schemas.

Acceptance:

- schemas validate current historical data;
- all finalist features are registered;
- manifests can be generated deterministically.

## Phase 2 — Baseline and model ladder

Tasks:

- complete M0–M4;
- add MLP M5;
- shared fit/predict/save/load interface;
- standardized preprocessing.

Acceptance:

- every core model runs locally on a small temporal split;
- winner and margin outputs are valid;
- checkpoints reload to identical or tolerance-matched predictions.

## Phase 3 — SGE matrix infrastructure

Tasks:

- generic manifests;
- chunk workers;
- submit wrappers;
- merge/status;
- disk guardrails;
- smoke tests.

Acceptance:

- 5–10 chunk smoke test succeeds;
- interrupted tasks can resume;
- merged results equal local reference runs.

## Phase 4 — Controlled historical experiment

Tasks:

- execute feature × model matrix;
- temporal outer folds;
- seed expansion;
- game-level finalist predictions;
- calibration;
- compute-cost tracking.

Acceptance:

- all matrix cells have valid results or documented infeasibility;
- no random game-level split is used for primary claims;
- all paired comparisons are reproducible.

## Phase 5 — Fingerprint diagnostics

Tasks:

- cumulative features;
- ablations;
- random controls;
- stability;
- uniqueness/redundancy;
- learning curves.

Acceptance:

- each claim about fingerprints has a direct experiment;
- no feature-family claim relies only on impurity importance.

## Phase 6 — Finalist selection and refit

Tasks:

- apply frozen selection rule;
- fit finalists on allowed historical data;
- calibrate;
- save model cards;
- hash all artifacts.

Acceptance:

- one compact confirmatory roster;
- every selected model has a complete inventory record;
- no manual undocumented model replacement.

## Phase 7 — Preseason freeze

Tasks:

- freeze bundle;
- signed commit/tag;
- immutable GitHub release;
- Zenodo archive/DOI;
- Sigstore signature;
- verification scripts;
- blog methodology package.

Acceptance:

- clean third-party checkout can verify all public artifacts;
- every weekly model ID resolves to a preseason checkpoint hash;
- the release predates the first covered kickoff.

## Phase 8 — 2025 historical rehearsal

Tasks:

- simulate week-by-week prediction bundles;
- score separately;
- introduce deliberate tampering;
- introduce late predictions;
- test schedule changes;
- rebuild final report only from frozen weekly bundles.

Acceptance:

- tampering is detected;
- post-kickoff rows are rejected;
- original prediction files are never mutated;
- season summary is reproducible.

## Phase 9 — 2026 weekly operation

Tasks:

- produce bundle;
- verify;
- publish;
- score;
- blog;
- append ledger;
- record anomalies.

Acceptance:

- every included game has exactly one prediction per frozen model;
- publication timestamps precede kickoff;
- amendments are append-only.

## Phase 10 — Postseason manuscript

Tasks:

- lock ledger;
- verify all hashes;
- recompute from source bundles;
- generate final tables/figures;
- write results and discussion;
- archive final reproducibility package.

Acceptance:

- no result depends on a mutable notebook state;
- all primary claims trace to frozen predictions;
- confirmatory and exploratory models are clearly separated.

---

# Part XV — Codex operating instructions

## 51. General instructions

Codex should:

1. inspect existing code before creating new modules;
2. reuse established project paths and naming where sensible;
3. avoid changing scientific behavior silently;
4. add tests for every schema and guardrail;
5. preserve backward compatibility when practical;
6. prefer configuration-driven experiment generation;
7. keep worker output compact;
8. generate documentation alongside implementation;
9. never submit a large SGE run automatically;
10. prepare commands and smoke tests, then stop before the full submit unless explicitly instructed.

## 52. Change reporting

For each implementation batch, Codex must report:

- files added;
- files modified;
- scientific behavior changed;
- compatibility impact;
- tests run;
- smoke-test results;
- SGE command prepared;
- expected task count;
- expected disk use;
- expected output locations;
- remaining blockers.

## 53. No silent result deletion

Before removing existing artifacts or output generation:

- identify consumers;
- identify whether the output is manuscript/blog relevant;
- preserve canonical source tables;
- document migration;
- update references.

## 54. Configuration precedence

Use this precedence:

```text
CLI override
> experiment config
> publication config
> project defaults
```

Write the resolved config into every run directory.

## 55. Naming convention

Suggested experiment ID:

```text
{objective}__{feature_tier}__{model_family}__{split}__{search_version}
```

Suggested final model ID:

```text
TDNET26-{TASK}-{FEATURE}-{MODEL}-v{N}
```

Suggested weekly bundle ID:

```text
TDNET26-W{WEEK:02d}-{CREATED_UTC}
```

---

# Part XVI — Immediate work queue

## P0 — Must complete before any giant rerun

1. Build feature registry and feature tiers.
2. Build model registry.
3. Implement regularized linear baselines cleanly.
4. Add boosted-tree baseline if absent.
5. Implement tabular MLP for winner and margin.
6. Freeze temporal split definitions.
7. Build generic SGE experiment manifests and workers.
8. Retain game-level predictions for finalists.
9. Create feature × model matrix config.
10. Create candidate-reduction logic for the existing hyperparameter search.
11. Build data leakage and feature-availability audit.
12. Create manuscript statistical analysis plan.

## P1 — Must complete before preseason freeze

1. Run the controlled historical matrix.
2. Run finalist seed/fold robustness.
3. Run cumulative-feature experiments.
4. Run ablations and negative controls.
5. Run calibration.
6. Run learning curves.
7. Compare market-only, football-only, and combined models.
8. Apply final selection rule.
9. Fit and inventory frozen models.
10. Complete 2025 weekly rehearsal.
11. Build preseason freeze bundle.
12. Publish immutable release, DOI archive, and signature proof.
13. Build public verification command.
14. Build preseason blog package.

## P2 — Complete during or after the season without contaminating confirmatory models

1. Structured neural network.
2. Alternative tabular neural architecture.
3. exploratory weekly refits;
4. additional injury or roster features;
5. new fingerprint families;
6. post-freeze ensembles;
7. expanded SHAP analyses;
8. conference-specific experimental models.

All P2 systems must have model IDs that clearly distinguish them from the preseason-frozen confirmatory roster.

---

# Part XVII — Definition of publication readiness

The project is ready for the 2026 prospective season when all statements below are true:

- The paper’s feature and model ladders are fixed.
- Every final feature has an availability rule.
- Historical validation uses temporal/grouped splits.
- Linear, forest, boosted-tree, and MLP models are implemented.
- The primary feature × model matrix is complete.
- Finalist predictions exist at game level.
- Fingerprint ablations and negative controls are complete.
- Calibration is complete.
- The final model-selection rule has been executed.
- The confirmatory roster is compact and documented.
- Every checkpoint, preprocessor, calibrator, config, and source state has a hash.
- A signed tag identifies the source release.
- An immutable GitHub release contains the public-safe bundle.
- A Zenodo record archives the release and provides a DOI.
- A Sigstore bundle proves the manifest signature and transparency-log inclusion.
- A clean verification script succeeds.
- The 2025 weekly rehearsal succeeds.
- The weekly 2026 pipeline can create, verify, publish, score, and summarize predictions without modifying the original bundle.
- Blog and manuscript figures are generated from canonical merged tables.
- Confirmatory and exploratory models cannot be confused.

---

## Sources for the freeze methodology

- GitHub, “Immutable releases”: https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases
- GitHub, “Preventing changes to your releases”: https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/establish-provenance-and-integrity/prevent-release-changes
- GitHub, “Verifying the integrity of a release”: https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/verify-release-integrity
- Zenodo, “Archive a release from GitHub”: https://help.zenodo.org/docs/github/archive-software/github-upload/
- Zenodo, “Digital Object Identifier”: https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/
- Sigstore, “Sigstore Quickstart with Cosign”: https://docs.sigstore.dev/quickstart/quickstart-cosign/
- Sigstore, “Transparency Log”: https://docs.sigstore.dev/logging/
- Sigstore, “Rekor”: https://docs.sigstore.dev/logging/overview/
- SLSA, “Security levels”: https://slsa.dev/spec/v1.1/levels
- SLSA, “Build provenance”: https://slsa.dev/spec/draft/build-provenance

---

## Recommended first Codex command

Ask Codex to begin with **Phase 0 and Phase 1 only**:

> Audit the repository against `TDNet_2026_Fingerprint_Engineering_Publication_Plan.md`. Produce `docs/publication_2026/current_state_audit.md`, then implement the canonical feature registry, feature ladders, model registry, split-config schemas, and experiment/prediction schemas. Reuse existing code and paths where possible. Add validation tests. Do not submit SGE jobs and do not begin the full hyperparameter rerun. End with a concrete mapping of existing components to plan requirements and a prioritized implementation diff for adding boosted trees and the tabular MLP.
