# TDNet Model Interface Requirements (Unified Training, Evaluation + Weekly Rankings)

This document defines the minimum contract each TDNet model family must
satisfy so we can:

1.  Run a single hyperparameter tuning / ablation runner across all
    models\
2.  Generate publication-quality benchmark tables across model families\
3.  Produce margin-based + probability-based power rankings\
4.  Compare models against Vegas\
5.  Generate explainability + calibration analyses\
6.  Run a weekly Top-25 notebook with cached logos

------------------------------------------------------------------------

# 1) Canonical Data Contract

All models operate on the same fingerprints + labels.

## 1.1 Fingerprints (Two-Team Input)

-   `X_home`: `(N, D)` float array\
-   `X_away`: `(N, D)` float array\
-   `feature_names: List[str]` length `D`

Requirements: - Numeric only (float32/float64) - Missing values handled
by shared preprocessor - Strict row alignment: - `X_home[i]`,
`X_away[i]`, `y_margin[i]`, `y_win[i]` refer to the same game

------------------------------------------------------------------------

## 1.2 Labels

To support both margin and win modeling:

-   `y_margin`: `(N,)` float
    -   Defined as:\
        `home_points - away_points`
-   `y_win`: `(N,)` binary
    -   1 if home_margin \> 0 else 0

Optional: - `y_total` - `y_spread_result`

------------------------------------------------------------------------

## 1.3 Metadata (Required for Evaluation + Ranking)

Must be joinable to predictions:

-   `keys_game_id`
-   `keys_season`
-   `keys_week`
-   `keys_team_home`
-   `keys_team_away`
-   Optional: `market_spread`, `market_moneyline_prob`

------------------------------------------------------------------------

# 2) Preprocessing Contract

All scaling/imputation must be external and reproducible.

Required:

    fit(X_train_home, X_train_away)
    transform(X_home, X_away)

Must guarantee: - identical feature ordering - identical imputation
rules - no column dropping

------------------------------------------------------------------------

# 3) Model Output Contract

Every model must support at least one of:

-   Margin prediction
-   Win probability prediction
-   Or both

------------------------------------------------------------------------

## 3.1 Required: predict_margin()

    predict_margin(X_home, X_away) -> np.ndarray shape (N,)

Returns: - float home_margin predictions - positive → home expected to
win - negative → home expected to lose

------------------------------------------------------------------------

## 3.2 Required: predict_proba()

    predict_proba(X_home, X_away) -> np.ndarray shape (N,)

Returns: - P(home wins) - values strictly in (0,1)

------------------------------------------------------------------------

## 3.3 Optional: predict_distribution()

    predict_distribution(X_home, X_away) -> dict

Used for: - Uncertainty intervals - Betting edge estimation - Ranking
confidence bands

------------------------------------------------------------------------

# 4) Fit Contract

    fit(X_train_home, X_train_away,
        y_margin=None,
        y_win=None,
        X_val_home=None,
        X_val_away=None,
        y_val_margin=None,
        y_val_win=None) -> self

Must: - Respect provided seed - Support margin-only, win-only, or joint
training - Support early stopping if validation provided

------------------------------------------------------------------------

# 5) Evaluation Requirements

Runner computes:

## Margin Metrics

-   MAE
-   RMSE
-   R²
-   ΔMAE vs Vegas
-   ΔRMSE vs Vegas
-   95% bootstrap CI

## Win Metrics

-   Accuracy
-   Log Loss
-   Brier Score
-   Expected Calibration Error (ECE)
-   ΔLogLoss vs Vegas implied probability
-   ATS accuracy (optional)

------------------------------------------------------------------------

# 6) Hyperparameter Search Interface

Required:

    @staticmethod
    suggest_params(trial) -> dict

Optional: - resource_profile() - sample_random()

------------------------------------------------------------------------

# 7) Reproducibility Contract

Each model must accept:

    params["seed"]

Saved artifact must include: - model weights - params - feature_names -
git commit - data hash - training date

------------------------------------------------------------------------

# 8) Synthetic Strength Operator

## Average Team Function

    F_avg = make_average_team(fps, by=("season","week"))

## Neutral-Field Rating

Margin models:

    m1 = model.predict_margin(T, Avg)
    m2 = model.predict_margin(Avg, T)
    rating_margin = (m1 - m2) / 2

Probability models:

    p = model.predict_proba(T, Avg)
    rating_logit = log(p / (1 - p))

------------------------------------------------------------------------

# 9) Weekly Ranking Notebook Requirements

For each model: 1. Compute synthetic rating 2. Rank teams 3. Output Top
25 with: - Rank - Team - Logo - Rating

Composite poll: - Rank 1 → 25 points ... Rank 25 → 1 point

------------------------------------------------------------------------

# 10) Explainability Interface

Top-performing models must support:

    get_feature_importance() -> Dict[str, float]

Optional: - SHAP integration - Coefficient export - Partial dependence

------------------------------------------------------------------------

# 11) Logo Caching

Directory:

    data/assets/logos/{team_slug}.png

Behavior: - Load local if exists - Download once if missing -
Placeholder if unknown

------------------------------------------------------------------------

# 12) Minimal Plug-and-Play Checklist

-   [ ] fit() trains without error\
-   [ ] predict_margin() returns (N,) float\
-   [ ] predict_proba() returns (N,) in (0,1)\
-   [ ] save/load round-trips\
-   [ ] suggest_params() exists\
-   [ ] respects feature order\
-   [ ] respects seed\
-   [ ] supports synthetic strength computation\
-   [ ] compatible with unified evaluation runner
