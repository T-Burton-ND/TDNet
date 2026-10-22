# Scientific 120-cell hyperparameter sensitivity study (prepared, unrun)

This is the proposed historical tuning-sensitivity study for F0–F19 × M1/M2/M3/M4/M5/M10. The frozen [configuration](../../../../configs/experiments/scientific_120_hps_v1.yaml), [candidate manifest](candidate_tasks.jsonl), [plan receipt](plan.json), and [input audit](input_audit.json) are preparation artifacts. No model has been fitted for this study.

## Design

- Each of the 120 fingerprint × architecture cells receives 100 unique random hyperparameter sets. The same 100 architecture-specific settings are reused across all 20 fingerprints, so comparisons are paired within architecture. This is 12,000 candidate evaluations × five rolling-origin folds = **60,000 planned fits**. An incumbent-settings control would add 600 fits after its provenance is bound.
- The five test seasons are 2021–2025. For each test year, training uses 2013 through the preceding season only. The target is the home-team margin. Preprocessing, early stopping, and any calibration must use only that fold's training years, with a still-earlier validation slice where needed. Never use the test year for early stopping.
- Primary selection is lowest equal-weight mean five-fold margin MAE, then lower fold standard deviation, then lower trial ID. This gives each season equal influence despite different game counts. Preserve all failed trials and their reasons. Retain the original per-cell settings as a separate control; do not silently count them among the 100 random candidates.
- The random-search space comes from the existing scientific HPS config, with the probability-only margin-temperature parameter removed and M10 neighbor count broadened to 1–150 so 100 distinct margin-relevant configurations exist. The study keeps F18/F19 representation choices fixed. Their prior representation search is part of their historical tuning advantage; the present study equalizes only **additional** hyperparameter opportunity.
- CPU only, one thread per fit, at most ten concurrent tasks for this study. The user's hard ceilings are 25 simultaneous tasks and two CPU cores per task. Set `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` in any future worker. The M5 PyTorch model is already CPU-first; reject a GPU backend in the worker receipt.

## Required source gate before any fit

The [audit](input_audit.json) currently says `blocked_before_fit`. The newer F09–F19 parent team-game file covers 2013–2025, including 632, 485, 562, 626, and 553 home-game rows in the proposed test seasons. The older F0–F8 archive contains 540 successful selected-fold summaries for 2015–2024, but those summaries are not per-game historical feature matrices. Build and freeze the F0–F8 feature matrices on the same five-year game intersection, with the original tier definitions, before launch. Also bind the F09–F19 derived matrices and F18/F19 frozen representation transforms to those game IDs.

For every cell, the launch gate must verify identical `target_game_id`, season, margin, home/away identity and kickoff; unique game IDs; 2013–2025 only; feature availability strictly before kickoff; and fold-specific fitting of any target-dependent transform. Save source-file SHA256, generated-matrix SHA256, feature manifest, missingness, market columns and quote timestamps. F7/F8/F17/F19 market inputs with unverified quote times are research-only and must be clearly separated from leakage-free claims. Do not fill holes with retrospective 2026 outcomes. The 2026 season is already inspected and is exploratory for this project.

The first five-fold search is a **development** selector, not an unbiased generalization estimate. Report its MAE curve as validation sensitivity. A later untouched, frozen forecast cohort is needed to claim tuned-versus-incumbent improvement. Brier, winner accuracy, ATS and upset metrics are secondary; probability calibration must be fit from inner historical out-of-fold predictions. Separate market-aware and market-free interpretations. Report season-paired differences and uncertainty, rank changes, and a best-score-versus-budget trace at 1/5/10/25/50/100 candidates. Do not call proximity to Vegas an irreducible-error bound.

## Metrics-only output contract

Any future worker writes one status row and five fold-metric rows per candidate. The aggregate `fold_metrics.csv` uses `fingerprint,architecture,trial_id,test_season,margin_mae` plus Brier, winner, ATS, game counts, runtime, CPU backend, input hash, and failure fields. Save no model checkpoints or serialized preprocessors. Save no prediction rows unless a separately approved research need arises. Save only numeric training history, fold scores, configuration, source hashes, and a compact status receipt.

For M5, save `epoch_history.csv` with `fingerprint,architecture,trial_id,test_season,epoch,train_loss,validation_loss` and optionally learning rate. Inner validation data must precede the outer test year. For M1/M2/M10 there is no optimizer epoch; M3/M4 estimator stages are architecture-specific and not directly comparable. The common all-model loss view is **validation MAE versus number of configurations tried**, with a mean line and five-fold interquartile band for each of the 120 cells. The separate M5 plot uses the one MAE-selected configuration per cell and shows average epoch training/validation loss with a five-fold interquartile band; it records the active-fold count because early stopping changes the denominator. Averaging raw losses across M5 configurations would mix Huber, MSE, and MAE units, so it is forbidden. Do not label the search-progress plot an epoch curve.

The prepared [plotter](../../../../scripts/publication/plot_scientific_120_hps_curves.py) accepts a complete fold-metric CSV and optional M5 epoch-history CSV, and writes six 20-panel search-progress figures plus the M5 epoch panel and their numeric summary CSVs. It refuses incomplete fold metrics. No figure is generated now because no HPS result exists.

## Reproduce the preparation

```bash
python3 scripts/publication/prepare_scientific_120_hps.py
python3 scripts/publication/audit_scientific_120_hps_inputs.py
```

The [plan receipt](plan.json) hashes the config, source search space, planner, and manifest. The input audit is deliberately a structural gate, not a full leakage proof. Implement and pass the full game-level source gate and CPU worker before submitting an SGE array; cap `-tc` at 10. In particular, the older `submit_new_fingerprint_hps_sge.sh` defaults to `-tc 50` and must not be reused for this study.
