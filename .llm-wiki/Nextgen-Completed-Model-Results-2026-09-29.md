---
type: synthesis
up: "[[Next-Generation-Fingerprint-Experiment]]"
related: "[[Nextgen-Results-Snapshot-2026-09-28]]"
tags: [nextgen, results, screening]
---

# Nextgen Completed Model Results — 2026-09-29

Saved results for all 360 previously submitted screening configurations are terminal: 356 successes and four F06 failures, with every successful prediction-file hash verified against its result record.

## Scope and aggregation

These are descriptive medians across successful configurations, computed independently for each metric. They are not metrics of one selected model or an ensemble. M2 is spline ridge; M4 is histogram gradient boosting. F denotes full inputs and R reduced inputs. Designs a/b/c are atomic/interactions/compressed football concepts. Each cell planned ten frozen configurations with seed 1701; F06 full a/b M2 each have eight successes, with two previously diagnosed memory failures. All other cells have ten successes.

2024 and 2025 are design-informed development years, not unbiased holdouts. F06/F09/F10 use 746 evaluation games in 2024 and 757 in 2025; F11/F12 use 626 and 553 respectively. Equal counts do not alone prove identical game identities. Do not infer improvement across the F10/F11 boundary from these raw medians. Existing common-game findings remain in the earlier empirical lessons; this table does not recompute paired comparisons.

The 2026-09-28 snapshot had 241 successful cells; this table includes 115 subsequently completed successes from the already-submitted work. The user reported increasing concurrency. No new model experiment, retraining, or model acceptance was performed for this table.

## Measured results

MAE is margin error in points (lower is better); winner and ATS values are accuracy. Each displayed metric is a median over the successful runs in that row.

| Fingerprint | Model | Successes | MAE 2024 | MAE 2025 | Winner 2025 | ATS 2025 |
|---|---|---:|---:|---:|---:|---:|
| F06_F_a | M2 | 8/10 | 13.473 | 13.080 | 71.7% | 50.9% |
| F06_F_a | M4 | 10/10 | 13.302 | 12.861 | 71.5% | 51.3% |
| F06_F_b | M2 | 8/10 | 13.481 | 13.128 | 71.5% | 50.9% |
| F06_F_b | M4 | 10/10 | 13.316 | 12.884 | 71.3% | 51.7% |
| F06_F_c | M2 | 10/10 | 13.386 | 13.020 | 71.9% | 50.9% |
| F06_F_c | M4 | 10/10 | 13.389 | 12.844 | 71.9% | 51.6% |
| F06_R_a | M2 | 10/10 | 13.435 | 12.985 | 71.7% | 50.8% |
| F06_R_a | M4 | 10/10 | 13.310 | 12.800 | 71.5% | 52.2% |
| F06_R_b | M2 | 10/10 | 13.427 | 13.011 | 72.3% | 50.8% |
| F06_R_b | M4 | 10/10 | 13.273 | 12.806 | 71.7% | 52.0% |
| F06_R_c | M2 | 10/10 | 13.344 | 12.938 | 72.3% | 52.4% |
| F06_R_c | M4 | 10/10 | 13.413 | 12.824 | 71.9% | 51.1% |
| F09_F_a | M2 | 10/10 | 13.448 | 13.037 | 71.5% | 50.9% |
| F09_F_a | M4 | 10/10 | 13.134 | 12.748 | 71.1% | 52.2% |
| F09_F_b | M2 | 10/10 | 13.478 | 13.184 | 71.9% | 50.2% |
| F09_F_b | M4 | 10/10 | 13.122 | 12.763 | 71.5% | 52.4% |
| F09_F_c | M2 | 10/10 | 13.379 | 12.921 | 71.7% | 50.6% |
| F09_F_c | M4 | 10/10 | 13.214 | 12.728 | 71.6% | 52.1% |
| F10_F_a | M2 | 10/10 | 13.393 | 12.897 | 72.4% | 51.8% |
| F10_F_a | M4 | 10/10 | 13.063 | 12.572 | 72.3% | 53.0% |
| F10_F_b | M2 | 10/10 | 13.464 | 13.037 | 72.3% | 52.0% |
| F10_F_b | M4 | 10/10 | 13.113 | 12.567 | 72.4% | 53.3% |
| F10_F_c | M2 | 10/10 | 13.417 | 12.762 | 71.5% | 52.0% |
| F10_F_c | M4 | 10/10 | 13.190 | 12.593 | 72.8% | 53.3% |
| F11_F_a | M2 | 10/10 | 13.707 | 12.537 | 71.0% | 53.3% |
| F11_F_a | M4 | 10/10 | 13.142 | 12.206 | 71.5% | 54.6% |
| F11_F_b | M2 | 10/10 | 13.831 | 12.670 | 70.6% | 52.6% |
| F11_F_b | M4 | 10/10 | 13.196 | 12.201 | 71.4% | 52.9% |
| F11_F_c | M2 | 10/10 | 13.711 | 12.371 | 71.2% | 53.1% |
| F11_F_c | M4 | 10/10 | 13.209 | 12.276 | 71.5% | 52.9% |
| F12_F_a | M2 | 10/10 | 13.824 | 12.729 | 69.8% | 53.1% |
| F12_F_a | M4 | 10/10 | 13.187 | 12.393 | 71.4% | 51.5% |
| F12_F_b | M2 | 10/10 | 13.968 | 12.916 | 69.1% | 51.9% |
| F12_F_b | M4 | 10/10 | 13.205 | 12.284 | 71.8% | 52.9% |
| F12_F_c | M2 | 10/10 | 13.746 | 12.595 | 70.6% | 52.7% |
| F12_F_c | M4 | 10/10 | 13.179 | 12.357 | 71.7% | 51.7% |

## Evidence and interpretation

The [complete CSV](evidence/Nextgen-Model-Results-2026-09-29.csv) also includes feature counts, evaluation counts, RMSE, Brier score, and upset accuracy for both years. [Verification evidence](evidence/Nextgen-Model-Results-2026-09-29.json) records all source result paths and hashes plus the verified prediction hashes. Results were read from `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen/experiments/`; repository HEAD at collection was `2c88a54`. Recorded metrics were aggregated without rerunning prediction scoring; hash verification is not a new full audit of training inputs or statistical claims.

These completed descriptive results do not themselves establish the best generation. F11's lower raw 2025 error uses a smaller evaluation cohort; the earlier common-game F11 A/M2 comparison was negative. Missing paired analyses for the newly completed cells remain follow-up work. No claims about causes of differences are established here.

See also: [Experiment](Next-Generation-Fingerprint-Experiment) · [Earlier snapshot](Nextgen-Results-Snapshot-2026-09-28).
