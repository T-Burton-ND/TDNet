---
type: analysis
up: "[[Next-Generation-Fingerprint-Experiment]]"
derived_from: ["[[Nextgen-Results-Snapshot-2026-09-28]]", "[[Fingerprint-F06]]", "[[Fingerprint-F09]]", "[[Fingerprint-F10]]", "[[Fingerprint-F11]]", "[[Fingerprint-F12]]"]
tags: [nextgen, empirical-lessons, measured-comparisons]
---

# Nextgen Empirical Lessons — 2026-09-28

The completed runs show architecture-dependent benefits from F09, useful F06 compression, and negative results for the currently tested F11/F12 A pipelines; this page records measured behavior rather than explanations assumed in advance.

## Question

What are we actually learning from the model results, including negative results and differences hidden by aggregate medians?

## Context

This is a read-only analysis of existing predictions from the [12:52 UTC results snapshot](Nextgen-Results-Snapshot-2026-09-28), not a new experiment or a claim about later-finishing runs. M2 = spline ridge; M4 = histogram gradient boosting. Fit years are 2010–2023; 2024/2025 are design-informed development years. No model was retrained, no feature trial was created, and no scheduler setting changed.

The frozen results Parquet SHA-256 is `8f8935f77bd39a6b185642b9bf20fca8f87293cd5387c0f44b3c9ffccee69dd4`. All 241 successful runs' prediction files were rechecked against the output hashes recorded in that frozen table. Comparisons below use terminal architecture matrices only. The [machine-readable comparisons and prediction hashes](evidence/nextgen-lessons-2026-09-28.json) and [analysis script](evidence/nextgen-lessons-2026-09-28.py) are published alongside this page. The raw frozen table/predictions live under `/groups/bsavoie2/tburton2/TDNet/fingerprint_nextgen`; those large artifacts are not uploaded to GitHub by this wiki update.

For each comparison, the script joins actual target-game IDs and seasons, verifies equal observed margins, and uses shared successful hyperparameter IDs. This controls evaluation games and configuration identity; it does **not** equalize training cohorts, feature spaces, or learned fits. F06 full A/B M2 each have eight shared successful configurations; other comparisons have ten. Configurations share data and are not independent statistical replications. No confidence interval or significance claim is made.

## Analysis

### 1. F09's strongest measured improvement is in M4; it is not uniform across models

F06 and F09 have **identical evaluation game IDs** in both development years: 746 games in 2024 and 757 in 2025. Their full historical row counts differ, but that does not prevent measuring predictive performance on these same evaluation games. This corrects the earlier overly broad suggestion that every cross-generation comparison was blocked by evaluation-cohort differences.

For M4, moving from each full F06 design to its full F09 descendant improves MAE in **all ten shared configurations in 2024**, in each of A/B/C. The median paired MAE changes are −0.139455 / −0.159766 / −0.120247 points for A/B/C. In 2025, improvements persist in 8/10, 7/10 and 6/10 configurations, with median paired changes −0.067242 / −0.070973 / −0.028712.

M2 is less consistent: F09 B in 2025 improves only 3/8 shared configurations, with a **+0.023923** median paired MAE change. Its difference of MAE medians is +0.056702. F09 A and C improve M2 medians in both years. The measured lesson is that the F09 pipeline helps M4 more consistently in this grid, while the B representation can hurt M2. These results do not identify interactions, overfitting, or any individual microstructure feature as the cause; feature additions and historical training coverage changed together.

### 2. F06 can be compressed substantially, but M2 benefits more consistently than M4

Reduced A/B/C retain 162/170/60 features versus 225/235/153. All use the same 746/757 evaluation games as their full references. C removes 60.8% of its source columns and remains within the recorded acceptance tolerance; it is the only accepted floor-sized reduction.

Across A/B/C respectively, M2 MAE improves in 6/8, 7/8, 8/10 shared configurations in 2024, and 8/8, 8/8, 9/10 in 2025. Thus the compression result survives matching A/B to their eight successful full-reference configurations; it is not merely an artifact of comparing ten candidate runs with eight reference runs.

M4 improvements are mixed. Reduced C improves just 4/10 configurations in **each** year. Its median paired changes are +0.040563 in 2024 and +0.015121 in 2025, even though its 2025 difference of medians is −0.020151. The evidence supports compression with small observed aggregate error changes, not a claim that reduction improves every model or that discarded features carry no signal.

### 3. A better median does not mean most matched configurations improved

Reduced B's M4 2024 median MAE falls from 13.316360 to 13.272906: a −0.043454 difference of medians. But only **4/10** matched configurations improve, and the **median paired difference is +0.003174**. Reduced C M4 2025 gives another sign reversal, documented above.

These are different estimands: `median(candidate) − median(reference)` can disagree with `median(candidate − reference)`. Both are computed from real results. The launch acceptance rule uses architecture medians, so those tolerance passes remain valid under that rule; they must not be described as consistent configuration-level gains. Earlier wording that “B improves all four MAE medians” is correct only at the aggregate level.

### 4. Supported F10 A improves the completed M2 results on the same evaluation games

F09 A and F10 A also have identical 746/757 evaluation game IDs. F10 A improves M2 MAE in **9/10** configurations in 2024 and **8/10** in 2025. Median paired changes are −0.068317 and −0.143402; differences of architecture medians are −0.054999 and −0.140236.

Secondary architecture medians move favorably too: Brier falls by 0.002555 / 0.002065 and winner accuracy on actual market-upset games rises by 1.628 / 3.846 percentage points. These are the implemented observed-use/recruiting/experience features and pipeline, not verified complete Week-0 roster or transfer coverage. M4 and B/C were not terminal at the frozen cutoff. The result supports the implemented A/M2 combination, not a conclusion about every roster design or the missing source families.

### 5. F11 A's apparent 2025 gain reverses when F10 is scored on the same games

The full-cohort M2 medians are 12.896856 for F10 A versus 12.537048 for F11 A in 2025. Comparing these directly makes F11 look better, but F11 evaluates only 553 of F10's 757 games.

Re-scoring the **already saved F10 predictions** on the same 553 games gives median MAE **12.446026**, versus **12.537048** for F11: F11 is worse by **0.091022**. Only 1/10 paired configurations improves. For 2024, on the same 626 games, F10 is **13.461476** versus F11 **13.706990**, a worsening of **0.245514**, with 0/10 improving.

This is a measured negative result for the currently tested F11 A/M2 pipeline. The favorable unmatched 2025 comparison was explained by evaluation selection strongly enough to reverse the ordering. It does not prove coaching information is useless: training coverage also changed, and current-assignment/interim/coordinator coverage is incomplete. F10 M4 was not terminal at the frozen cutoff, so no analogous M4 conclusion is drawn.

### 6. F12 A worsens MAE and Brier relative to F11 A on identical evaluation games

F11 A and F12 A share all **626 games in 2024 and 553 in 2025**, verified by IDs and targets. Their historical training coverage differs, but their development evaluation cohorts do not.

M2 worsens in **all ten configurations in both years**. Median paired MAE changes are **+0.121238 / +0.162777**; architecture-median changes are +0.116589 / +0.192169. M4 worsens in **7/10 configurations in each year**, with median paired changes **+0.082665 / +0.193884** and architecture-median changes +0.045133 / +0.186558.

Architecture-median Brier also worsens for both models in both years. M4's 2025 ATS median falls from 0.546125 to 0.514760, a **3.137 percentage-point** decline. M2 2024 ATS improves despite worse MAE/Brier, so deterioration is not universal across every secondary metric. The observed lesson is that this F12 A pipeline has not delivered an MAE/Brier gain over F11 A; the evidence does not support assuming later generations must be better. It is not a causal ablation of unit features, and the untested B/C designs or missing defensive-unit sources remain separate questions.

### 7. MAE-preserving compression can sacrifice upset recognition

F06 reduced C M2 improves median 2025 MAE by 0.082251 points, but median winner accuracy on **actual market-upset games** falls from 0.271795 to 0.241026: **3.077 percentage points**. M4's same upset metric falls from 0.228205 to 0.220513, and its ATS median falls from 0.515520 to 0.511471. Conversely, M2 2025 ATS improves from 0.509447 to 0.524291.

The upset metric is conditioned on games where the actual winner opposed the market favorite; it is not precision among predicted upsets. These mixed results demonstrate that passing the MAE reduction rule does not preserve every secondary behavior. They do not establish betting profitability, statistical significance, or an unbiased future edge.

### Full paired-MAE audit

All deltas are candidate minus reference, in margin points; negative is better. “Improved” counts shared configurations with strictly lower MAE. “Same IDs” means the original evaluation sets match; otherwise both predictions are evaluated on their intersection. “Δ medians” and “median paired Δ” deliberately remain separate.

| Reference → candidate | Model | Year | Games | Same IDs | Configs | Δ medians | Median paired Δ | Improved |
|---|---|---:|---:|---|---:|---:|---:|---:|
| F06_F_a → F06_R_a | M2 | 2024 | 746 | yes | 8 | -0.038282 | -0.030349 | 6/8 |
| F06_F_a → F06_R_a | M2 | 2025 | 757 | yes | 8 | -0.095192 | -0.081893 | 8/8 |
| F06_F_a → F06_R_a | M4 | 2024 | 746 | yes | 10 | +0.008451 | +0.005218 | 4/10 |
| F06_F_a → F06_R_a | M4 | 2025 | 757 | yes | 10 | -0.060956 | -0.028518 | 6/10 |
| F06_F_b → F06_R_b | M2 | 2024 | 746 | yes | 8 | -0.053614 | -0.047761 | 7/8 |
| F06_F_b → F06_R_b | M2 | 2025 | 757 | yes | 8 | -0.116111 | -0.107085 | 8/8 |
| F06_F_b → F06_R_b | M4 | 2024 | 746 | yes | 10 | -0.043454 | +0.003174 | 4/10 |
| F06_F_b → F06_R_b | M4 | 2025 | 757 | yes | 10 | -0.077768 | -0.030268 | 5/10 |
| F06_F_c → F06_R_c | M2 | 2024 | 746 | yes | 10 | -0.042948 | -0.038734 | 8/10 |
| F06_F_c → F06_R_c | M2 | 2025 | 757 | yes | 10 | -0.082251 | -0.103255 | 9/10 |
| F06_F_c → F06_R_c | M4 | 2024 | 746 | yes | 10 | +0.024707 | +0.040563 | 4/10 |
| F06_F_c → F06_R_c | M4 | 2025 | 757 | yes | 10 | -0.020151 | +0.015121 | 4/10 |
| F06_F_a → F09_F_a | M2 | 2024 | 746 | yes | 8 | -0.024928 | -0.041771 | 7/8 |
| F06_F_a → F09_F_a | M2 | 2025 | 757 | yes | 8 | -0.043310 | -0.060112 | 8/8 |
| F06_F_a → F09_F_a | M4 | 2024 | 746 | yes | 10 | -0.167766 | -0.139455 | 10/10 |
| F06_F_a → F09_F_a | M4 | 2025 | 757 | yes | 10 | -0.113502 | -0.067242 | 8/10 |
| F06_F_b → F09_F_b | M2 | 2024 | 746 | yes | 8 | -0.002634 | -0.011735 | 6/8 |
| F06_F_b → F09_F_b | M2 | 2025 | 757 | yes | 8 | +0.056702 | +0.023923 | 3/8 |
| F06_F_b → F09_F_b | M4 | 2024 | 746 | yes | 10 | -0.194306 | -0.159766 | 10/10 |
| F06_F_b → F09_F_b | M4 | 2025 | 757 | yes | 10 | -0.120150 | -0.070973 | 7/10 |
| F06_F_c → F09_F_c | M2 | 2024 | 746 | yes | 10 | -0.007353 | -0.020762 | 7/10 |
| F06_F_c → F09_F_c | M2 | 2025 | 757 | yes | 10 | -0.099151 | -0.040536 | 9/10 |
| F06_F_c → F09_F_c | M4 | 2024 | 746 | yes | 10 | -0.174538 | -0.120247 | 10/10 |
| F06_F_c → F09_F_c | M4 | 2025 | 757 | yes | 10 | -0.115525 | -0.028712 | 6/10 |
| F09_F_a → F10_F_a | M2 | 2024 | 746 | yes | 10 | -0.054999 | -0.068317 | 9/10 |
| F09_F_a → F10_F_a | M2 | 2025 | 757 | yes | 10 | -0.140236 | -0.143402 | 8/10 |
| F10_F_a → F11_F_a | M2 | 2024 | 626 | intersection | 10 | +0.245514 | +0.245514 | 0/10 |
| F10_F_a → F11_F_a | M2 | 2025 | 553 | intersection | 10 | +0.091022 | +0.122226 | 1/10 |
| F11_F_a → F12_F_a | M2 | 2024 | 626 | yes | 10 | +0.116589 | +0.121238 | 0/10 |
| F11_F_a → F12_F_a | M2 | 2025 | 553 | yes | 10 | +0.192169 | +0.162777 | 0/10 |
| F11_F_a → F12_F_a | M4 | 2024 | 626 | yes | 10 | +0.045133 | +0.082665 | 3/10 |
| F11_F_a → F12_F_a | M4 | 2025 | 553 | yes | 10 | +0.186558 | +0.193884 | 3/10 |

## Conclusion

The strongest supported lessons are that F09's measured gains are most consistent for M4, F06 compression benefits M2 more consistently than M4, and the completed F10 A/M2 combination improves on F09 A. F11 A/M2 and F12 A show real negative comparisons after controlling evaluation games. Aggregate medians can conceal configuration-level reversals, and MAE acceptance can coexist with worse upset recognition.

These conclusions are observations of the tested fitted pipelines and development data. Their mechanisms and prospective persistence remain untested. They supersede any expectation-based statement that richer generations necessarily improve models or that all cross-generation comparisons are impossible because total dataset sizes differ.

## Open follow-ups

The experiment goal remains paused. No follow-up experiment is authorized by this analysis. Still unresolved are causal separation of feature additions from training-coverage changes, prospective generalization, missing source families, incomplete frozen-cutoff architecture matrices and later reductions. A/B minimum representations remain unproven. No new acceptance receipt or final program recommendation was issued.

See also: [Frozen result inventory](Nextgen-Results-Snapshot-2026-09-28) · [Experiment](Next-Generation-Fingerprint-Experiment) · [F06](Fingerprint-F06) · [F09](Fingerprint-F09) · [F10](Fingerprint-F10) · [F11](Fingerprint-F11) · [F12](Fingerprint-F12).
