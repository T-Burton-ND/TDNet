# 2026 scientific roster: quality gallery

All 18 figures are together in this directory. Figures 01–16 contain the same 120 F0–F19 scientific cells (M1–M5 and M10) and one Vegas point; Figures 17–18 show selected forecasters as curves. F17 is the `F17-market` research fingerprint in the source roster. F19/M4 is pink (`edgePink`); Vegas is gold (`brass`). The other model colors, typography, grid, and background follow [`gridiron.palette.json`](../../../../gridiron.palette.json) and [`gridiron_light.mplstyle`](../../../../docs/style/color_palettes/gridiron_light.mplstyle).

## Roster and distribution views

| Figure | Main question | Encoding |
| --- | --- | --- |
| [01 Error landscape](01_error_landscape.png) | Who approaches Vegas in both margin and probability error? | MAE × Brier; straight-up wins by point size; architecture by color and shape. |
| [02 Accuracy landscape](02_accuracy_landscape.png) | Does picking more winners track lower margin error? | MAE × straight-up accuracy; Brier by blue shade; architecture by shape. |
| [03 Balanced quality ladder](03_balanced_quality_ladder.png) | What is the complete order under one explicit summary index? | One common horizontal quality scale, with every point separately ranked. |
| [04 Roster matrix](04_roster_matrix.png) | Which fingerprint and architecture pairs account for the pattern? | Unique fingerprint × architecture position; quality by blue shade. |
| [05 Architecture swarm](05_architecture_swarm.png) | How broad is each architecture's quality distribution? | Quality on a common horizontal scale; deterministic vertical packing for nearby points. |
| [06 Fingerprint progression](06_fingerprint_progression.png) | How does performance vary across the fingerprint generations? | Fingerprint position × quality; architecture by color and shape. |
| [07 Architecture violin](07_architecture_violin.png) | What are each architecture's mean, median, and quartiles across fingerprints? | Smoothed density plus every point, mean circle, median tick, and interquartile bar. |
| [08 Fingerprint swarm](08_fingerprint_swarm.png) | How do the six architectures spread within each fingerprint? | A distinct point for every model in each fingerprint row. |
| [09 Fingerprint violin](09_fingerprint_violin.png) | Which fingerprints are robust to architecture choice? | Six exact points, mean, median, and quartiles per fingerprint with illustrative smoothing. |

The [paired score data](paired_quality.csv), [architecture summary](architecture_distribution.csv), and [fingerprint summary](fingerprint_distribution.csv) contain the exact values. Rebuild these nine figures with [`plot_2026_scientific_total_quality.py`](../../../../scripts/publication/plot_2026_scientific_total_quality.py). The [manifest](manifest.json) records the input and output hashes.

## Game-level diagnostic views

These analyze distinct aspects of quality, not another mix of MAE, Brier, and straight-up accuracy.

| Figure | Question | Measure |
| --- | --- | --- |
| [10 Discrimination and penalty](10_probability_discrimination.png) | Do probabilities rank winners and avoid confident mistakes? | ROC AUC × log loss. |
| [11 Calibration and sharpness](11_calibration_sharpness.png) | Are forecasts both trustworthy and decisive? | Ten-bin calibration error × mean distance from 50%. |
| [12 Selective prediction](12_selective_prediction.png) | Does each model's confidence locate safer picks? | Area under risk–coverage curve × accuracy on the 93 most-confident games. |
| [13 Margin tail risk](13_margin_tail_risk.png) | Does a good typical error hide costly misses? | Median absolute error × mean of the worst 10% of absolute errors. |
| [14 Gamewise Vegas duel](14_gamewise_market_duel.png) | Is a model closer than Vegas on most games? | Paired margin-error win share × mean margin-error advantage. |
| [15 Market regimes](15_market_regimes.png) | Does performance change with market confidence? | Brier advantage on near-tossups × strong-favorite games. |
| [16 Market blend probe](16_market_blend_probe.png) | Does a fixed blend of model and Vegas improve Vegas? | Model log-loss gap × 50/50 blended Brier gain. |
| [17 Reliability profiles](17_probability_reliability.png) | Where do probability forecasts deviate from observed frequency? | Ten equal-count bins for Vegas, F19/M4, F19/M1, and the 120-model consensus. |
| [18 Confidence–coverage curves](18_confidence_risk_coverage.png) | What accuracy remains as lower-confidence games are added? | Same four selected forecasters, ranked by their own probability confidence. |

SVG versions have the same base names. The [diagnostic score data](diagnostic_quality.csv) contains all 121 forecasters, and the [model-game loss ledger](diagnostic_model_game_loss.parquet) preserves every paired probability-game loss. [Reliability-bin values](probability_reliability_bins.csv), [coverage checkpoints](confidence_coverage_checkpoints.csv), and [diagnostic hashes](diagnostics_manifest.json) support reproduction. Rebuild with [`analyze_2026_scientific_quality_diagnostics.py`](../../../../scripts/publication/analyze_2026_scientific_quality_diagnostics.py).

## Metric and comparison method

All margin MAE and straight-up results use the same **379 spread-covered games**. All Brier results use the same **372 moneyline-covered games**. The figure script checks that every model has every paired game ID before scoring. The Vegas point has 11.3892-point MAE, 294/379 straight-up wins (77.57%), and 0.157786 Brier. F19/M4 has 11.3882-point MAE, 296/379 wins (78.10%), and 0.159724 Brier. Its MAE edge is about 0.0010 point; its Brier is worse than Vegas.

The descriptive **balanced relative skill** index is the arithmetic mean of three percentage changes versus Vegas:

```text
MAE skill   = 100 × (Vegas MAE − model MAE) / Vegas MAE
Brier skill = 100 × (Vegas Brier − model Brier) / Vegas Brier
SU skill    = 100 × (model SU accuracy − Vegas SU accuracy) / Vegas SU accuracy
Balanced relative skill = (MAE skill + Brier skill + SU skill) / 3
```

Zero is Vegas. Higher is better. This index is a visualization convenience, not a significance test or a betting utility. Equal relative weighting is a design choice; Brier and straight-up accuracy both reflect winner prediction, so the index should not replace the two raw-metric landscapes. Neither the small F19/M4 MAE edge nor its two extra correct winners establishes superiority over Vegas.

The architecture distributions support a more precise observation than a simple-versus-complex rule. M1 linear has the strongest **median** balanced skill across the 20 fingerprints (−8.35% versus Vegas), but a wide **9.59-percentage-point IQR**. M3 tree has the narrowest IQR (4.32 points); M2 spline and M4 boosted are close (4.86 and 5.01). M4 has the best single point, F19/M4 at −0.18%. M5 neural has the weakest family median (−23.19%). Fingerprint violins each contain only six architecture scores, so their smooth shape should not be read as a measured population density; use the points and quartile marks.

## What the additional diagnostics find

All comparisons below are descriptive through Week 7. On the **372 shared moneyline games**, Vegas has **0.47756 log loss** and **0.83851 ROC AUC**; no individual scientific cell beats either value. No cell has a lower area under the probability-confidence risk–coverage curve than Vegas. On the **379 shared spread games**, F19/M4's margin forecast is closer to the final margin than Vegas on **47.49%** of games (ties count as half), even though its mean absolute error is **0.00103 point lower**. Its median absolute margin error is lower than Vegas (9.32 versus 9.75), but its own worst-decile mean error is higher (31.10 versus 30.77). Different error summaries therefore tell different parts of the story.

On the **80 market-probability near-tossups** (home-win probability 40–60%), four of 120 models have lower Brier than Vegas. On the **183 strong-favorite games** (market probability ≤25% or ≥75%), none does. A fixed, untuned 50/50 blend of model and Vegas probabilities improves Vegas Brier for only two cells: F19/M1 by **0.000396** and F19/M2 by **0.000023** on the 372 shared games. These small observed differences across 120 inspected cells do not establish incremental information or prospective profit. The one model with higher top-quartile confidence accuracy than Vegas selects a *different* set of 93 games; that difference is not a paired head-to-head win.

Calibration error in Figure 11 is an **equal-count ten-bin estimate**, with 37–38 games in each bin. Ten cells have a lower numerical binned error than Vegas, but that statistic is sensitive to bin choices and sampling noise; read it with the reliability curves, log loss, and sharpness. Probability log loss clips only exact endpoints to [0.000001, 0.999999] to keep the score finite. Selective prediction orders games by `max(p, 1−p)` and breaks equal-confidence ties by game ID. The Vegas point in Figure 14 is the neutral 50% gamewise comparison of Vegas with itself; other points use exact paired game errors.

## Design research

[Cleveland and McGill's graphical-perception experiments](https://www.tandfonline.com/doi/abs/10.1080/01621459.1984.10478080) motivate using positions on common axes for the main comparisons. Their [scatterplot methods paper](https://www.tandfonline.com/doi/abs/10.1080/01621459.1984.10477098) discusses adding category codes and point sizes to a scatterplot, motivating the two metric landscapes. [Matplotlib's scatter example](https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_demo2.html) documents the color and size encodings used here. The matrix preserves one unambiguous position per model when scatter points overlap; the ladder and swarm preserve a common quality axis for comparing all 121 points.

The diagnostic suite follows [Gneiting and Raftery's proper-scoring framework](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) for log loss and the distinction between calibration and sharpness. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) show why accuracy alone does not establish calibrated probabilities; [Arrieta-Ibarra et al.](https://www.jmlr.org/papers/v23/22-0658.html) explain the binning problem in reliability metrics. [Selective-prediction research](https://proceedings.mlr.press/v267/zhou25y.html) motivates risk–coverage analysis. A [sports-betting model-selection study](https://arxiv.org/abs/2303.06021) motivates examining calibration alongside accuracy, while the present retrospective and timing-limited data do not support a betting-profit claim. We did not compute continuous ranked probability score or interval coverage because the saved margin forecasts are point predictions, not full predictive distributions.

## Provenance limits

These are through-Week-7 **retrospective** comparisons. Weeks 6–7 scientific backfill was run after games; Week 7 market quotes were retrieved after games, and provider quote times are unverified. F17-market and F19 use market inputs. The figures are descriptive performance summaries, and searching 120 related cells creates a selection problem. They cannot be used as leakage-free prospective evidence or as proof that one model beats Vegas. See the Week 7 source files and their provenance columns for the underlying game records.
