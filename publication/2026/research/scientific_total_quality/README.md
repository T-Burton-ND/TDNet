# 2026 scientific roster: quality gallery

All 18 core figures and two publication violin variants are together in this directory. Figures 01–16 show the same 120 F0–F19 scientific cells (M1–M5 and M10), with Vegas as a point on the two-metric scatterplots and as the zero-skill threshold on the roster summaries. Figures 17–18 show selected forecasters as curves. F17 is the `F17-market` research fingerprint in the source roster. Every scientific cell uses its architecture's standard marker and color, including F19/M4 as the M4 teal diamond; Vegas is gold (`brass`). The other model colors, typography, grid, and background follow [`gridiron.palette.json`](../../../../gridiron.palette.json) and [`gridiron_light.mplstyle`](../../../../docs/style/color_palettes/gridiron_light.mplstyle).

## Rebuild the saved Week 7 gallery

Run from the repository root with Python and the `numpy`, `pandas`, `pyarrow`, and `matplotlib` packages. The repository's [`pyproject.toml`](../../../../pyproject.toml) declares the project dependencies. The saved images were produced with Python 3.9.25, NumPy 1.23.5, pandas 2.3.3, PyArrow 21.0.0, and Matplotlib 3.9.4; those versions and the installed DejaVu Sans font matter if byte-for-byte image reproduction is required. The source hashes and generated-file hashes are in [manifest.json](manifest.json) and [diagnostics_manifest.json](diagnostics_manifest.json).

```bash
MPLCONFIGDIR=/tmp/tdnet-mpl python3 scripts/publication/plot_2026_scientific_total_quality.py
MPLCONFIGDIR=/tmp/tdnet-mpl python3 scripts/publication/analyze_2026_scientific_quality_diagnostics.py
```

The default input is `publication/2026/week_07/post_game/retrospective_replay/cumulative/scientific_f0_f19/`; the default output is this directory. Run the roster script first and the diagnostic script second. The first writes Figures 01–09, two publication violin variants, and the paired/distribution tables; the second writes Figures 10–18 and the diagnostic tables. Core figures are saved as 200-dpi PNGs and SVGs; the publication variants have 300-dpi PNGs and SVGs. SVG metadata has the fixed `2026-10-21` render date, and both scripts fix the Matplotlib SVG hash salt so a rerun in the same environment is deterministic. The two manifests hash their own source files and generated outputs; they do not hash the README or each other.

## Rebuild after the 2026 season

First freeze a **new, fully scored common-cohort export** in its own directory, with the three filenames and columns below. Include only completed games with final margins; preserve the archived forecast and market values for each game. Use the same game IDs and home-team orientation in all three files. This gallery does not fetch odds, generate forecasts, or build the source cohort itself. The Week 7 export was assembled by [`publish_2026_week06_07_f0_f19_backfill.py`](../../../../scripts/publication/publish_2026_week06_07_f0_f19_backfill.py), which is specific to that catch-up window; it is a reference for the source schema, not a final-season export command.

| Input file | Required content |
| --- | --- |
| `common_120_model_game_results.parquet` | One row per `game_id` × `fingerprint` × `model_id` for **every** one of the 120 cells on **every** common-cohort game. Required columns: `game_id`, `week`, `fingerprint`, `model_id`, `pred_home_margin`, `pred_home_win_probability`, `actual_home_margin`, `margin_error`, `winner_correct`, `brier_error`. The fingerprint keys are F0–F16, `F17-market`, F18, F19; model keys are M1–M5 and M10. |
| `common_cohort_vegas_games.csv` | One row per common-cohort game. Required columns: `game_id`, `week`, `actual_home_margin`, `market_home_spread`, `market_home_probability`, `margin_error`, `winner_correct`, `brier_error`, `market_provenance`. A missing spread excludes that game from margin and winner comparisons; a missing probability excludes it from probability comparisons. A home spread is negative when the home team is favored, so Vegas's predicted home margin is `−market_home_spread`. |
| `common_120_consensus_game_results.csv` | One row per common-cohort game with `game_id`, `week`, `actual_home_margin`, and `pred_home_win_probability`. The consensus is used only for Figures 17–18; the other figures use individual models and Vegas. |

`margin_error`, `winner_correct`, and `brier_error` are the frozen grading fields in the source. The roster script uses the model and Vegas `margin_error`/`winner_correct`/`brier_error` columns; the diagnostic script recomputes probability and tail measures from the raw predictions and final margin, and checks the archived Vegas margin error and Brier against those values. Before exporting, verify that `pred_home_win_probability` and `market_home_probability` are home-win probabilities on `[0,1]`, and that the market values reflect the intended archived quote cutoff. The scripts reject missing model cells, duplicate IDs, mismatched final margins, and incomplete paired cohorts. A final-season export must retain the same **120-cell roster**; a roster change needs a separate analysis definition.

Choose a new output directory to keep the Week 7 snapshot intact. Replace the example paths, date, and market note with the actual frozen snapshot details; pass the **same four arguments to both scripts**:

```bash
MPLCONFIGDIR=/tmp/tdnet-mpl python3 scripts/publication/plot_2026_scientific_total_quality.py \
  --source-dir publication/2026/week_FINAL/post_game/cumulative/scientific_f0_f19 \
  --output-dir publication/2026/research/scientific_total_quality_final \
  --render-date 2027-01-15 \
  --market-note 'Market source and quote timing: describe the frozen final-season snapshot here.'

MPLCONFIGDIR=/tmp/tdnet-mpl python3 scripts/publication/analyze_2026_scientific_quality_diagnostics.py \
  --source-dir publication/2026/week_FINAL/post_game/cumulative/scientific_f0_f19 \
  --output-dir publication/2026/research/scientific_total_quality_final \
  --render-date 2027-01-15 \
  --market-note 'Market source and quote timing: describe the frozen final-season snapshot here.'
```

`week_FINAL` and the date are placeholders, not existing paths or a scheduled run. The scripts read the largest `week` in the new market file for the figure footers, count the spread and moneyline cohorts separately, and compute the confidence and regime counts from those games. Inspect the new `manifest.json` and `diagnostics_manifest.json` for the source hashes, cohort counts, and output hashes. Confirm that there are 20 PNGs and 20 SVGs (18 core figures plus two publication variants), 121 rows in each model/market summary, 20 architecture observations per model family, and six architecture observations per fingerprint. Keep the original Week 7 manifests for historical comparison; source changes should produce new hashes.

From the repository root, this check verifies every recorded source and output hash and the expected figure/summary counts; replace the argument with the chosen output directory:

```bash
python3 - publication/2026/research/scientific_total_quality_final <<'PY'
import hashlib, json, sys
from pathlib import Path
import pandas as pd

out = Path(sys.argv[1])
for name in ('manifest.json', 'diagnostics_manifest.json'):
    record = json.loads((out / name).read_text())
    for filename, digest in record['source'].items():
        source = Path(record['source_directory']) / filename
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, source
    for filename, digest in record['outputs'].items():
        artifact = out / filename
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest, artifact
assert len(list(out.glob('[0-9][0-9]_*.png'))) == 20
assert len(list(out.glob('[0-9][0-9]_*.svg'))) == 20
assert len(pd.read_csv(out / 'paired_quality.csv')) == 121
assert len(pd.read_csv(out / 'diagnostic_quality.csv')) == 121
print('Gallery source files, 18 core figure pairs, two publication pairs, score tables, and hashes verified')
PY
```

For a poster or paper, audit the new `market_provenance` and archived quote timestamps before interpreting a Vegas comparison as prospective. The final-season plots remain descriptive if forecasts or market lines were filled after the games. Do not combine the original Week 7 retrospective figures with a later prospective cohort without labeling the timing difference.

## Roster and distribution views

| Figure | Main question | Encoding |
| --- | --- | --- |
| [01 Error landscape](01_error_landscape.png) | Who approaches Vegas in both margin and probability error? | MAE × Brier; straight-up wins by point size; architecture by color and shape. |
| [02 Accuracy landscape](02_accuracy_landscape.png) | Does picking more winners track lower margin error? | MAE × straight-up accuracy; Brier by blue shade; architecture by shape. |
| [03 Balanced quality ladder](03_balanced_quality_ladder.png) | What is the complete model order under one explicit summary index? | Rank all 120 cells on one horizontal scale; gold line is Vegas at zero. |
| [04 Roster matrix](04_roster_matrix.png) | Which fingerprint and architecture pairs account for the pattern? | Unique fingerprint × architecture position; quality by blue shade; gold line at zero on the colorbar. |
| [05 Architecture swarm](05_architecture_swarm.png) | How broad is each architecture's quality distribution? | Quality on a common horizontal scale; deterministic vertical packing; gold Vegas line. |
| [06 Fingerprint progression](06_fingerprint_progression.png) | How does performance vary across the fingerprint generations? | Fingerprint position × quality; architecture by color and shape; horizontal Vegas line. |
| [07 Architecture violin](07_architecture_violin.png) | What are each architecture's mean, median, and quartiles across fingerprints? | Smoothed density plus every point, mean circle, median tick, interquartile bar, and gold Vegas line. |
| [08 Fingerprint swarm](08_fingerprint_swarm.png) | How do the six architectures spread within each fingerprint? | A distinct point for every model in each fingerprint row; gold Vegas line. |
| [09 Fingerprint violin](09_fingerprint_violin.png) | Which fingerprints are robust to architecture choice? | Six exact points and quartiles per fingerprint with illustrative smoothing and a gold Vegas line. |

The [paired score data](paired_quality.csv), [architecture summary](architecture_distribution.csv), and [fingerprint summary](fingerprint_distribution.csv) contain the exact values. Rebuild these nine figures with [`plot_2026_scientific_total_quality.py`](../../../../scripts/publication/plot_2026_scientific_total_quality.py). The [manifest](manifest.json) records the input and output hashes.

### Publication violin exports

| Export | Print layout |
| --- | --- |
| [Architecture violin PNG](07_architecture_violin_publication.png) · [SVG](07_architecture_violin_publication.svg) | 15 × 8.5 inches, 300-dpi PNG; 20 model points per architecture. |
| [Fingerprint violin PNG](09_fingerprint_violin_publication.png) · [SVG](09_fingerprint_violin_publication.svg) | 15 × 15 inches, 300-dpi PNG; six model points per fingerprint. |

These use the same paired scores, model colors, point shapes, violin smoothing, quartiles, means, and gold Vegas threshold as Figures 07 and 09. They omit the internal title, subtext, and footer; enlarge the axis labels, tick text, and legend; and add legend keys for the interquartile bar, median tick, and mean circle. The script generates them with the normal gallery command and records their hashes in `manifest.json`. Give a paper or poster caption the cohort and market-timing limitation stated in **Provenance limits** below, since the clean image omits that footer.

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

All margin MAE and straight-up results use the same **379 spread-covered games**. All Brier results use the same **372 moneyline-covered games**. The figure script checks that every model has every paired game ID before scoring. The Vegas baseline has 11.3892-point MAE, 294/379 straight-up wins (77.57%), and 0.157786 Brier. F19/M4 has 11.3882-point MAE, 296/379 wins (78.10%), and 0.159724 Brier. Its MAE edge is about 0.0010 point; its Brier is worse than Vegas.

The descriptive **balanced relative skill** index is the arithmetic mean of three percentage changes versus Vegas:

```text
MAE skill   = 100 × (Vegas MAE − model MAE) / Vegas MAE
Brier skill = 100 × (Vegas Brier − model Brier) / Vegas Brier
SU skill    = 100 × (model SU accuracy − Vegas SU accuracy) / Vegas SU accuracy
Balanced relative skill = (MAE skill + Brier skill + SU skill) / 3
```

Zero is Vegas. Higher is better. This index is a visualization convenience, not a significance test or a betting utility. Equal relative weighting is a design choice; Brier and straight-up accuracy both reflect winner prediction, so the index should not replace the two raw-metric landscapes. Neither the small F19/M4 MAE edge nor its two extra correct winners establishes superiority over Vegas.

The roster script reads the frozen `margin_error`, `winner_correct`, and `brier_error` grading columns on the shared market-covered games. For each forecast, MAE is the mean absolute home-margin error, straight-up accuracy is the fraction of `winner_correct` rows, and Brier is the mean squared home-win probability error. Fingerprint and architecture groups are **descriptive collections of different model cells**: each architecture has 20 fingerprint scores, and each fingerprint has six architecture scores. The violin bar spans the 25th to 75th percentiles (pandas linear interpolation), its white tick is the median, its open circle is the arithmetic mean, and the translucent shape is Matplotlib's default kernel-density smoothing. The swarm retains the exact horizontal skill value for every model and applies deterministic vertical packing only when nearby points would overlap. Vegas is the zero-skill reference: a brighter gold vertical line in Figures 03, 05, 07–09, a gold horizontal line in Figure 06, and a gold zero line on Figure 04's colorbar. F19/M4 uses the same teal diamond, size, and outline as the other M4 cells; the matrix uses the same score-color encoding as every other cell. The ladder ranks the 120 model cells; the two metric landscapes retain Vegas as a gold star at its two measured scores. For later cohorts, those metric landscapes use automatic axis limits so improved or worsened scores remain visible.

The architecture distributions support a more precise observation than a simple-versus-complex rule. M1 linear has the strongest **median** balanced skill across the 20 fingerprints (−8.35% versus Vegas), but a wide **9.59-percentage-point IQR**. M3 tree has the narrowest IQR (4.32 points); M2 spline and M4 boosted are close (4.86 and 5.01). M4 has the best single point, F19/M4 at −0.18%. M5 neural has the weakest family median (−23.19%). Fingerprint violins each contain only six architecture scores, so their smooth shape should not be read as a measured population density; use the points and quartile marks.

## What the additional diagnostics find

All comparisons below are descriptive through Week 7. On the **372 shared moneyline games**, Vegas has **0.47756 log loss** and **0.83851 ROC AUC**; no individual scientific cell beats either value. No cell has a lower area under the probability-confidence risk–coverage curve than Vegas. On the **379 shared spread games**, F19/M4's margin forecast is closer to the final margin than Vegas on **47.49%** of games (ties count as half), even though its mean absolute error is **0.00103 point lower**. Its median absolute margin error is lower than Vegas (9.32 versus 9.75), but its own worst-decile mean error is higher (31.10 versus 30.77). Different error summaries therefore tell different parts of the story.

On the **80 market-probability near-tossups** (home-win probability 40–60%), four of 120 models have lower Brier than Vegas. On the **183 strong-favorite games** (market probability ≤25% or ≥75%), none does. A fixed, untuned 50/50 blend of model and Vegas probabilities improves Vegas Brier for only two cells: F19/M1 by **0.000396** and F19/M2 by **0.000023** on the 372 shared games. These small observed differences across 120 inspected cells do not establish incremental information or prospective profit. The one model with higher top-quartile confidence accuracy than Vegas selects a *different* set of 93 games; that difference is not a paired head-to-head win.

Calibration error in Figure 11 is an **equal-count ten-bin estimate**, with 37–38 games in each bin. Ten cells have a lower numerical binned error than Vegas, but that statistic is sensitive to bin choices and sampling noise; read it with the reliability curves, log loss, and sharpness. Probability log loss clips only exact endpoints to [0.000001, 0.999999] to keep the score finite. Selective prediction orders games by `max(p, 1−p)` and breaks equal-confidence ties by game ID. The Vegas point in Figure 14 is the neutral 50% gamewise comparison of Vegas with itself; other points use exact paired game errors.

### Diagnostic definitions used by Figures 10–18

Let `y = 1` if the home team won and `0` otherwise; `p` is its forecast home-win probability. Every model is paired with Vegas on identical game IDs for each market-covered measure.

| Measure | Exact construction |
| --- | --- |
| Log loss and ROC AUC (10) | Mean `−y·log(p) − (1−y)·log(1−p)`, clipping `p` to `[10⁻⁶, 1−10⁻⁶]`; ROC AUC is the rank statistic across winning versus losing home teams, with tied probability ranks averaged. |
| Calibration and decisiveness (11, 17) | Sort by `p`, then `game_id` for ties; split into ten nearly equal-count contiguous bins. Calibration error is `Σ (bin size/N) × |mean(p in bin) − mean(y in bin)|`. The horizontal Figure 11 measure is that estimate; the vertical measure is `mean(|p−0.5|)`. Figure 17 plots each bin's `mean(p)` against `mean(y)`. |
| Selective prediction (12, 18) | Rank games by `max(p,1−p)`, breaking ties by `game_id`; call home when `p ≥ 0.5`. At each retained count `k`, risk is the fraction of mistakes in the first `k` games. Area under the risk–coverage curve (AURC) is the arithmetic mean of these risks over `k = 1…N`; lower is better. Figure 12 also shows accuracy among the first `ceil(0.25N)` games. Figure 18 plots `1−risk` for coverage from 5% to 100%, with table checkpoints at 25%, 50%, and 100%. |
| Margin tails (13) | Compute `|predicted home margin − actual home margin|`; plot the median against the mean of the largest `ceil(0.10N)` errors. The summary CSV also records the 90th percentile and mean signed home-margin error. |
| Paired gamewise comparison (14) | For each spread-covered game, score a model 1 if its absolute margin error is smaller than Vegas's, 0.5 for a tie within `10⁻¹⁰`, and 0 otherwise; average these scores. Vertical position is `mean(Vegas absolute error − model absolute error)` in points. Vegas's self-comparison is placed at (0.5, 0). |
| Market-confidence regimes (15) | Using **Vegas** home-win probability, define near-tossups as `0.40 ≤ p ≤ 0.60` and strong favorites as `p ≤ 0.25` or `p ≥ 0.75`. For each regime, plot `mean(Vegas Brier loss − model Brier loss)`; positive favors the model. |
| Untuned blend probe (16) | For each game set blend probability to `(model p + Vegas p)/2`. Plot `mean(Vegas Brier loss − blend Brier loss)` versus `mean(model log loss − Vegas log loss)`. The weight is fixed at 50/50 before evaluation. |

Figure 17 and Figure 18 draw curves only for Vegas, F19/M4, F19/M1, and the saved 120-model consensus. Figures 10–16 contain **all 120 individual cells plus Vegas**. `diagnostic_quality.csv` has one row per individual cell plus Vegas; `diagnostic_model_game_loss.parquet` has one row per model per shared moneyline game and includes the market, model, and blend losses and the regime label. Figure 11's binned calibration error and Figure 12's top-confidence accuracy compare different model-selected game sets, so read them with the reliability and risk–coverage curves rather than as paired win counts.

## Design research

[Cleveland and McGill's graphical-perception experiments](https://www.tandfonline.com/doi/abs/10.1080/01621459.1984.10478080) motivate using positions on common axes for the main comparisons. Their [scatterplot methods paper](https://www.tandfonline.com/doi/abs/10.1080/01621459.1984.10477098) discusses adding category codes and point sizes to a scatterplot, motivating the two metric landscapes. [Matplotlib's scatter example](https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_demo2.html) documents the color and size encodings used here. The matrix preserves one unambiguous position per model when scatter points overlap; the ladder and swarm preserve a common quality axis for comparing 120 model points with a zero-skill Vegas line.

The diagnostic suite follows [Gneiting and Raftery's proper-scoring framework](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) for log loss and the distinction between calibration and sharpness. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) show why accuracy alone does not establish calibrated probabilities; [Arrieta-Ibarra et al.](https://www.jmlr.org/papers/v23/22-0658.html) explain the binning problem in reliability metrics. [Selective-prediction research](https://proceedings.mlr.press/v267/zhou25y.html) motivates risk–coverage analysis. A [sports-betting model-selection study](https://arxiv.org/abs/2303.06021) motivates examining calibration alongside accuracy, while the present retrospective and timing-limited data do not support a betting-profit claim. We did not compute continuous ranked probability score or interval coverage because the saved margin forecasts are point predictions, not full predictive distributions.

## Provenance limits

These are through-Week-7 **retrospective** comparisons. Weeks 6–7 scientific backfill was run after games; Week 7 market quotes were retrieved after games, and provider quote times are unverified. F17-market and F19 use market inputs. The figures are descriptive performance summaries, and searching 120 related cells creates a selection problem. They cannot be used as leakage-free prospective evidence or as proof that one model beats Vegas. See the Week 7 source files and their provenance columns for the underlying game records.
