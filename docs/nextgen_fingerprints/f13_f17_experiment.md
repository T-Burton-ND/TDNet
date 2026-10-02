# F13–F17 exploratory training, 2026

Status: **completed exploratory A screen on 2026-10-02**. All 140 primary
M2/M4 runs, 100 one-family/market ablations and 60 market-residual variants
succeeded. These are candidate A representations. No F13–F17 lineage,
reduced fingerprint or prospective deployment has been accepted.

The saved F09–F12 artifacts and 356 successful earlier runs are preserved. A
separate F12 correction-only model replaces exactly the inherited F09 A values
after rebuilding the F09 sufficient statistics with the repaired play-type
classifier. Comparisons then add one family at a time:

| Stage | Added source information |
| --- | --- |
| F12 original | Saved F12 A input, on the new common cohort |
| F12 corrected | Repaired F09 A inputs only |
| F13 | Prior-season context-adjusted yards, success and tails |
| F14 | Adjacent observed-play success/recovery transitions |
| F15 | Verified situational rushing/reception actor concentration |
| F16 | Recent changes in F13 residuals and dispersion of F14 recovery |
| F17 market | F16 plus archived spread, total, opening line, movement, moneyline, implied probability, provider dispersion and quote count |

All seven stages use the same 7,358 F12 A games from 2013–2025. Fitting is on
the same 6,179 games through 2023; 2024 and 2025 are reported separately on
626 and 553 games. These later seasons are design-informed development
evidence. The 2026 season is quarantined. M2 and M4 use their ten previously
frozen setpoints and seed 1701. Probability calibration uses only rolling
origin residuals from training years. Every paired comparison checks game IDs,
actual outcomes and training-year counts.

F13 context bins are fixed play class, early/late down, short/medium/long
distance, three field zones and close versus wider score. A season's expected
yardage and success use only *earlier seasons*, with a 50-play shrinkage toward
the earlier play-class mean. F14 counts only adjacent events within drives;
ambiguous order and skipped administrative/no-play events cannot create a
transition. F15 uses documented `/plays/stats` Rush and Reception associations
matched to the canonical play/team. Conflicting associations are excluded,
and a game-role needs at least two observed events and 80% event coverage.
F16 compares the last three available games with the last twelve, requiring
minimum support. All dynamic states use completed regular-season games and a
48-hour reconstructed reporting lag before the target kickoff.

F17 is a **retrospective market-assisted comparator**. The historical line
archive does not include quote timestamps. A line's recorded value cannot be
certified as the value available at a live pregame forecast cutoff. Market
inputs do not enter F12–F16. Opening lines and moneylines begin in 2021 in
the archived coverage; missing earlier values remain missing. Historical
score fields in line records are never selected as inputs.

This first screen trains the A/full candidates. Pair-closed reduction,
architecture-normalized SHAP, alternate B/C representations and ancestry
acceptance require separate measured evidence. The original program's hard
ten-feature reduction floor is not silently applied to these small candidate
families; any reduction needs a documented floor disposition.

Implementation: `scripts/nextgen_rounds_prepare.py`,
`scripts/nextgen_rounds_train.py`, `scripts/nextgen_rounds_ablate.py`,
`scripts/nextgen_rounds_market_residual.py`,
`scripts/nextgen_rounds_report.py`, and
`src/gridiron_ml/experiments/nextgen_rounds_features.py`. Raw caches and the
original fingerprint artifacts are read-only. New private experiment outputs
are under the ignored `data/nextgen_rounds_2026/` directory.

## Measured results

The table reports the median MAE across ten frozen configurations within each
architecture. It compares models on the same training game IDs and the same
2024 and 2025 evaluation game IDs. The archived spread is an untrained
historical benchmark, with predicted home margin equal to negative home
spread. Lower MAE is better.

| Input or model | M2 2024 | M2 2025 | M4 2024 | M4 2025 |
| --- | ---: | ---: | ---: | ---: |
| Original F12 A | 13.824 | 12.729 | 13.187 | 12.393 |
| Corrected F12 A | 13.825 | 12.816 | 13.207 | 12.404 |
| F13 cumulative | 13.855 | 12.845 | 13.249 | 12.475 |
| F14 cumulative | 13.955 | 12.902 | 13.266 | 12.424 |
| F15 cumulative | 14.013 | 12.942 | 13.202 | 12.439 |
| F16 cumulative | 14.034 | 12.951 | 13.250 | 12.457 |
| F17 plus market | 13.339 | 12.669 | 12.439 | 11.788 |
| F17 market-anchored residual | 13.231 | 12.625 | 12.231 | 11.713 |
| Archived spread alone | 12.240 | 11.648 | 12.240 | 11.648 |

The correction-only comparison changed M2 median paired MAE by +0.003 in
2024 and +0.082 in 2025. M4 changes were -0.002 and -0.008. The corrected
F12 reference therefore separates the taxonomy repair from the new families;
it does not explain away F13's result.

F13 worsened median paired MAE in all four architecture/year cells, with only
two of ten configurations improving in each cell. F14 worsened M2 in both
years, while M4 showed small paired gains in 2024/2025 (-0.007/-0.036).
F15 worsened M2 and had mixed M4 results (-0.053 in 2024, +0.010 in 2025).
F16 worsened median paired MAE in all four cells. Individual-family ablations
against corrected F12 found no consistent gain for F14, F15 or F16. These
signals do not support promoting a market-free F13–F16 lineage from this A
screen.

Adding market features to F16 improved all ten configurations in both models
and both years. Median paired MAE changes were -0.671/-0.255 for M2 and
-0.812/-0.641 for M4 in 2024/2025. That measures the value of archived market
context relative to F16, not the value of F13–F16. The F17 direct M4 median
MAE (12.439/11.788) remained above the spread alone (12.240/11.648).
Anchoring predictions to the spread reduced the M4 median to 12.231/11.713:
it narrowly beat the spread in 2024 by 0.009, but missed it in 2025 by 0.065.
Seven of ten anchored full M4 configurations beat the spread in 2024 and none
did so in 2025. No market-assisted variant established a consistent two-year
improvement over the spread benchmark.

## Coverage and interpretation limits

The [all-generation scatter figure](figures/README.md) shows M2 and M4 MAE,
winner accuracy and upset recall from F0 through `F17_market`. Marker shapes
separate the historical rolling-fold study from the later 2025 screens. Its
late-stage band shows the market-free A plateau and the market fork's
MAE/upset-recall tradeoff; results across different evaluation settings are
descriptive only.

Preparation retained 14,716 target-team rows. The observed-player audit
counted 844,916 eligible role events in successful attribution-game scope,
802,539 matched role events, 23,603 conflicting or unverified actor rows,
3,762 game-role summaries excluded for low coverage, and 62,255 supported
game-role summaries. F14 excluded five drives whose sequence order was
ambiguous. These are measured source diagnostics; source completeness is not
proved by the presence of a cache or by a model score.

The report is `data/nextgen_rounds_2026/report.json`, with per-run metrics in
`run_results.csv`, paired comparisons in `paired_results.csv`, and companion
ablation/residual CSVs. The report verifies output hashes, actual outcomes,
training counts and common evaluation IDs before computing paired changes.
All 19 focused/relevant regression tests passed. The full archive, prepared
features, predictions, and model results remain local and ignored by Git.

These A candidates are exploratory. F13 did not add a past-only opponent
strength baseline or B/C representations, and this screen did not perform
pair-closed reduction or complete SHAP voting. Those untested designs may be
researched separately; the present scores do not authorize an accepted new
fingerprint. The 2024/2025 cohorts were design-informed, and no 2026 result
was used or reported.
