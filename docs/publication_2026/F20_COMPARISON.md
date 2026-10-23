# F20 A/B/C matched developmental comparison

## Selection view (2026-10-23)

Lower Brier score is better. Every row in a cohort uses the same game IDs for
market-only, F20-A, F20-B, and F20-C.

| Arm | Inputs | 2021–2024 rolling-origin Brier, 2,216 games | 2025 exploratory Brier, 534 games |
| --- | --- | ---: | ---: |
| Market-only | Tracker market margin and missingness | **0.176184** | 0.180584 |
| F20-A | Market plus five tracker-reported expert margins | 0.176874 | **0.180073** |
| F20-B | A plus 20 TDNet features selected within each training fold | 0.178208 | 0.180318 |
| F20-C | Full 472-column historical F19 matrix plus expert block | 0.181223 | 0.182342 |

On the 2021–2024 development games, B−A is +0.001334 Brier points (95% paired
game-bootstrap interval −0.000008 to +0.002670) and C−A is +0.004349
(+0.001061 to +0.007700). A−market is +0.000690 (−0.000685 to +0.002047).
On the already-consumed 2025 games, B−A is +0.000244 (−0.002040 to
+0.002513), C−A is +0.002269 (−0.003768 to +0.008635), and A−market is
−0.000511 (−0.002366 to +0.001356). These intervals resample individual
games; games within a week may be correlated.

**Interpretation:** The measured comparison does not show a B or C advantage
over A. The market-only control is slightly better than A in the 2021–2024
development folds; A is slightly better in the consumed 2025 check. The
expert increment itself is therefore unresolved. If an expert-bearing F20
must be chosen from these arms now, A has the strongest developmental evidence
and the only working prospective feature path. No arm is selected canonical by
this report; the owner will choose after reviewing the table.

## Reproduction and limits

The comparison intersects tracker historical games with F19 historical games
by CFBD game ID and confirms matching season and outcome. The complete overlap
is 4,887 games from 2017–2025; 2,216 appear in rolling-origin test seasons
2021–2024 and 534 in the 2025 check. Each arm uses a logistic probability
model with training-only median imputation and standardization. F20-B selects
its 20 eligible, non-market TDNet fields on each fold's training seasons only.
The same logistic-C grid was offered to every arm: 0.0001, 0.0003, 0.001,
0.003, 0.01, 0.1, 1, 10, and 100. Development Brier selected C=0.1 for the
market control, 100 for A, 0.01 for B, and 0.001 for C. The reported
2021–2024 Brier also selected C, so it is mildly optimistic. The 2025 season
had already been consumed by earlier TDNet work and is not untouched
confirmation. Preliminary fixed-C and narrower-grid iterations are preserved
privately; grid expansion was prompted by boundary optima in development.
At fixed C=1, F20-C scored 0.209137 on the same 2,216 development games;
stronger regularization reduced it to 0.181223. F20-A's selected C=100 is at
the upper grid edge, but C=10 scored 0.176883 versus 0.176874 at C=100 on
those same development games, a small sensitivity.

Historical tracker publication times and F19 market quote times are
unverified. These are developmental comparisons, not point-in-time validated
prospective scores. F20-C's large input set and its historical market fields
have different provenance from tracker market fields in A/B. The trained B/C
candidate models are archived for inspection but cannot be used prospectively
until their weekly TDNet/F19 feature path is audited. No 2026 outcomes entered
training or selection.

Machine-readable metrics, per-season Brier, paired intervals, input hashes,
selected features, and model hashes are in `f20_abc_comparison_v4.json`.
Game-level predictions, the full C-grid predictions, and candidate models are
private at `/groups/bsavoie2/tburton2/TDNet/f20_2026/analysis/abc_comparison_v4/`.
Regenerate with `scripts/publication/compare_f20_abc.py --c-grid
0.0001,0.0003,0.001,0.003,0.01,0.1,1,10,100 --output <new-private-dir>
--summary <new-summary-file>`; the script refuses to overwrite artifacts.
