# F19 confidence and betting profit, 2026 Weeks 1–5

This extends the $10 flat-stake F19 betting replay. It describes how realized profit changes when a wager is placed only above each confidence cutoff. The models, consensus, odds, and game outcomes are unchanged.

## Meaning of a confidence percentage

- ATS: empirical chance that the chosen side covers, estimated from each strategy's 2,226 frozen 2022–2025 out-of-fold margin residuals. The residual CDF is evaluated at that game's predicted margin plus pregame spread. This is a historical error estimate, not a verified calibrated cover probability. The ledger also carries the fraction of six F19 models agreeing with the ATS side; that is agreement, not probability.
- Moneyline: the frozen F19 probability assigned to the picked winner. The ledger also shows the selected quote's break-even probability and model-minus-price gap. A high win probability does not alone imply favorable odds.
- Cutoffs run from 50% through 100% in 1-point steps. The 0% row is the original all-priced-bets baseline. Bets remain $10; skipped bets cost $0. ROI equals net profit divided by dollars actually staked.

## Price and research limits

ATS prices remain assumed -110. The broad moneyline scenario has 56 frozen pregame Week 4 prices and 194 retrospectively archived prices with unverified timing. A separate Week 4 moneyline scenario uses only its 56 frozen prices. F19 forecasts were produced after these games, and earlier project work had inspected 2026 results. Historical F19 market quote timing is also unverified. Cutoff rankings and correlations here reuse the same 2026 outcomes; they are not a tested forward betting rule. Many ATS probabilities cluster near 50%, so high cutoffs often leave very few games.

## Observed relationship

- Consensus moneyline confidence had Spearman correlation +0.357 with winning the game, but -0.365 with dollars won per bet on the 250 quoted-price games. The ≥90% cutoff kept 57 bets and netted $1.78; all 250 quoted bets netted $86.77.
- ATS M3 at an estimated ≥53% cover cutoff kept 138 bets and netted $91.82, versus $17.27 on all 263 ATS bets. That cutoff was found by examining these outcomes.
- In the temporal cutoff check, Weeks 1–3 select the net-maximizing cutoff with at least 30 training bets (including an all-bets option); Weeks 4–5 are then settled without changing the cutoff. This isolates threshold selection within the replay, but the underlying F19 forecasts and most odds remain retrospective.

## Weeks 1–3 selection, Weeks 4–5 check

| Market | Strategy | Selected cutoff | Check bets | Check net | All-bets check net |
|---|---|---:|---:|---:|---:|
| ATS | M3 | 53% | 47 | +$17.27 | +$7.27 |
| ATS | F19 equal consensus | all bets | 114 | -$69.09 | -$69.09 |
| moneyline | M10 | 57% | 95 | +$65.28 | +$139.54 |
| moneyline | F19 equal consensus | all bets | 112 | +$17.74 | +$17.74 |

## Selected examples

The following rows maximize observed net profit among cutoffs retaining at least 30 bets; the selection is entirely in-sample. Compare with the all-bets baseline rather than treating a selected cutoff as a recommendation.

| Market | Scenario | Strategy | Cutoff | Bets | Net | Change vs all bets |
|---|---|---|---:|---:|---:|---:|
| ATS | all_eligible | M3 | 53% | 138 | +$91.82 | +$74.55 |
| ATS | all_eligible | F19 equal consensus | 56% | 48 | -$21.82 | +$37.27 |
| moneyline | quoted_odds | M10 | 50% | 250 | +$185.36 | +$0.00 |
| moneyline | quoted_odds | F19 equal consensus | 59% | 211 | +$87.70 | +$0.93 |
| moneyline | frozen_pregame_subset | F19 equal consensus | 54% | 51 | +$72.77 | +$32.00 |

## Files

- `confidence_ledger.csv`: game-level confidence, model agreement, quoted break-even probability, and unchanged outcomes.
- `threshold_sweep.csv`: all strategies, scenarios, cutoffs, net, ROI, and bet counts.
- `confidence_bins.csv`: disjoint confidence bands; these do not double-count games.
- `correlations.csv`: descriptive Spearman rank correlations with win and net per bet.
- `week_split_check.csv`: thresholds selected on Weeks 1–3, then checked on Weeks 4–5.
- `best_cutoffs_in_sample.csv`: 30-bet-minimum observed net maxima, not recommended rules.
- `historical_f19_oof_residuals.csv`: audited source for ATS percentages.
- `confidence_thresholds.png`: profit, ROI, and surviving-bet views.
- `manifest.json`: immutable input and output hashes.
