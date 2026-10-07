# F19 $10 flat-stake betting replay, 2026 Weeks 1–5

This is a retrospective, counterfactual settlement of the six F19 scientific models and the frozen equal F19 consensus. The F19 forecasts were produced after the games and the broader project had previously inspected 2026 outcomes. These totals are not a live betting track record or a forward profitability estimate.

## Rules and coverage

- Stake $10 on every eligible game, independently for each strategy and market. Each strategy starts at $0 net profit; a hypothetical $1,000 bankroll is also shown.
- ATS: take the home side when predicted home margin plus the archived pregame home spread is positive; otherwise take the away side. A zero edge means no bet. A graded tie is a push. All ATS bets assume American odds of -110 because the archive lacks spread-side prices.
- Moneyline: take home when its frozen forecast win probability exceeds 50%, otherwise away. An exact 50% probability means no bet. The selected side needs an actual quoted American moneyline; no quoted price means no bet. For negative odds, winning net profit is $10 × 100 / |odds|; for positive odds it is $10 × odds / 100. Losses cost $10.
- Quoted-odds scenario: use the frozen Week 4 pregame moneyline snapshot; use the best available provider price in the raw CFBD archive for Weeks 1–3 and 5. Those 194 archive prices have unverified timing and may not have been available when a bet could have been placed. Prices from different providers are selected independently for each side; availability, limits, fees, and movement are not modeled.
- Frozen-pregame subset: settle only the 56 Week 4 games with archived pregame moneyline prices. The prices are frozen pregame; the F19 predictions still are not.
- There are 271 completed target games, 263 with F19 forecasts and pregame spreads, and eight unavailable F19 games. The quoted-odds scenario has 250 priced games (56 frozen Week 4, 194 with retrospective prices); 13 available-forecast games lack a selected-side moneyline.

## Net results

| Strategy | ATS bets | ATS net | ATS ROI | ML bets | ML net | ML ROI | Week 4 ML net |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 263 | -$135.45 | -5.15% | 250 | +$89.78 | +3.59% | +$40.77 |
| M2 | 263 | -$154.55 | -5.88% | 250 | +$65.32 | +2.61% | +$40.77 |
| M3 | 263 | +$17.27 | +0.66% | 250 | +$71.28 | +2.85% | +$40.77 |
| M4 | 263 | -$59.09 | -2.25% | 250 | +$151.27 | +6.05% | +$40.77 |
| M5 | 263 | -$307.27 | -11.68% | 250 | +$5.73 | +0.23% | -$8.82 |
| M10 | 263 | -$230.91 | -8.78% | 250 | +$185.36 | +7.41% | +$119.31 |
| F19 equal consensus | 263 | -$59.09 | -2.25% | 250 | +$86.77 | +3.47% | +$40.77 |

The Week 4 subset contains 56 bets per strategy. Cumulative curves show net profit above or below the starting bankroll, not total account balance.

## Files

- `bet_ledger.csv`: every target game × strategy × scenario, including skipped bets and their reasons, price source, side, outcome, stake, and cumulative results.
- `summary.csv`: strategy totals with win/loss/push and ROI.
- `weekly.csv`: weekly totals and cumulative bankroll from an initial $1,000.
- `cumulative_net_profit.png`: ATS and broad quoted-moneyline profit paths (left axes) plus the grey dotted cumulative amount wagered per strategy (right axes). All seven strategies stake the same total within each panel.
- `manifest.json`: source and output SHA-256 hashes.
