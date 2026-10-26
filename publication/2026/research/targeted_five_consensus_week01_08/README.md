# Retrospective targeted five-cell 2026 consensus

Equal weights on F20/M1, F7/M1, F19/M1, F19/M2, and F19/M10. For each game, the target's home-win probability is the arithmetic mean of the five saved probabilities and its predicted home margin is the arithmetic mean of the five saved margins. Straight-up picks use probability ≥ 0.5; ATS uses the mean margin against the saved Vegas spread. No models were refitted or reweighted.

All five forecasts exist on 420 completed CFBD Weeks 1–8 games. The primary four-metric view uses the **same 412 games** with both a saved Vegas spread and probability. The 419-game spread-paired scorecard adds seven games lacking a saved Vegas probability; the full 420-game scorecard adds one game lacking a spread. Use the separate metric denominators in those files when comparing to Vegas.

| Metric | Targeted five | F19 six-model consensus | Vegas |
| --- | ---: | ---: | ---: |
| SU, 419 spread-paired games | 317/419 | 314/419 | 317/419 |
| ATS, 419 spread-paired games | 204–203–12 (50.12%) | 213–194–12 (52.33%) | N/A |
| Brier, 412 fully paired games ↓ | 0.167187 | 0.167427 | 0.166405 |
| Margin MAE, 419 spread-paired games ↓ | 11.257012 | 11.249956 | 11.184964 |

The target is 317/420 straight up on its full available cohort. Its paired Brier is +0.000782 relative to Vegas (positive is worse), and its spread-paired margin MAE is +0.072048 points relative to Vegas (positive is worse). Vegas has no ATS record against its own line. At hypothetical -110 odds, ATS needs 52.38% wins to break even; the target's 50.12% is below that threshold.

`cumulative_fully_paired_412.csv` and the PNG track the same 412-game universe through each week, comparing the target, F19 consensus, and Vegas. `weekly_fully_paired_412.csv` gives the separate weekly increments. The 9-row scope scorecards also include F20 consensus and each of the five selected cells. These are aggregate public results; source game-level predictions remain private.

**Interpretation:** The five cells were selected after inspecting their 2026 results, so this is an in-sample exploratory combination. F20 Week 7 was reconstructed with models fitted after those games; the source archive was obtained after games and original expert and market quote times remain unverified. Some F0–F19 weeks and Week 8 F20 margins also have retrospective timing limits. This result cannot establish prospective superiority or a realizable betting edge.
