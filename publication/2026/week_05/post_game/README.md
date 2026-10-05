# TDNet 2026 Week 5 postgame package

This Week 5 retrospective contains the full Week 3-style postgame figure suite. Reader-facing figures live in `figures/`, source tables in `tables/`, and the paper-oriented scientific rosters in `scientific/`. It covers completed Weeks 0–5, including 267 scored game predictions for the 33-model operational roster and matching cumulative scientific cohorts. Week 4 was reconstructed only from its immutable frozen bundle and completed cached results; the completeness record is in `data/publication/2026/weekly_operations/week_04/postgame_results_completeness.json`.

The comparison reference is **AP Top 25 (Post-Week 5)** and is labeled that way in every retained comparison. Poll points and consensus power ratings remain independent.

The consensus bankroll figures compare flat $10 ATS and moneyline bets for the margin-wide, F0–F6 scientific, and full F0–F8 scientific consensuses. ATS uses an explicit -110 assumption because CFBD does not publish spread-side prices; moneyline returns use the best available CFBD quote.

Separate confidence-scaled figures use model support for ATS confidence and picked-team win probability for moneyline confidence, scaling linearly from $0 at 49.9% to $25 at 100%.

Season-to-date confidence threshold sweeps show flat-$10 profit and ROI at each minimum confidence cutoff. These are descriptive in-sample diagnostics, not forward-validated betting rules.

Separate cumulative model-calibration overlays reproduce the historical diagnostic: binned predicted home-win probability versus observed home-win rate, one line per individual frozen model, with probability density below.

Weekly and cumulative margin parity plots label all four TDNet home/away pick and realized home/away winner quadrants.

The season-to-date scorecards rank every operational model and both scientific cohorts. The cumulative charts include the evaluation-only Vegas baselines. Week 6 matchup predictions and draft social copy are separately stored in `publication/2026/week_06/pre_game`.
