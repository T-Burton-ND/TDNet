# TDNet 2026 Week 3 postgame package

This lean Sunday release mirrors the pregame layout: reader-facing wide-margin output lives in `figures/`, its source tables live in `tables/`, and the paper-oriented roster lives in `scientific/`. Per-model PNGs and redundant diagnostic renders are not retained.

The comparison reference is **AP Top 25 (Post-Week 3)** and is labeled that way in every retained comparison. Poll points and consensus power ratings remain independent.

The consensus bankroll figures compare flat $10 ATS and moneyline bets for the margin-wide, F0–F6 scientific, and full F0–F8 scientific consensuses. ATS uses an explicit -110 assumption because CFBD does not publish spread-side prices; moneyline returns use the best available CFBD quote. Historical moneyline prices are labeled retrospective when no pregame snapshot was retained.

Separate confidence-scaled figures use model support for ATS confidence and picked-team win probability for moneyline confidence. Stakes scale linearly from $0 at 49.9% to $25 at 100%.

Season-to-date confidence threshold sweeps show flat-$10 profit and ROI at each minimum confidence cutoff. These are descriptive in-sample diagnostics, not forward-validated betting rules.

Separate cumulative model-calibration overlays reproduce the historical diagnostic: binned predicted home-win probability versus observed home-win rate, one line per individual frozen model, with probability density below.

Weekly and cumulative margin parity plots label all four TDNet home/away pick and realized home/away winner quadrants.

Week 4 matchup predictions are intentionally absent; those belong in `publication/2026/week_04/pre_game`.
