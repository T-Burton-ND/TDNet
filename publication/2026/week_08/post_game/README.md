# TDNet 2026 Week 8 postgame package

## Three roster views

- **Margin-wide publication roster:** `figures/` and `tables/`. All 33 saved model forecasts are scored on the 53 games remaining at the October 21 freeze; the refreshed Week 9 state produces 33 ranking ballots.
- **Original frozen scientific roster:** `scientific/`. All 42 F0–F6 × M1–M5/M10 forecasts are scored on those same 53 games; the refreshed Week 9 state produces 42 ranking ballots.
- **Expanded scientific roster:** `scientific/expanded_f0_f20/`. Its 126 F0–F20 × M1–M5/M10 scorecards, violin and matrix figures use the 49 games common to F20. F0–F19 also have a separate 53-game scorecard. The expanded research power rankings, Top 25, and Top 10 use 353 shared games from Weeks 1–6 and 8, covering 136 FBS teams. They are descriptive fits to saved predicted margins, not live Week 9 team states.

The expanded roster's [`full_roster_scorecards/`](scientific/expanded_f0_f20/full_roster_scorecards/README.md) adds Week 8 and season-to-date tables with all 126 model cells, a Vegas reference, 21 fingerprint consensuses, and six architecture consensuses. Both latest-available and fully matched cohorts are provided, with game and probability denominators on every row.

The [three-roster graphics checklist](three_roster_graphics/README.md) maps 27 postgame graphic types to each roster and adds the missing Week 8 views. Its betting figures compare the three roster consensuses on the same 49 games. The new F0–F6 cumulative views recover its frozen Week 6 cells from the saved F0–F8 component; the expanded F0–F20 cumulative views use 353 shared games from Weeks 1–6 and 8. The checklist records each graphic's cohort and source status.

The `scientific/full_f0_f8/` directory is an F0–F8 diagnostic component used by the existing calibration and bankroll renderer. It is part of the expanded research path, not a fourth roster. `three_roster_comparison/` directly compares the three roster consensuses on the same 49 games, including SU, ATS, Brier, margin error, and flat-stake bankroll views.

CFBD certified final scores for every one of the 53 frozen games. The Week 9 AP Top 25 came from the refreshed CFBD rankings feed. The [Prediction Tracker season results](https://www.thepredictiontracker.com/ncaaresults.php) extend through October 25, while its [Last Week view](https://www.thepredictiontracker.com/ncaaresults.php?orderby=wpct+desc&type=3&year=26) still reports results through October 18; archived game-level pregame captures, rather than that aggregate page, supply F20 inputs.

F20's 49 probability forecasts were issued as late pregame research revisions on October 23, after the official weekly cutoff. The six margin models were fitted before those games kicked off, then applied after the games to the archived pregame feature vectors. Their margin and ATS grades are research evaluations, not issued margin picks. F20 has no Week 7 expert forecast coverage, so the cumulative matched comparison excludes Week 7.

This lean Sunday release mirrors the pregame layout: reader-facing wide-margin output lives in `figures/`, aggregate scorecards live in `tables/`, and the paper-oriented roster lives in `scientific/`. Game-level predictions, raw provider responses, and ballots stay in ignored local data paths. Per-model PNGs and redundant diagnostic renders are not retained.

The comparison reference is **Week 9 AP Top 25** and is labeled that way in every retained comparison. Poll points and consensus power ratings remain independent.

The earlier figures in `figures/` compare flat $10 ATS and moneyline bets for the margin-wide, F0–F6 scientific, and F0–F8 component consensuses. The new three-roster graphics checklist adds matched Week 8 bankroll, scaled-stake, and threshold views for margin-wide, F0–F6, and F0–F20. ATS uses an explicit -110 assumption because CFBD does not publish spread-side prices; moneyline returns use the saved pregame quote.

Separate confidence-scaled figures use model support for ATS confidence and picked-team win probability for moneyline confidence, scaling linearly from $0 at 49.9% to $25 at 100%.

Season-to-date confidence threshold sweeps show flat-$10 profit and ROI at each minimum confidence cutoff. These are descriptive in-sample diagnostics, not forward-validated betting rules.

Separate cumulative model-calibration overlays reproduce the historical diagnostic: binned predicted home-win probability versus observed home-win rate, one line per individual frozen model, with probability density below.

Weekly and cumulative margin parity plots label all four TDNet home/away pick and realized home/away winner quadrants.

Week 9 matchup predictions are intentionally absent; those belong in `publication/2026/week_09/pre_game`.
