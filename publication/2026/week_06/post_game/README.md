# TDNet 2026 Week 6 postgame package

This package was completed on October 21. Week 6 game scoring uses the saved pregame freeze. The Week 7 state polls were reconstructed on October 21 from Week 6 features; their provider field timing before Week 7 was not independently verified. Rankings are retrospective, and no Week 7 picks were issued at the time.

This lean Sunday release mirrors the pregame layout: reader-facing wide-margin output lives in `figures/`, its source tables live in `tables/`, and the paper-oriented roster lives in `scientific/`. Per-model PNGs and redundant diagnostic renders are not retained.

The comparison reference is **Week 7 AP Top 25** and is labeled that way in every retained comparison. Poll points and consensus power ratings remain independent.

The consensus bankroll figures compare flat $10 ATS and moneyline bets for the margin-wide, F0–F6 scientific, and full F0–F8 scientific consensuses. ATS uses an explicit -110 assumption because CFBD does not publish spread-side prices; moneyline returns use the best available CFBD quote; historical quote timing is not fully verified.

Separate confidence-scaled figures use model support for ATS confidence and picked-team win probability for moneyline confidence, scaling linearly from $0 at 49.9% to $25 at 100%.

Season-to-date confidence threshold sweeps show flat-$10 profit and ROI at each minimum confidence cutoff. These are descriptive in-sample diagnostics, not forward-validated betting rules.

Separate cumulative model-calibration overlays reproduce the historical diagnostic: binned predicted home-win probability versus observed home-win rate, one line per individual frozen model, with probability density below.

Weekly and cumulative margin parity plots label all four TDNet home/away pick and realized home/away winner quadrants.

Week 7 had no timely pregame package. Its after-the-fact replay is separately labeled in `publication/2026/week_07/post_game/retrospective_replay`.
