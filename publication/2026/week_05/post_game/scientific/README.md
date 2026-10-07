# TDNet scientific postgame package

The performance files score the immutable 42-model pregame scientific predictions; no prediction model was rerun. The ranking files use the refreshed postgame team state.

- `scientific_tdnet_top25.csv` is the model-ballot poll.
- `scientific_full_ballots.csv` contains one complete all-team ballot per scientific model.
- `scientific_consensus_power_rankings.csv` independently aggregates predicted margin versus the constructed average FBS team.
- `scientific_poll_power_divergence.json` records differences without forcing the two rankings to agree.
- `scientific_cumulative_model_scorecard.csv` and `.png` rank the season-to-date scientific roster and include the evaluation-only Vegas baselines. In the full cohort, F7 remains in the rolling audit CSV but is omitted from scorecards and figures because it is the market-only tier.
- No next-week matchup predictions are generated in this directory.

The separate retrospective `current_season_f0_f19/` research export scores all 120 F0–F19 scientific model × fingerprint cells through Week 5 and contains per-game, weekly, cumulative, and consensus tables. Its eight unavailable F19 market-snapshot games are explicit; see its README for coverage and interpretation.

The F0–F19 research export also contains `scientific_2026_full_f0_f19_cumulative_performance.png`, the three-panel Brier/winner/ATS curve figure, and `scientific_2026_full_f0_f19_cumulative_model_scorecard.png`, a 120-model table ranked on the common 263-game cohort. The table's companion CSV preserves unrounded values. These are research figures; the published 42-model scorecard above remains intact.

The full-roster scorecard's `Upset calls (hit%)` column reports correct underdog winner picks divided by all underdog winner picks against the archived spread favorite, with the count shown beside the percentage.
