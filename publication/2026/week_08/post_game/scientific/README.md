# TDNet scientific postgame package

The performance files score the immutable 42-model pregame scientific predictions; no prediction model was rerun. The ranking files use the refreshed postgame team state.

- `scientific_tdnet_top25.csv` is the model-ballot poll.
- `scientific_full_ballots.csv` contains one complete all-team ballot per scientific model.
- `scientific_consensus_power_rankings.csv` independently aggregates predicted margin versus the constructed average FBS team.
- `scientific_poll_power_divergence.json` records differences without forcing the two rankings to agree.
- `scientific_cumulative_model_scorecard.csv` and `.png` rank the season-to-date scientific roster and include the evaluation-only Vegas baselines. In the full cohort, F7 remains in the rolling audit CSV but is omitted from scorecards and figures because it is the market-only tier.
- No next-week matchup predictions are generated in this directory.
